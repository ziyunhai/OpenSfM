# pyre-strict
"""Raw + clean depthmaps (pipeline phases 1–2).

Phase 1 runs multi-scale PatchMatch with geometric consistency per cluster on
GPU; phase 2 cleans each raw depthmap against an N-hop neighbour set on GPU.
GPU device discovery and the raw-phase work-queue/thread loop live here.
"""

import gc
import logging
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from opensfm import context, log, pydense, pymap, types
from opensfm.dataset import UndistortedDataSet

from .common import scale_down_image, select_cluster_views, _save_depthmap_as_ply

logger: logging.Logger = logging.getLogger(__name__)

# Maximum concurrent clusters running on a single device.
_MAX_PARALLEL_PER_DEVICE: int = 1


def discover_gpu_devices(config: Dict[str, Any]) -> List[int]:
    """Return the ordered list of usable GPU device indices.

    Intel devices are skipped when ``opencl_ignore_intel_device`` is set.
    Returns an empty list when no OpenCL GPU is available.
    """
    num_devices = 0
    if pydense.DepthmapClusterEstimator.is_available():
        num_devices = pydense.DepthmapClusterEstimator.num_devices()
    if num_devices == 0:
        logger.warning("No OpenCL devices found — cannot compute depthmaps")
        return []

    ignore_intel_device = config.get("opencl_ignore_intel_device", True)

    # Build ordered device list: GPUs only
    gpu_devs: List[int] = []
    for di in range(num_devices):
        dname = pydense.DepthmapClusterEstimator.device_name(di)
        if "Intel(R)" in dname and ignore_intel_device:
            logger.info(
                f"  Device {di}: {dname} (ignored by config, skipping)"
            )
            continue
        dmem = pydense.DepthmapClusterEstimator.device_memory_bytes(di)
        if pydense.DepthmapClusterEstimator.device_is_gpu(di):
            logger.info(
                f"  Device {di}: {dname} ({dmem / (1024**3):.1f} GiB)"
            )
            gpu_devs.append(di)

    total_slots = _MAX_PARALLEL_PER_DEVICE * len(gpu_devs)
    logger.info(
        f"{num_devices} device(s), {_MAX_PARALLEL_PER_DEVICE} GPU slots/device → {total_slots} total parallel cluster(s)"
    )
    return gpu_devs


def run_raw_depthmaps(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    best_neighbors: Dict[str, List[pymap.Shot]],
    clusters: List[List[str]],
    depth_ranges: Dict[str, Tuple[float, float]],
    processable: List[str],
    device_order: List[int],
) -> Set[str]:
    """Phase 1: per-cluster raw depthmaps on GPU.

    A shared work queue is consumed independently by per-device workers; each
    GPU runs ``_MAX_PARALLEL_PER_DEVICE`` threads that pull clusters on demand.
    Returns the set of processed shot ids.
    """
    processed_shots: Set[str] = set()
    errors: List[Tuple[int, Exception]] = []
    results_lock = threading.Lock()

    work_q: queue.Queue[Tuple[int, List[str]]] = queue.Queue()
    for ci, cluster_shots in enumerate(clusters):
        work_q.put((ci, cluster_shots))

    def _device_loop(device_idx: int) -> None:
        """Pull clusters from the shared queue until empty."""
        log.setup()
        while True:
            try:
                batch_num, cluster_shots = work_q.get_nowait()
            except queue.Empty:
                return
            try:
                if all(data.raw_depthmap_exists(s) or data.clean_depthmap_exists(s) for s in cluster_shots):
                    logger.info(
                        f"Cluster {batch_num}: all {len(cluster_shots)} raw depthmaps exist, skipping"
                    )
                    with results_lock:
                        processed_shots.update(cluster_shots)
                    continue
                context.log_memory(
                    f"raw cluster {batch_num} start (dev {device_idx})"
                )
                ret = _run_cluster_raw(
                    data, graph, reconstruction, best_neighbors,
                    cluster_shots, depth_ranges, batch_num,
                    device_idx=device_idx,
                )
                context.log_memory(f"raw cluster {batch_num} end")
                with results_lock:
                    processed_shots.update(ret)
                logger.info(
                    f"Cluster {batch_num} raw done on dev {device_idx} ({len(processed_shots)}/{len(processable)} views)"
                )
            except Exception as exc:
                logger.error(
                    f"Cluster batch {batch_num} failed", exc_info=True,
                )
                with results_lock:
                    errors.append((batch_num, exc))

    # Launch per-device threads.
    threads: List[threading.Thread] = []
    for di in device_order:
        per_device = _MAX_PARALLEL_PER_DEVICE
        for wi in range(per_device):
            t = threading.Thread(
                target=_device_loop, args=(di,),
                name=f"dev{di}-w{wi}", daemon=True,
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    if errors:
        logger.warning(f"{len(errors)} cluster(s) failed in raw phase:")
        for batch_num, exc in errors:
            logger.warning(f"  batch {batch_num}: {exc}")

    return processed_shots


def _setup_cluster_params(
    cluster: Any,
    config: Dict[str, Any],
    debug_dir: str = "",
) -> None:
    """Apply shared PatchMatch parameters to a cluster object."""
    cluster.set_max_iterations(config["depthmap_max_iterations"])
    cluster.set_patch_size(config["depthmap_patch_size"])
    cluster.set_max_image_size(config["depthmap_max_image_size"])
    cluster.set_sigma_spatial(config["depthmap_sigma_spatial"])
    cluster.set_sigma_color(config["depthmap_sigma_color"])
    cluster.set_top_k(config["depthmap_num_matching_views"])
    cluster.set_use_census(config["depthmap_use_census"])
    cluster.set_hierarchy_levels(config["depthmap_hierarchy_levels"])
    cluster.set_smooth_weight(config["depthmap_smooth_weight"])
    cluster.set_anchor_views(config["depthmap_anchor_views"])
    cluster.set_checkerboard_filter(config["depthmap_checkerboard_filter"])
    cluster.set_speckle_min_size(config["depthmap_speckle_min_size"])
    cluster.set_gap_max_size(config["depthmap_gap_max_size"])
    cluster.set_geom_consistency_weight(
        config["depthmap_geom_consistency_weight"]
    )


def _run_cluster_raw(
    data: UndistortedDataSet,
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    best_neighbors: Dict[str, List[pymap.Shot]],
    cluster_shot_ids: List[str],
    depth_ranges: Dict[str, Tuple[float, float]],
    batch_num: int,
    device_idx: int = 0,
) -> List[str]:
    """Run raw depthmaps (no cleaning, no fusion) for a cluster.

    Saves raw depthmaps to disk immediately and returns the list of
    processed shot IDs.  This avoids accumulating all cluster results
    in memory.
    """
    config = data.config

    if not pydense.DepthmapClusterEstimator.is_available():
        raise RuntimeError(
            "PatchMatch requires OpenCL but no OpenCL device was found."
        )

    logger.info(
        f"Cluster {batch_num} raw ({len(cluster_shot_ids)} views) on device {device_idx}: {', '.join(cluster_shot_ids)}"
    )

    num_neighbors: int = config["depthmap_num_matching_views"]
    max_size: int = config["depthmap_max_image_size"]

    cluster = pydense.DepthmapClusterEstimator()
    debug_dir = ""
    if config.get("depthmap_save_debug_ply", False) and config.get("depthmap_segmentation_enabled", False):
        debug_dir = data._depthmap_path()
        os.makedirs(debug_dir, exist_ok=True)
    _setup_cluster_params(cluster, config, debug_dir=debug_dir)
    cluster.set_device(device_idx)

    # -- Build cluster: add ref views and their source views --
    ref_idx_map: Dict[str, int] = {}
    ref_nbr_ids: Dict[int, List[str]] = {}

    for ref_id in cluster_shot_ids:
        neighbors = best_neighbors[ref_id]
        shot = reconstruction.shots[ref_id]
        min_d, max_d = depth_ranges[ref_id]

        # Load and prepare the reference image (grayscale + color).
        ref_gray = cv2.cvtColor(
            data.load_undistorted_image(shot.id), cv2.COLOR_RGB2GRAY)
        oh, ow = ref_gray.shape
        if ow > max_size or oh > max_size:
            factor = max_size / max(ow, oh)
            w, h = int(ow * factor), int(oh * factor)
        else:
            w, h = ow, oh
        image = scale_down_image(ref_gray, w, h)
        K = shot.camera.get_K_in_pixel_coordinates(w, h)
        R = shot.pose.get_rotation_matrix()
        t = shot.pose.translation

        idx = cluster.begin_ref_view(K, R, t, image, min_d, max_d)
        ref_idx_map[ref_id] = idx

        # SfM planar prior.
        if config["depthmap_sfm_planar_prior"]:
            sfm_points: List[NDArray] = []
            for track in graph.get_shot_observations(shot.id):
                if track in reconstruction.points:
                    sfm_points.append(reconstruction.points[track].coordinates)
            if sfm_points:
                cluster.set_sfm_points(np.array(sfm_points, dtype=np.float64))

        # Add source views.
        nbr_ids: List[str] = [ref_id]
        for s in neighbors[1: num_neighbors + 1]:
            assert s.camera.projection_type == "perspective"
            src_gray = cv2.cvtColor(
                data.load_undistorted_image(s.id), cv2.COLOR_RGB2GRAY)
            soh, sow = src_gray.shape
            if sow > max_size or soh > max_size:
                factor = max_size / max(sow, soh)
                sw, sh = int(sow * factor), int(soh * factor)
            else:
                sw, sh = sow, soh
            src_img = scale_down_image(src_gray, sw, sh)
            sK = s.camera.get_K_in_pixel_coordinates(sw, sh)
            sR = s.pose.get_rotation_matrix()
            st = s.pose.translation
            cluster.add_source_view(sK, sR, st, src_img)
            nbr_ids.append(s.id)

        ref_nbr_ids[idx] = nbr_ids

    context.log_memory("cluster data loaded")

    # -- Register geometric consistency links --
    for ref_id, idx in ref_idx_map.items():
        nbr_ids = ref_nbr_ids[idx]
        for other_id, other_idx in ref_idx_map.items():
            if other_id == ref_id:
                continue
            if other_id in nbr_ids:
                src_pos = nbr_ids.index(other_id) - 1
                if src_pos >= 0:
                    cluster.add_geom_link(idx, src_pos, other_idx)

    # -- Run raw depthmaps only --
    results = cluster.run()
    cluster.clear()

    context.log_memory("cluster raw depthmaps computed")

    # -- Save raw depthmaps to disk in parallel --
    processed: List[str] = []
    save_items = []
    max_cost: float = config["depthmap_max_cost"]
    for i, ref_id in enumerate(cluster_shot_ids):
        depth, normal, cost, confidence = results[i]
        depth = np.asarray(depth, dtype=np.float32)
        normal = np.asarray(normal, dtype=np.float32)
        cost = np.asarray(cost, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)

        # Sanitize NaN/Inf depths: zero out invalid pixels.
        nan_mask = ~np.isfinite(depth)
        n_nan = int(np.count_nonzero(nan_mask))
        if n_nan > 0:
            logger.warning(
                f"Raw depthmap {ref_id}: {n_nan} NaN/Inf pixels zeroed"
            )
            depth[nan_mask] = 0.0
            # Zero out corresponding normals too.
            nan_mask_3 = np.broadcast_to(
                nan_mask[:, :, np.newaxis], normal.shape
            )
            normal[nan_mask_3] = 0.0

        # Zero out textureless / low-confidence pixels
        if max_cost > 0:
            high_cost = cost > max_cost
            n_high = int(np.count_nonzero(high_cost))
            if n_high > 0:
                depth[high_cost] = 0.0
                logger.debug(
                    f"Raw depthmap {ref_id}: {n_high} pixels zeroed "
                    f"(cost > {max_cost:.2f})"
                )

        nghbrs = [n.id for n in best_neighbors.get(ref_id, [])]
        score = cost
        nghbr = np.zeros((0,), dtype=np.int32)
        save_items.append((ref_id, depth, normal, score, nghbr, nghbrs,
                           confidence))
        processed.append(ref_id)

    data.save_raw_depthmaps_parallel(save_items)

    # Export each raw depthmap as a binary PLY.
    if config["depthmap_save_debug_ply"]:
        for ref_id, dep, _nrm, _sc, _ng, _ngs, _conf in save_items:
            _save_depthmap_as_ply(data, reconstruction, ref_id, dep, "raw")

    del save_items
    del cluster
    del results
    gc.collect()

    context.log_memory("cluster raw depthmaps saved to disk")

    logger.info(
        f"Cluster {batch_num}: raw {len(processed)} views saved to disk"
    )
    return processed


def clean_depthmaps(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    clusters: List[List[str]],
    all_neighbors: Dict[str, List[pymap.Shot]],
    processable: List[str],
    config: Dict[str, Any],
    max_neighbors: int = 16,
    device_order: List[int] | None = None,
) -> None:
    """Clean all raw depthmaps on GPU using N-hop expanded neighbors.

    Clusters are distributed across GPU devices via a shared work queue,
    each device running ``_MAX_PARALLEL_PER_DEVICE`` worker threads.
    Each worker creates its own ``GPUDepthmapCleaner`` bound to one device
    so multiple GPUs work in parallel.
    """
    available = [
        sid for sid in processable if data.raw_depthmap_exists(sid)
    ]
    if not available:
        return

    available_set: Set[str] = set(available)
    if device_order is None:
        device_order = [0]

    def _gpu_clean_and_save_cluster(
        cleaner: Any,
        cluster_shots: List[str],
    ) -> int:
        """Load views, GPU-clean, save results, release everything.

        The *cleaner* is reused across batches (kernel compiled once);
        ``clear()`` is called at entry to discard the previous batch's
        views.  All loaded data is released before returning so memory
        stays flat between batches.

        The C++ cleaner has a hard limit of 16 neighbor views per ref
        (kMaxCleanSources).  We select the best ``max_neighbors``
        neighbors per ref from ``all_neighbors`` (already ranked by
        overlap quality), and only load the union of needed views.
        """
        cleaner.clear()

        # Bounded, coverage-ranked neighbour set: cap the TOTAL views loaded per
        # batch (peak RAM blows up on spread clusters otherwise); per-ref lists
        # stay capped at max_neighbors (the C++ kMaxCleanSources limit).
        ordered_shots, per_ref_nbrs = select_cluster_views(
            cluster_shots, all_neighbors, available_set,
            max_total=config["depthmap_max_cluster_views"],
            per_ref_cap=max_neighbors,
        )
        shot_to_idx: Dict[str, int] = {
            sid: i for i, sid in enumerate(ordered_shots)
        }

        logger.info(
            f"GPU clean cluster: {len(cluster_shots)} ref shots, "
            f"{len(ordered_shots) - len(cluster_shots)} neighbors, "
            f"{len(ordered_shots)} total"
        )

        # ── Load raw depthmaps in parallel & feed the GPU cleaner ──
        raw_normals: Dict[str, NDArray] = {}
        raw_confidence: Dict[str, Optional[NDArray]] = {}
        cluster_shots_set = set(cluster_shots)

        loaded = data.load_raw_depthmaps_parallel(ordered_shots)
        # Keep raw depths for potential pass-2 reload.
        raw_depths: Dict[str, NDArray] = {}
        for sid, (depth, normal, _, _, _, confidence) in zip(
            ordered_shots, loaded
        ):
            raw_depths[sid] = depth
            shot = reconstruction.shots[sid]
            h, w = depth.shape[:2]
            K = shot.camera.get_K_in_pixel_coordinates(w, h)
            R = shot.pose.get_rotation_matrix()
            t = shot.pose.translation
            # Pass normals for ref shots (enables grazing-angle detection).
            if sid in cluster_shots_set and normal is not None:
                # PixelData3f expects (3, H*W) column-major (F-contiguous).
                norm_3n = np.asfortranarray(
                    normal.reshape(-1, 3).T.astype(np.float32)
                )
                cleaner.add_view_with_normal(K, R, t, depth, norm_3n)
            else:
                cleaner.add_view(K, R, t, depth)

            if sid in cluster_shots_set:
                raw_normals[sid] = normal
                raw_confidence[sid] = confidence

        del loaded

        two_pass = config.get("depthmap_carving_two_pass", True)

        # ── PASS 1: consistency-only (carving disabled) ────────────
        if two_pass:
            cleaner.set_max_carved_views(999)  # effectively disable carving
        # else: use configured max_carved_views (single-pass behavior)

        pass1_depths: Dict[str, NDArray] = {}
        use_segment_filter = config.get("depthmap_segmentation_enabled", False)
        for sid in cluster_shots:
            nbr_indices = np.array(
                [shot_to_idx[n] for n in per_ref_nbrs[sid]],
                dtype=np.int32,
            )
            cleaned_depth = cleaner.clean(shot_to_idx[sid], nbr_indices)
            cleaned_arr = np.asarray(cleaned_depth, dtype=np.float32)

            pass1_depths[sid] = cleaned_arr

        if not two_pass:
            # Single-pass: save directly.
            save_items = []
            for sid in cluster_shots:
                norm = raw_normals.pop(sid)
                conf = raw_confidence.pop(sid, None)
                score = np.zeros_like(pass1_depths[sid])
                save_items.append((sid, pass1_depths[sid], norm, score, conf))

            data.save_clean_depthmaps_parallel(save_items)
            if config["depthmap_save_debug_ply"]:
                for sid, cleaned, _norm, _sc, _conf in save_items:
                    _save_depthmap_as_ply(
                        data, reconstruction, sid, cleaned, "clean")
            del raw_normals, raw_depths, pass1_depths
            cleaner.clear()
            gc.collect()
            return len(save_items)

        # ── PASS 2: carving with cleaned neighbor depths ───────────
        # Reload the cleaner with pass-1 cleaned depths for all views.
        # Reference shots use their pass-1 result; neighbor-only shots
        # that were not ref in this cluster keep their raw depth (they
        # may have been cleaned in another cluster — but we don't have
        # their clean result here, so raw is the best approximation).
        cleaner.clear()
        cleaner.set_max_carved_views(config["depthmap_max_carved_views"])

        ordered_shots2 = ordered_shots  # same bounded view set as pass 1
        shot_to_idx2: Dict[str, int] = {
            sid: i for i, sid in enumerate(ordered_shots2)
        }

        for sid in ordered_shots2:
            shot = reconstruction.shots[sid]
            # Use pass-1 cleaned depth if available, otherwise raw.
            depth = pass1_depths.get(sid, raw_depths[sid])
            h, w = depth.shape[:2]
            K = shot.camera.get_K_in_pixel_coordinates(w, h)
            R = shot.pose.get_rotation_matrix()
            t = shot.pose.translation
            if sid in cluster_shots_set and raw_normals.get(sid) is not None:
                norm_3n = np.asfortranarray(
                    raw_normals[sid].reshape(-1, 3).T.astype(np.float32)
                )
                cleaner.add_view_with_normal(K, R, t, depth, norm_3n)
            else:
                cleaner.add_view(K, R, t, depth)

        del raw_depths

        logger.info(
            f"GPU clean cluster pass 2 (carving): "
            f"{len(cluster_shots)} ref shots"
        )

        save_items = []
        for sid in cluster_shots:
            nbr_indices = np.array(
                [shot_to_idx2[n] for n in per_ref_nbrs[sid]],
                dtype=np.int32,
            )
            cleaned_depth = cleaner.clean(shot_to_idx2[sid], nbr_indices)
            cleaned_arr = np.asarray(cleaned_depth, dtype=np.float32)
            norm = raw_normals.pop(sid)
            conf = raw_confidence.pop(sid, None)
            score = np.zeros_like(cleaned_arr)
            save_items.append((sid, cleaned_arr, norm, score, conf))

        del pass1_depths
        data.save_clean_depthmaps_parallel(save_items)

        # Export each clean depthmap as a binary PLY.
        if config["depthmap_save_debug_ply"]:
            for sid, cleaned, _norm, _sc, _conf in save_items:
                _save_depthmap_as_ply(
                    data, reconstruction, sid, cleaned, "clean")

        # Release everything for this batch.
        del raw_normals
        cleaner.clear()
        gc.collect()
        return len(save_items)

    # -- parallel GPU pipeline via shared work queue -----------------------
    cleaned_count = 0
    clean_errors: List[Tuple[int, Exception]] = []
    clean_lock = threading.Lock()

    work_q: queue.Queue[Tuple[int, List[str]]] = queue.Queue()
    for bi, cluster in enumerate(clusters):
        cluster_valid = list(set(cluster) & available_set)
        if not cluster_valid:
            continue
        if all(data.clean_depthmap_exists(sid) for sid in cluster_valid):
            logger.info(
                f"Clean batch {bi} skipped: all shots already cleaned"
            )
            continue
        work_q.put((bi, cluster_valid))

    total_work = work_q.qsize()
    if total_work == 0:
        logger.info("GPU cleaning: nothing to do")
        return

    context.log_memory("starting GPU cleaning batches")

    def _clean_device_loop(device_idx: int) -> None:
        """Pull clusters from the shared queue until empty."""
        log.setup()
        cleaner = pydense.GPUDepthmapCleaner()
        cleaner.set_same_depth_threshold(
            config["depthmap_same_depth_threshold"]
        )
        cleaner.set_min_consistent_views(
            config["depthmap_min_consistent_views"]
        )
        cleaner.set_carving_threshold(
            config["depthmap_carving_threshold"]
        )
        cleaner.set_max_carved_views(
            config["depthmap_max_carved_views"]
        )
        cleaner.set_grazing_cos_threshold(
            config["depthmap_grazing_cos_threshold"]
        )
        cleaner.set_device(device_idx)
        try:
            while True:
                try:
                    bi, cluster_valid = work_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    context.log_memory(
                        f"clean batch {bi} start (dev {device_idx})"
                    )
                    n = _gpu_clean_and_save_cluster(cleaner, cluster_valid)
                    context.log_memory(f"clean batch {bi} end")
                    with clean_lock:
                        nonlocal cleaned_count
                        cleaned_count += n
                    logger.info(
                        f"Clean batch {bi} done ({n} views) on dev {device_idx}, "
                        f"{cleaned_count}/{len(available)} total"
                    )
                except Exception as exc:
                    logger.error(
                        f"Clean batch {bi} failed on dev {device_idx}",
                        exc_info=True,
                    )
                    with clean_lock:
                        clean_errors.append((bi, exc))
        finally:
            del cleaner
            gc.collect()

    # Launch one thread per device (the GPU cleaner holds an OpenCL
    # context that cannot be shared across threads on the same device).
    threads: List[threading.Thread] = []
    for di in device_order:
        t = threading.Thread(
            target=_clean_device_loop, args=(di,),
            name=f"clean-dev{di}", daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if clean_errors:
        logger.warning(
            f"{len(clean_errors)} cluster(s) failed in clean phase:"
        )
        for bi, exc in clean_errors:
            logger.warning(f"  batch {bi}: {exc}")

    logger.info(
        f"GPU cleaning done: {cleaned_count} views",
    )
    context.log_memory("GPU cleaning complete")
