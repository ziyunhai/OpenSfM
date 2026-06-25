# pyre-strict
"""Per-cluster SVO fusion with sub-volume splitting (pipeline phase 3).

Each cluster is augmented with N-hop neighbours, pre-scanned on CPU, split into
GPU-budget-bounded sub-volumes, fused (TSDF) with optional photometric refine,
and extracted to a ``fused_batch_*.ply``.  When DSM is enabled it also rasters
the per-cluster DSM/ortho tile (``dsm_ortho_batch_*.npz``): Pass 1 builds the
DSM geometry, then the corrected DSM drives the Pass-2 ortho colour bake.
The interleaving (recursive split → refine → Pass 1 → Pass 2) is what keeps
peak GPU/CPU memory bounded.
"""

import gc
import logging
import os
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from opensfm import context, log, pydense, pymap, types
from opensfm.dataset import UndistortedDataSet

from .common import (
    _CLUSTER_COLORS,
    _sample_bbox_wireframe,
    _sample_frustum_wireframe,
    scale_down_image,
    select_cluster_views,
)
from .dsm_ortho import (
    _dsm_point_normals,
    _fill_dsm_holes,
    _fill_ortho_holes,
    _ortho_gated_median,
    _shock_dsm_edges,
)

logger: logging.Logger = logging.getLogger(__name__)

# GPU memory model for SVO-fuser retention budgeting — mirrors the C++ layout in svo_opencl.{h,cc}.
# sizeof(GPUVoxelSlot); see svo_opencl.h static_assert
_SVO_SLOT_BYTES: int = 36
# Refine images kept resident for BakeColors, per pixel per view: color RGBA8 (4) + TSDF-rendered depth f32 (4) + clean depth f32 (4).
_SVO_REFINE_IMG_BYTES_PX_VIEW: int = 12
# Transient refine scratch, per slot: grad (4) + grad_w (4) + adam m/v (8).
_SVO_REFINE_SCRATCH_BYTES_SLOT: int = 16

# Coverage safety factor applied to the median per-pixel surface footprint when deriving the "fine" voxel size.
_SAMPLING_COVERAGE_FACTOR: float = 2.0

# SVO voxel resolution levels: voxel = fine_sampling * multiplier. "fine" tracks
# the depthmaps' median surface sampling; "half"/"quarter" coarsen it 2x/4x.
_VOXEL_LEVEL_MULTIPLIER: Dict[str, float] = {
    "fine": 1.0,
    "half": 2.0,
    "quarter": 4.0,
}


class _SubVolume(NamedTuple):
    """Axis-aligned sub-volume for bounded SVO fusion."""
    core_min: NDArray   # (3,) world-space core bbox min
    core_max: NDArray   # (3,) world-space core bbox max
    ext_min: NDArray    # (3,) extended bbox (core + margin)
    ext_max: NDArray    # (3,) extended bbox (core + margin)
    view_ids: Set[str]  # views contributing to this sub-volume


def _prescan_coarse_grid(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    voxel_size: float,
    coarse_factor: int,
    subsample: int = 8,
    clean_cache: Optional[
        Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]]
    ] = None,
    depth_factor: float = 0.0,
) -> Dict[Tuple[int, int, int], Set[int]]:
    """CPU pre-scan: subsample cleaned depthmaps, project to world, bucket.

    For each view, loads the cleaned depth, subsamples, back-projects
    to world coordinates, and records which views contribute to each
    coarse grid cell.

    Args:
        data: dataset for loading cleaned depthmaps.
        view_ids: ordered list of shot IDs to scan.
        reconstruction: for camera parameters.
        voxel_size: SVO fine voxel size.
        coarse_factor: coarse cell = coarse_factor * voxel_size.
        subsample: take every Nth pixel in each dimension.

    Returns:
        Grid mapping coarse cell coordinate to set of view indices
        (indices into ``view_ids``).
    """
    coarse_size = voxel_size * coarse_factor
    inv_coarse = 1.0 / coarse_size
    grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)

    for view_idx, sid in enumerate(view_ids):
        if clean_cache is not None:
            cached = clean_cache.get(sid)
            if cached is None:
                continue
            depth = cached[0]
        else:
            if not data.clean_depthmap_exists(sid):
                continue
            depth, _normal, _score, _conf = data.load_clean_depthmap(sid)
        h, w = depth.shape

        # Subsample pixel coordinates.
        rs = np.arange(0, h, subsample)
        cs = np.arange(0, w, subsample)
        rr, cc = np.meshgrid(rs, cs, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()
        d = depth[rr, cc]
        valid = d > 0
        rr, cc, d = rr[valid], cc[valid], d[valid]
        del depth  # release full depth

        if len(d) == 0:
            continue

        # Depth clamp (robust to grazing): drop samples far beyond this view's typical viewing distance.
        if depth_factor > 0.0:
            med = float(np.median(d))
            if med > 0.0:
                keep = d <= depth_factor * med
                rr, cc, d = rr[keep], cc[keep], d[keep]
                if len(d) == 0:
                    continue

        shot = reconstruction.shots[sid]
        K = shot.camera.get_K_in_pixel_coordinates(w, h)
        R = shot.pose.get_rotation_matrix()
        t_vec = shot.pose.translation

        Kinv = np.linalg.inv(K)
        pts_px = np.vstack(
            [cc.astype(np.float64), rr.astype(np.float64), np.ones(len(d))]
        )
        pts_cam = Kinv @ pts_px * d
        Rinv = R.T
        pts_world = Rinv @ (pts_cam - t_vec[:, None])

        cx = np.floor(pts_world[0] * inv_coarse).astype(np.int32)
        cy = np.floor(pts_world[1] * inv_coarse).astype(np.int32)
        cz = np.floor(pts_world[2] * inv_coarse).astype(np.int32)

        # Unique coarse cells touched by this view.
        keys = np.column_stack([cx, cy, cz])
        unique_keys = np.unique(keys, axis=0)
        for ukey in unique_keys:
            grid[(int(ukey[0]), int(ukey[1]), int(ukey[2]))].add(view_idx)

    return grid


def _split_into_subvolumes(
    grid: Dict[Tuple[int, int, int], Set[int]],
    view_ids: List[str],
    voxel_size: float,
    coarse_factor: int,
    max_voxels: int,
    trunc_factor: float,
) -> List[_SubVolume]:
    """Split occupied coarse cells into sub-volumes fitting GPU budget.

    Uses recursive KD-tree style median-split along the longest axis.
    Each sub-volume gets a margin of ``trunc_factor * voxel_size`` for
    correct TSDF at boundaries.  Views are assigned to a sub-volume if
    they have coarse cells inside its extended (core + margin) bbox.
    """
    if not grid:
        return []

    coarse_size = voxel_size * coarse_factor
    voxels_per_cell = coarse_factor ** 3
    margin_world = trunc_factor * voxel_size
    margin_cells = int(np.ceil(trunc_factor / coarse_factor))
    max_cells = max(1, max_voxels // voxels_per_cell)

    all_coords = np.array(list(grid.keys()), dtype=np.int32)  # (N, 3)
    all_keys = list(grid.keys())

    def _make_subvolume(cell_indices: NDArray) -> _SubVolume:
        """Build a SubVolume from a set of coarse cell indices."""
        cells = all_coords[cell_indices]
        cmin = cells.min(axis=0)
        cmax = cells.max(axis=0)

        core_min = cmin.astype(np.float64) * coarse_size
        core_max = (cmax + 1).astype(np.float64) * coarse_size
        ext_min = (core_min - margin_world).astype(np.float32)
        ext_max = (core_max + margin_world).astype(np.float32)
        core_min = core_min.astype(np.float32)
        core_max = core_max.astype(np.float32)

        # Gather views from all cells in the extended bbox (core + margin).
        ext_cmin = cmin - margin_cells
        ext_cmax = cmax + margin_cells
        sv_views: Set[str] = set()
        for idx in range(len(all_coords)):
            c = all_coords[idx]
            if np.all(c >= ext_cmin) and np.all(c <= ext_cmax):
                for vi in grid[all_keys[idx]]:
                    sv_views.add(view_ids[vi])

        return _SubVolume(
            core_min=core_min,
            core_max=core_max,
            ext_min=ext_min,
            ext_max=ext_max,
            view_ids=sv_views,
        )

    def _split(cell_indices: NDArray) -> List[_SubVolume]:
        if len(cell_indices) <= max_cells:
            return [_make_subvolume(cell_indices)]

        cells = all_coords[cell_indices]
        spans = cells.max(axis=0) - cells.min(axis=0)
        axis = int(np.argmax(spans))

        if spans[axis] == 0:
            # Can't split further — accept over-budget.
            return [_make_subvolume(cell_indices)]

        median = int(np.median(cells[:, axis]))
        left_mask = cells[:, axis] <= median
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            # All cells on one side; force split at midpoint.
            mid = (cells[:, axis].min() + cells[:, axis].max()) // 2
            left_mask = cells[:, axis] <= mid
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                return [_make_subvolume(cell_indices)]

        return (
            _split(cell_indices[left_mask])
            + _split(cell_indices[right_mask])
        )

    return _split(np.arange(len(all_coords)))


def _render_dsm_patch_from_sv(
    fuser: "pydense.SVOFuser",
    sv: "_SubVolume",
    dsm_grid: NDArray,
    owner_grid: NDArray,
    leaf_idx: int,
    origin_x: float,
    origin_y: float,
    gsd: float,
    grid_w: int,
    grid_h: int,
    z_min: float,
    z_max: float,
) -> None:
    """Render the DSM for a leaf's core XY footprint and composite (MAX-z).

    Records per-cell ownership (``owner_grid`` = ``leaf_idx`` where this leaf
    won the height) so Pass 2 can bake the ortho from the post-processed DSM
    using exactly the views that produced each cell.  No colour is baked here.
    """
    # Compute the sub-grid pixel range corresponding to this sub-volume's core.
    col_start = max(0, int(np.floor((sv.core_min[0] - origin_x) / gsd)))
    col_end = min(grid_w, int(np.ceil((sv.core_max[0] - origin_x) / gsd)))
    row_start = max(0, int(np.floor((sv.core_min[1] - origin_y) / gsd)))
    row_end = min(grid_h, int(np.ceil((sv.core_max[1] - origin_y) / gsd)))

    patch_w = col_end - col_start
    patch_h = row_end - row_start
    if patch_w <= 0 or patch_h <= 0:
        return

    patch_origin_x = origin_x + col_start * gsd
    patch_origin_y = origin_y + row_start * gsd

    # Render the DSM for this patch (colour is baked later, in Pass 2).
    dsm_patch, _hillshade, _normals = fuser.render_dsm_ortho(
        patch_origin_x, patch_origin_y, gsd,
        patch_w, patch_h, z_min, z_max,
    )
    dsm_patch = np.asarray(dsm_patch, dtype=np.float32)

    # Composite DSM: MAX-z (highest surface wins) and record ownership.
    valid = ~np.isnan(dsm_patch)
    dst_slice = dsm_grid[row_start:row_end, col_start:col_end]
    higher = valid & (
        np.isnan(dst_slice) | (dsm_patch > dst_slice)
    )
    dst_slice[higher] = dsm_patch[higher]
    owner_grid[row_start:row_end, col_start:col_end][higher] = leaf_idx

    logger.info(
        f"    DSM patch {patch_w}×{patch_h}: "
        f"{int(np.count_nonzero(valid))} valid, "
        f"{int(np.count_nonzero(higher))} updated"
    )


def _pack_coarse_cells(
    cells: NDArray, origin: NDArray, span: NDArray
) -> NDArray:
    """Pack integer coarse-cell coords into unique int64 keys for fast lookup.

    Coordinates are offset by ``origin`` into ``[0, span)`` per axis and folded
    into a single int64.  Cells outside that range get key ``-1`` (so they never
    match a real, non-negative owned key).
    """
    c = cells - origin
    in_range = np.all((c >= 0) & (c < span), axis=1)
    keys = np.full(len(c), -1, dtype=np.int64)
    if np.any(in_range):
        cc = c[in_range]
        keys[in_range] = (cc[:, 0] * span[1] + cc[:, 1]) * span[2] + cc[:, 2]
    return keys


def _compute_cell_ownership(
    data: UndistortedDataSet,
    clusters: List[List[str]],
    reconstruction: types.Reconstruction,
    voxel_size: float,
    coarse_factor: int,
    fusable_set: Set[str],
) -> Dict[Tuple[int, int, int], int]:
    """Assign every occupied coarse cell to one owning cluster (fusion Plan B).

    Each processable shot belongs to exactly one cluster, so prescanning every
    cluster's OWN cleaned depthmaps covers the whole surface exactly once.  A
    coarse cell is *covered* by a cluster when that cluster's own cameras
    project depth into it; the cell is awarded to the covering cluster whose
    camera centroid is nearest (ties broken toward the lower cluster index).

    Restricting each cluster's fusion to the cells it owns then yields disjoint,
    gap-free territories: no two clusters fuse the same cell (no fuse-then-
    discard waste), and every cell is fused by a cluster that can actually see
    it (no holes).  The coarse grid is floored against the world origin with a
    shared cell size, so cell coordinates are consistent across clusters and
    with the per-cluster fusion prescan.

    Returns ``cell_owner`` mapping coarse-cell coord → owning cluster index.
    """
    coarse_size = voxel_size * coarse_factor

    centroids: List[Optional[NDArray]] = []
    for cl in clusters:
        origins = [
            reconstruction.shots[s].pose.get_origin()
            for s in cl if s in reconstruction.shots
        ]
        centroids.append(
            np.mean(origins, axis=0).astype(np.float64) if origins else None
        )

    cell_owner: Dict[Tuple[int, int, int], int] = {}
    cell_best: Dict[Tuple[int, int, int], float] = {}

    for ci, cl in enumerate(clusters):
        own = [s for s in cl if s in fusable_set]
        cen = centroids[ci]
        if not own or cen is None:
            continue

        # Load this cluster's own cleaned depthmaps (bounded — one cluster at a
        # time) and prescan them at the shared coarse resolution.
        loaded = data.load_clean_depthmaps_parallel(own)
        cache: Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]] = {
            sid: (d, n, c) for sid, (d, n, _s, c) in zip(own, loaded)
        }
        del loaded
        grid = _prescan_coarse_grid(
            data, own, reconstruction, voxel_size, coarse_factor,
            clean_cache=cache,
            depth_factor=data.config["dsm_territory_depth_factor"],
        )
        del cache
        if not grid:
            continue

        # Nearest-centroid award (vectorised distance, strict `<` so an earlier
        # / lower-index cluster keeps the cell on ties).
        keys = list(grid.keys())
        cells = np.array(keys, dtype=np.int64)
        centers = (cells.astype(np.float64) + 0.5) * coarse_size
        d2 = np.sum((centers - cen) ** 2, axis=1)
        for key, dist in zip(keys, d2):
            prev = cell_best.get(key)
            if prev is None or dist < prev:
                cell_best[key] = float(dist)
                cell_owner[key] = ci
        del grid

    return cell_owner


def _estimate_sampling_voxel_size(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    subsample: int = 8,
    max_views: int = 200,
) -> float:
    """Median surface sampling distance of the cleaned depthmaps (world units).

    Each valid depth pixel back-projects to a surface footprint of about
    ``depth / focal_px`` — the finest detail the data actually resolves. We take
    each view's median footprint, then the median across views, giving a robust,
    redundancy-agnostic estimate of the surface sampling, scaled by
    ``_SAMPLING_COVERAGE_FACTOR`` to keep ray-driven integration hole-free. Views are
    scanned in parallel and only a subsampled, transient slice of each depthmap
    is held, so peak RAM stays bounded.

    At most ``max_views`` views are sampled (seeded, so the derived size is
    reproducible across runs); the median is stable on a sample this size, so a
    large scene's voxel size costs a bounded scan rather than one per view.
    """
    if len(view_ids) > max_views:
        idx = np.random.default_rng(0).choice(
            len(view_ids), size=max_views, replace=False
        )
        view_ids = [view_ids[i] for i in sorted(idx.tolist())]

    def _view_median(sid: str) -> Optional[float]:
        if not data.clean_depthmap_exists(sid):
            return None
        depth, _normal, _score, _conf = data.load_clean_depthmap(sid)
        h, w = depth.shape
        d = depth[::subsample, ::subsample].ravel()
        d = d[d > 0]
        if d.size == 0:
            return None
        K = reconstruction.shots[sid].camera.get_K_in_pixel_coordinates(w, h)
        focal_px = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
        if focal_px <= 0.0:
            return None
        return _SAMPLING_COVERAGE_FACTOR * float(np.median(d)) / focal_px

    with ThreadPoolExecutor(max_workers=data.config["io_processes"]) as pool:
        medians = [m for m in pool.map(
            _view_median, view_ids) if m is not None]

    if not medians:
        raise RuntimeError(
            "Cannot derive SVO voxel size: no cleaned depthmaps with valid "
            "depth were found."
        )
    return float(np.median(medians))


def _resolve_voxel_size(
    data: UndistortedDataSet,
    view_ids: List[str],
    reconstruction: types.Reconstruction,
    config: Dict[str, Any],
) -> float:
    """Resolve the SVO voxel size from ``depthmap_fusion_svo_voxel_level``.

    The "fine" sampling is measured once over all fusable cleaned depthmaps and
    scaled by the level multiplier (fine=1x, half=2x, quarter=4x). One global
    size keeps the cell-ownership grid, sub-volumes, and DSM raster consistent
    across clusters.
    """
    level = str(config["depthmap_fusion_svo_voxel_level"]).lower()
    if level not in _VOXEL_LEVEL_MULTIPLIER:
        raise ValueError(
            f"Unknown depthmap_fusion_svo_voxel_level '{level}'; expected one "
            f"of {sorted(_VOXEL_LEVEL_MULTIPLIER)}."
        )
    fine = _estimate_sampling_voxel_size(data, view_ids, reconstruction)
    voxel_size = fine * _VOXEL_LEVEL_MULTIPLIER[level]
    logger.info(
        "SVO voxel size: level=%s, fine sampling=%.4fm -> voxel=%.4fm",
        level, fine, voxel_size,
    )
    return voxel_size


def fuse_clusters(
    data: UndistortedDataSet,
    neighbors: Dict[str, List[pymap.Shot]],
    clusters: List[List[str]],
    cluster_bboxes: List[Tuple[NDArray, NDArray]],
    processable: List[str],
    config: Dict[str, Any],
    reconstruction: types.Reconstruction,
    depth_ranges: Dict[str, Tuple[float, float]],
    best_neighbors: Dict[str, List[pymap.Shot]],
) -> None:
    """Fuse cleaned depthmaps per cluster, each using N-hop expanded
    neighbors, producing a ``fused_batch_XXXX.ply`` per cluster.
    Points outside the cluster's bounding box are discarded.

    Cleaned depthmaps and color images are loaded from disk on-demand
    for each cluster and released after the cluster is fused.  Clusters
    are processed in parallel to overlap I/O with CPU fusion.
    """
    fusable = [
        sid for sid in processable if data.clean_depthmap_exists(sid)
    ]
    if not fusable:
        logger.warning("No cleaned views available for fusion.")
        return

    fusable_set: Set[str] = set(fusable)

    # Derive the global SVO voxel size from the depthmaps' surface sampling.
    # One value is shared by every cluster (ownership grid, sub-volumes, DSM).
    voxel_size = _resolve_voxel_size(data, fusable, reconstruction, config)
    logger.info(
        "Fusing %d clusters with %d fusable views at %.4fm voxel size",
        len(clusters), len(fusable), voxel_size,
    )

    # Pre-assign every occupied coarse cell to the nearest covering cluster, so each cluster fuses only its own territory.
    ownership_voxel_size = voxel_size
    ownership_coarse_factor = config["depthmap_fusion_svo_coarse_factor"]
    context.log_memory("fusion: cell-ownership pre-pass start")
    cell_owner = _compute_cell_ownership(
        data, clusters, reconstruction,
        ownership_voxel_size, ownership_coarse_factor, fusable_set,
    )
    n_owners = len(set(cell_owner.values()))
    logger.info(
        f"Territory ownership: {len(cell_owner)} occupied coarse cells "
        f"assigned across {n_owners}/{len(clusters)} clusters"
    )
    context.log_memory("fusion: cell-ownership pre-pass done")

    def _fuse_single_cluster(batch_num: int) -> None:
        """Fuse one cluster: load views from disk, run fuser, save PLY."""
        log.setup()
        cluster_shots = clusters[batch_num]

        # ── Batch-preload helper ──────────────────────────────────────
        def _prepare_views(
            view_ids: List[str],
            clean_cache: Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]],
        ) -> Dict[str, Tuple[
            NDArray, NDArray, NDArray, NDArray,
            NDArray, NDArray, NDArray, Optional[NDArray],
        ]]:
            """Batch-load cleaned depthmaps + color images in parallel.

            Returns a dict mapping shot id → (K, R, t, depth, normal,
            color, vmask, confidence) for every successfully loaded view.
            """
            loadable = [sid for sid in view_ids if sid in clean_cache]
            if not loadable:
                return {}

            # Depthmaps were already decompressed once (shared with the
            # pre-scan); reuse them here instead of loading a second time.
            loaded = [clean_cache[sid] for sid in loadable]

            # Load validity masks in parallel.
            vmask_map = data.load_undistorted_validity_masks_parallel(
                loadable,
            )

            sizes = {
                sid: (depth.shape[1], depth.shape[0])  # (width, height)
                for sid, (depth, _n, _c) in zip(loadable, loaded)
            }

            def _load_scaled_color(sid: str) -> Tuple[str, NDArray]:
                w, h = sizes[sid]
                return sid, scale_down_image(
                    data.load_undistorted_image(sid), w, h
                )

            with ThreadPoolExecutor(
                max_workers=data.config["io_processes"]
            ) as pool:
                color_map = dict(pool.map(_load_scaled_color, loadable))

            result: Dict[str, Tuple[
                NDArray, NDArray, NDArray, NDArray,
                NDArray, NDArray, NDArray, Optional[NDArray],
            ]] = {}
            for sid, (depth, normal, confidence) in zip(
                loadable, loaded,
            ):
                h, w = depth.shape[:2]

                shot = reconstruction.shots[sid]
                K = shot.camera.get_K_in_pixel_coordinates(w, h)
                R = shot.pose.get_rotation_matrix()
                t = shot.pose.translation

                combined = depth.ravel() > 0

                if sid in depth_ranges:
                    _, max_d = depth_ranges[sid]
                    extreme = depth.ravel() > max_d
                    n_ext = int(np.count_nonzero(extreme & combined))
                    if n_ext > 0:
                        logger.warning(
                            "Shot %s: masking %d pixels with depth > %.1f",
                            sid, n_ext, max_d,
                        )
                    combined &= ~extreme

                if sid in vmask_map:
                    vmask_raw = vmask_map[sid]
                    mh, mw = vmask_raw.shape[:2]
                    if (mh, mw) != (h, w):
                        vmask_raw = cv2.resize(
                            vmask_raw, (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    combined &= vmask_raw.ravel() > 0

                reject = ~combined.reshape(h, w)
                n_rejected = int(np.count_nonzero(reject & (depth > 0)))
                if n_rejected > 0:
                    logger.info(
                        "Shot %s: combined mask rejected %d pixels",
                        sid, n_rejected,
                    )
                depth[reject] = 0.0

                vmask = (combined.reshape(h, w).astype(np.uint8)) * 255

                color = color_map[sid]  # pre-scaled to (w, h) at load time

                result[sid] = (
                    K, R, t, depth, normal, color, vmask, confidence,
                )

            return result

        # ── SVO fusion with sub-volume splitting ─────────────────
        # voxel_size is resolved once for the whole run (see fuse_clusters top).
        trunc_factor_cfg = config["depthmap_fusion_svo_trunc_factor"]
        max_voxels = config["depthmap_fusion_svo_max_voxels"]
        coarse_factor = config["depthmap_fusion_svo_coarse_factor"]
        n_augment = config["depthmap_fusion_svo_augment_neighbors"]

        # Step 1: bounded neighbour augmentation — coverage-ranked and capped at
        # depthmap_max_cluster_views total so a spread cluster can't load a huge
        # disjoint neighbour set.
        augmented, _ = select_cluster_views(
            cluster_shots, neighbors, fusable_set,
            max_total=config["depthmap_max_cluster_views"],
            per_ref_cap=n_augment,
        )
        n_extra = len(augmented) - len(cluster_shots)
        logger.info(
            f"Cluster {batch_num}: {len(cluster_shots)} shots + "
            f"{n_extra} augmented = {len(augmented)} total"
        )

        # Decompress each augmented view's cleaned depthmap ONCE (parallel)
        loadable_aug = [s for s in augmented if data.clean_depthmap_exists(s)]
        _loaded_dm = data.load_clean_depthmaps_parallel(loadable_aug)
        clean_cache: Dict[str, Tuple[NDArray, NDArray, Optional[NDArray]]] = {
            sid: (d, n, c)
            for sid, (d, n, _s, c) in zip(loadable_aug, _loaded_dm)
        }
        del _loaded_dm
        context.log_memory(f"cluster {batch_num}: preloaded cleaned depthmaps")

        # Step 2: Pre-scan on CPU (subsampled depth projection).
        grid = _prescan_coarse_grid(
            data, augmented, reconstruction,
            voxel_size, coarse_factor, clean_cache=clean_cache,
            depth_factor=config["dsm_territory_depth_factor"],
        )
        coarse_size = voxel_size * coarse_factor
        logger.info(
            f"Cluster {batch_num}: pre-scan found {len(grid)} "
            f"coarse cells (cell_size={coarse_size:.3f}m)"
        )
        context.log_memory(f"cluster {batch_num}: pre-scan grid built")

        n_cells_all = len(grid)
        grid = {
            cell: views for cell, views in grid.items()
            if cell_owner.get(cell) == batch_num
        }
        logger.info(
            f"Cluster {batch_num}: territory owns {len(grid)}/{n_cells_all} "
            f"occupied coarse cells"
        )
        if not grid:
            logger.info(
                f"Cluster {batch_num}: owns no cells, nothing to fuse"
            )
            gc.collect()
            return

        # Owned-cell lookup for the extraction clip.
        owned_cells = np.array(list(grid.keys()), dtype=np.int64)
        owned_origin = owned_cells.min(axis=0)
        owned_span = owned_cells.max(axis=0) - owned_origin + 1
        owned_keys = np.sort(
            _pack_coarse_cells(owned_cells, owned_origin, owned_span)
        )
        inv_coarse = 1.0 / coarse_size

        # Step 3: Split into sub-volumes.
        subvolumes = _split_into_subvolumes(
            grid, augmented, voxel_size, coarse_factor,
            max_voxels, trunc_factor_cfg,
        )
        logger.info(
            f"Cluster {batch_num}: split into "
            f"{len(subvolumes)} sub-volume(s)"
        )

        # Step 4: Fuse each sub-volume independently.
        #         If the GPU counting pass reveals more voxels than
        #         the budget, recursively split along the longest
        #         core axis.
        all_points: List[NDArray] = []
        all_normals: List[NDArray] = []
        all_colors: List[NDArray] = []
        # Leaf sub-volumes actually fused (post-split), for the Pass 2 ortho
        # bake; owner_grid[cell] = index into `leaves` of the leaf that won the
        # cell's height (MAX-z), so each cell is re-coloured by the views that
        # produced it.

        leaves: List["_SubVolume"] = []
        # Pass-1 fusers kept alive (bounded) for the Pass-2 ortho bake.
        leaf_fusers: List[Optional["pydense.SVOFuser"]] = []
        max_reuse_fusers = config[
            "depthmap_fusion_svo_bake_reuse_max_fusers"
        ]
        # VRAM-aware retention budget: a retained fuser keeps its GPU hash table (capacity * slot) plus the refine images BakeColors needs resident
        reuse_vram_budget = int(
            pydense.DepthmapClusterEstimator.device_memory_bytes(0)
            * config["depthmap_fusion_svo_bake_reuse_vram_fraction"]
        )
        retained_gpu_bytes = 0  # running sum of retained fusers' resident bytes

        # --- DSM/ortho grid (populated per sub-volume if svo method) ---
        dsm_enabled = config["dsm_enabled"]
        dsm_gsd = voxel_size / _SAMPLING_COVERAGE_FACTOR
        dsm_grid: Optional[NDArray] = None
        ortho_grid: Optional[NDArray] = None
        owner_grid: Optional[NDArray] = None
        dsm_origin_x: float = 0.0
        dsm_origin_y: float = 0.0
        dsm_z_min: float = 0.0
        dsm_z_max: float = 0.0
        dsm_w: int = 0
        dsm_h: int = 0
        dsm_global_origin_x: float = 0.0
        dsm_global_origin_y: float = 0.0
        dsm_global_w: int = 0
        dsm_global_h: int = 0
        win_r0: int = 0
        win_c0: int = 0

        if dsm_enabled and reconstruction.points:
            coords = np.array(
                [p.coordinates for p in reconstruction.points.values()]
            )
            min_xyz = coords.min(axis=0).astype(np.float32)
            max_xyz = coords.max(axis=0).astype(np.float32)
            extent_xy = max_xyz[:2] - min_xyz[:2]
            margin_xy = np.maximum(
                extent_xy * 0.05, dsm_gsd * 2
            ).astype(np.float32)
            # Global grid: defines the shared georeference + size every cluster's
            # tile composites into (merge_dsm_ortho_batches).
            dsm_global_origin_x = float(min_xyz[0] - margin_xy[0])
            dsm_global_origin_y = float(min_xyz[1] - margin_xy[1])
            max_x = float(max_xyz[0] + margin_xy[0])
            max_y = float(max_xyz[1] + margin_xy[1])
            dsm_global_w = int(
                np.ceil((max_x - dsm_global_origin_x) / dsm_gsd))
            dsm_global_h = int(
                np.ceil((max_y - dsm_global_origin_y) / dsm_gsd))
            z_margin = (max_xyz[2] - min_xyz[2]) * 0.1 + voxel_size * 5
            dsm_z_min = float(min_xyz[2] - z_margin)
            dsm_z_max = float(max_xyz[2] + z_margin)

            # Re-fit the raster to this cluster's territory: the owned cells' XY
            # AABB → a window into the global grid (padded a few px so DSM/ortho
            # hole-filling has context at the territory edge).  This keeps the
            # in-RAM rasters proportional to the cluster, not the whole scene.
            owned_xy_min = owned_origin[:2].astype(np.float64) * coarse_size
            owned_xy_max = (
                owned_cells.max(axis=0)[:2] + 1
            ).astype(np.float64) * coarse_size
            pad = 4
            win_c0 = max(0, int(np.floor(
                (owned_xy_min[0] - dsm_global_origin_x) / dsm_gsd)) - pad)
            win_r0 = max(0, int(np.floor(
                (owned_xy_min[1] - dsm_global_origin_y) / dsm_gsd)) - pad)
            win_c1 = min(dsm_global_w, int(np.ceil(
                (owned_xy_max[0] - dsm_global_origin_x) / dsm_gsd)) + pad)
            win_r1 = min(dsm_global_h, int(np.ceil(
                (owned_xy_max[1] - dsm_global_origin_y) / dsm_gsd)) + pad)
            dsm_w = max(0, win_c1 - win_c0)
            dsm_h = max(0, win_r1 - win_r0)
            dsm_origin_x = dsm_global_origin_x + win_c0 * dsm_gsd
            dsm_origin_y = dsm_global_origin_y + win_r0 * dsm_gsd

            if dsm_w > 0 and dsm_h > 0:
                dsm_grid = np.full((dsm_h, dsm_w), np.nan, dtype=np.float32)
                ortho_grid = np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
                owner_grid = np.full((dsm_h, dsm_w), -1, dtype=np.int32)
                logger.info(
                    f"Cluster {batch_num}: DSM window {dsm_w}×{dsm_h} at "
                    f"(r{win_r0},c{win_c0}) of global {dsm_global_w}×"
                    f"{dsm_global_h}, GSD={dsm_gsd:.4f}, "
                    f"Z=[{dsm_z_min:.2f}, {dsm_z_max:.2f}]"
                )
            context.log_memory(f"cluster {batch_num}: DSM grid allocated")

        # Batch-load all views upfront (parallel I/O).
        view_cache = _prepare_views(augmented, clean_cache)
        logger.info(
            f"Cluster {batch_num}: loaded {len(view_cache)} views"
        )
        context.log_memory(f"cluster {batch_num}: views loaded")

        def _build_fuser(
            sv: "_SubVolume",
        ) -> Tuple["pydense.SVOFuser", int]:
            """Create a configured fuser for a sub-volume and load its views."""
            fuser = pydense.SVOFuser()
            fuser.set_voxel_size(voxel_size)
            fuser.set_trunc_factor(trunc_factor_cfg)
            fuser.set_min_weight(config["depthmap_fusion_svo_min_weight"])
            fuser.set_num_levels(config["depthmap_fusion_svo_num_levels"])
            fuser.set_decimate_flat(
                config["depthmap_fusion_svo_decimate_flat"])
            fuser.set_edge_threshold(
                config["depthmap_fusion_svo_edge_threshold"]
            )
            fuser.set_min_count(config["depthmap_fusion_svo_min_count"])
            fuser.set_relative_min_weight(
                config["depthmap_fusion_svo_relative_min_weight"]
            )
            fuser.set_dsm_wall_cull_nz(config["dsm_wall_cull_nz"])
            fuser.set_device(0)
            fuser.set_bbox(sv.ext_min, sv.ext_max)
            n_loaded = 0
            for sid in sorted(sv.view_ids):
                result = view_cache.get(sid)
                if result is None:
                    continue
                K, R, t, depth, normal, color, vmask, conf = result
                fuser.add_view(
                    K, R, t, depth, normal, color, vmask,
                    confidence=conf, name=sid,
                )
                n_loaded += 1
            return fuser, n_loaded

        def _fuse_subvolume(sv: _SubVolume) -> None:
            """Fuse a sub-volume (geometry only), splitting if over budget.

            Renders the DSM patch and records per-cell ownership; the ortho is
            baked later (Pass 2) from the post-processed DSM.
            """
            nonlocal retained_gpu_bytes
            fuser, n_loaded = _build_fuser(sv)

            logger.info(
                f"  Sub-volume ({n_loaded} views), core "
                f"[{sv.core_min[0]:.1f},{sv.core_min[1]:.1f},"
                f"{sv.core_min[2]:.1f}]-"
                f"[{sv.core_max[0]:.1f},{sv.core_max[1]:.1f},"
                f"{sv.core_max[2]:.1f}]"
            )

            if n_loaded == 0:
                return

            # Count actual voxels on GPU.
            n_voxels = fuser.count_voxels()
            if n_voxels > max_voxels:
                logger.info(
                    f"  → {n_voxels} voxels > budget "
                    f"{max_voxels}, splitting …"
                )
                del fuser
                # Binary split along longest core axis.
                spans = sv.core_max - sv.core_min
                axis = int(np.argmax(spans))
                mid = (sv.core_min[axis] + sv.core_max[axis]) / 2.0
                margin = trunc_factor_cfg * voxel_size

                # Left half.
                left_core_max = sv.core_max.copy()
                left_core_max[axis] = mid
                left_ext_min = (
                    sv.core_min - margin
                ).astype(np.float32)
                left_ext_max = (
                    left_core_max + margin
                ).astype(np.float32)
                left_views = {
                    sid for sid in sv.view_ids
                    if sid in fusable_set
                }
                left_sv = _SubVolume(
                    core_min=sv.core_min.copy(),
                    core_max=left_core_max.astype(np.float32),
                    ext_min=left_ext_min,
                    ext_max=left_ext_max,
                    view_ids=left_views,
                )

                # Right half.
                right_core_min = sv.core_min.copy()
                right_core_min[axis] = mid
                right_ext_min = (
                    right_core_min - margin
                ).astype(np.float32)
                right_ext_max = (
                    sv.core_max + margin
                ).astype(np.float32)
                right_sv = _SubVolume(
                    core_min=right_core_min.astype(np.float32),
                    core_max=sv.core_max.copy(),
                    ext_min=right_ext_min,
                    ext_max=right_ext_max,
                    view_ids=left_views,  # same views
                )

                _fuse_subvolume(left_sv)
                _fuse_subvolume(right_sv)
                return

            # Voxels within budget — proceed with integration.
            refine_enabled = config[
                "depthmap_fusion_svo_refine_enabled"
            ]
            if refine_enabled:
                fuser.fuse_only()
                refine_nbrs = {
                    sid: [s.id for s in best_neighbors.get(sid, [])]
                    for sid in sorted(sv.view_ids)
                }
                fuser.refine_geometry(
                    iters=config[
                        "depthmap_fusion_svo_refine_iters"
                    ],
                    lambda_reg=config[
                        "depthmap_fusion_svo_refine_lambda_reg"
                    ],
                    neighbors=refine_nbrs,
                    lambda_anchor=config[
                        "depthmap_fusion_svo_refine_lambda_anchor"
                    ],
                    early_stop_rel=config[
                        "depthmap_fusion_svo_refine_early_stop_rel"
                    ],
                )
            else:
                fuser.fuse_only()

            # --- Render DSM patch (geometry) for this leaf's core footprint;
            #     the ortho is baked in Pass 2 from the corrected DSM. ---
            keep_for_bake = False
            if dsm_grid is not None and owner_grid is not None:
                leaf_idx = len(leaves)
                leaves.append(sv)
                _render_dsm_patch_from_sv(
                    fuser, sv, dsm_grid, owner_grid, leaf_idx,
                    dsm_origin_x, dsm_origin_y, dsm_gsd,
                    dsm_w, dsm_h, dsm_z_min, dsm_z_max,
                )
                # Retain this leaf's fuser for the Pass-2 ortho bake, bounded by BOTH a hard count cap and the live VRAM budget.
                cap = fuser.capacity()
                # Refine images are at depth resolution, uniform across a leaf's views; read W*H off any one of them.
                sample = next(
                    (view_cache[s] for s in sv.view_ids if s in view_cache),
                    None,
                )
                img_px = (
                    int(sample[3].shape[0]) * int(sample[3].shape[1])
                    if sample is not None
                    else 0
                )
                table_b = cap * _SVO_SLOT_BYTES
                img_b = n_loaded * img_px * _SVO_REFINE_IMG_BYTES_PX_VIEW
                scratch_b = cap * _SVO_REFINE_SCRATCH_BYTES_SLOT
                retained_cost = table_b  # table only after release
                next_active_peak = table_b + img_b + scratch_b
                n_retained = sum(f is not None for f in leaf_fusers)
                fits_vram = (
                    retained_gpu_bytes + retained_cost + next_active_peak
                    <= reuse_vram_budget
                )
                keep_for_bake = n_retained < max_reuse_fusers and fits_vram
                if keep_for_bake:
                    retained_gpu_bytes += retained_cost
                elif n_retained < max_reuse_fusers and not fits_vram:
                    logger.info(
                        f"  Pass-2 reuse throttled by VRAM: "
                        f"{retained_gpu_bytes / 1e9:.1f} GB tables retained + "
                        f"{retained_cost / 1e9:.1f} GB this table + "
                        f"{next_active_peak / 1e9:.1f} GB next-leaf peak > "
                        f"{reuse_vram_budget / 1e9:.1f} GB budget → "
                        f"rebuilding this leaf's fuser in Pass 2"
                    )
                leaf_fusers.append(fuser if keep_for_bake else None)

            # Always bake colors (voxels no longer store color).
            pts_arr, nrm_arr, clr_arr = fuser.extract_and_bake()
            pts = np.asarray(pts_arr, dtype=np.float32)
            nrm = np.asarray(nrm_arr, dtype=np.float32)
            clr = np.asarray(clr_arr, dtype=np.uint8)

            if len(pts) > 0:
                inside_core = np.all(
                    (pts >= sv.core_min) & (pts <= sv.core_max),
                    axis=1,
                )
                pts = pts[inside_core]
                nrm = nrm[inside_core]
                clr = clr[inside_core]

            if len(pts) > 0:
                # Restrict to cells this cluster actually owns (exact, jagged-
                # boundary-safe dedup across clusters; the AABB core clip above
                # already separates sibling sub-volumes within this cluster).
                pcells = np.floor(
                    pts.astype(np.float64) * inv_coarse
                ).astype(np.int64)
                owned_mask = np.isin(
                    _pack_coarse_cells(pcells, owned_origin, owned_span),
                    owned_keys,
                )
                pts = pts[owned_mask]
                nrm = nrm[owned_mask]
                clr = clr[owned_mask]

            if len(pts) > 0:
                all_points.append(pts)
                all_normals.append(nrm)
                all_colors.append(clr)
                logger.info(
                    f"    → {len(pts)} points extracted"
                )

            if not keep_for_bake:
                del fuser
            else:
                # Strip the retained fuser to its hash table only — the refine images + grad/adam are dead weight between passes
                fuser.release_refine_buffers()

        # --- Pass 1: build the global DSM geometry (no colour yet). ---
        for sv in subvolumes:
            _fuse_subvolume(sv)
            gc.collect()

        # --- Finalize DSM, then bake the ortho from the CORRECTED DSM. ---
        # Pass 1 produced the raw composited DSM + per-cell ownership.  We first
        # post-process the DSM (fill no-data holes, then sharpen building edges
        # with the shock filter), and only THEN bake the ortho (Pass 2): each
        # leaf re-fuses its views and colours the cells it owns by projecting
        # the FINAL heights into its images.  This keeps DSM↔ortho consistent
        # and gives the ortho the sharpened edges (no bake↔DSM chicken-and-egg).
        if dsm_grid is not None:
            valid_count = int(np.count_nonzero(~np.isnan(dsm_grid)))
            logger.info(
                f"Cluster {batch_num}: DSM mesh raster, "
                f"{valid_count}/{dsm_grid.size} valid cells"
            )

            # debug: which cells were reconstructed BEFORE the hole-fill (vs
            # hole-filled afterwards) — used by the bake-category raster below.
            orig_valid = ~np.isnan(dsm_grid)
            dsm_grid, dsm_extrap = _fill_dsm_holes(dsm_grid, config)
            logger.info(
                "  Filled DSM footprint holes (diffuse tiny + Delaunay pockets)"
            )

            dsm_grid = _shock_dsm_edges(dsm_grid, dsm_gsd, config)
            logger.info("  Sharpened DSM edges (shock filter)")

            # --- Pass 2: bake the ortho from the corrected DSM. ---
            # Every cell that carries a height — reconstructed OR hole-filled — is
            # coloured by REPROJECTING its final DSM height into source images and
            # taking the best-resolution inlier views (real multi-view baking, NOT
            # colour interpolation).
            if ortho_grid is not None and owner_grid is not None:
                n_final = int(config["ortho_bake_n_final_views"])
                irls_iters = int(config["ortho_bake_irls_iterations"])
                valid = ~np.isnan(dsm_grid)
                # DSM heightfield for per-view horizon occlusion of filled cells:
                # the bake marches it from each cell toward the camera and drops
                # views a taller surface (e.g. a roof) blocks — so occluded ground
                # gets real colour from clear nadir views, not a roof ghost.
                dsm_occ_arr = None
                dsm_max_z = 0.0
                if config["ortho_bake_dsm_occlusion"]:
                    dsm_occ_arr = np.ascontiguousarray(
                        dsm_grid, dtype=np.float32)
                    dsm_max_z = float(np.nanmax(dsm_grid))
                # Cells that received a REAL (non-black) bake.  A cell is baked by exactly one leaf
                baked_mask = np.zeros((dsm_h, dsm_w), dtype=bool)
                n_filled_tot = 0  # diagnostics: filled cells + how many baked
                n_filled_black = 0  # filled cells the bake still left black
                n_owned_black = 0  # reconstructed cells left black
                for leaf_idx, leaf_sv in enumerate(leaves):
                    # This leaf's core cell footprint (same formula as Pass 1).
                    cs = max(0, int(np.floor(
                        (leaf_sv.core_min[0] - dsm_origin_x) / dsm_gsd)))
                    ce = min(dsm_w, int(np.ceil(
                        (leaf_sv.core_max[0] - dsm_origin_x) / dsm_gsd)))
                    rs = max(0, int(np.floor(
                        (leaf_sv.core_min[1] - dsm_origin_y) / dsm_gsd)))
                    re_ = min(dsm_h, int(np.ceil(
                        (leaf_sv.core_max[1] - dsm_origin_y) / dsm_gsd)))
                    if ce <= cs or re_ <= rs:
                        continue
                    # view → writes back
                    sub_owner = owner_grid[rs:re_, cs:ce]
                    sub_valid = valid[rs:re_, cs:ce]
                    # Cells this leaf bakes: the ones it won, plus still-unowned
                    # filled holes inside its footprint (owner < 0 & has height).
                    fill_local = (sub_owner < 0) & sub_valid
                    sub_mask = ((sub_owner == leaf_idx)
                                | fill_local) & sub_valid
                    if not sub_mask.any():
                        continue
                    lr, lc = np.where(sub_mask)
                    rows_idx = lr + rs
                    cols_idx = lc + cs
                    pts = np.empty((len(rows_idx), 3), dtype=np.float32)
                    pts[:, 0] = dsm_origin_x + (cols_idx + 0.5) * dsm_gsd
                    pts[:, 1] = dsm_origin_y + (rows_idx + 0.5) * dsm_gsd
                    pts[:, 2] = dsm_grid[rows_idx, cols_idx]
                    nrm = _dsm_point_normals(
                        dsm_grid, rows_idx, cols_idx, dsm_gsd
                    )
                    # Filled holes carry interpolated geometry: bake them with a flat up-normal (no false grazing rejects)
                    is_filled = fill_local[lr, lc]
                    if is_filled.any():
                        nrm[is_filled] = (0.0, 0.0, 1.0)
                    # Mask-relax filled cells so they sample real colour where no
                    # view has depth — EXCEPT cells flagged no-relax: extrapolated
                    # (linear) or OCCLUDED ground under a tall roof (low_flat), to
                    # avoid ghosting roof colour onto occluded ground.
                    relax = (
                        is_filled & ~dsm_extrap[rows_idx, cols_idx]
                    ).astype(np.uint8)
                    baker = leaf_fusers[leaf_idx]
                    if baker is None:
                        # Not retained from Pass 1 (over the GPU cap) — rebuild and re-fuse, as before.
                        baker, n_loaded = _build_fuser(leaf_sv)
                        if n_loaded == 0:
                            del baker
                            continue
                        baker.count_voxels()
                        baker.fuse_only()
                    colors = np.asarray(
                        baker.bake_colors(
                            pts, nrm, n_final=n_final, irls_iters=irls_iters,
                            relax_occlusion=relax,
                            dsm_occ=dsm_occ_arr,
                            dsm_origin_x=dsm_origin_x,
                            dsm_origin_y=dsm_origin_y,
                            dsm_gsd=dsm_gsd,
                            dsm_max_z=dsm_max_z,
                        ),
                        dtype=np.uint8,
                    )
                    ortho_grid[rows_idx, cols_idx, :] = colors
                    # A cell is "really baked" iff the kernel found >=1 view
                    # (n_valid==0 emits pure black); record that, not sum>0 on
                    # the final ortho (which a legitimately-dark bake would fail).
                    ok = colors.any(axis=1)
                    baked_mask[rows_idx, cols_idx] = ok
                    n_filled_tot += int(is_filled.sum())
                    n_filled_black += int((is_filled & ~ok).sum())
                    n_owned_black += int((~is_filled & ~ok).sum())
                    # Claim the filled holes so a neighbour leaf (footprints can
                    # overlap by ~1 cell) does not re-bake them.
                    sub_owner[fill_local] = leaf_idx
                    del baker
                    # release the retained GPU fuser
                    leaf_fusers[leaf_idx] = None
                    gc.collect()
                n_unbaked = int((valid & ~baked_mask).sum())
                logger.info(
                    f"  Baked ortho from corrected DSM ({len(leaves)} leaves); "
                    f"{n_unbaked} cell(s) saw no view → residual fill"
                )
                logger.info(
                    f"    of which filled-cell black: {n_filled_black}/"
                    f"{n_filled_tot} filled, reconstructed black: "
                    f"{n_owned_black}"
                )
                # DEBUG (dsm_save_cluster_tiles): classify every cell so the
                # fan artefact can be localised in QGIS BEFORE the residual fill
                # repaints anything.
                # green = reconstructed geometry, baked;
                # red = reconstructed but no view saw it;
                # cyan = tiny hole filled by DIFFUSION (a smooth ramp), baked;
                # blue = large hole filled  FLAT-LOW, baked;
                # yellow = hole-filled but no view (→ residual fill).  Overlay on the fan:
                # cyan/blue ⇒ it's a hole-fill bake,
                # green ⇒ it's real (oblique-painted facade) geometry.
                if config.get("dsm_save_cluster_tiles", False):
                    from scipy import ndimage as _ndi
                    recon = orig_valid
                    filled = valid & ~recon
                    lbl, _ = _ndi.label(filled)
                    csz = np.bincount(lbl.ravel())
                    small_max = int(config["hole_fill_small_area_max"])
                    tiny = filled & (csz[lbl] <= small_max)
                    cat = np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
                    cat[recon & baked_mask] = (0, 170, 0)
                    cat[recon & ~baked_mask] = (200, 0, 0)
                    cat[tiny & baked_mask] = (0, 200, 200)
                    cat[(filled & ~tiny) & baked_mask] = (0, 90, 255)
                    cat[filled & ~baked_mask] = (255, 210, 0)
                    data.save_ortho(
                        cat, dsm_origin_x, dsm_origin_y, dsm_gsd,
                        reconstruction.reference, nodata_mask=~valid,
                        path=data.ortho_cluster_file(batch_num).replace(
                            "ortho_cluster_", "ortho_dbg_cluster_"),
                    )
                    logger.info(
                        "  [debug] wrote bake-category raster: green=recon, "
                        "red=recon-noview, cyan=diffusion-fill, blue=flatlow-"
                        "fill, yellow=fill-noview"
                    )
                ortho_grid = _fill_ortho_holes(
                    ortho_grid, dsm_grid, config, baked_mask=baked_mask
                )
                logger.info(
                    "  Filled residual ortho holes (occluded in every view)")

                ortho_grid = _ortho_gated_median(ortho_grid, dsm_grid, config)
                logger.info("  Despeckled ortho (gated 3x3 median)")

            # Release any Pass-1 fusers still held (skipped leaves, or ortho
            # disabled) before clearing the view cache they borrow from.
            leaf_fusers = []
            view_cache.clear()

            if not config.get("dsm_merge_feather", True):
                owned_xy = np.unique(owned_cells[:, :2], axis=0)
                oxy_origin = owned_xy.min(axis=0)
                oxy_span = owned_xy.max(axis=0) - oxy_origin + 1
                owned_xy_keys = np.sort(
                    (owned_xy[:, 0] - oxy_origin[0]) * oxy_span[1]
                    + (owned_xy[:, 1] - oxy_origin[1])
                )
                vr, vc = np.where(~np.isnan(dsm_grid))
                cx = np.floor(
                    (dsm_origin_x + (vc + 0.5) * dsm_gsd) * inv_coarse
                ).astype(np.int64) - oxy_origin[0]
                cy = np.floor(
                    (dsm_origin_y + (vr + 0.5) * dsm_gsd) * inv_coarse
                ).astype(np.int64) - oxy_origin[1]
                in_range = (
                    (cx >= 0) & (cx < oxy_span[0])
                    & (cy >= 0) & (cy < oxy_span[1])
                )
                col_keys = np.where(in_range, cx * oxy_span[1] + cy, -1)
                spill = ~np.isin(col_keys, owned_xy_keys)
                if spill.any():
                    dsm_grid[vr[spill], vc[spill]] = np.nan
                    if ortho_grid is not None:
                        ortho_grid[vr[spill], vc[spill]] = 0
                    logger.info(
                        f"Cluster {batch_num}: masked {int(spill.sum())} "
                        f"DSM/ortho spill cell(s) (hard territory mask)"
                    )

            # Persist this cluster's finished DSM+ortho as a compact tile;
            ortho_tile = (
                ortho_grid if ortho_grid is not None
                else np.zeros((dsm_h, dsm_w, 3), dtype=np.uint8)
            )
            # The tile carries the GLOBAL georeference + shape and its window
            # offset, so the merge places this territory-sized raster correctly.
            data.save_dsm_ortho_batch(
                batch_num, dsm_grid, ortho_tile,
                dsm_global_origin_x, dsm_global_origin_y, dsm_gsd,
                base_offset=(win_r0, win_c0),
                global_shape=(dsm_global_h, dsm_global_w),
                confidence=orig_valid,
            )
            logger.info(
                f"Cluster {batch_num}: DSM/ortho tile saved → "
                f"{data.dsm_ortho_batch_file(batch_num)}"
            )

            # Debug: also dump this cluster's own DSM+ortho window as standalone
            # georeferenced GeoTIFFs (they overlay the final raster in GIS), to
            # isolate per-cluster boundary / grazing-bake artefacts.
            if config.get("dsm_save_cluster_tiles", False):
                data.save_dsm(
                    dsm_grid, dsm_origin_x, dsm_origin_y, dsm_gsd,
                    reconstruction.reference,
                    path=data.dsm_cluster_file(batch_num),
                )
                data.save_ortho(
                    ortho_tile, dsm_origin_x, dsm_origin_y, dsm_gsd,
                    reconstruction.reference,
                    nodata_mask=np.isnan(dsm_grid),
                    path=data.ortho_cluster_file(batch_num),
                )
                logger.info(
                    f"Cluster {batch_num}: debug DSM/ortho GeoTIFFs → "
                    f"{data.ortho_cluster_file(batch_num)}"
                )
        else:
            view_cache.clear()

        # Concatenate all sub-volume results.
        if all_points:
            points = np.concatenate(all_points)
            normals = np.concatenate(all_normals)
            colors = np.concatenate(all_colors)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            normals = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)
        logger.info(
            f"Cluster {batch_num}: {len(points)} total fused points "
            f"from {len(subvolumes)} sub-volume(s)"
        )
        context.log_memory("fused cluster data in memory")
        logger.info("Batch %d: %d fused points", batch_num, len(points))

        if len(points) > 0:
            labels = np.zeros(len(points), dtype=np.uint8)
            filename = f"fused_batch_{batch_num:04d}.ply"
            data.save_point_cloud(
                points, normals, colors, labels, filename=filename
            )

        if config.get("depthmap_save_debug_ply", False) and len(points) > 0:
            cr, cg, cb = _CLUSTER_COLORS[
                batch_num % len(_CLUSTER_COLORS)
            ]

            bbox_pts = _sample_bbox_wireframe(
                cluster_bboxes[batch_num], step=0.01
            )
            n_bbox = len(bbox_pts)
            bbox_normals = np.zeros((n_bbox, 3), dtype=np.float32)
            bbox_colors = np.full(
                (n_bbox, 3), (cr, cg, cb), dtype=np.uint8
            )

            frustum_parts: List[NDArray] = []
            for sid in cluster_shots:
                if sid in reconstruction.shots:
                    fpts = _sample_frustum_wireframe(
                        reconstruction.shots[sid],
                        depth=0.3, step=0.01,
                    )
                    frustum_parts.append(fpts)
            if frustum_parts:
                frustum_pts = np.concatenate(frustum_parts)
            else:
                frustum_pts = np.empty((0, 3), dtype=np.float32)
            n_frust = len(frustum_pts)
            frustum_normals = np.zeros(
                (n_frust, 3), dtype=np.float32
            )
            frustum_colors = np.full(
                (n_frust, 3), (cr, cg, cb), dtype=np.uint8,
            )

            dbg_points = np.concatenate([points, bbox_pts, frustum_pts])
            dbg_normals = np.concatenate(
                [normals, bbox_normals, frustum_normals],
            )
            dbg_colors = np.concatenate(
                [colors, bbox_colors, frustum_colors],
            )
            dbg_labels = np.zeros(len(dbg_points), dtype=np.uint8)
            data.save_point_cloud(
                dbg_points, dbg_normals, dbg_colors, dbg_labels,
                filename=f"fused_batch_{batch_num:04d}_debug.ply",
            )

        gc.collect()

    # Run cluster fusions in parallel.
    n_fusion_workers = 1  # min(4, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(
        max_workers=n_fusion_workers, thread_name_prefix="fuse"
    ) as pool:
        futures: Dict[Future, int] = {}
        for batch_num in range(len(clusters)):
            fut = pool.submit(_fuse_single_cluster, batch_num)
            futures[fut] = batch_num

        for fut in as_completed(futures):
            batch_num = futures[fut]
            try:
                fut.result()
                logger.info(f"Fusion cluster {batch_num} complete")
            except Exception as exc:
                logger.error(f"Fusion cluster {batch_num} failed: {exc}")
