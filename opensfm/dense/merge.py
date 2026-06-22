# pyre-strict
"""Merge stage (pipeline phase 4) and final-product export.

Streams the per-cluster ``fused_batch_*.ply`` into the final ``fused.ply``,
composites the per-cluster DSM/ortho tiles (max-z) into ``dsm.tif`` /
``ortho.tif``, and optionally exports the cloud as LAS/LAZ and as octree tiles
for the viewer.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opensfm import io, pypointcloud
from opensfm.dataset import UndistortedDataSet

logger: logging.Logger = logging.getLogger(__name__)


def merge_fusion_batches(
    data: UndistortedDataSet,
    batch_nums: List[int],
) -> None:
    """Stream all batch PLYs into the final ``fused.ply``.

    Each batch is loaded, appended to the output stream, and released
    before the next is read, so peak memory is one batch — not the whole
    merged cloud (which previously had to be held twice across the
    ``np.concatenate``).  The output is byte-identical to the old path.

    When ``depthmap_delete_fusion_batches`` is set (default), the per-batch
    PLYs are deleted once the complete ``fused.ply`` has been written; they
    are fully consumed here and redundant afterwards.
    """
    if not batch_nums:
        logger.warning("No fusion batches to merge.")
        return

    def _batch_file(batch_num: int) -> str:
        return f"fused_batch_{batch_num:04d}.ply"

    delete_batches = data.config.get("depthmap_delete_fusion_batches", True)

    # Resume guard: ``fused.ply`` is only created after a *complete* write,
    # so if it already exists but some batches are missing we are resuming
    # an interrupted cleanup — never rebuild the final cloud from a partial
    # batch set (which would silently drop points).
    fused_exists = data.io_handler.isfile(data.point_cloud_file("fused.ply"))
    present = [
        bn for bn in sorted(batch_nums)
        if data.io_handler.isfile(data.point_cloud_file(_batch_file(bn)))
    ]
    if fused_exists and len(present) < len(batch_nums):
        logger.info(
            "fused.ply already complete; skipping rebuild and cleaning up "
            f"{len(present)} leftover batch PLY(s)"
        )
        if delete_batches:
            for bn in present:
                data.io_handler.rm_if_exist(
                    data.point_cloud_file(_batch_file(bn)))
        return

    # Pass 1: header-only read to get each batch's vertex count → total.
    batch_files: List[Tuple[str, int]] = []
    total = 0
    for bn in present:
        filename = _batch_file(bn)
        with data.io_handler.open_rb(data.point_cloud_file(filename)) as fp:
            n = io.read_ply_vertex_count(fp)
        if n == 0:
            continue
        batch_files.append((filename, n))
        total += n

    if total == 0:
        logger.warning("No points found in any batch PLY.")
        return

    # Pass 2: stream each batch into fused.ply (one batch resident at a time).
    data.io_handler.mkdir_p(data._depthmap_path())
    written = 0
    with data.io_handler.open_wb(data.point_cloud_file("fused.ply")) as out:
        out.write(io._ply_header(total).encode("ascii"))
        for filename, _n in batch_files:
            p, nrm, c, lbl = data.load_point_cloud(filename)
            if len(p) == 0:
                continue
            out.write(io.pack_point_cloud_rows(p, nrm, c, lbl))
            written += len(p)
            del p, nrm, c, lbl
            gc.collect()

    logger.info(
        f"Merged {len(batch_files)} batches → fused.ply ({written} points)"
    )

    # Reclaim the now-redundant per-batch PLYs (only after the full write).
    if delete_batches:
        for filename, _n in batch_files:
            data.io_handler.rm_if_exist(data.point_cloud_file(filename))
        logger.info(f"Removed {len(batch_files)} merged batch PLY(s)")


def merge_dsm_ortho_batches(
    data: UndistortedDataSet,
    batch_nums: List[int],
    reference: Optional[Any],
) -> None:
    """Composite per-cluster DSM/ortho tiles into the final dsm.tif / ortho.tif.

    Each cluster wrote a tight-cropped tile (``save_dsm_ortho_batch``) keyed on
    its offset into the shared global grid.  We allocate the global grid once
    and merge every tile by **max-z**: a cell takes a tile's height+colour when
    the tile has a surface there AND (the global cell is still empty OR the
    tile's height is greater).  This keeps the higher surface where cluster
    footprints overlap — the same arbitration used within a cluster.  The ortho
    colour follows the winning height, so DSM and ortho stay consistent.

    Tiles are deleted afterwards when ``depthmap_delete_fusion_batches`` is set,
    matching the PLY batch cleanup.
    """
    present = [
        bn for bn in sorted(batch_nums)
        if data.dsm_ortho_batch_exists(bn)
    ]
    if not present:
        logger.warning("No DSM/ortho tiles to merge.")
        return

    # Global extent + georeference come from the tiles themselves (identical
    # across clusters); read the first to size the output grid.
    _, _, _, _, (gh, gw), (origin_x, origin_y, gsd) = \
        data.load_dsm_ortho_batch(present[0])
    dsm = np.full((gh, gw), np.nan, dtype=np.float32)
    ortho = np.zeros((gh, gw, 3), dtype=np.uint8)

    n_merged = 0
    for bn in present:
        d_win, o_win, r0, c0, gshape, _geo = data.load_dsm_ortho_batch(bn)
        if gshape != (gh, gw):
            logger.warning(
                f"DSM tile {bn} global shape {gshape} != {(gh, gw)}; skipping"
            )
            continue
        h, w = d_win.shape[:2]
        win_dsm = dsm[r0:r0 + h, c0:c0 + w]
        win_ortho = ortho[r0:r0 + h, c0:c0 + w]
        # Take cells where this tile has a surface and either the global grid is
        # empty there or this tile's height wins (NaN-safe: NaN comparisons are
        # False, so the isnan term carries empty cells).
        take = ~np.isnan(d_win) & (np.isnan(win_dsm) | (d_win > win_dsm))
        win_dsm[take] = d_win[take]
        win_ortho[take] = o_win[take]
        n_merged += 1
        del d_win, o_win
        gc.collect()

    data.save_dsm(dsm, origin_x, origin_y, gsd, reference)
    # No-data (transparent) wherever no cluster produced a surface.
    data.save_ortho(
        ortho, origin_x, origin_y, gsd, reference,
        nodata_mask=np.isnan(dsm),
    )
    n_valid = int(np.count_nonzero(~np.isnan(dsm)))
    logger.info(
        f"Merged {n_merged} DSM/ortho tile(s) → {data.dsm_file()} / "
        f"{data.ortho_file()} ({n_valid} valid cells)"
    )

    if data.config.get("depthmap_delete_fusion_batches", True):
        for bn in present:
            data.io_handler.rm_if_exist(data.dsm_ortho_batch_file(bn))
        logger.info(f"Removed {len(present)} merged DSM/ortho tile(s)")


def export_pointcloud_formats(
    data: UndistortedDataSet,
    config: Dict[str, Any],
) -> None:
    """Stream the merged ``fused.ply`` into LAS and/or LAZ when configured.

    fused.ply stays the canonical product / octree input; these are additional
    archival/interchange copies. Conversion is chunked (bounded memory) and any
    failure is logged without aborting the pipeline.
    """
    formats = []
    if config.get("dense_pointcloud_export_las", False):
        formats.append("las")
    if config.get("dense_pointcloud_export_laz", False):
        formats.append("laz")
    if not formats:
        return

    src = data.point_cloud_file("fused.ply")
    if not data.io_handler.isfile(src):
        logger.warning("No fused.ply found; skipping LAS/LAZ export.")
        return

    for fmt in formats:
        out_path = data.point_cloud_file(f"fused.{fmt}")
        try:
            reader = pypointcloud.open_reader(src)
            has_normals, has_colors, has_labels = reader.attributes()
            header = pypointcloud.PointCloudHeader()
            header.point_count = reader.total_count
            header.has_normals = has_normals
            header.has_colors = has_colors
            header.has_labels = has_labels
            writer = pypointcloud.open_writer(out_path, header)

            total = 0
            while True:
                chunk = reader.read_chunk(1_000_000)
                if chunk is None:
                    break
                pos, nrm, col, lbl = chunk
                kwargs: Dict[str, Any] = {}
                if nrm is not None:
                    kwargs["normals"] = nrm
                if col is not None:
                    kwargs["colors"] = col
                if lbl is not None:
                    kwargs["labels"] = lbl
                writer.write_chunk(pos, **kwargs)
                total += len(pos)
            writer.finalize()
            logger.info(f"Exported {total} points to fused.{fmt}")
        except Exception:
            logger.warning(f"Failed to export fused.{fmt}", exc_info=True)


def export_octree_tiles(
    data: UndistortedDataSet,
    config: Dict[str, Any],
) -> None:
    """Convert the dense point cloud PLY to octree tiles for the viewer.

    Looks for fused.ply or merged.ply (in that order) and builds an octree
    tile set under ``point_cloud/`` next to the reconstruction.

    The octree uses Morton ordering and LOD subsampling so that the viewer
    can progressively stream tiles based on the camera frustum.
    """

    # Find the source PLY.
    for ply_name in ("fused.ply", "merged.ply"):
        ply_path = data.point_cloud_file(ply_name)
        if data.io_handler.isfile(ply_path):
            break
    else:
        logger.warning("No dense point cloud found for octree export.")
        return

    logger.info(f"Building octree tiles from {ply_name} …")

    # Output directory: next to the undistorted data, under point_cloud/.
    output_dir = os.path.join(data.data_path, "point_cloud")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the builder.
    builder_config = pypointcloud.OctreeBuilderConfig()
    builder_config.output_dir = output_dir
    builder_config.max_points_per_tile = config.get(
        "octree_max_points_per_tile", 50000
    )
    builder_config.max_depth = config.get("octree_max_depth", 15)
    builder_config.lod_sample_count = config.get(
        "octree_lod_sample_count", 10000
    )

    # Out-of-core build: the PLY is memory-mapped and processed in bounded RAM
    # (independent of the point count), so multi-gigabyte clouds never need to
    # be loaded into memory. Output tiles are format-identical to before.
    meta = pypointcloud.build_octree_from_file(
        cloud_path=ply_path,
        config=builder_config,
        split_depth=config.get("octree_split_depth", 4),
        max_bucket_points=config.get("octree_max_bucket_points", 8_000_000),
        temp_dir=os.path.join(output_dir, "_ooc_tmp"),
    )

    if meta.total_points == 0:
        logger.warning("Point cloud is empty, skipping octree export.")
        return

    logger.info(
        f"Octree export complete: {meta.total_points} points, depth {meta.max_depth}, {output_dir}"
    )
