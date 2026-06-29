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
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from opensfm import geo, io, pypointcloud
from opensfm.dataset import UndistortedDataSet

logger: logging.Logger = logging.getLogger(__name__)

# Binary PLY record of the dense point-cloud layout: 6 float32 (xyz + normal)
# + 4 uint8 (rgb + class) = 28 bytes (mirrors io.pack_point_cloud_rows).
_PLY_ROW_DT: np.dtype = np.dtype([("f", "<f4", 6), ("c", "u1", 4)])


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

    delete_batches = data.config["depthmap_delete_fusion_batches"]

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


def merge_mesh_batches(
    data: UndistortedDataSet,
    batch_nums: List[int],
) -> None:
    """Stream all per-cluster ``mesh_batch_*.ply`` into the final ``mesh.ply``.

    A PLY lists all vertices before all faces, so this is a two-sweep stream:
    sweep 1 writes every batch's vertices, sweep 2 writes every batch's faces
    with their indices shifted by the running vertex base.  Peak memory is one
    batch at a time, mirroring ``merge_fusion_batches``.
    """
    if not batch_nums:
        return

    def _batch_file(batch_num: int) -> str:
        return f"mesh_batch_{batch_num:04d}.ply"

    delete_batches = data.config["depthmap_fusion_mesh_delete_batches"]

    mesh_exists = data.io_handler.isfile(data.mesh_file("mesh.ply"))
    present = [
        bn for bn in sorted(batch_nums)
        if data.io_handler.isfile(data.mesh_file(_batch_file(bn)))
    ]
    if mesh_exists and len(present) < len(batch_nums):
        logger.info("mesh.ply already complete; cleaning up leftover batches")
        if delete_batches:
            for bn in present:
                data.io_handler.rm_if_exist(data.mesh_file(_batch_file(bn)))
        return

    # Sweep 0: header-only counts → totals.
    batch_info: List[Tuple[str, int, int]] = []
    total_v = 0
    total_f = 0
    for bn in present:
        filename = _batch_file(bn)
        with data.io_handler.open_rb(data.mesh_file(filename)) as fp:
            nv, nf = io.read_mesh_ply_counts(fp)
        if nv == 0:
            continue
        batch_info.append((filename, nv, nf))
        total_v += nv
        total_f += nf

    if total_v == 0:
        logger.warning("No vertices found in any mesh batch.")
        return

    data.io_handler.mkdir_p(data._depthmap_path())
    with data.io_handler.open_wb(data.mesh_file("mesh.ply")) as out:
        out.write(io.mesh_ply_header(total_v, total_f).encode("ascii"))
        # Sweep 1: all vertices, in batch order.
        for filename, _nv, _nf in batch_info:
            v, n, c, _f = data.load_mesh(filename)
            out.write(io.pack_mesh_vertex_rows(v, n, c))
            del v, n, c, _f
            gc.collect()
        # Sweep 2: all faces, offset by each batch's vertex base.
        vbase = 0
        for filename, nv, _nf in batch_info:
            _v, _n, _c, f = data.load_mesh(filename)
            out.write(io.pack_mesh_face_rows(f, index_offset=vbase))
            vbase += nv
            del _v, _n, _c, f
            gc.collect()

    logger.info(
        f"Merged {len(batch_info)} mesh batches → mesh.ply "
        f"({total_v} verts, {total_f} faces)"
    )

    if delete_batches:
        for filename, _nv, _nf in batch_info:
            data.io_handler.rm_if_exist(data.mesh_file(filename))
        logger.info(f"Removed {len(batch_info)} merged mesh batch PLY(s)")


def merge_dsm_ortho_batches(
    data: UndistortedDataSet,
    batch_nums: List[int],
    reference: Optional[Any],
    output_crs: Optional[str] = None,
) -> None:
    """Composite per-cluster DSM/ortho tiles into the final dsm.tif / ortho.tif.

    Each cluster wrote a tight-cropped tile (``save_dsm_ortho_batch``) keyed on
    its offset into the shared global grid.

    With ``dsm_merge_feather`` (default) the tiles are left overlapping at fusion
    time and composited by **distance-transform feather blending**: each tile's
    weight is its pixel distance into its own valid region (deepest interior =
    highest weight, ramping to ~0 at tile edges), and the output is the weighted
    average of all tiles covering a cell.  Overlaps cross-fade, so cluster seams
    lose both their tonal step and the boundary speckle that a hard territory
    mask produced.

    The blend is **two-tier by confidence**: a cell's REAL (reconstructed) tiles
    feather among themselves and win outright; the FILL (hole-filled) tier is
    used only where no tile reconstructed the cell.  This stops a good roof
    height from being averaged toward an overlapping tile that filled the same
    spot from the ground (the DSM staircase).  Otherwise (legacy) tiles are
    composited by hard **max-z**.

    Tiles are deleted afterwards when ``depthmap_delete_geotiff_batches`` is set.
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
    _, _, _, _, (gh, gw), (origin_x, origin_y, gsd), _ = \
        data.load_dsm_ortho_batch(present[0])

    feather = data.config["dsm_merge_feather"]

    # Metadata pass: each tile's window offset + shape (one decompress each,
    # pixel data discarded), so the band loop below knows which tiles touch
    # each band without holding them all resident.
    meta: List[Tuple[int, int, int, int, int]] = []  # (bn, r0, c0, h, w)
    for bn in present:
        d_win, o_win, r0, c0, gshape, _geo, _conf = \
            data.load_dsm_ortho_batch(bn)
        if gshape != (gh, gw):
            logger.warning(
                f"DSM tile {bn} global shape {gshape} != {(gh, gw)}; skipping"
            )
        else:
            meta.append((bn, r0, c0, d_win.shape[0], d_win.shape[1]))
        del d_win, o_win

    # Band height bounded by the RAM budget: each band holds accumulators of
    # band_rows × gw.  Feather keeps TWO tiers (real + fill), each w + h + RGB
    # ≈ 20 B/cell → ~40 B/cell plus outputs/flips.
    max_ram = int(data.config["dsm_merge_max_ram_mb"]) * (1 << 20)
    band_rows = max(1, min(int(gh), max(64, max_ram // max(1, gw * 50))))

    def fill_band(rs: int, re_: int) -> Tuple[NDArray, NDArray]:
        """Composite all tiles overlapping grid rows [rs, re_) into one band."""
        bh = re_ - rs
        if feather:
            # Two confidence tiers per cell: REAL (reconstructed) and FILL
            # (hole-filled / interpolated).  Real cells feather-blend among
            # themselves; the fill tier is consulted ONLY where no real tile
            # covers the cell — so a good roof estimate is never dragged toward
            # an overlapping tile's ground-level fill (no staircase seams).
            accR_w = np.zeros((bh, gw), dtype=np.float32)
            accR_h = np.zeros((bh, gw), dtype=np.float32)
            accR_c = np.zeros((bh, gw, 3), dtype=np.float32)
            accF_w = np.zeros((bh, gw), dtype=np.float32)
            accF_h = np.zeros((bh, gw), dtype=np.float32)
            accF_c = np.zeros((bh, gw, 3), dtype=np.float32)
        else:
            dsm_b = np.full((bh, gw), np.nan, dtype=np.float32)
            ortho_b = np.zeros((bh, gw, 3), dtype=np.uint8)
        for bn, r0, c0, h, w in meta:
            if r0 >= re_ or r0 + h <= rs:
                continue  # tile does not intersect this band
            d_win, o_win, _, _, _, _, conf = data.load_dsm_ortho_batch(bn)
            # Row ranges: tile-local [ts, te) ↔ band-local [bs, be).
            ts = max(rs, r0) - r0
            te = min(re_, r0 + h) - r0
            bs = max(rs, r0) - rs
            be = min(re_, r0 + h) - rs
            cs, ce = c0, c0 + w
            if feather:
                valid = ~np.isnan(d_win)
                # Feather weight computed on the FULL tile (a padded False
                # border so its outer edges ramp to ~0), then sliced — so band
                # boundaries don't introduce artificial weight edges.
                padded = np.zeros((h + 2, w + 2), dtype=bool)
                padded[1:-1, 1:-1] = valid
                wgt = distance_transform_edt(
                    padded
                )[1:-1, 1:-1].astype(np.float32) * valid
                wseg = wgt[ts:te]
                vseg = valid[ts:te]
                dseg = np.where(vseg, d_win[ts:te], 0.0).astype(np.float32)
                oseg = o_win[ts:te].astype(np.float32)
                # Split this tile's valid cells into the two tiers.  A legacy
                # tile (conf is None) is treated as all-real → plain feather.
                if conf is not None:
                    real = vseg & (conf[ts:te] > 0)
                else:
                    real = vseg
                wR = wseg * real
                wF = wseg * (vseg & ~real)
                accR_w[bs:be, cs:ce] += wR
                accR_h[bs:be, cs:ce] += wR * dseg
                accR_c[bs:be, cs:ce] += wR[..., None] * oseg
                accF_w[bs:be, cs:ce] += wF
                accF_h[bs:be, cs:ce] += wF * dseg
                accF_c[bs:be, cs:ce] += wF[..., None] * oseg
            else:
                dseg = d_win[ts:te]
                cur = dsm_b[bs:be, cs:ce]
                curo = ortho_b[bs:be, cs:ce]
                take = ~np.isnan(dseg) & (np.isnan(cur) | (dseg > cur))
                cur[take] = dseg[take]
                curo[take] = o_win[ts:te][take]
            del d_win, o_win
        if feather:
            useR = accR_w > 0.0                       # real tile(s) cover it
            useF = (~useR) & (accF_w > 0.0)           # fill only where no real
            dsm_b = np.full((bh, gw), np.nan, dtype=np.float32)
            dsm_b[useR] = accR_h[useR] / accR_w[useR]
            dsm_b[useF] = accF_h[useF] / accF_w[useF]
            ortho_b = np.zeros((bh, gw, 3), dtype=np.uint8)
            ortho_b[useR] = np.clip(
                accR_c[useR] / accR_w[useR][:, None] + 0.5, 0.0, 255.0
            ).astype(np.uint8)
            ortho_b[useF] = np.clip(
                accF_c[useF] / accF_w[useF][:, None] + 0.5, 0.0, 255.0
            ).astype(np.uint8)
        gc.collect()
        return dsm_b, ortho_b

    n_valid = data.save_dsm_ortho_streamed(
        gh, gw, origin_x, origin_y, gsd, fill_band, band_rows,
        reference=reference, output_crs=output_crs,
    )
    crs_msg = f" in {output_crs}" if output_crs else " (topocentric)"
    logger.info(
        f"Merged {len(meta)} DSM/ortho tile(s) → {data.dsm_file()} / "
        f"{data.ortho_file()} ({n_valid} valid cells, band_rows={band_rows})"
        f"{crs_msg}"
    )

    if data.config["depthmap_delete_geotiff_batches"]:
        for bn in present:
            data.io_handler.rm_if_exist(data.dsm_ortho_batch_file(bn))
        logger.info(f"Removed {len(present)} merged DSM/ortho tile(s)")


def export_pointcloud_formats(
    data: UndistortedDataSet,
    config: Dict[str, Any],
    reference: Optional[Any] = None,
    output_crs: Optional[str] = None,
) -> None:
    """Stream the merged ``fused.ply`` into LAS and/or LAZ when configured.

    fused.ply stays the canonical product / octree input (topocentric); these are
    additional archival/interchange copies.  When ``output_crs`` (and
    ``reference``) are given the points/normals are reprojected to that CRS and
    the CRS is embedded in the LAS/LAZ header; otherwise they stay topocentric.

    The PLY is read with plain buffered file reads in fixed-size record chunks
    (NOT memory-mapped), so an arbitrarily large cloud never has its page cache
    balloon resident — the cause of OOM on the previous mmap-based reader.  The
    LAS/LAZ writers already stream to disk.  Any failure is logged without
    aborting the pipeline.
    """
    formats = []
    if config["dense_pointcloud_export_las"]:
        formats.append("las")
    if config["dense_pointcloud_export_laz"]:
        formats.append("laz")
    if not formats:
        return

    src = data.point_cloud_file("fused.ply")
    if not data.io_handler.isfile(src):
        logger.warning("No fused.ply found; skipping LAS/LAZ export.")
        return

    # Optional reprojection topocentric → output CRS (positions + normals), using
    # the exact per-point PROJ transform (not a single affine, which drifts by
    # metres over large extents — mostly in Z, from the tangent-plane curvature).
    crs_wkt = ""
    offset = None
    transformer = None
    if output_crs is not None and reference is not None:
        transformer = geo.construct_proj_transformer(output_crs, inverse=True)
        crs_wkt = geo.crs_to_wkt(output_crs)
        # Place the LAS integer offset near the data; projected eastings/northings
        # would otherwise overflow the int32 (value-offset)/scale encoding.
        origin = geo.transform_to_proj([0.0, 0.0, 0.0], reference, transformer)
        offset = [float(np.floor(c)) for c in origin]

    rec = _PLY_ROW_DT.itemsize
    chunk_pts = 1_000_000
    # fused.ply always carries the full dense layout (xyz + normal + rgb + class).
    for fmt in formats:
        out_path = data.point_cloud_file(f"fused.{fmt}")
        try:
            with data.io_handler.open_rb(src) as fp:
                n_total = io.read_ply_vertex_count(fp)
                header = pypointcloud.PointCloudHeader()
                header.point_count = n_total
                header.has_normals = True
                header.has_colors = True
                header.has_labels = True
                if crs_wkt:
                    header.crs_wkt = crs_wkt
                    header.offset = offset
                writer = pypointcloud.open_writer(out_path, header)

                written = 0
                while written < n_total:
                    k = min(chunk_pts, n_total - written)
                    buf = fp.read(k * rec)
                    if not buf:
                        break
                    rows = np.frombuffer(
                        buf, dtype=_PLY_ROW_DT, count=len(buf) // rec
                    )
                    pos = rows["f"][:, :3].astype(np.float64)
                    nrm = np.ascontiguousarray(
                        rows["f"][:, 3:]).astype(np.float32)
                    if transformer is not None:
                        pos, nrm = geo.transform_points_normals_to_proj(
                            reference, transformer, pos, nrm)
                        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
                    col = np.ascontiguousarray(rows["c"][:, :3])
                    lbl = np.ascontiguousarray(rows["c"][:, 3])
                    writer.write_chunk(
                        pos, normals=nrm, colors=col, labels=lbl
                    )
                    written += len(rows)
                    del buf, rows, pos, nrm, col, lbl
                writer.finalize()
            crs_msg = f" in {output_crs}" if crs_wkt else " (topocentric)"
            logger.info(
                f"Exported {written} points to fused.{fmt} (streamed){crs_msg}"
            )
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
    builder_config.max_points_per_tile = config["octree_max_points_per_tile"]
    builder_config.max_depth = config["octree_max_depth"]
    builder_config.lod_sample_count = config["octree_lod_sample_count"]

    # Out-of-core build: the PLY is memory-mapped and processed in bounded RAM
    # (independent of the point count), so multi-gigabyte clouds never need to
    # be loaded into memory. Output tiles are format-identical to before.
    meta = pypointcloud.build_octree_from_file(
        cloud_path=ply_path,
        config=builder_config,
        split_depth=config["octree_split_depth"],
        max_bucket_points=config["octree_max_bucket_points"],
        temp_dir=os.path.join(output_dir, "_ooc_tmp"),
    )

    if meta.total_points == 0:
        logger.warning("Point cloud is empty, skipping octree export.")
        return

    logger.info(
        f"Octree export complete: {meta.total_points} points, depth {meta.max_depth}, {output_dir}"
    )
