# pyre-strict
"""DSM / ortho post-processing helpers.

Pure grid operations used by the fusion stage to finish the DSM and ortho:
GPU Perona-Malik diffusion, per-component linear hole fill, footprint
detection, two-pass hole filling, edge-sharpening shock filter, gated median
despeckle, and DSM-derived point normals.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from opensfm import pydense

logger: logging.Logger = logging.getLogger(__name__)


def _gpu_diffuse_grid(
    grid_nan: NDArray, iters: int, kappa: float, dt: float, device: int = 0
) -> NDArray:
    """Bounded GPU Perona-Malik diffusion of a single-channel float grid.

    NaN cells are holes (seeded from valid neighbours, one ring per
    iteration, then relaxed); non-NaN cells are frozen (Dirichlet BC).
    Returns the diffused grid as float32.
    """
    diff = pydense.GPUDiffuser()
    diff.set_device(int(device))
    diff.upload_grid(np.ascontiguousarray(grid_nan, dtype=np.float32))
    # Empty guide → self-guided (edge magnitude from the grid itself).
    empty_guide = np.empty((0,), dtype=np.float32)
    res = diff.diffuse(empty_guide, int(iters), float(kappa), float(dt))
    return np.asarray(res, dtype=np.float32)


def _linear_fill_components(
    grid: NDArray,
    labels: NDArray,
    fill_ids: NDArray,
    sample_valid: NDArray,
    ring_iters: int = 3,
    extrap_out: Optional[NDArray] = None,
) -> None:
    """Fill large hole components by linear (Delaunay) interpolation.

    For each labelled component in ``fill_ids`` a thin ring of valid samples
    around it (``sample_valid`` cells within ``ring_iters`` of the hole) is
    triangulated and the hole interior is linearly interpolated across it.
    Cells outside the ring's convex hull fall back to nearest-neighbour.
    Works per-component so colours/heights never bleed between holes.
    ``grid`` (``(H,W)`` or ``(H,W,C)`` float) is modified in place.

    ``extrap_out`` (bool ``(H,W)``, optional): set True at cells that fell back
    to nearest-neighbour (outside the convex hull) — i.e. EXTRAPOLATED, hence
    geometrically unreliable (typically the open side of boundary concavities).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    multichannel = grid.ndim == 3
    H, W = labels.shape
    pad = ring_iters + 1
    # All per-component work is cropped to the component's bounding box (+pad
    # for the boundary ring) so cost scales with hole size, not grid size.
    boxes = ndimage.find_objects(labels)
    for lid in fill_ids:
        box = boxes[lid - 1]
        if box is None:
            continue
        y0 = max(box[0].start - pad, 0)
        y1 = min(box[0].stop + pad, H)
        x0 = max(box[1].start - pad, 0)
        x1 = min(box[1].stop + pad, W)

        sub_grid = grid[y0:y1, x0:x1]  # a view → writes propagate to grid
        comp = labels[y0:y1, x0:x1] == lid
        ring = ndimage.binary_dilation(
            comp, iterations=ring_iters) & sample_valid[y0:y1, x0:x1]
        ys, xs = np.nonzero(ring)
        if ys.size < 4:
            continue
        pts = np.column_stack((xs, ys)).astype(np.float64)  # local frame
        qy, qx = np.nonzero(comp)
        q = np.column_stack((qx, qy)).astype(np.float64)
        vals = sub_grid[ys, xs, :] if multichannel else sub_grid[ys, xs]

        try:
            filled = LinearNDInterpolator(pts, vals)(q)
        except Exception:
            filled = np.full(
                (q.shape[0],) + vals.shape[1:], np.nan, dtype=np.float64
            )

        bad = (
            np.isnan(filled).any(axis=1) if multichannel else np.isnan(filled)
        )
        if np.any(bad):
            filled[bad] = NearestNDInterpolator(pts, vals)(q[bad])
            if extrap_out is not None:
                extrap_out[y0:y1, x0:x1][qy[bad], qx[bad]] = True

        if multichannel:
            sub_grid[qy, qx, :] = filled.astype(grid.dtype)
        else:
            sub_grid[qy, qx] = filled.astype(grid.dtype)


def _fill_holes_2pass(
    values: NDArray,
    sample_valid: NDArray,
    hole_mask: NDArray,
    *,
    small_area_max: int,
    diffuse_iters: int,
    kappa: float,
    dt: float,
    device: int = 0,
) -> Tuple[NDArray, NDArray]:
    """Two-stage hole fill on a float grid (post-process).

    Stage 1 — tiny holes (connected components <= ``small_area_max`` cells):
    bounded GPU Perona-Malik diffusion seeded from the surrounding data.
    Stage 2 — larger holes: per-component linear (Delaunay) interpolation
    from their boundary ring.

    ``values``      : float ``(H,W)`` or ``(H,W,C)``.
    ``sample_valid``: bool ``(H,W)``, True where real data exists (samples,
                      frozen during diffusion).
    ``hole_mask``   : bool ``(H,W)``, cells to fill.  Cells that are neither a
                      sample nor a hole are left untouched (never used as a
                      sample, never frozen).  Every cell in ``hole_mask`` is
                      filled — the caller is responsible for excluding the
                      genuine no-data exterior (DSM: the footprint test; ortho:
                      ``has_dsm``).
    Returns ``(filled_copy_float32, extrapolated_mask_bool)`` where the second
    is True at cells filled by nearest-neighbour extrapolation (unreliable).
    """
    out = values.astype(np.float32, copy=True)
    extrap = np.zeros(values.shape[:2], dtype=bool)
    if not hole_mask.any():
        return out, extrap

    labels, n_comp = ndimage.label(hole_mask)
    if n_comp == 0:
        return out, extrap

    comp_size = np.bincount(labels.ravel())
    small_mask = hole_mask & (comp_size[labels] <= small_area_max)

    # Every hole in ``hole_mask`` is fillable here — the genuine no-data exterior
    # was already excluded by the caller (DSM footprint / ortho ``has_dsm``).
    large_ids = np.nonzero(comp_size > small_area_max)[0]
    large_ids = large_ids[large_ids != 0]  # drop the non-hole label 0

    multichannel = out.ndim == 3
    nchan = out.shape[2] if multichannel else 1

    # Stage 1: GPU diffusion for tiny holes.  Only sample cells are frozen;
    # holes and ignored cells are NaN so they neither seed nor bleed.
    diffuse_ok = True
    if small_mask.any():
        frozen = ~sample_valid
        for c in range(nchan):
            ch = out[..., c] if multichannel else out
            grid = ch.copy()
            grid[frozen] = np.nan
            try:
                filled = _gpu_diffuse_grid(
                    grid, diffuse_iters, kappa, dt, device)
            except Exception as e:  # OpenCL unavailable → fall back to linear
                logger.warning(
                    f"GPU diffuse hole-fill unavailable ({e}); "
                    "routing tiny holes through linear interpolation"
                )
                diffuse_ok = False
                break
            if multichannel:
                out[..., c][small_mask] = filled[small_mask]
            else:
                out[small_mask] = filled[small_mask]

    # Stage 2: per-component linear interpolation for enclosed large holes
    # (and enclosed tiny holes too if the diffusion pass was unavailable).
    if diffuse_ok:
        stage2_ids = large_ids
        stage2_valid = sample_valid | small_mask  # tiny holes now hold values
    else:
        all_ids = np.nonzero(comp_size != 0)[0]
        stage2_ids = all_ids[all_ids != 0]
        stage2_valid = sample_valid

    n_tiny = int(small_mask.sum()) if diffuse_ok else 0
    n_linear = int(comp_size[stage2_ids].sum()) if stage2_ids.size else 0
    n_nodata = int(hole_mask.sum()) - n_tiny - n_linear
    logger.info(
        f"Hole fill: {n_tiny} tiny cells (diffusion), {n_linear} enclosed "
        f"cells (linear), {n_nodata} cells left as no-data"
    )

    if stage2_ids.size:
        _linear_fill_components(
            out, labels, stage2_ids, stage2_valid, extrap_out=extrap
        )

    return out, extrap


def _dsm_footprint(valid: NDArray, close_iters: int) -> NDArray:
    """Reconstructed footprint of the DSM: the valid region morphologically
    closed (to bridge ragged-boundary concavity mouths) and hole-filled.

    No-data INSIDE the footprint are fillable pockets/concavities; no-data
    OUTSIDE it is the genuine exterior (kept as no-data).
    """
    if close_iters > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        closed = ndimage.binary_closing(
            valid, structure=structure, iterations=int(close_iters)
        )
    else:
        closed = valid
    return ndimage.binary_fill_holes(closed)


def _fill_dsm_holes(
    dsm_grid: NDArray, config: Dict[str, Any]
) -> Tuple[NDArray, NDArray]:
    """Fill DSM no-data holes that lie INSIDE the reconstructed footprint: tiny
    ones by edge-aware GPU diffusion, larger pockets/boundary concavities by
    per-component linear (Delaunay) interpolation.  The no-data EXTERIOR outside
    the footprint is kept as no-data (so the ortho stays transparent there).

    Returns ``(filled_dsm, extrapolated_mask)`` — the mask marks cells filled by
    nearest-neighbour extrapolation (unreliable geometry, e.g. the open side of
    boundary concavities); the ortho bake must NOT mask-relax those.
    """
    valid = ~np.isnan(dsm_grid)
    empty = np.zeros(dsm_grid.shape, dtype=bool)
    if valid.all() or not valid.any():
        return dsm_grid, empty

    footprint = _dsm_footprint(valid, int(config["hole_fill_footprint_close"]))
    fillable = (~valid) & footprint
    if not fillable.any():
        return dsm_grid, empty

    return _fill_holes_2pass(
        dsm_grid,
        sample_valid=valid,
        hole_mask=fillable,
        small_area_max=config["hole_fill_small_area_max"],
        diffuse_iters=config["hole_fill_diffuse_iters"],
        kappa=0.5,  # metres: large steps (building edges) stop diffusion
        dt=0.2,
    )


def _fill_ortho_holes(
    ortho_grid: NDArray,
    dsm_grid: NDArray,
    config: Dict[str, Any],
    baked_mask: Optional[NDArray] = None,
) -> NDArray:
    """Residual ortho colour fill: cells where a DSM surface exists but the bake
    produced no colour (occluded in every view → black).  Most filled-DSM cells
    are coloured by the Pass-2 reprojection bake; this only mops up the few the
    bake could not see.  Tiny holes by (near-isotropic) GPU diffusion, larger
    ones by linear interpolation.  Cells with no DSM stay untouched (black).

    ``baked_mask`` (if given) marks cells that received a real bake; only the
    complement (DSM but not baked) is filled, and the fill is seeded from the
    baked cells — so a successfully-baked cell is never overwritten, even a
    legitimately dark one.  Falls back to ``ortho.sum > 0`` when not provided."""
    has_dsm = ~np.isnan(dsm_grid)
    if baked_mask is not None:
        has_color = has_dsm & baked_mask
    else:
        has_color = has_dsm & (ortho_grid.sum(axis=2) > 0)
    fillable = has_dsm & ~has_color
    if not fillable.any():
        return ortho_grid
    filled, _extrap = _fill_holes_2pass(
        ortho_grid,
        sample_valid=has_color,  # only real colour seeds the fill
        hole_mask=fillable,
        small_area_max=config["hole_fill_small_area_max"],
        diffuse_iters=config["hole_fill_diffuse_iters"],
        kappa=1e9,  # near-isotropic: smooth colour fill, no false edges
        dt=0.2,
    )
    return np.clip(filled, 0, 255).astype(ortho_grid.dtype)


def _ortho_gated_median(
    ortho_grid: NDArray, dsm_grid: NDArray, config: Dict[str, Any]
) -> NDArray:
    """Gated 3x3 GPU median despeckle of the ortho.

    Replaces a pixel with its local median only when it differs by more than
    ``ortho_median_threshold`` (per channel), removing isolated speckle while
    preserving real texture/edges.  No-data cells (DSM NaN) are ignored.
    """
    threshold = float(config["ortho_median_threshold"])
    if threshold <= 0.0:
        return ortho_grid
    valid = (~np.isnan(dsm_grid)).astype(np.uint8)
    ortho_f = np.ascontiguousarray(ortho_grid, dtype=np.float32)
    try:
        f = pydense.GPUDiffuser()
        f.set_device(0)
        out = f.gated_median(ortho_f, valid, threshold)
        return np.clip(out, 0, 255).astype(ortho_grid.dtype)
    except Exception as e:
        logger.warning(f"Ortho gated median unavailable ({e}); skipping")
        return ortho_grid


def _shock_dsm_edges(
    dsm_grid: NDArray, gsd: float, config: Dict[str, Any]
) -> NDArray:
    """Sharpen DSM edges with a coherence-enhancing shock filter (GPU).

    Steepens fattened roof/ground height ramps into steps using only the DSM
    (no ortho → breaks the bake↔DSM chicken-and-egg).  Steered by a local
    structure tensor so the resulting edges are sharp and straight, and gated by
    local slope so smooth gradients are not terraced into staircases.  No-data
    (NaN) cells are preserved.
    """
    iters = int(config["dsm_shock_iterations"])
    if iters <= 0:
        return dsm_grid
    win = int(config["dsm_shock_window"])
    dt = float(config["dsm_shock_dt"])
    coherence = float(config["dsm_shock_coherence"])
    edge_slope = float(config["dsm_shock_edge_slope"])
    dsm_in = np.ascontiguousarray(dsm_grid, dtype=np.float32)
    try:
        f = pydense.GPUDiffuser()
        f.set_device(0)
        out = f.shock_filter(
            dsm_in, iters, win, dt, coherence, float(gsd), edge_slope
        )
        return np.asarray(out, dtype=np.float32)
    except Exception as e:
        logger.warning(f"DSM shock filter unavailable ({e}); skipping")
        return dsm_grid


def _dsm_point_normals(
    dsm: NDArray, rows: NDArray, cols: NDArray, gsd: float
) -> NDArray:
    """Per-point up-facing surface normals from DSM central differences.

    Used to colour-bake the corrected DSM cells (Pass 2).  Cells whose
    neighbours are no-data fall back to a flat (0, 0, 1) normal.
    """
    h, w = dsm.shape
    cl = np.maximum(cols - 1, 0)
    cr = np.minimum(cols + 1, w - 1)
    ru = np.maximum(rows - 1, 0)
    rd = np.minimum(rows + 1, h - 1)
    dzdx = (dsm[rows, cr] - dsm[rows, cl]) / (2.0 * gsd)
    dzdy = (dsm[rd, cols] - dsm[ru, cols]) / (2.0 * gsd)
    nx = np.where(np.isnan(dzdx), 0.0, -dzdx).astype(np.float32)
    ny = np.where(np.isnan(dzdy), 0.0, -dzdy).astype(np.float32)
    norm = np.sqrt(nx * nx + ny * ny + 1.0).astype(np.float32)
    out = np.empty((len(rows), 3), dtype=np.float32)
    out[:, 0] = nx / norm
    out[:, 1] = ny / norm
    out[:, 2] = 1.0 / norm
    return out
