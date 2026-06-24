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


def _smooth_fill_components(
    grid: NDArray,
    labels: NDArray,
    fill_ids: NDArray,
    sample_valid: NDArray,
    ring_iters: int = 3,
) -> None:
    """Fill large COLOUR-hole components by solving Laplace's equation inside each
    hole with the surrounding valid cells as the Dirichlet boundary — a HARMONIC
    fill, the smooth-fill replacement for ``_linear_fill_components`` on the
    ortho.

    Linear (Delaunay) interpolation triangulates the boundary samples, and across
    a wide irregular occluded strip those triangles render as radiating *fan*
    streaks.  A fast-march inpaint (``cv2.inpaint``) removes the fans but leaves
    its own medial-axis streaks on wide regions.  The harmonic solution is the
    smoothest interpolant of the border colour — no triangulation, no streaks —
    and (unlike bounded diffusion) fills any width exactly in one direct solve.

    Per-component, cropped to the bbox (+``ring_iters``+2 pad) so cost scales with
    hole size, not grid size; ``grid`` (``(H,W,C)`` colour) is modified in place.
    Only ``sample_valid`` cells inside the crop pin the solution (Dirichlet); the
    no-data exterior and other holes are free (Neumann), so nothing bleeds in.
    """
    from scipy import sparse as sp
    from scipy.sparse.linalg import splu

    H, W = labels.shape
    nchan = grid.shape[2]
    pad = ring_iters + 2
    boxes = ndimage.find_objects(labels)
    for lid in fill_ids:
        box = boxes[lid - 1]
        if box is None:
            continue
        y0 = max(box[0].start - pad, 0)
        y1 = min(box[0].stop + pad, H)
        x0 = max(box[1].start - pad, 0)
        x1 = min(box[1].stop + pad, W)

        sub = grid[y0:y1, x0:x1]  # a view → writes propagate to grid
        comp = labels[y0:y1, x0:x1] == lid
        seed = sample_valid[y0:y1, x0:x1]  # Dirichlet boundary values live here
        if not seed.any():
            continue
        hN, wN = comp.shape
        ys, xs = np.nonzero(comp)
        n = ys.size
        if n == 0:
            continue
        uidx = np.full((hN, wN), -1, np.int64)
        uidx[ys, xs] = np.arange(n)

        # 5-point Laplacian over the unknown (hole) cells.  Each row:
        # diag·u - Σ neighbour_unknown = Σ neighbour_seed_value.  Neighbours that
        # are neither unknown nor seed (exterior / other holes) and out-of-bounds
        # neighbours are dropped (homogeneous Neumann → no-flux), which keeps the
        # fill anchored only to real colour.
        diag = np.zeros(n)
        rhs = np.zeros((n, nchan))
        off_r: list = []
        off_c: list = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = ys + dy
            nx = xs + dx
            inb = (ny >= 0) & (ny < hN) & (nx >= 0) & (nx < wN)
            nyc = np.clip(ny, 0, hN - 1)
            nxc = np.clip(nx, 0, wN - 1)
            nb_u = inb & (uidx[nyc, nxc] >= 0)
            nb_s = inb & seed[nyc, nxc] & ~nb_u
            diag += (nb_u | nb_s).astype(np.float64)
            uu = np.nonzero(nb_u)[0]
            off_r.append(uu)
            off_c.append(uidx[nyc[uu], nxc[uu]])
            ss = np.nonzero(nb_s)[0]
            if ss.size:
                rhs[ss] += sub[nyc[ss], nxc[ss], :]
        diag[diag == 0.0] = 1.0  # isolated cell with no valid neighbour → keep 0

        rows = np.concatenate([np.arange(n)] + off_r)
        cols = np.concatenate([np.arange(n)] + off_c)
        vals = np.concatenate([diag] + [-np.ones(u.size) for u in off_r])
        amat = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        lu = splu(amat)
        out = np.empty((n, nchan))
        for ch in range(nchan):
            out[:, ch] = lu.solve(rhs[:, ch])
        sub[ys, xs, :] = np.clip(out, 0, 255).astype(grid.dtype)


def _low_flat_fill_components(
    grid: NDArray,
    labels: NDArray,
    fill_ids: NDArray,
    sample_valid: NDArray,
    low_percentile: float,
    ring_iters: int = 3,
    occlusion_drop: float = 0.0,
    no_relax_out: Optional[NDArray] = None,
) -> None:
    """Fill each large hole as a gravity-aligned FLAT surface at the LOW altitude
    of its boundary ring (single-channel DSM only).

    Models occlusion: the higher bordering surface (a roof — likely NOT occluded,
    so correctly reconstructed) keeps its height and meets the hole with a SHARP
    edge, while the hole — assumed to be occluded ground — is completed flat at
    the lower (``low_percentile``-th) altitude of its surrounding ring, rather
    than being slanted up toward the roof by linear interpolation.
    ``grid`` ``(H,W)`` float is modified in place; work is cropped per component.

    ``occlusion_drop``/``no_relax_out``: when a hole's ring spans more than
    ``occlusion_drop`` (high border minus the low fill altitude), the filled
    ground sits in a bordering roof's shadow → it is OCCLUDED, so its cells are
    marked in ``no_relax_out`` and the bake keeps the strict mask there (no
    roof-colour ghosting; falls to the smooth residual fill instead).
    """
    H, W = labels.shape
    pad = ring_iters + 1
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
        ring_vals = sub_grid[ring]
        if ring_vals.size == 0:
            continue
        # Low altitude of the border = the (occluded) ground level to continue.
        low = np.percentile(ring_vals, low_percentile)
        sub_grid[comp] = np.float32(low)

        # A tall border (roof) above this ground → the fill is occluded ground;
        # don't let the bake mask-relax it (would ghost roof colour onto it).
        if occlusion_drop > 0.0 and no_relax_out is not None:
            high = np.percentile(ring_vals, 100.0 - low_percentile)
            if high - low > occlusion_drop:
                no_relax_out[y0:y1, x0:x1][comp] = True


def _fill_holes_2pass(
    values: NDArray,
    sample_valid: NDArray,
    hole_mask: NDArray,
    *,
    small_area_max: int,
    diffuse_iters: int,
    kappa: float,
    dt: float,
    large_fill: str = "linear",
    low_percentile: float = 20.0,
    occlusion_drop: float = 0.0,
    device: int = 0,
) -> Tuple[NDArray, NDArray]:
    """Two-stage hole fill on a float grid (post-process).

    Stage 1 — tiny holes (connected components <= ``small_area_max`` cells):
    bounded GPU Perona-Malik diffusion seeded from the surrounding data.
    Stage 2 — larger holes: ``large_fill="linear"`` per-component linear
    (Delaunay) interpolation from their boundary ring, or ``"low_flat"`` a
    gravity-aligned flat fill at the ``low_percentile`` border altitude
    (DSM only — continues occluded ground flat with a sharp roof edge).

    ``values``      : float ``(H,W)`` or ``(H,W,C)``.
    ``sample_valid``: bool ``(H,W)``, True where real data exists (samples,
                      frozen during diffusion).
    ``hole_mask``   : bool ``(H,W)``, cells to fill.  Cells that are neither a
                      sample nor a hole are left untouched (never used as a
                      sample, never frozen).  Every cell in ``hole_mask`` is
                      filled — the caller is responsible for excluding the
                      genuine no-data exterior (DSM: the footprint test; ortho:
                      ``has_dsm``).
    Returns ``(filled_copy_float32, no_relax_mask_bool)`` where the second marks
    cells the ortho bake must NOT mask-relax: linear → cells extrapolated by
    nearest-neighbour (outside the hull); low_flat → occluded holes (a tall roof
    border above the ground fill).
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
    n_large = int(comp_size[stage2_ids].sum()) if stage2_ids.size else 0
    n_nodata = int(hole_mask.sum()) - n_tiny - n_large
    mode = {"low_flat": "flat-low", "inpaint": "inpaint"}.get(
        large_fill, "linear")
    logger.info(
        f"Hole fill: {n_tiny} tiny cells (diffusion), {n_large} enclosed "
        f"cells ({mode}), {n_nodata} cells left as no-data"
    )

    if stage2_ids.size:
        if large_fill == "low_flat":
            # Flat gravity-aligned fill (DSM).  `extrap` here carries the
            # no-mask-relax cells = the OCCLUDED holes (ground in a roof's
            # shadow), so the bake won't ghost roof colour onto them.
            _low_flat_fill_components(
                out, labels, stage2_ids, stage2_valid, low_percentile,
                occlusion_drop=occlusion_drop, no_relax_out=extrap,
            )
        elif large_fill == "inpaint":
            # Smooth HARMONIC (Laplace) fill (ortho colour): no Delaunay fan
            # streaks across wide occluded strips, and no fast-march medial
            # streaks either.  Leaves `extrap` empty (unused by the ortho fill).
            _smooth_fill_components(out, labels, stage2_ids, stage2_valid)
        else:
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
    ones by edge-aware GPU diffusion, larger ones by a gravity-aligned FLAT fill
    at the low border altitude (continues occluded ground flat, leaving a sharp
    edge under the bordering roof — not a linear slant).  The no-data EXTERIOR
    outside the footprint is kept as no-data (ortho stays transparent there).

    Returns ``(filled_dsm, no_relax_mask)``; the flat fill marks the OCCLUDED
    holes (ground sitting below a tall roof border, > ``hole_fill_occlusion_drop``
    m) so the bake keeps the strict mask there — no roof-colour ghosting; those
    fall to the smooth residual fill.  Open/shallow filled holes are left
    mask-relaxed (real colour, nadir-only gate keeps it height-robust).
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
        large_fill="low_flat",  # continue occluded ground flat, sharp roof edge
        low_percentile=float(config["hole_fill_low_percentile"]),
        occlusion_drop=float(config["hole_fill_occlusion_drop"]),
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
    ones by a smooth fast-march inpaint (NOT Delaunay linear — its triangulation
    fanned out as streaks across wide occluded strips).  Cells with no DSM stay
    untouched (black).

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
        large_fill="inpaint",  # smooth fast-march fill, no Delaunay fan streaks
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
