# pyre-strict
"""DSM / ortho post-processing helpers.

Pure grid operations used by the fusion stage to finish the DSM and ortho:
GPU Perona-Malik diffusion, per-component linear hole fill, footprint
detection, two-pass hole filling, edge-sharpening shock filter, gated median
despeckle, and DSM-derived point normals.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy import sparse as sp
from scipy.sparse.linalg import cg
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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
        # Dirichlet boundary values live here
        seed = sample_valid[y0:y1, x0:x1]
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
        # isolated cell with no valid neighbour → keep 0
        diag[diag == 0.0] = 1.0

        rows = np.concatenate([np.arange(n)] + off_r)
        cols = np.concatenate([np.arange(n)] + off_c)
        vals = np.concatenate([diag] + [-np.ones(u.size) for u in off_r])
        amat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        m_inv = sp.diags(1.0 / diag)
        maxiter = 4 * int(np.ceil(np.sqrt(n))) + 200
        out = np.empty((n, nchan))
        for ch in range(nchan):
            sol, _info = cg(
                amat, rhs[:, ch], M=m_inv, rtol=1e-3, atol=0.0, maxiter=maxiter
            )
            out[:, ch] = sol
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


def _component_elongation(labels: NDArray, n_comp: int) -> NDArray:
    """Per-component elongation: ratio of the MAJOR to MINOR principal-axis
    extent of each hole's cells (≈ aspect ratio; 1 = round/compact, large = long
    and thin).
    """
    elong = np.ones(n_comp + 1, dtype=np.float64)
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return elong
    lab = labels[ys, xs]
    ml = n_comp + 1
    yf = ys.astype(np.float64)
    xf = xs.astype(np.float64)
    n = np.bincount(lab, minlength=ml).astype(np.float64)
    nn = np.where(n >= 1.0, n, 1.0)
    my = np.bincount(lab, weights=yf, minlength=ml) / nn
    mx = np.bincount(lab, weights=xf, minlength=ml) / nn
    cyy = np.bincount(lab, weights=yf * yf, minlength=ml) / nn - my * my
    cxx = np.bincount(lab, weights=xf * xf, minlength=ml) / nn - mx * mx
    cxy = np.bincount(lab, weights=xf * yf, minlength=ml) / nn - mx * my
    # Eigenvalues of the 2x2 symmetric covariance [[cyy, cxy], [cxy, cxx]].
    tr = 0.5 * (cyy + cxx)
    disc = np.sqrt(np.maximum(0.0, (0.5 * (cyy - cxx)) ** 2 + cxy * cxy))
    lam1 = tr + disc                                    # major-axis variance
    lam2 = tr - disc                                    # minor-axis variance
    eps = 1e-6
    ratio = np.sqrt(np.maximum(lam1, 0.0) / np.maximum(lam2, eps))
    use = (n >= 2.0) & (lam1 > eps)
    elong[use] = ratio[use]
    return elong


def _component_thickness(
    hole_mask: NDArray, labels: NDArray, n_comp: int
) -> NDArray:
    """Per-component morphological THICKNESS: the radius (in cells) of the largest
    disk that fits inside each hole — the maximum of the Euclidean distance
    transform (distance to the nearest non-hole cell) over the component.

    Returns ``(n_comp + 1,)`` indexed by label id (0.0 for label 0 / absent).
    """
    thick = np.zeros(n_comp + 1, dtype=np.float64)
    if not hole_mask.any():
        return thick
    dt = ndimage.distance_transform_edt(np.pad(hole_mask, 1))[1:-1, 1:-1]
    maxes = ndimage.maximum(dt, labels=labels, index=np.arange(1, n_comp + 1))
    thick[1:] = np.asarray(maxes, dtype=np.float64)
    return thick


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
    max_aspect: float = 0.0,
    min_thickness: float = 0.0,
    device: int = 0,
) -> Tuple[NDArray, NDArray]:
    """Two-stage hole fill on a float grid (post-process).

    Stage 1 — bounded GPU Perona-Malik diffusion seeded from the surrounding
    data (smooth, edge-aware fill that follows the local terrain).
    Stage 2 — per-component fill of the remaining holes: ``large_fill="linear"``
    per-component linear (Delaunay) interpolation from their boundary ring, or
    ``"low_flat"`` a gravity-aligned flat fill at the ``low_percentile`` border
    altitude (DSM only — continues occluded ground flat with a sharp roof edge).

    Which hole goes to which stage (DSM ``low_flat`` path): a hole is flat-filled
    (Stage 2) only if it passes EVERY active geometric gate, else it is diffused
    (Stage 1 — follows the terrain).  Flat-filling at the single low border
    altitude is only valid for a CHUNKY, COMPACT hole; on a long or thin one that
    altitude is unrelated to the far end and carves a canyon.  The gates:
      * COMPACT — ``max_aspect > 0``: cell aspect ratio (major/minor axis) <=
        ``max_aspect``.  Rejects long elongated holes.
      * THICK — ``min_thickness > 0``: largest inscribed-disk radius (max distance
        transform, in cells) >= ``min_thickness``.  Rejects THIN / feathery /
        branching holes that the aspect ratio misses — a wiggly feather has a
        near-round bounding box but is thin everywhere.

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
    multichannel = out.ndim == 3
    nchan = out.shape[2] if multichannel else 1

    # ── Route each hole to one of the two stages ──
    if large_fill == "low_flat" and (max_aspect > 0.0 or min_thickness > 0.0):
        to_flat = np.ones(n_comp + 1, dtype=bool)
        to_flat[0] = False                              # label 0 = background
        if max_aspect > 0.0:
            to_flat &= _component_elongation(labels, n_comp) <= max_aspect
        if min_thickness > 0.0:
            to_flat &= _component_thickness(
                hole_mask, labels, n_comp) >= min_thickness
        large_ids = np.nonzero(to_flat)[0]
        diffuse_comp = ~to_flat
        diffuse_comp[0] = False
        diffuse_mask = hole_mask & diffuse_comp[labels]
    else:
        # Every hole in ``hole_mask`` is fillable here — the genuine no-data
        # exterior was already excluded by the caller (DSM footprint / ortho
        # ``has_dsm``).
        diffuse_mask = hole_mask & (comp_size[labels] <= small_area_max)
        large_ids = np.nonzero(comp_size > small_area_max)[0]
        large_ids = large_ids[large_ids != 0]  # drop the non-hole label 0

    # Stage 1: GPU diffusion for the smooth (or, legacy, tiny) holes.  Only
    # sample cells are frozen; holes and ignored cells are NaN so they neither
    # seed nor bleed.
    diffuse_ok = True
    if diffuse_mask.any():
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
                out[..., c][diffuse_mask] = filled[diffuse_mask]
            else:
                out[diffuse_mask] = filled[diffuse_mask]

    # Stage 2: per-component fill for the remaining (large / stepped) holes
    # (and the diffused holes too if the diffusion pass was unavailable).
    if diffuse_ok:
        stage2_ids = large_ids
        stage2_valid = sample_valid | diffuse_mask  # diffused holes now hold values
    else:
        all_ids = np.nonzero(comp_size != 0)[0]
        stage2_ids = all_ids[all_ids != 0]
        stage2_valid = sample_valid

    n_diffused = int(diffuse_mask.sum()) if diffuse_ok else 0
    n_large = int(comp_size[stage2_ids].sum()) if stage2_ids.size else 0
    n_nodata = int(hole_mask.sum()) - n_diffused - n_large
    mode = {"low_flat": "flat-low", "inpaint": "inpaint"}.get(
        large_fill, "linear")
    logger.info(
        f"Hole fill: {n_diffused} cells (diffusion), {n_large} cells "
        f"({mode}), {n_nodata} cells left as no-data"
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

    ch0 = out[..., 0] if multichannel else out
    residual = hole_mask & ~np.isfinite(ch0)
    n_residual = int(residual.sum())
    if n_residual:
        src = np.isfinite(ch0) & ~residual  # every cell that now holds a value
        if src.any():
            pad = 64
            H, W = out.shape[:2]
            rlbl, n_res_comp = ndimage.label(residual)
            boxes = ndimage.find_objects(rlbl)
            n_mopped = 0
            for cid in range(1, n_res_comp + 1):
                box = boxes[cid - 1]
                if box is None:
                    continue
                y0 = max(box[0].start - pad, 0)
                y1 = min(box[0].stop + pad, H)
                x0 = max(box[1].start - pad, 0)
                x1 = min(box[1].stop + pad, W)
                sub_src = src[y0:y1, x0:x1]
                if not sub_src.any():
                    continue  # no filled cell within reach (left as no-data)
                sub_res = rlbl[y0:y1, x0:x1] == cid
                _, (iy, ix) = ndimage.distance_transform_edt(
                    ~sub_src, return_indices=True)
                sub_out = out[y0:y1, x0:x1]
                sub_out[sub_res] = sub_out[iy[sub_res], ix[sub_res]]
                n_mopped += int(sub_res.sum())
            logger.info(
                f"Hole fill: mopped up {n_mopped}/{n_residual} residual cell(s) "
                "(filler bailed) from nearest filled neighbour"
            )
        else:
            logger.info(
                f"Hole fill: {n_residual} residual cell(s) left as no-data "
                "(no filled neighbour to source from)"
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
        n = int(close_iters)
        padded = np.pad(valid, n)
        closed = ndimage.binary_closing(
            padded, structure=structure, iterations=n
        )[n:-n, n:-n]
    else:
        closed = valid
    return ndimage.binary_fill_holes(closed)


def _fill_dsm_holes(
    dsm_grid: NDArray,
    config: Dict[str, Any],
    footprint: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray]:
    """Fill DSM no-data holes that lie INSIDE the completable footprint: tiny
    ones by edge-aware GPU diffusion, larger ones by a gravity-aligned FLAT fill
    at the low border altitude (continues occluded ground flat, leaving a sharp
    edge under the bordering roof — not a linear slant).  The no-data EXTERIOR
    outside the footprint is kept as no-data (ortho stays transparent there).

    ``footprint`` (bool ``(H, W)``) overrides which no-data is fillable.  Callers
    pass the chunk's completion TERRITORY here: on a disjoint KD-tree chunk the
    big cross-chunk holes touch the tile edge, so the default morphological
    footprint (``_dsm_footprint``) treats them as exterior and never fills them;
    the territory mask marks them fillable instead.  ``None`` → derive the
    footprint from the reconstructed cells (legacy / full-extent behaviour).

    Each hole is routed between the smooth diffusion fill and the flat-low fill
    by its SHAPE: only a COMPACT (``hole_fill_low_flat_max_aspect``) and THICK
    (``hole_fill_low_flat_min_thickness``) hole gets the flat fill (sharp
    occluded-ground edge); elongated OR thin/feathery holes are diffused (follow
    the terrain; avoids carving a ridge or a thin string into a flat canyon).
    See ``_fill_holes_2pass``.

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

    # Local enclosure: holes the RENDERED surface encloses, at pixel
    # granularity.  This alone catches the coarse-cell voids inside the surface
    # (an empty coarse cell renders as a NaN square; binary_fill_holes encloses
    # it) — so they never survive as coarse-grid speckle.
    local = _dsm_footprint(valid, int(config["hole_fill_footprint_close"]))
    # ``footprint`` (the chunk's completion territory) EXTENDS the local
    # enclosure to the big cross-chunk holes that touch the tile edge and thus
    # look "exterior" locally.  Union, so a cell is fillable if locally enclosed
    # OR in the territory — neither bucket can drop an interior hole.
    footprint = local if footprint is None else (local | footprint)
    fillable = (~valid) & footprint
    # Diagnostic: split the no-data into FILLABLE (inside the completion
    # footprint — local enclosure or territory) vs genuine EXTERIOR (outside it,
    # kept transparent).  If the void squares you see correspond to the
    # "exterior" count, the cause is the MASK (render produced an empty coarse
    # cell the footprint doesn't reach); if to "fillable" residual after the
    # fill, the cause is the FILLER bailing (mopped up below).
    n_nd = int((~valid).sum())
    n_fill = int(fillable.sum())
    n_local = int(((~valid) & local).sum())
    logger.info(
        f"DSM holes: {n_nd} no-data → {n_fill} fillable "
        f"({n_local} via local enclosure) + {n_nd - n_fill} exterior (kept)"
    )

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
        max_aspect=float(config["hole_fill_low_flat_max_aspect"]),
        min_thickness=float(config["hole_fill_low_flat_min_thickness"]),
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


def _inject_ortho_detail(
    ortho_grid: NDArray,
    sharp_grid: NDArray,
    baked_mask: NDArray,
    config: Dict[str, Any],
) -> NDArray:
    """Add the single-sharpest-view's high-frequency texture back onto the blend.

    The multi-view colour bake gives a SMOOTH, robust, seam-free blend
    (``ortho_grid``) but low-passes texture; ``sharp_grid`` holds, per cell, the
    raw colour of the single sharpest inlier view (full detail but, taken alone,
    patchy where the chosen view changes — a LOW-frequency exposure/colour step).
    We composite the two by frequency band:

        final = blend + strength * (sharp - lowpass(sharp))

    The high-pass ``sharp - lowpass(sharp)`` carries texture but not exposure, so
    the sharp view's single-source patchiness (all low-frequency) is removed and
    never appears — only its detail is injected.  The low-pass is a *masked*
    Gaussian (normalised by the blurred coverage mask) so no-data borders do not
    bleed dark halos into the detail.  Only ``baked_mask`` cells are touched.
    """
    sigma = float(config["ortho_detail_sigma"])
    strength = float(config["ortho_detail_strength"])
    if sigma <= 0.0 or strength <= 0.0 or not baked_mask.any():
        return ortho_grid

    m = baked_mask.astype(np.float32)
    s = sharp_grid.astype(np.float32)
    # Masked Gaussian low-pass: blur(sharp*mask) / blur(mask) keeps the average
    # anchored to real colour, so cells near the no-data boundary are not pulled
    # toward black (which would inject a bright false-edge halo).
    blur_s = cv2.GaussianBlur(s * m[..., None], (0, 0), sigma)
    blur_m = cv2.GaussianBlur(m, (0, 0), sigma)
    low = blur_s / np.maximum(blur_m, 1e-6)[..., None]
    high = s - low  # high-frequency texture (exposure-free)

    out = ortho_grid.astype(np.float32) + strength * high
    out = np.clip(out, 0.0, 255.0).astype(ortho_grid.dtype)
    # Only inject where a real sharp source exists; elsewhere keep the blend.
    return np.where(baked_mask[..., None], out, ortho_grid)


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
