# pyre-strict
"""Radiometric equalization of the image set from SfM track correspondences.

A track is one 3-D point seen in several images; its observations therefore
record the SAME surface colour as measured by different images.  Differences
between those measurements are the images' radiometric mismatch — exposure /
white-balance drift (per-image, per-channel gain) and lens vignetting (a radial
brightness falloff).  This module estimates, per image, a multiplicative
correction that brings every image onto a common radiometric frame, so the
downstream colour bake no longer sees exposure/vignette steps between source
images (the low-frequency mismatch that makes ortho source-switching visible).

Model (per colour channel, solved independently so white balance is corrected
too).  For an observation of track ``t`` in image ``i`` at normalised radius
``rho`` (0 at the principal point, 1 at the image corner) with measured
intensity ``I``::

    log I  =  alpha_i  +  sum_k gamma_{i,k} * rho^{2k}  +  lambda_t

  alpha_i        per-image log-gain (exposure + white balance)
  gamma_{i,k}    per-image vignetting log-coefficients (basis rho^2, rho^4, …);
                 V_i(rho) = exp(sum_k gamma_{i,k} rho^{2k}), V_i(0) = 1 by
                 construction (the constant term is the gain alpha_i)
  lambda_t       per-track log-albedo (the unknown true colour; nuisance var)

This is LINEAR in the unknowns, so the whole image set is solved jointly as one
large sparse least-squares per channel via preconditioned conjugate gradient
(PCG) on the normal equations, wrapped in IRLS / Huber reweighting so specular
highlights, moving objects and mis-tracks are down-weighted.  The global
scale/gauge (adding a constant to every alpha and subtracting it from every
lambda is unobservable) is fixed by a soft ``sum(alpha) = 0`` constraint, which
also keeps the mean brightness unchanged.

A purely multiplicative (gain + vignette) model is used — no additive black
level: an additive offset is poorly constrained by sparse track colours and
risks washing out contrast; the per-channel gain already covers exposure and
white balance.

The per-image result (gain + vignetting coefficients + the radius normalisation)
is saved so the colour bake can divide it out per sampled pixel.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from scipy.sparse.linalg import cg

from opensfm import pymap, types

logger: logging.Logger = logging.getLogger(__name__)

# Channel order matches the observation colour triplet (RGB).
_NUM_CHANNELS: int = 3


def _camera_radius_norm(camera: "Any") -> Tuple[NDArray, float]:
    """Vignetting centre (principal point) and corner radius in NORMALISED image
    coordinates (OpenSfM convention: pixel offset from the image centre divided
    by ``max(width, height)``), so a measured ``obs.point`` maps to ``rho`` in
    ``[0, 1]`` via ``|obs.point - pp| / rmax``.
    """
    w = float(getattr(camera, "width", 0) or 0)
    h = float(getattr(camera, "height", 0) or 0)
    if w <= 0.0 or h <= 0.0:
        # No pixel size known — normalised coords already ~[-0.5, 0.5]; the
        # half-diagonal of a unit-longer-side frame is 0.5*sqrt(2).
        rmax = 0.5 * np.sqrt(2.0)
    else:
        rmax = 0.5 * np.sqrt(w * w + h * h) / max(w, h)
    pp = np.zeros(2, dtype=np.float64)
    try:
        # Some camera models (e.g. perspective) carry no principal-point
        # parameter and raise rather than return None — fall back to centre (0).
        ppt = camera.principal_point
        pp = np.asarray(ppt, dtype=np.float64).reshape(-1)[:2]
    except Exception:
        pass
    return pp, float(rmax)


class _Observations:
    """Flat arrays of the track observations used by the solver.

    ``img`` (Nobs,)   image index, ``trk`` (Nobs,) track index, ``rho2`` (Nobs,)
    squared normalised radius, ``logI``/``color`` (Nobs, 3) measured colour
    (log and raw).  ``shot_ids`` indexes ``img``; ``n_tracks`` sizes ``trk``.
    """

    def __init__(
        self,
        img: NDArray,
        trk: NDArray,
        rho2: NDArray,
        color: NDArray,
        shot_ids: List[str],
        n_tracks: int,
    ) -> None:
        self.img = img
        self.trk = trk
        self.rho2 = rho2
        self.color = color
        self.logI: NDArray = np.log(np.clip(color, 1.0, 255.0))
        self.shot_ids = shot_ids
        self.n_tracks = n_tracks


def _collect_observations(
    tracks_manager: "pymap.TracksManager",
    reconstruction: "types.Reconstruction",
    min_track_length: int,
    max_observations: int,
) -> Optional[_Observations]:
    """Gather per-(image, track) colours + radii for reconstructed shots.

    Only tracks seen by ``>= min_track_length`` reconstructed images contribute
    (a track in one image couples no images).  If the total exceeds
    ``max_observations`` a deterministic random subset of TRACKS is kept (whole
    tracks, so every kept track keeps its cross-image constraints).
    """
    shots = reconstruction.shots
    shot_index: Dict[str, int] = {
        sid: i for i, sid in enumerate(sorted(shots.keys()))
    }
    shot_ids: List[str] = sorted(shots.keys())

    # Precompute per-camera radius normalisation (pp, rmax).
    cam_norm: Dict[str, Tuple[NDArray, float]] = {}
    for sid in shot_ids:
        cam = shots[sid].camera
        cam_norm[sid] = _camera_radius_norm(cam)

    track_ids = list(tracks_manager.get_track_ids())
    rng = np.random.default_rng(0)
    rng.shuffle(track_ids)

    img_list: List[int] = []
    trk_list: List[int] = []
    rho2_list: List[float] = []
    color_list: List[Tuple[float, float, float]] = []
    n_tracks = 0
    for tid in track_ids:
        obs = tracks_manager.get_track_observations(tid)
        kept = [(sid, o) for sid, o in obs.items() if sid in shot_index]
        if len(kept) < min_track_length:
            continue
        t_local = n_tracks
        n_tracks += 1
        for sid, o in kept:
            pp, rmax = cam_norm[sid]
            p = np.asarray(o.point, dtype=np.float64).reshape(-1)[:2]
            d = p - pp
            rho = float(np.sqrt(d[0] * d[0] + d[1] * d[1]) / max(rmax, 1e-9))
            col = np.asarray(o.color, dtype=np.float64).reshape(-1)[:3]
            img_list.append(shot_index[sid])
            trk_list.append(t_local)
            rho2_list.append(rho * rho)
            color_list.append((col[0], col[1], col[2]))
        if len(img_list) >= max_observations:
            logger.info(
                "Equalize: reached the %d-observation cap; using %d tracks",
                max_observations, n_tracks,
            )
            break

    if n_tracks == 0 or not img_list:
        return None

    return _Observations(
        img=np.asarray(img_list, dtype=np.int64),
        trk=np.asarray(trk_list, dtype=np.int64),
        rho2=np.asarray(rho2_list, dtype=np.float64),
        color=np.asarray(color_list, dtype=np.float64),
        shot_ids=shot_ids,
        n_tracks=n_tracks,
    )


def _build_design(
    obs: _Observations, vignette_order: int
) -> Tuple[sp.csr_matrix, int, int]:
    """Sparse design matrix ``M`` (Nobs × Nunknowns) for one channel's data term.

    Unknown layout: ``[alpha (Nimg) | gamma_k (Nimg) for k=1..V | lambda (Ntrk)]``.
    Row for observation o = (image i, track t, rho): ``1`` at ``alpha_i``,
    ``rho^{2k}`` at each ``gamma_{i,k}``, ``1`` at ``lambda_t``.  ``M`` is shared
    by all three channels (only the RHS / weights differ).
    """
    n_img = len(obs.shot_ids)
    n_trk = obs.n_tracks
    n_obs = len(obs.img)
    v = int(vignette_order)
    n_unknown = n_img * (1 + v) + n_trk
    alpha0 = 0
    gamma0 = n_img            # gamma_k block starts here; image i, order k at
    lambda0 = n_img * (1 + v)  # gamma0 + (k-1)*n_img + i

    rows_per = 2 + v  # alpha + lambda + v vignette terms
    rows = np.empty(n_obs * rows_per, dtype=np.int64)
    cols = np.empty(n_obs * rows_per, dtype=np.int64)
    vals = np.empty(n_obs * rows_per, dtype=np.float64)
    ar = np.arange(n_obs, dtype=np.int64)

    # alpha_i term
    rows[0::rows_per] = ar
    cols[0::rows_per] = alpha0 + obs.img
    vals[0::rows_per] = 1.0
    # lambda_t term
    rows[1::rows_per] = ar
    cols[1::rows_per] = lambda0 + obs.trk
    vals[1::rows_per] = 1.0
    # vignette terms rho^{2k}
    rho2k = obs.rho2.copy()
    for k in range(v):
        off = 2 + k
        rows[off::rows_per] = ar
        cols[off::rows_per] = gamma0 + k * n_img + obs.img
        vals[off::rows_per] = rho2k
        rho2k = rho2k * obs.rho2  # rho^(2(k+1)) for the next order

    M = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_obs, n_unknown)
    )
    return M, n_img, n_trk


def _huber_weights(residual: NDArray, delta_sigma: float) -> NDArray:
    """Huber IRLS weights from residuals: 1 inside the robust band, falling off
    as ``delta*sigma/|r|`` outside it.  ``sigma`` is a robust MAD scale."""
    absr = np.abs(residual)
    mad = float(np.median(absr)) if absr.size else 0.0
    sigma = max(1.4826 * mad, 1e-3)
    d = delta_sigma * sigma
    w = np.ones_like(absr)
    big = absr > d
    w[big] = d / absr[big]
    return w


def _solve_channel(
    M: sp.csr_matrix,
    y: NDArray,
    base_w: NDArray,
    n_img: int,
    n_trk: int,
    vignette_order: int,
    config: Dict[str, Any],
) -> NDArray:
    """IRLS + PCG solve of one channel's weighted least squares with gauge and
    vignette regularisation.  Returns the unknown vector ``x``."""
    n_unknown = M.shape[1]
    v = int(vignette_order)
    gamma0 = n_img
    irls_iters = int(config["equalize_irls_iterations"])
    delta_sigma = float(config["equalize_huber_delta"])
    lam_vig = float(config["equalize_vignette_regularization"])
    lam_gain = float(config["equalize_gain_regularization"])
    gauge_w = float(config["equalize_gauge_weight"])

    # Fixed regularisation as extra weighted rows appended to the system:
    #   gauge:    sum(alpha) = 0                       (resolve the global scale)
    #   vignette: gamma_{i,k} -> 0                     (prior: gentle falloff)
    #   gain:     alpha_i -> 0  (tiny, conditioning)
    reg_rows: List[sp.csr_matrix] = []
    reg_rhs: List[NDArray] = []
    reg_w: List[NDArray] = []
    # gauge: one row of ones over the alpha block
    g_row = sp.csr_matrix(
        (np.ones(n_img), (np.zeros(n_img, dtype=np.int64), np.arange(n_img))),
        shape=(1, n_unknown),
    )
    reg_rows.append(g_row)
    reg_rhs.append(np.zeros(1))
    reg_w.append(np.array([gauge_w]))
    # vignette ridge over the whole gamma block
    n_gamma = n_img * v
    if n_gamma and lam_vig > 0.0:
        gcols = np.arange(gamma0, gamma0 + n_gamma)
        gmat = sp.csr_matrix(
            (np.ones(n_gamma), (np.arange(n_gamma), gcols)),
            shape=(n_gamma, n_unknown),
        )
        reg_rows.append(gmat)
        reg_rhs.append(np.zeros(n_gamma))
        reg_w.append(np.full(n_gamma, lam_vig))
    # tiny gain ridge over the alpha block (conditioning only)
    if lam_gain > 0.0:
        amat = sp.csr_matrix(
            (np.ones(n_img), (np.arange(n_img), np.arange(n_img))),
            shape=(n_img, n_unknown),
        )
        reg_rows.append(amat)
        reg_rhs.append(np.zeros(n_img))
        reg_w.append(np.full(n_img, lam_gain))

    R = sp.vstack(reg_rows, format="csr")
    r_rhs = np.concatenate(reg_rhs)
    r_w = np.concatenate(reg_w)

    x = np.zeros(n_unknown)
    w = base_w.copy()
    for it in range(max(1, irls_iters)):
        # Weighted normal equations  (Mᵀ W M + Rᵀ Wr R) x = Mᵀ W y + Rᵀ Wr r
        Wm = M.multiply(w[:, None]).tocsr()
        A = (M.T @ Wm) + (R.T @ R.multiply(r_w[:, None]))
        b = M.T @ (w * y) + R.T @ (r_w * r_rhs)
        A = A.tocsr()
        diag = A.diagonal()
        diag[diag <= 0.0] = 1.0
        precond = sp.diags(1.0 / diag)
        maxiter = int(config["equalize_pcg_max_iterations"])
        x, info = cg(A, b, x0=x, rtol=float(config["equalize_pcg_tol"]),
                     atol=0.0, maxiter=maxiter, M=precond)
        if it + 1 < irls_iters:
            resid = M @ x - y
            w = base_w * _huber_weights(resid, delta_sigma)
    return x


def estimate_image_corrections(
    tracks_manager: "pymap.TracksManager",
    reconstruction: "types.Reconstruction",
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Estimate per-image gain + radial vignetting from the track correspondences.

    Returns ``{shot_id: {"gain": [gR,gG,gB], "vignette": [[k1,k2]_R, _G, _B],
    "pp": [x,y], "rmax": r, "vignette_order": V}}`` where the per-pixel
    correction for a sample at normalised point ``p`` is, per channel::

        rho = |p - pp| / rmax
        corrected = I / (gain * exp(sum_k vignette[k] * rho^{2(k+1)}))

    With the ``sum(alpha)=0`` gauge the gains are centred on 1 (geometric mean),
    so the overall brightness is preserved.
    """
    vignette_order = int(config["equalize_vignette_order"])
    sat = float(config["equalize_saturation_margin"])

    obs = _collect_observations(
        tracks_manager, reconstruction,
        min_track_length=int(config["equalize_min_track_length"]),
        max_observations=int(config["equalize_max_observations"]),
    )
    if obs is None:
        logger.warning(
            "Equalize: no multi-view tracks among reconstructed shots; "
            "emitting identity corrections."
        )
        return _identity_corrections(reconstruction, vignette_order)

    n_img = len(obs.shot_ids)
    logger.info(
        "Equalize: %d observations over %d tracks and %d images "
        "(vignette order %d)",
        len(obs.img), obs.n_tracks, n_img, vignette_order,
    )

    M, n_img, n_trk = _build_design(obs, vignette_order)

    # Per-channel solve.  Saturated/clipped measurements carry no reliable log
    # value → zero base weight (kept in the system so indices stay aligned).
    gains = np.ones((n_img, _NUM_CHANNELS))
    vignettes = np.zeros((n_img, _NUM_CHANNELS, vignette_order))
    rms_before: List[float] = []
    rms_after: List[float] = []
    for c in range(_NUM_CHANNELS):
        y = obs.logI[:, c]
        base_w = ((obs.color[:, c] > sat) &
                  (obs.color[:, c] < 255.0 - sat)).astype(np.float64)
        if base_w.sum() < n_img + n_trk:
            logger.warning(
                "Equalize: channel %d has few usable observations; "
                "result may be weak", c)
        x = _solve_channel(
            M, y, base_w, n_img, n_trk, vignette_order, config)
        gains[:, c] = np.exp(x[:n_img])
        for k in range(vignette_order):
            vignettes[:, c, k] = x[n_img + k * n_img: n_img + (k + 1) * n_img]
        # Consistency diagnostic: residual spread of observations about their
        # track mean, before vs after correction (lower = better aligned).
        m = base_w > 0
        rms_before.append(_track_consistency(obs, y, m))
        rms_after.append(_track_consistency(obs, y - (M @ x), m))

    logger.info(
        "Equalize: per-channel track-colour RMS (log) %.4f -> %.4f "
        "(R/G/B before %s, after %s)",
        float(np.mean(rms_before)), float(np.mean(rms_after)),
        np.round(rms_before, 4).tolist(), np.round(rms_after, 4).tolist(),
    )

    result: Dict[str, Dict[str, Any]] = {}
    for i, sid in enumerate(obs.shot_ids):
        cam = reconstruction.shots[sid].camera
        pp, rmax = _camera_radius_norm(cam)
        result[sid] = {
            "gain": [float(gains[i, c]) for c in range(_NUM_CHANNELS)],
            "vignette": [
                [float(vignettes[i, c, k]) for k in range(vignette_order)]
                for c in range(_NUM_CHANNELS)
            ],
            "pp": [float(pp[0]), float(pp[1])],
            "rmax": float(rmax),
            "vignette_order": vignette_order,
        }
    return result


def _track_consistency(
    obs: _Observations, values: NDArray, mask: NDArray
) -> float:
    """RMS spread of ``values`` about their per-track mean (over ``mask``).

    Measures how consistent a track's observations are with each other.  Passing
    the raw log-intensity gives the BEFORE figure; passing the fit residual
    ``logI - M x`` (whose per-track mean is ~0) gives the AFTER figure — both are
    the spread of the corrected colours about the track's common albedo."""
    vals = values[mask]
    trk = obs.trk[mask]
    if vals.size == 0:
        return 0.0
    n_trk = obs.n_tracks
    sums = np.bincount(trk, weights=vals, minlength=n_trk)
    cnts = np.bincount(trk, minlength=n_trk).astype(np.float64)
    cnts[cnts == 0] = 1.0
    means = sums / cnts
    centred = vals - means[trk]
    return float(np.sqrt(np.mean(centred * centred)))


def apply_equalization(
    image: NDArray, corr: Dict[str, Any], highlight_knee: float = 235.0
) -> NDArray:
    """Apply a per-image radiometric correction to an undistorted colour image.

    Divides out the estimated per-channel gain and radial vignetting so the image
    matches the common radiometric frame::

        rho   = |norm_xy - pp| / rmax              (normalised image coords)
        out_c = I_c / (gain_c * exp(sum_k vignette_c[k] * rho^{2(k+1)}))

    The normalisation is resolution-independent (offset from the image centre
    over ``max(w, h)``), so it is correct even though the bake samples a
    DOWNSCALED copy of the image.  ``corr`` is one entry of
    ``DataSet.load_equalization()``.

    Highlight preservation: a single per-image gain that lifts the midtones also
    multiplies the already-bright pixels, which then clip to white (a flat
    "burn").  ``highlight_knee`` rolls the correction off toward identity as the
    CORRECTED luminance rises above the knee, so highlights keep ~their original
    value (where the multiplicative model is unreliable anyway) instead of
    burning, while midtones get the full correction.  ``>= 255`` disables it.

    Returns a clipped array of ``image``'s dtype.
    """
    h, w = image.shape[:2]
    size = float(max(w, h))
    pp = corr["pp"]
    rmax = max(float(corr["rmax"]), 1e-9)
    gain = corr["gain"]
    vign = corr["vignette"]

    xs = (np.arange(w, dtype=np.float64) + 0.5 - 0.5 * w) / size - float(pp[0])
    ys = (np.arange(h, dtype=np.float64) + 0.5 - 0.5 * h) / size - float(pp[1])
    rho2 = (xs[None, :] ** 2 + ys[:, None] ** 2) / (rmax * rmax)  # (h, w)

    src = image.astype(np.float32)
    nch = min(_NUM_CHANNELS, src.shape[2])
    out = src.copy()
    for c in range(nch):
        logv = np.zeros_like(rho2)
        power = rho2.copy()
        for k in range(len(vign[c])):
            logv += float(vign[c][k]) * power
            power = power * rho2
        factor = 1.0 / (max(float(gain[c]), 1e-6) * np.exp(logv))
        out[..., c] = src[..., c] * factor.astype(np.float32)

    # Highlight roll-off: blend the corrected colour back toward the original as
    # the corrected luminance approaches white, so a brightening gain cannot burn
    # already-bright regions.  Luminance-based (shared across channels) so hue is
    # preserved.  s = 1 below the knee (full correction) → 0 at white (identity).
    if highlight_knee < 255.0 and nch >= 3:
        lum = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])
        s = np.clip((255.0 - lum) / (255.0 - highlight_knee), 0.0, 1.0)
        s = s[..., None]
        out[..., :nch] = out[..., :nch] * s + src[..., :nch] * (1.0 - s)

    return np.clip(out, 0.0, 255.0).astype(image.dtype)


def _identity_corrections(
    reconstruction: "types.Reconstruction", vignette_order: int
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sid in reconstruction.shots:
        cam = reconstruction.shots[sid].camera
        pp, rmax = _camera_radius_norm(cam)
        out[sid] = {
            "gain": [1.0, 1.0, 1.0],
            "vignette": [[0.0] * vignette_order for _ in range(_NUM_CHANNELS)],
            "pp": [float(pp[0]), float(pp[1])],
            "rmax": float(rmax),
            "vignette_order": vignette_order,
        }
    return out
