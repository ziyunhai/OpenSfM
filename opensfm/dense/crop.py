# pyre-strict
"""Crop the dense outputs to the SfM footprint.

The dense point cloud and the DSM/ortho rasters extend wherever *any* depthmap
reconstructed a surface — which includes a sparse, noisy fringe well beyond the
area the cameras actually surveyed (grazing background, sky-line bleed, a few
far outlier points).  To trim the products back to the genuinely covered area we
clip everything to the convex hull of the SfM points on the ground (XY) plane.

The hull is made robust to outlier points by trimming the extremities in an
**oriented (PCA) frame** before hulling: the XY points are rotated onto their
two principal axes, the outer ``percentile`` fraction along each axis is dropped,
and the convex hull is taken of what survives.  Rotating first means the trim
follows the dominant scene orientation (e.g. a flight strip) instead of the
world axes, so an elongated survey is not over-cropped on its long side.

The hull is computed by ``dense_clustering`` (the stage that already holds the
reconstruction) and consumed by ``dense_merging`` to clip the point cloud and
mask the DSM/ortho cells.  Everything is in topocentric XY — the frame shared by
the fused point cloud and the DSM/ortho grid *before* any georeferencing
reprojection — so a single hull crops every product consistently.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from opensfm import types

logger: logging.Logger = logging.getLogger(__name__)


def crop_hull_from_points(
    points_xyz: NDArray,
    percentile: float,
) -> Optional[NDArray]:
    """Convex hull (CCW topocentric XY polygon) of ``points_xyz`` after trimming
    outliers in the oriented PCA frame.

    Args:
        points_xyz: ``(N, 3)`` (or ``(N, 2)``) point coordinates; only XY is used.
        percentile: fraction (in percent, e.g. ``2.0``) trimmed off *each*
            extremity along each principal axis before hulling.  ``0`` disables
            the trim (plain convex hull of all points).

    Returns:
        ``(K, 2)`` hull vertices in counter-clockwise order, or ``None`` when
        there are too few points or they are degenerate (collinear) — in which
        case the caller should apply no crop.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        logger.warning(
            "Too few points (%d) for a crop hull; skipping", len(pts))
        return None
    xy = pts[:, :2]

    # Oriented (PCA) frame: rotate onto the two principal axes so the percentile
    # trim follows the dominant scene orientation rather than the world axes.
    center = xy.mean(axis=0)
    centered = xy - center
    cov = centered.T @ centered / max(1, len(centered) - 1)
    # eigh → columns are orthonormal principal axes (order/sign irrelevant here,
    # the per-axis percentile trim is symmetric).
    _evals, evecs = np.linalg.eigh(cov)
    proj = centered @ evecs  # (N, 2) coordinates in the PCA frame

    p = float(np.clip(percentile, 0.0, 49.0))
    if p > 0.0:
        lo = np.percentile(proj, p, axis=0)
        hi = np.percentile(proj, 100.0 - p, axis=0)
        keep = np.all((proj >= lo) & (proj <= hi), axis=1)
        if int(keep.sum()) >= 3:
            xy = xy[keep]
        else:
            logger.warning(
                "Percentile trim left <3 points; hulling the full set instead")

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(xy)
    except Exception:
        logger.warning(
            "Convex hull failed (degenerate points); applying no crop")
        return None
    # ConvexHull.vertices are counter-clockwise for 2-D inputs.
    return xy[hull.vertices].astype(np.float64)


def compute_sfm_crop_hull(
    reconstruction: types.Reconstruction,
    percentile: float,
) -> Optional[NDArray]:
    """Crop hull of the reconstruction's SfM points (see ``crop_hull_from_points``)."""
    pts = np.array(
        [p.coordinates for p in reconstruction.points.values()],
        dtype=np.float64,
    )
    if len(pts) < 3:
        logger.warning(
            "Reconstruction has too few points (%d) for a crop hull", len(pts))
        return None
    return crop_hull_from_points(pts, percentile)


def hull_contains(hull: NDArray, x: NDArray, y: NDArray) -> NDArray:
    """Vectorized point-in-convex-polygon test.

    A point is inside a counter-clockwise convex polygon iff it lies to the left
    of (or on) every directed edge.  ``hull`` is ``(K, 2)`` CCW vertices; ``x``
    and ``y`` are broadcastable arrays of query coordinates.  Returns a boolean
    array (the broadcast shape) — True where the point is inside or on the hull.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    inside = np.ones(np.broadcast(x, y).shape, dtype=bool)
    k = len(hull)
    for i in range(k):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % k]
        # cross((b - a), (p - a)) >= 0 on/left of the edge; small negative
        # tolerance keeps boundary points and float noise inside.
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        inside &= cross >= -1e-9
    return inside
