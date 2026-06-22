# pyre-strict
"""Shared low-level utilities and constants for the dense pipeline.

Image scaling, per-shot depth-range computation, depthmap→PLY export, debug
wireframe samplers, and cluster colours — reused across the clustering,
depthmaps, and fusion stages.
"""

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from opensfm import io, pymap, types
from opensfm.dataset import UndistortedDataSet


def select_cluster_views(
    cluster_shots: List[str],
    all_neighbors: Dict[str, List[Any]],
    available_set: Set[str],
    max_total: int,
    per_ref_cap: int,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Pick a bounded set of views to load for one cluster batch.

    The cluster's own shots are always included.  Candidate neighbours are
    ranked by *coverage* — how many cluster refs list them, weighted by per-ref
    rank — and added until the total reaches ``max_total``.  Favouring
    high-coverage neighbours keeps the context shared across refs and bounds
    peak RAM regardless of how spread the cluster is.  Each ref's neighbour list
    (capped at ``per_ref_cap``) is then restricted to the chosen set.

    Returns ``(sorted_view_ids, per_ref_neighbor_ids)``.  ``per_ref_neighbor_ids``
    is keyed by cluster shot id; consumers that only need the load set (fusion)
    can ignore it.
    """
    cluster_set = set(cluster_shots)
    votes: Dict[str, float] = defaultdict(float)
    per_ref_ranked: Dict[str, List[str]] = {}
    for sid in cluster_shots:
        ranked: List[str] = []
        for nbr in all_neighbors.get(sid, []):
            nid = nbr.id if hasattr(nbr, "id") else nbr
            if nid != sid and nid in available_set and nid not in cluster_set:
                ranked.append(nid)
                if len(ranked) >= per_ref_cap:
                    break
        per_ref_ranked[sid] = ranked
        for rank, nid in enumerate(ranked):
            votes[nid] += 1.0 / (rank + 1.0)

    needed: Set[str] = set(cluster_shots)
    budget = max(0, max_total - len(needed))
    chosen = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)[:budget]
    needed.update(chosen)

    per_ref_nbrs = {
        sid: [nid for nid in per_ref_ranked[sid] if nid in needed]
        for sid in cluster_shots
    }
    return sorted(needed), per_ref_nbrs


# Distinct colors for up to 12 clusters; cycles for more.
_CLUSTER_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 0, 128),    # rose
    (128, 255, 0),    # lime
    (0, 128, 255),    # sky blue
]


def _sample_edges_wireframe(
    corners: NDArray,
    edges: List[Tuple[int, int]],
    step: float = 0.01,
) -> NDArray:
    """Sample points along edges defined by pairs of corner indices."""
    pts: List[NDArray] = []
    for i, j in edges:
        length = float(np.linalg.norm(corners[j] - corners[i]))
        n_samples = max(int(np.ceil(length / step)), 2)
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
        pts.append(corners[i] + t * (corners[j] - corners[i]))
    if not pts:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _sample_bbox_wireframe(
    bbox: Tuple[NDArray, NDArray],
    step: float = 0.01,
) -> NDArray:
    """Sample points along the 12 edges of an axis-aligned bounding box."""
    bb_min, bb_max = bbox
    corners = np.array(
        [
            [bb_min[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_min[1], bb_min[2]],
            [bb_max[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_max[1], bb_min[2]],
            [bb_min[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_min[1], bb_max[2]],
            [bb_max[0], bb_max[1], bb_max[2]],
            [bb_min[0], bb_max[1], bb_max[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    return _sample_edges_wireframe(corners, edges, step)


def _sample_frustum_wireframe(
    shot: pymap.Shot,
    depth: float = 0.3,
    step: float = 0.01,
) -> NDArray:
    """Sample points along the 8 edges of a camera frustum.

    The frustum is defined by the camera origin and four far-plane
    corners at the given *depth* along the viewing direction.
    """
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation
    origin = (-R.T @ t).astype(np.float32)  # camera center in world

    w = int(shot.camera.width)
    h = int(shot.camera.height)
    K = shot.camera.get_K_in_pixel_coordinates(w, h)
    K_inv = np.linalg.inv(K)

    # Four image corners in pixel coords → normalized camera rays.
    pixel_corners = np.array(
        [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]],
        dtype=np.float64,
    ).T  # (3, 4)
    cam_rays = K_inv @ pixel_corners  # (3, 4)
    # Normalize so Z=1, then scale by depth.
    cam_rays = cam_rays / cam_rays[2:3, :] * depth
    # Transform to world coordinates.
    world_pts = (R.T @ (cam_rays - t[:, np.newaxis])).T.astype(np.float32)

    # corners: 0=origin, 1..4=far plane TL, TR, BR, BL
    corners = np.vstack([origin[np.newaxis, :], world_pts])  # (5, 3)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # origin to far corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # far plane rectangle
    ]
    return _sample_edges_wireframe(corners, edges, step)


def compute_depth_range(
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    shot: pymap.Shot,
    config: Dict[str, Any],
    down: float = 0.1,
    up: float = 0.9,
    extent: bool = True,
) -> Tuple[float, float]:
    """Compute min and max depth based on reconstruction points."""
    depths = []
    for track in tracks_manager.get_shot_observations(shot.id):
        if track in reconstruction.points:
            p = reconstruction.points[track].coordinates
            z = shot.pose.transform(p)[2]
            depths.append(z)
    min_depth = np.percentile(depths, down * 100) * \
        ((1 - down) if extent else 1)
    max_depth = np.percentile(depths, up * 100) * \
        ((1 + (1 - up)) if extent else 1)

    config_min_depth: float = config["depthmap_min_depth"]
    config_max_depth: float = config["depthmap_max_depth"]

    return config_min_depth or min_depth, config_max_depth or max_depth


def scale_image(
    image: NDArray, width: int, height: int, interpolation: int
) -> NDArray:
    return cv2.resize(image, (width, height), interpolation=interpolation)


def scale_down_image(
    image: NDArray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> NDArray:
    width = min(width, image.shape[1])
    height = min(height, image.shape[0])
    return scale_image(image, width, height, interpolation)


def _save_depthmap_as_ply(
    data: UndistortedDataSet,
    reconstruction: types.Reconstruction,
    shot_id: str,
    depth: NDArray,
    prefix: str,
) -> None:
    """Back-project a depthmap to 3-D and save as a binary PLY."""
    shot = reconstruction.shots[shot_id]
    h, w = depth.shape[:2]
    K = shot.camera.get_K_in_pixel_coordinates(w, h)
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation

    y, x = np.mgrid[:h, :w]
    v = np.vstack((x.ravel(), y.ravel(), np.ones(w * h)))
    cam_coords = depth.reshape((1, -1)) * np.linalg.inv(K).dot(v)
    pts_world = (R.T @ (cam_coords - t[:, np.newaxis])).T.astype(np.float32)

    valid = depth.ravel() > 0
    points = pts_world[valid]

    color = data.load_undistorted_image(shot_id)
    color = scale_down_image(color, w, h)
    colors = color.reshape(-1, 3)[valid].astype(np.uint8)

    normals = np.zeros_like(points)
    labels = np.zeros(len(points), dtype=np.uint8)
    data.save_point_cloud(
        points, normals, colors, labels,
        filename=f"{prefix}_{shot_id}.ply",
    )


def depthmap_to_ply(shot: pymap.Shot, depth: NDArray, image: NDArray) -> str:
    """Export depthmap points as a PLY string."""
    height, width = depth.shape
    K = shot.camera.get_K_in_pixel_coordinates(width, height)
    R = shot.pose.get_rotation_matrix()
    t = shot.pose.translation
    y, x = np.mgrid[:height, :width]
    v = np.vstack((x.ravel(), y.ravel(), np.ones(width * height)))
    camera_coords = depth.reshape((1, -1)) * np.linalg.inv(K).dot(v)
    points = R.T.dot(camera_coords - t.reshape(3, 1))

    vertices = []
    for p, c, d in zip(points.T, image.reshape(-1, 3), depth.reshape(-1, 1)):
        if d != 0:
            s = "{} {} {} {} {} {}".format(p[0], p[1], p[2], c[0], c[1], c[2])
            vertices.append(s)

    return io.points_to_ply_string(vertices)
