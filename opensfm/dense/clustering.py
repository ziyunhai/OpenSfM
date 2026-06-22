# pyre-strict
"""View clustering for geometric consistency.

Groups shots into bounded, spatially-compact, covisibility-connected clusters
(super-point covisibility → spatial edge gate → Leiden/CPM → compact connected
size-cap split → small-cluster absorption) and provides per-cluster bounding
boxes and the cluster debug PLY export.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import igraph as ig
import leidenalg
import numpy as np
from numpy.typing import NDArray

from opensfm import context, pymap, pysfm, types
from opensfm.dataset import UndistortedDataSet

from .common import _CLUSTER_COLORS, _sample_frustum_wireframe

logger: logging.Logger = logging.getLogger(__name__)


def _split_connected(
    members: List[int],
    adj: Dict[int, Dict[int, float]],
    positions: NDArray,
    target_size: int,
) -> List[List[int]]:
    """Partition a community into connected, spatially-compact pieces of size
    <= ``target_size``.

    Region growing on the covisibility subgraph for connectivity, but the
    frontier node added at each step is the one spatially CLOSEST to the current
    piece's camera centroid.  Restricting the frontier to covisibility
    neighbours keeps every piece connected; choosing the nearest one grows
    compact 2-D blocks instead of long flight-line strips (which pure
    covisibility-weight growth produces), for better multi-view geometry and
    smaller fusion bounding boxes.  A node with no intra-community edge falls out
    as its own singleton.

    Operates in shot-index space; ``adj[i]`` maps neighbour index → edge weight,
    ``positions[i]`` is shot *i*'s camera centre.
    """
    unassigned: Set[int] = set(members)
    pieces: List[List[int]] = []
    while unassigned:
        # Seed: the unassigned node best connected to the rest of the community.
        seed = max(
            unassigned,
            key=lambda v: sum(
                w for n, w in adj.get(v, {}).items() if n in unassigned
            ),
        )
        piece: List[int] = [seed]
        unassigned.discard(seed)
        centroid_sum = positions[seed].astype(np.float64).copy()
        frontier: Set[int] = {
            n for n in adj.get(seed, {}) if n in unassigned
        }
        while len(piece) < target_size and frontier:
            mean = centroid_sum / len(piece)
            u = min(
                frontier,
                key=lambda v: float(
                    np.dot(positions[v] - mean, positions[v] - mean)
                ),
            )
            frontier.discard(u)
            if u not in unassigned:
                continue
            piece.append(u)
            unassigned.discard(u)
            centroid_sum += positions[u]
            for n in adj.get(u, {}):
                if n in unassigned:
                    frontier.add(n)
        pieces.append(piece)
    return pieces


def _covis_component_sizes(
    member_idxs: List[int], adj: Dict[int, Dict[int, float]]
) -> List[int]:
    """Connected-component sizes (descending) of a cluster in the covisibility
    subgraph induced by its members."""
    members: Set[int] = set(member_idxs)
    seen: Set[int] = set()
    comps: List[int] = []
    for start in member_idxs:
        if start in seen:
            continue
        size = 0
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            size += 1
            for v in adj.get(u, {}):
                if v in members and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(size)
    return sorted(comps, reverse=True)


def _log_cluster_stats(
    label: str,
    clusters: List[List[str]],
    adj: Dict[int, Dict[int, float]],
    idx_of: Dict[str, int],
    reconstruction: types.Reconstruction,
    detail: bool = False,
) -> None:
    """Log size, covisibility-connectivity and spatial-extent stats for a set of
    clusters.  Used to diagnose disconnected / spatially-spread clusters at each
    clustering stage (raw Leiden → split → final)."""
    n = len(clusters)
    if n == 0:
        logger.info(f"[clusters {label}] 0 clusters")
        return
    sizes = np.array([len(c) for c in clusters])
    comp_counts: List[int] = []
    diags: List[float] = []
    disconnected: List[Tuple[int, int, List[int], float]] = []
    for ci, c in enumerate(clusters):
        comps = _covis_component_sizes([idx_of[s] for s in c], adj)
        comp_counts.append(len(comps))
        origins = np.array(
            [reconstruction.shots[s].pose.get_origin()
             for s in c if s in reconstruction.shots]
        )
        diag = (
            float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
            if len(origins) >= 2 else 0.0
        )
        diags.append(diag)
        if len(comps) > 1:
            disconnected.append((ci, len(c), comps, diag))
    diag_arr = np.array(diags)
    # Size histogram in quartiles of the cap.
    logger.info(
        f"[clusters {label}] n={n} | size min/med/mean/max="
        f"{int(sizes.min())}/{int(np.median(sizes))}/{sizes.mean():.1f}/"
        f"{int(sizes.max())} | singletons={int((sizes == 1).sum())} | "
        f"disconnected={len(disconnected)}/{n} "
        f"(mean {np.mean(comp_counts):.2f} covis-comps/cluster) | "
        f"cam-bbox diag(m) med/mean/max="
        f"{np.median(diag_arr):.1f}/{diag_arr.mean():.1f}/{diag_arr.max():.1f}"
    )
    if detail and disconnected:
        for ci, sz, comps, diag in disconnected[:25]:
            logger.info(
                f"    cluster {ci}: {sz} shots split into covis-components "
                f"{comps}, cam-bbox diag {diag:.1f} m"
            )
        if len(disconnected) > 25:
            logger.info(f"    … and {len(disconnected) - 25} more")


def cluster_views(
    processable: List[str],
    tracks_manager: pymap.TracksManager,
    reconstruction: types.Reconstruction,
    max_cluster_size: int,
    fuse_knn: int = 15,
    fuse_radius_factor: float = 0.5,
    edge_max_factor: float = 0.0,
) -> Tuple[List[List[str]], NDArray, List[Set[str]], List[List[str]], "pysfm.CovisibilityGraph"]:
    """Cluster shots by shared super-point visibility (Leiden/CPM).

    Steps 1–3 (super-point fusion, kNN, pairwise covisibility) are
    delegated to C++ via ``pysfm.build_covisibility_graph`` for speed
    and low memory usage.  Step 4 (Leiden partitioning + small-cluster
    absorption) remains in Python.

    Returns ``(clusters, sp_coords, sp_vis, sp_track_ids, covis, leiden_raw)``
    where *covis* is the C++ ``CovisibilityGraph`` object (needed for
    downstream neighbor selection) and *leiden_raw* is the raw Leiden partition
    before the connected size-cap split / absorption (empty when Leiden does not
    run), exposed so callers can dump it for clustering diagnostics.
    """
    N = len(processable)

    # ── C++ super-point fusion + covisibility graph (multithreaded) ──
    covis: pysfm.CovisibilityGraph = pysfm.build_covisibility_graph(
        tracks_manager,
        reconstruction.map,
        processable,
        fuse_knn,
        fuse_radius_factor,
    )

    # Unpack super-points for downstream debug PLY and other consumers.
    sp_coords_list: List[NDArray] = []
    sp_vis_list: List[Set[str]] = []
    sp_track_ids_list: List[List[str]] = []
    for sp in covis.super_points:
        sp_coords_list.append(np.asarray(sp.coord, dtype=np.float64))
        sp_vis_list.append(set(sp.vis))
        sp_track_ids_list.append(list(sp.tracks))
    sp_coords_arr = (
        np.array(sp_coords_list)
        if sp_coords_list
        else np.empty((0, 3), dtype=np.float64)
    )

    logger.info(
        f"C++ covisibility: {len(covis.super_points)} super-points, "
        f"{len(covis.edges)} edges"
    )

    if N <= max_cluster_size:
        return [list(processable)], sp_coords_arr, sp_vis_list, sp_track_ids_list, covis, []
    if N <= 1 or max_cluster_size <= 1:
        return (
            [[s] for s in processable],
            sp_coords_arr, sp_vis_list, sp_track_ids_list, covis, [],
        )

    # ── Leiden partitioning (Python — igraph/leidenalg, C under hood) ─
    edges = covis.edges
    weights = covis.weights

    if not edges:
        logger.warning("No covisibility edges — single cluster")
        return [list(processable)], sp_coords_arr, sp_vis_list, sp_track_ids_list, covis, []

    positions: NDArray = np.array(
        [reconstruction.shots[sid].pose.get_origin() for sid in processable],
        dtype=np.float64,
    )

    # ── Spatial edge gate (geometry-agnostic, density-adaptive) ────────
    if edge_max_factor and edge_max_factor > 0.0:
        from scipy.spatial import cKDTree

        ea = np.asarray(edges, dtype=np.int64)
        baselines = np.linalg.norm(
            positions[ea[:, 0]] - positions[ea[:, 1]], axis=1
        )
        # k small but > 1 for robustness to near-duplicate / burst shots.
        k_local = min(N - 1, 3)
        nn_dists, _ = cKDTree(positions).query(positions, k=k_local + 1)
        local_scale = np.asarray(nn_dists)[:, k_local]  # col 0 is self (0 m)
        edge_scale = np.minimum(local_scale[ea[:, 0]], local_scale[ea[:, 1]])
        keep = baselines <= edge_max_factor * edge_scale
        n_drop = int((~keep).sum())
        if n_drop:
            edges = [e for e, k in zip(edges, keep) if k]
            weights = [w for w, k in zip(weights, keep) if k]
            logger.info(
                f"Spatial edge gate: pruned {n_drop}/{len(keep)} covisibility "
                f"edges with baseline > {edge_max_factor:.1f}× local camera "
                f"spacing (median {float(np.median(local_scale)):.1f} m; "
                f"median edge baseline {float(np.median(baselines)):.1f} m) "
                f"for clustering"
            )
        if not edges:
            logger.warning("All edges pruned by spatial gate — single cluster")
            return [list(processable)], sp_coords_arr, sp_vis_list, sp_track_ids_list, covis, []

    ig_graph = ig.Graph(n=N, edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    # Covisibility adjacency (shot-index space) for connectivity-aware
    # splitting and absorption.
    idx_of: Dict[str, int] = {sid: i for i, sid in enumerate(processable)}
    adj: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (ei, ej), ew in zip(edges, weights):
        adj[ei][ej] = ew
        adj[ej][ei] = ew

    mean_w = float(np.mean(weights)) if weights else 1.0
    resolution = mean_w * 0.5

    context.log_memory("starting Leiden partitioning")

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        max_comm_size=int(max_cluster_size),
        n_iterations=-1,
        seed=42,
    )

    part_map: Dict[int, List[str]] = {}
    for i, comm in enumerate(partition.membership):
        part_map.setdefault(comm, []).append(processable[i])
    final = list(part_map.values())
    logger.info(
        f"Leiden partitioning: {N} shots → {len(final)} communities, "
        f"quality {partition.quality():.1f}, resolution {resolution:.1f}"
    )
    _log_cluster_stats(
        "leiden-raw", final, adj, idx_of, reconstruction, detail=True)
    # Snapshot the raw partition (before split/absorb) for diagnostics.
    leiden_raw: List[List[str]] = [list(c) for c in final]

    # Enforce the size cap while keeping every cluster covisibility-connected.
    # leidenalg's ``max_comm_size`` is only a soft constraint (its aggregation
    # phase violates it) and its communities are not guaranteed connected, so we
    # re-split every community on its covisibility subgraph into balanced,
    # connected pieces of size <= max_cluster_size.
    capped: List[List[str]] = []
    for comm in final:
        members = [idx_of[s] for s in comm]
        n = len(members)
        k = max(1, -(-n // int(max_cluster_size)))      # ceil(n / max)
        target = -(-n // k)                              # balanced, <= max
        for piece in _split_connected(members, adj, positions, target):
            capped.append([processable[i] for i in piece])
    if len(capped) != len(final):
        logger.info(
            f"Connected size-cap split: {len(final)} → {len(capped)} clusters "
            f"(cap {int(max_cluster_size)} views)"
        )
    final = capped
    _log_cluster_stats("after-split", final, adj, idx_of, reconstruction)

    # Absorb undersized clusters into their best COVISIBLE neighbour, never past
    # the size cap.  Requiring strictly positive shared covisibility keeps merged
    # clusters connected — a zero-overlap cluster is left on its own rather than
    # glued onto an arbitrary neighbour (which produced disconnected clusters).
    changed = True
    while changed:
        changed = False
        for ci in range(len(final)):
            if len(final[ci]) >= int(max_cluster_size * 0.75):
                continue
            best_ti = -1
            best_sim_val = 0.0  # require strictly positive covisibility to merge
            for ti in range(len(final)):
                if ti == ci:
                    continue
                if len(final[ti]) + len(final[ci]) > max_cluster_size:
                    continue
                s = sum(
                    adj[idx_of[a]].get(idx_of[b], 0.0)
                    for a in final[ci]
                    for b in final[ti]
                )
                if s > best_sim_val:
                    best_sim_val = s
                    best_ti = ti
            if best_ti < 0:
                continue  # no covisible cluster with room → keep as its own
            final[best_ti].extend(final[ci])
            logger.info(
                f"  Absorbed cluster {ci} (size {len(final[ci])}) into cluster {best_ti} (new size {len(final[best_ti])})"
            )
            del final[ci]
            changed = True
            break

    sizes = [len(c) for c in final]
    logger.info(
        f"Super-point clustering: {N} shots → {len(final)} clusters, "
        f"sizes {sizes}",
    )
    _log_cluster_stats(
        "final", final, adj, idx_of, reconstruction, detail=True)
    return final, sp_coords_arr, sp_vis_list, sp_track_ids_list, covis, leiden_raw


def _save_cluster_debug_ply(
    data: UndistortedDataSet,
    clusters: List[List[str]],
    reconstruction: types.Reconstruction,
    sp_coords: NDArray,
    sp_vis: List[Set[str]],
    frustum_step: float = 0.005,
    filename_prefix: str = "cluster_debug",
) -> None:
    """Write one debug PLY per cluster with colored camera frustums,
    white neighbor frustums, and super-points.

    Each super-point is colored by the cluster(s) it is observed from.
    A super-point observed by cameras in cluster *i* gets cluster *i*'s
    color.  If observed by multiple clusters it gets a blended color.
    """
    # Build shot → cluster index lookup.
    shot_cluster: Dict[str, int] = {}
    for ci, cl in enumerate(clusters):
        for sid in cl:
            shot_cluster[sid] = ci

    # Pre-assign each super-point a dominant cluster (majority vote)
    # and its blended color.
    sp_cluster: List[int] = []  # dominant cluster per super-point
    sp_color: List[Tuple[int, int, int]] = []
    for vis in sp_vis:
        votes: Dict[int, int] = {}
        for sid in vis:
            ci = shot_cluster.get(sid, -1)
            if ci >= 0:
                votes[ci] = votes.get(ci, 0) + 1
        if not votes:
            sp_cluster.append(-1)
            sp_color.append((128, 128, 128))
            continue
        # Dominant cluster.
        dom = max(votes, key=lambda c: votes[c])
        sp_cluster.append(dom)
        # Blend colors weighted by vote count.
        total = sum(votes.values())
        r = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][0]
                for c in votes) // total
        g = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][1]
                for c in votes) // total
        b = sum(votes[c] * _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)][2]
                for c in votes) // total
        sp_color.append((r, g, b))

    for ci, cl in enumerate(clusters):
        cr, cg, cb = _CLUSTER_COLORS[ci % len(_CLUSTER_COLORS)]
        color = np.array([cr, cg, cb], dtype=np.uint8)

        cl_pts: List[NDArray] = []
        cl_colors: List[NDArray] = []

        # Camera frustum wireframes.
        for sid in cl:
            if sid not in reconstruction.shots:
                continue
            shot = reconstruction.shots[sid]
            frustum_depth = 5
            fpts = _sample_frustum_wireframe(shot, depth=frustum_depth,
                                             step=frustum_step)
            cl_pts.append(fpts)
            cl_colors.append(np.tile(color, (len(fpts), 1)))

        # Super-points whose dominant cluster is this one.
        if len(sp_coords) > 0:
            mask = [i for i, dom in enumerate(sp_cluster) if dom == ci]
            if mask:
                sp_pts = sp_coords[mask].astype(np.float32)
                sp_cols = np.array(
                    [sp_color[i] for i in mask], dtype=np.uint8,
                )
                cl_pts.append(sp_pts)
                cl_colors.append(sp_cols)

        if not cl_pts:
            continue

        points = np.concatenate(cl_pts).astype(np.float32)
        colors = np.concatenate(cl_colors).astype(np.uint8)
        normals = np.zeros_like(points)
        labels = np.zeros(len(points), dtype=np.uint8)
        filename = f"{filename_prefix}_{ci:04d}.ply"
        data.save_point_cloud(
            points, normals, colors, labels, filename=filename,
        )
        logger.info(
            f"Cluster {ci} debug PLY saved ({len(points)} points, "
            f"{len(cl)} shots)"
        )


def _compute_cluster_bbox(
    cluster_shots: List[str],
    graph: pymap.TracksManager,
    reconstruction: types.Reconstruction,
) -> Tuple[NDArray, NDArray]:
    """Axis-aligned bounding box of 3-D points visible in the cluster.

    Uses the tracks manager to collect every reconstruction point
    observed by at least one shot in the cluster.  Falls back to
    camera positions if no points are found.

    Returns ``(min_coords, max_coords)``, each shape ``(3,)``.
    """
    seen_tracks: Set[str] = set()
    for sid in cluster_shots:
        for tid in graph.get_shot_observations(sid):
            if tid in reconstruction.points:
                seen_tracks.add(tid)

    if seen_tracks:
        coords = np.array(
            [reconstruction.points[tid].coordinates for tid in seen_tracks]
        )
    else:
        coords = np.array(
            [reconstruction.shots[sid].pose.get_origin()
             for sid in cluster_shots]
        )
    return coords.min(axis=0), coords.max(axis=0)
