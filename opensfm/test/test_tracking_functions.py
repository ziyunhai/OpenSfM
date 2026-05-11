# pyre-strict
"""Tests for pure functions in opensfm.tracking."""

import numpy as np
from opensfm import pymap, tracking


# ── _good_track ──────────────────────────────────────────────────────


def test_good_track_accepts_long_enough() -> None:
    """Track with >= min_length unique images is accepted."""
    track = [("im1", 0), ("im2", 1), ("im3", 2)]
    assert tracking._good_track(track, min_length=2)
    assert tracking._good_track(track, min_length=3)


def test_good_track_rejects_too_short() -> None:
    """Track shorter than min_length is rejected."""
    track = [("im1", 0)]
    assert not tracking._good_track(track, min_length=2)


def test_good_track_rejects_duplicate_images() -> None:
    """Track with duplicate image names is rejected."""
    track = [("im1", 0), ("im1", 1), ("im2", 2)]
    assert not tracking._good_track(track, min_length=2)


def test_good_track_empty() -> None:
    """Empty track is rejected."""
    assert not tracking._good_track([], min_length=1)


# ── common_tracks ────────────────────────────────────────────────────


def _make_tracks_manager_for_common() -> pymap.TracksManager:
    """Build a TracksManager where im1 and im2 share tracks t0 and t1,
    and im2 has an additional track t2 not in im1.
    """
    tm = pymap.TracksManager()
    # Track t0: observed by im1 and im2
    tm.add_observation("im1", "t0", pymap.Observation(
        0.1, 0.2, 1.0, 0, 0, 0, 0))
    tm.add_observation("im2", "t0", pymap.Observation(
        0.3, 0.4, 1.0, 0, 0, 0, 1))
    # Track t1: observed by im1 and im2
    tm.add_observation("im1", "t1", pymap.Observation(
        0.5, 0.6, 1.0, 0, 0, 0, 2))
    tm.add_observation("im2", "t1", pymap.Observation(
        0.7, 0.8, 1.0, 0, 0, 0, 3))
    # Track t2: only im2
    tm.add_observation("im2", "t2", pymap.Observation(
        0.9, 1.0, 1.0, 0, 0, 0, 4))
    return tm


def test_common_tracks_returns_shared() -> None:
    """common_tracks returns only tracks seen by both images."""
    tm = _make_tracks_manager_for_common()
    tracks, p1, p2 = tracking.common_tracks(tm, "im1", "im2")
    assert set(tracks) == {"t0", "t1"}
    assert p1.shape == (2, 2)
    assert p2.shape == (2, 2)


def test_common_tracks_feature_values() -> None:
    """Feature coordinates match what was stored."""
    tm = _make_tracks_manager_for_common()
    tracks, p1, p2 = tracking.common_tracks(tm, "im1", "im2")
    idx_t0 = tracks.index("t0")
    assert np.allclose(p1[idx_t0], [0.1, 0.2])
    assert np.allclose(p2[idx_t0], [0.3, 0.4])


def test_common_tracks_no_overlap() -> None:
    """No common tracks returns empty arrays."""
    tm = pymap.TracksManager()
    tm.add_observation("im1", "t0", pymap.Observation(0, 0, 1, 0, 0, 0, 0))
    tm.add_observation("im2", "t1", pymap.Observation(0, 0, 1, 0, 0, 0, 0))
    tracks, p1, p2 = tracking.common_tracks(tm, "im1", "im2")
    assert len(tracks) == 0


# ── as_weighted_graph ────────────────────────────────────────────────


def test_as_weighted_graph_nodes() -> None:
    """Graph nodes correspond to shot IDs."""
    tm = _make_tracks_manager_for_common()
    g = tracking.as_weighted_graph(tm)
    assert set(g.nodes) == {"im1", "im2"}


def test_as_weighted_graph_edge_weight() -> None:
    """Edge weight equals the number of common tracks."""
    tm = _make_tracks_manager_for_common()
    g = tracking.as_weighted_graph(tm)
    assert g.has_edge("im1", "im2")
    assert g["im1"]["im2"]["weight"] == 2


def test_as_weighted_graph_isolated() -> None:
    """Image with no common tracks to others is still a node."""
    tm = pymap.TracksManager()
    tm.add_observation("im1", "t0", pymap.Observation(0, 0, 1, 0, 0, 0, 0))
    tm.add_observation("im2", "t1", pymap.Observation(0, 0, 1, 0, 0, 0, 0))
    g = tracking.as_weighted_graph(tm)
    assert "im1" in g.nodes
    assert "im2" in g.nodes
    assert not g.has_edge("im1", "im2")


# ── as_graph (bipartite) ────────────────────────────────────────────


def test_as_graph_bipartite() -> None:
    """Bipartite graph has shot and track nodes with correct edges."""
    tm = pymap.TracksManager()
    tm.add_observation("im1", "t0", pymap.Observation(
        0.1, 0.2, 1.0, 10, 20, 30, 0))
    tm.add_observation("im2", "t0", pymap.Observation(
        0.3, 0.4, 1.0, 40, 50, 60, 1))
    g = tracking.as_graph(tm)

    # Shot nodes have bipartite=0, track nodes have bipartite=1
    assert g.nodes["im1"]["bipartite"] == 0
    assert g.nodes["t0"]["bipartite"] == 1
    # Edges connect shots to tracks
    assert g.has_edge("im1", "t0")
    assert g.has_edge("im2", "t0")
    # Edge attributes contain feature info
    edge = g["im1"]["t0"]
    assert np.allclose(edge["feature"], [0.1, 0.2])


# ── all_common_tracks ───────────────────────────────────────────────


def test_all_common_tracks_with_features() -> None:
    """all_common_tracks with include_features returns tracks and points."""
    tm = _make_tracks_manager_for_common()
    result = tracking.all_common_tracks(
        tm, include_features=True, min_common=1)
    assert ("im1", "im2") in result
    tracks, p1, p2 = result[("im1", "im2")]
    assert len(tracks) == 2


def test_all_common_tracks_min_common_filters() -> None:
    """Pairs with fewer than min_common tracks are excluded."""
    tm = _make_tracks_manager_for_common()
    result = tracking.all_common_tracks(
        tm, include_features=True, min_common=10)
    assert len(result) == 0


def test_all_common_tracks_without_features() -> None:
    """all_common_tracks_without_features returns only track IDs."""
    tm = _make_tracks_manager_for_common()
    result = tracking.all_common_tracks_without_features(tm, min_common=1)
    assert ("im1", "im2") in result
    tracks = result[("im1", "im2")]
    assert isinstance(tracks, (list, np.ndarray))
    assert len(tracks) == 2
