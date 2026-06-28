# pyre-strict
from pathlib import Path

import numpy as np
from opensfm import dense, io, pygeometry, types
from opensfm.dataset import UndistortedDataSet
from opensfm.dense.dsm_ortho import (
    _dsm_footprint,
    _fill_holes_2pass,
    _smooth_fill_components,
)
from opensfm.dense.fusion import (
    _assign_footprint_holes,
    _dilate_core_xy,
    _kdtree_chunk_labels,
    _pack_coarse_cells,
    _prescan_coarse_grid,
    _select_chunk_views,
    _trim_sparse_ends,
)


def test_depthmap_to_ply() -> None:
    height, width = 2, 3

    camera = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    camera.id = "cam1"
    camera.height = height
    camera.width = width
    r = types.Reconstruction()
    r.add_camera(camera)
    shot = r.create_shot(
        "shot1",
        camera.id,
        pygeometry.Pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )

    image = np.zeros((height, width, 3))
    depth = np.ones((height, width))

    ply = dense.depthmap_to_ply(shot, depth, image)
    assert len(ply.splitlines()) == 16


def _one_shot_recon(width: int = 32, height: int = 32) -> types.Reconstruction:
    camera = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    camera.id = "cam1"
    camera.width = width
    camera.height = height
    r = types.Reconstruction()
    r.add_camera(camera)
    r.create_shot(
        "s0",
        camera.id,
        pygeometry.Pose(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )
    return r


def test_prescan_coarse_grid_buckets_views() -> None:
    # One view with all-valid depth projects into a non-empty set of coarse
    # cells, each recording the (only) contributing view index 0.
    r = _one_shot_recon()
    depth = np.full((32, 32), 5.0, dtype=np.float32)
    grid = _prescan_coarse_grid(
        None, ["s0"], r, voxel_size=2.0, coarse_factor=8,
        clean_cache={"s0": (depth, depth, None)},
    )
    assert len(grid) > 0
    assert all(views == {0} for views in grid.values())


def test_kdtree_chunk_labels_partitions_all_cells() -> None:
    # A line of 100 cells split at max_cells=10 → a clean partition: every cell
    # labelled, contiguous chunk ids, each chunk within budget.
    cells = np.array([(x, 0, 0) for x in range(100)], dtype=np.int32)
    labels = _kdtree_chunk_labels(cells, max_cells=10)
    sizes = np.bincount(labels)
    assert sizes.sum() == 100          # every cell assigned
    assert (sizes > 0).all()           # no empty chunk ids
    assert sizes.max() <= 10           # GPU/coherence budget respected


class _StubDataset:
    """Minimal duck type for ``UndistortedDataSet.list_batch_indices``."""

    def __init__(self, path: str) -> None:
        self._path = path
        self.io_handler = io.IoFilesystemDefault()

    def _depthmap_path(self) -> str:
        return self._path


def test_list_batch_indices_discovers_gaps_and_skips_debug(tmp_path: Path) -> None:
    # Chunks 0 and 2 produced a fused PLY (1 produced none → a gap); a _debug
    # variant and an unrelated file must NOT be counted as batches.
    for name in (
        "fused_batch_0000.ply",
        "fused_batch_0002.ply",
        "fused_batch_0001_debug.ply",
        "mesh_batch_0000.ply",
        "reconstruction.json",
    ):
        (tmp_path / name).write_text("")
    s = _StubDataset(str(tmp_path))
    fused = UndistortedDataSet.list_batch_indices(s, "fused_batch_", ".ply")
    assert fused == [0, 2]                          # gap preserved, debug skipped
    assert UndistortedDataSet.list_batch_indices(
        s, "mesh_batch_", ".ply") == [0]
    assert UndistortedDataSet.list_batch_indices(
        s, "dsm_ortho_batch_", ".npz") == []        # none present


def test_list_batch_indices_missing_folder_is_empty() -> None:
    s = _StubDataset("/no/such/depthmap/path")
    assert UndistortedDataSet.list_batch_indices(s, "fused_batch_", ".ply") == []


def test_assign_footprint_holes_assigns_enclosed_gap() -> None:
    # A 7x7 ring of occupied cells with a hollow 3x3 centre: the centre is an
    # enclosed interior hole and must be handed out for completion (single chunk).
    occ = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(occ, dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)
    holes = _assign_footprint_holes(cells, labels, 1, close_cells=0)
    expected = {(x, y) for x in range(2, 5) for y in range(2, 5)}
    assert holes[0] == expected


def test_trim_sparse_ends_cuts_only_extremities() -> None:
    # Dense core with thin tapering ends: only the sparse extremities are cut,
    # and an interior low-count dip is preserved.  Threshold is frac * max.
    counts = np.array([1, 1, 50, 60, 3, 55, 50, 1, 1], dtype=np.int64)
    # max = 60, frac=0.1 → thr=6; ends [1,1] are < 6, idx2 (50) stops the walk.
    assert _trim_sparse_ends(counts, 0.1) == (2, 7)  # interior dip (3) kept
    # frac=0 disables trimming.
    assert _trim_sparse_ends(counts, 0.0) == (0, len(counts))
    # all-zero axis → no trim (degenerate).
    assert _trim_sparse_ends(np.zeros(5, np.int64), 0.5) == (0, 5)


def test_assign_footprint_holes_trims_sparse_outlier() -> None:
    # A 7x7 ring with a hollow 3x3 centre, plus a lone outlier far in +x.  A
    # large close_cells WITHOUT trimming bridges the whole 33-cell gap to the
    # outlier (inventing a wide sparse fringe); WITH trimming the outlier column
    # is cut, so only the ring's genuine enclosed centre is completed.
    ring = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(ring + [(40, 3, 0)], dtype=np.int64)
    labels = np.zeros(len(cells), dtype=np.int64)
    center = {(x, y) for x in range(2, 5) for y in range(2, 5)}

    holes = _assign_footprint_holes(
        cells, labels, 1, close_cells=64, trim_fraction=0.5
    )
    assert center <= holes[0]                       # enclosed centre completed
    assert all(x <= 6 for (x, _y) in holes[0])      # outlier fringe cut

    # Without trimming, the large closing DOES bridge into the sparse fringe.
    holes_notrim = _assign_footprint_holes(
        cells, labels, 1, close_cells=64, trim_fraction=0.0
    )
    assert any(x > 6 for (x, _y) in holes_notrim[0])


def test_assign_footprint_holes_none_when_solid() -> None:
    cells = np.array(
        [(x, y, 0) for x in range(4) for y in range(4)], dtype=np.int64
    )
    labels = np.zeros(len(cells), dtype=np.int64)
    holes = _assign_footprint_holes(cells, labels, 1, close_cells=0)
    assert holes[0] == set()


def test_assign_footprint_holes_partition_is_disjoint_and_complete() -> None:
    # Same ring, but two chunks split by x: every enclosed empty cell is owned
    # by exactly one chunk (disjoint), and together they cover the whole hole.
    occ = [
        (x, y, 0) for x in range(7) for y in range(7)
        if not (1 < x < 5 and 1 < y < 5)
    ]
    cells = np.array(occ, dtype=np.int64)
    labels = (cells[:, 0] > 3).astype(np.int64)  # 2 chunks by x median
    holes = _assign_footprint_holes(cells, labels, 2, close_cells=0)
    expected = {(x, y) for x in range(2, 5) for y in range(2, 5)}
    assert (holes[0] | holes[1]) == expected
    assert holes[0].isdisjoint(holes[1])


def test_dsm_footprint_is_extensive_with_large_close() -> None:
    # The padded closing must be EXTENSIVE — never erode the footprint inward
    # from the array border (the binary_closing border_value pitfall, which the
    # default border_value=0 would collapse to empty at 32 iterations).
    valid = np.zeros((60, 60), bool)
    valid[3:57, 3:57] = True
    fp = _dsm_footprint(valid, close_iters=32)
    assert fp[valid].all()
    assert int(fp.sum()) >= int(valid.sum())


def test_dsm_footprint_does_not_balloon_convex_block() -> None:
    # A solid convex block has no concavity/hole to close, so the footprint must
    # equal the block at ANY close_iters — i.e. the closing does NOT balloon out
    # to the array rectangle (the border_value=1 failure this padded closing
    # replaces).  This is what makes a large dsm_footprint_close_cells safe.
    valid = np.zeros((60, 60), bool)
    valid[3:57, 3:57] = True
    for n in (8, 32, 256):
        fp = _dsm_footprint(valid, close_iters=n)
        assert int(fp.sum()) == int(valid.sum()), f"ballooned at n={n}"


def test_dsm_footprint_bridges_narrow_gap() -> None:
    # Two blocks separated by a 4-wide gap (<= 2*close_iters) are bridged — the
    # intended effect of close_iters — while the data extent is preserved.
    valid = np.zeros((40, 40), bool)
    valid[5:35, 8:19] = True    # left block (cols 8..18)
    valid[5:35, 23:34] = True   # right block (cols 23..33); gap cols 19..22
    closed = _dsm_footprint(valid, close_iters=4)
    # interior of the gap is filled (assert away from the open top/bottom ends)
    assert closed[10:30, 19:23].all()
    assert int(closed.sum()) > int(valid.sum())


def test_smooth_fill_components_harmonic_cg() -> None:
    # The colour inpaint solves Laplace's equation matrix-free with CG (no splu
    # fill-in).  Between two constant-colour borders the harmonic solution is a
    # linear ramp: the hole centre is ~the border average, the row is monotone,
    # values stay in [0, 255], and no cell is left NaN.
    from scipy import ndimage as ndi

    grid = np.zeros((30, 80, 3), np.float32)
    grid[:, :20] = (200.0, 100.0, 50.0)
    grid[:, 60:] = (50.0, 150.0, 250.0)
    valid = np.zeros((30, 80), bool)
    valid[:, :20] = True
    valid[:, 60:] = True
    hole = np.zeros((30, 80), bool)
    hole[:, 20:60] = True
    labels, n = ndi.label(hole)

    _smooth_fill_components(grid, labels, np.arange(1, n + 1), valid)

    assert np.isfinite(grid[hole]).all()
    assert grid.min() >= 0.0 and grid.max() <= 255.0
    centre = grid[15, 40]
    assert np.allclose(centre, (125.0, 125.0, 150.0), atol=8.0)
    row_r = grid[15, 20:60, 0]
    assert np.all(np.diff(row_r) <= 0.5)  # monotone non-increasing ramp


def test_fill_holes_2pass_mops_up_ringless_component() -> None:
    # A fillable component whose ring holds no valid sample makes the low_flat
    # filler bail (`ring_vals.size == 0`), leaving it NaN — the source of
    # coarse-grid void squares.  The guaranteed mop-up must close it from the
    # nearest filled cell, WITHOUT touching the genuine exterior (outside the
    # hole mask), which stays no-data so the ortho remains transparent there.
    g = np.full((40, 40), np.nan, np.float32)
    g[0:10, 0:10] = 5.0
    valid = ~np.isnan(g)
    hole_a = np.zeros((40, 40), bool)
    hole_a[0:14, 11:30] = True  # 266 cells, ring touches the valid block
    hole_b = np.zeros((40, 40), bool)
    hole_b[23:40, 23:40] = True  # 289 cells, ring is all exterior → bails
    hole_mask = hole_a | hole_b

    out, _extrap = _fill_holes_2pass(
        g, sample_valid=valid, hole_mask=hole_mask,
        small_area_max=256, diffuse_iters=8, kappa=0.5, dt=0.2,
        large_fill="low_flat", low_percentile=20.0, occlusion_drop=0.0,
    )

    # No fillable cell may survive as no-data.
    assert not (hole_mask & ~np.isfinite(out)).any()
    # The genuine exterior (NaN, outside the hole mask) is left untouched.
    exterior = (~valid) & (~hole_mask)
    assert np.isnan(out[exterior]).all()


def test_select_chunk_views_coverage_before_quality() -> None:
    # 6 chunk cells along x. Two high-weight views (A, B) redundantly cover cells
    # 0-3; a LOW-weight view C is the sole observer of cells 4,5.  Naive top-2 by
    # weight = [A, B] would leave 4,5 unfused (→ holes).  Coverage-first must
    # include C, then fill remaining budget with the best skipped views.
    cells = np.array([[i, 0, 0] for i in range(6)], dtype=np.int64)
    origin = cells.min(0)
    span = cells.max(0) - origin + 1
    cpk = np.sort(_pack_coarse_cells(cells, origin, span))

    def uk(idxs: list) -> np.ndarray:
        return np.array([[i, 0, 0] for i in idxs], dtype=np.int64)

    view_ukey = {
        "A": uk([0, 1, 2, 3]),
        "B": uk([0, 1, 2, 3]),
        "C": uk([4, 5]),
        "D": uk([0, 1]),
    }
    weighted = ["A", "B", "D", "C"]  # weight desc; C is worst

    views, n_unc = _select_chunk_views(cpk, weighted, view_ukey, origin, span, 2)
    assert "C" in views and n_unc == 0  # sole observer of 4,5 kept

    views, n_unc = _select_chunk_views(cpk, weighted, view_ukey, origin, span, 4)
    assert set(views) == {"A", "B", "C", "D"} and n_unc == 0
    assert views[0] == "A" and views[1] == "C"  # coverage picks come first

    views, n_unc = _select_chunk_views(cpk, weighted, view_ukey, origin, span, 1)
    assert views == ["A"] and n_unc == 2  # budget-limited, reported


def test_select_chunk_views_min_two_observers() -> None:
    # Ensure each cell gets >= 2 observers.  A covers all 4 cells; B covers 0,1;
    # C (low weight) is the only OTHER view of cells 2,3.  Naive top-2 = [A, B]
    # leaves cells 2,3 with a single observer (A) — which the SVO would reject.
    # min_obs=2 must pull in C so 2,3 reach two observers.
    cells = np.array([[i, 0, 0] for i in range(4)], dtype=np.int64)
    origin = cells.min(0)
    span = cells.max(0) - origin + 1
    cpk = np.sort(_pack_coarse_cells(cells, origin, span))

    def uk(idxs: list) -> np.ndarray:
        return np.array([[i, 0, 0] for i in idxs], dtype=np.int64)

    view_ukey = {
        "A": uk([0, 1, 2, 3]),
        "B": uk([0, 1]),
        "C": uk([2, 3]),
    }
    weighted = ["A", "B", "C"]  # C is worst

    # min_obs=1: A alone covers everything once.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 3, min_obs=1
    )
    assert views[0] == "A" and n_unc == 0

    # min_obs=2: A then B (0,1 → 2 obs) then C (2,3 → 2 obs); all satisfied.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 3, min_obs=2
    )
    assert set(views) == {"A", "B", "C"} and n_unc == 0

    # min_obs=2, budget=2: only A,B fit → cells 2,3 stay single-observer.
    views, n_unc = _select_chunk_views(
        cpk, weighted, view_ukey, origin, span, 2, min_obs=2
    )
    assert views == ["A", "B"] and n_unc == 2

    # A cell with a single observer in the whole candidate set cannot reach 2.
    view_ukey2 = {"A": uk([0, 1, 2, 3]), "B": uk([0, 1, 2])}  # cell 3: only A
    views, n_unc = _select_chunk_views(
        cpk, weighted[:2], view_ukey2, origin, span, 5, min_obs=2
    )
    assert n_unc == 1  # cell 3 can never get a 2nd observer


def test_dilate_core_xy_grows_into_observed_neighbours() -> None:
    # A 3x3 core plus observed cells one ring out (same Z) and a far cell.
    # margin=1 grows the core by one XY ring (into the observed neighbours) but
    # never reaches the far cell, and ignores Z (2.5D plan-view dilation).
    core = {(x, y, 0) for x in range(3) for y in range(3)}
    ring = {(-1, 1, 0), (3, 1, 5), (1, -1, 0)}  # adjacent, varied Z
    far = {(20, 20, 0)}
    observed = core | ring | far
    out = _dilate_core_xy(observed, core, margin=1)
    assert core <= out                 # core always kept
    assert ring <= out                 # one-ring neighbours pulled in
    assert far.isdisjoint(out)         # far cell never reached
    assert out <= observed             # only ever observed cells


def test_dilate_core_xy_zero_margin_is_core_only() -> None:
    core = {(0, 0, 0), (1, 0, 0)}
    observed = core | {(2, 0, 0)}
    assert _dilate_core_xy(observed, core, margin=0) == core


def test_kdtree_chunk_labels_separates_far_blobs() -> None:
    # Two compact blobs far apart; with a budget that fits each, the split keeps
    # each blob whole and in its own chunk (spatially coherent, not scattered).
    a = np.array([(x, y, 0) for x in range(5) for y in range(5)], dtype=np.int32)
    b = a + np.array([1000, 0, 0], dtype=np.int32)
    cells = np.vstack([a, b])
    labels = _kdtree_chunk_labels(cells, max_cells=25)
    assert len(set(labels[:25].tolist())) == 1   # blob A is one chunk
    assert len(set(labels[25:].tolist())) == 1   # blob B is one chunk
    assert labels[0] != labels[25]               # different chunks
