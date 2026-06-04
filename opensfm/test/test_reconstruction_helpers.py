# pyre-strict
"""Tests for pure functions in opensfm.reconstruction_helpers."""

import numpy as np
import pytest
from opensfm import reconstruction_helpers


# ── guess_gravity_up_from_orientation_tag ────────────────────────────


@pytest.mark.parametrize(
    "orientation, expected",
    [
        (1, [0, -1, 0]),
        (2, [0, -1, 0]),
        (3, [0, 1, 0]),
        (4, [0, 1, 0]),
        (5, [-1, 0, 0]),
        (6, [-1, 0, 0]),
        (7, [1, 0, 0]),
        (8, [1, 0, 0]),
    ],
)
def test_guess_gravity_up_all_orientations(
    orientation: int, expected: list
) -> None:
    result = reconstruction_helpers.guess_gravity_up_from_orientation_tag(
        orientation)
    assert np.array_equal(result, expected)


def test_guess_gravity_up_invalid_raises() -> None:
    """Orientation outside 1-8 raises RuntimeError."""
    with pytest.raises(RuntimeError):
        reconstruction_helpers.guess_gravity_up_from_orientation_tag(0)
    with pytest.raises(RuntimeError):
        reconstruction_helpers.guess_gravity_up_from_orientation_tag(9)


# ── compute_focal (standalone helper) ───────────────────────────────


def test_compute_focal_from_35mm() -> None:
    from opensfm import exif
    f35, ratio = exif.compute_focal(50.0, None, None, None)
    assert abs(ratio - 50.0 / 36.0) < 1e-6


def test_compute_focal_no_info() -> None:
    from opensfm import exif
    f35, ratio = exif.compute_focal(None, None, None, None)
    assert f35 == 0.0
    assert ratio == 0.0
