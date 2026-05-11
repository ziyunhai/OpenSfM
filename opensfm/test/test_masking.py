# pyre-strict
"""Tests for pure functions in opensfm.masking."""

import numpy as np
from opensfm import masking


# ── mask_from_segmentation ───────────────────────────────────────────


def test_mask_from_segmentation_single_ignore() -> None:
    """Pixels matching ignore value are zeroed out."""
    seg = np.array(
        [[0, 1, 2, 0],
         [2, 2, 1, 0],
         [0, 1, 0, 2],
         [2, 0, 1, 1]],
        dtype=np.uint8,
    )
    mask = masking.mask_from_segmentation(seg, ignore_values=[2])
    expected = (seg != 2).astype(np.uint8)
    assert np.array_equal(mask, expected)


def test_mask_from_segmentation_multiple_ignore() -> None:
    """Multiple ignore values are all zeroed."""
    seg = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    mask = masking.mask_from_segmentation(seg, ignore_values=[1, 3])
    assert np.array_equal(mask, [[1, 0, 1, 0]])


def test_mask_from_segmentation_no_ignore() -> None:
    """No ignore values → all ones."""
    seg = np.array([[5, 6, 7]], dtype=np.uint8)
    mask = masking.mask_from_segmentation(seg, ignore_values=[])
    assert np.all(mask == 1)


# ── combine_masks ────────────────────────────────────────────────────


def test_combine_masks_both_none() -> None:
    """None + None → None."""
    assert masking.combine_masks(None, None) is None


def test_combine_masks_first_none() -> None:
    """None + mask → mask."""
    m = np.array([[1, 0, 1]], dtype=np.uint8)
    result = masking.combine_masks(None, m)
    assert result is not None
    assert np.array_equal(result, m)


def test_combine_masks_second_none() -> None:
    """mask + None → mask."""
    m = np.array([[0, 1, 1]], dtype=np.uint8)
    result = masking.combine_masks(m, None)
    assert result is not None
    assert np.array_equal(result, m)


def test_combine_masks_both_present() -> None:
    """mask & mask → bitwise AND."""
    m1 = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    m2 = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    result = masking.combine_masks(m1, m2)
    assert result is not None
    assert np.array_equal(result, [[1, 0, 0, 0]])


# ── _resize_masks_to_match ──────────────────────────────────────────


def test_resize_masks_to_match_same_size() -> None:
    """Same-size masks are returned unchanged."""
    m1 = np.ones((10, 10), dtype=np.uint8)
    m2 = np.zeros((10, 10), dtype=np.uint8)
    r1, r2 = masking._resize_masks_to_match(m1, m2)
    assert r1.shape == (10, 10)
    assert r2.shape == (10, 10)


def test_resize_masks_to_match_different_sizes() -> None:
    """Smaller mask is resized to match the larger one."""
    m1 = np.ones((5, 5), dtype=np.uint8)
    m2 = np.ones((10, 10), dtype=np.uint8)
    r1, r2 = masking._resize_masks_to_match(m1, m2)
    assert r1.shape == (10, 10)
    assert r2.shape == (10, 10)


def test_combine_masks_different_sizes() -> None:
    """combine_masks handles masks of different sizes via resize."""
    m1 = np.ones((4, 4), dtype=np.uint8)
    m2 = np.ones((8, 8), dtype=np.uint8)
    m2[0, 0] = 0
    result = masking.combine_masks(m1, m2)
    assert result is not None
    assert result.shape == (8, 8)
    # The resized m1 should be all-ones, so result == m2
    assert result[0, 0] == 0
