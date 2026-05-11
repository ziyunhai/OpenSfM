# pyre-strict
"""Tests for pure functions in opensfm.features."""

from io import BytesIO
from typing import Any, Dict

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import features


# ── normalized / denormalized image coordinates ──────────────────────


def test_normalized_denormalized_roundtrip() -> None:
    """Round-trip: pixel → normalized → pixel recovers original coords."""
    width, height = 640, 480
    pixels = np.array([[0.0, 0.0], [320.0, 240.0], [639.0, 479.0]])
    normed = features.normalized_image_coordinates(pixels, width, height)
    recovered = features.denormalized_image_coordinates(normed, width, height)
    assert np.allclose(recovered, pixels, atol=1e-10)


def test_normalized_center_pixel() -> None:
    """Center pixel of a square image maps to (0, 0)."""
    width, height = 100, 100
    center = np.array([[49.5, 49.5]])  # exact center: (w/2 - 0.5, h/2 - 0.5)
    normed = features.normalized_image_coordinates(center, width, height)
    assert np.allclose(normed, [[0.0, 0.0]], atol=1e-10)


def test_denormalized_origin() -> None:
    """Normalized (0, 0) maps back to center pixel."""
    width, height = 200, 100
    normed = np.array([[0.0, 0.0]])
    pixels = features.denormalized_image_coordinates(normed, width, height)
    assert np.allclose(pixels, [[99.5, 49.5]], atol=1e-10)


# ── resized_image ────────────────────────────────────────────────────


def test_resized_image_noop_when_small() -> None:
    """No resize when image is smaller than max_size."""
    img = np.zeros((50, 80, 3), dtype=np.uint8)
    result = features.resized_image(img, 100)
    assert result.shape == img.shape


def test_resized_image_downscales() -> None:
    """Image is downscaled when larger than max_size."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    result = features.resized_image(img, 200)
    # max dim is 400, scale to 200 → new shape (100, 200)
    assert result.shape[1] == 200
    assert result.shape[0] == 100


# ── root_feature ─────────────────────────────────────────────────────


def test_root_feature_values() -> None:
    """root_feature applies L1-normalized square root."""
    desc = np.array([[4.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = features.root_feature(desc)
    # sum = 4, sqrt(4/4) = 1, sqrt(0/4) = 0
    assert np.allclose(result, [[1.0, 0.0, 0.0, 0.0]])


def test_root_feature_uniform() -> None:
    """Uniform descriptor produces uniform root feature."""
    desc = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    result = features.root_feature(desc)
    expected = np.sqrt(0.25)
    assert np.allclose(result, [[expected, expected, expected, expected]])


def test_root_feature_with_l2_normalization() -> None:
    """L2 normalization divides by row norm before root mapping."""
    desc = np.array([[3.0, 4.0]], dtype=np.float32)
    result = features.root_feature(desc, l2_normalization=True)
    # After L2 norm: [3/5, 4/5] = [0.6, 0.8]
    # sum = 1.4, sqrt([0.6/1.4, 0.8/1.4])
    l2 = np.array([0.6, 0.8])
    s = l2.sum()
    expected = np.sqrt(l2 / s)
    assert np.allclose(result, [expected], atol=1e-6)


# ── root_feature_surf ────────────────────────────────────────────────


def test_root_feature_surf_64dim() -> None:
    """root_feature_surf processes 64-dim descriptors."""
    desc = np.abs(np.random.RandomState(42).randn(
        2, 64).astype(np.float32)) + 0.01
    result = features.root_feature_surf(desc.copy())
    # Result should have same shape
    assert result.shape == (2, 64)
    # Non-negative absolute values should remain non-negative after sqrt
    assert np.all(np.abs(result) >= 0)


def test_root_feature_surf_noop_non64() -> None:
    """root_feature_surf is a no-op for non-64-dim descriptors."""
    desc = np.ones((3, 32), dtype=np.float32)
    result = features.root_feature_surf(desc.copy())
    assert np.allclose(result, desc)


# ── normalize_features ───────────────────────────────────────────────


def test_normalize_features_transforms_coords() -> None:
    """normalize_features scales coordinates by image size."""
    width, height = 200, 100
    points = np.array([[100.0, 50.0, 10.0, 0.0]], dtype=np.float32)
    desc = np.array([[1.0, 2.0]], dtype=np.float32)
    colors = np.array([[128, 64, 32]], dtype=np.uint8)

    pts, d, c = features.normalize_features(
        points, desc, colors, width, height)

    # x: (100 + 0.5 - 100) / 200 = 0.5/200 = 0.0025
    assert np.allclose(pts[0, 0], 0.0025, atol=1e-6)
    # size: 10 / 200 = 0.05
    assert np.allclose(pts[0, 2], 0.05, atol=1e-6)
    # desc and colors pass through
    assert np.array_equal(d, desc)
    assert np.array_equal(c, colors)


# ── _in_mask ─────────────────────────────────────────────────────────


def test_in_mask_inside() -> None:
    """Point inside the non-zero region returns True."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    point = np.array([5.0, 5.0])  # pixel coords
    assert features._in_mask(point, 10, 10, mask)


def test_in_mask_outside() -> None:
    """Point outside the non-zero region returns False."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    point = np.array([0.0, 0.0])
    assert not features._in_mask(point, 10, 10, mask)


# ── SemanticData ─────────────────────────────────────────────────────


def test_semantic_data_mask() -> None:
    """SemanticData.mask filters segmentation and instances."""
    seg = np.array([1, 2, 3, 4, 5])
    inst = np.array([10, 20, 30, 40, 50])
    labels = [{"name": "a"}]
    sd = features.SemanticData(seg, inst, labels)

    mask = np.array([True, False, True, False, True])
    masked = sd.mask(mask)
    assert np.array_equal(masked.segmentation, [1, 3, 5])
    assert masked.instances is not None
    assert np.array_equal(masked.instances, [10, 30, 50])


def test_semantic_data_has_instances() -> None:
    """has_instances reflects whether instances array is set."""
    sd_with = features.SemanticData(np.array([1]), np.array([10]), [])
    assert sd_with.has_instances()

    sd_without = features.SemanticData(np.array([1]), None, [])
    assert not sd_without.has_instances()


# ── FeaturesData.mask ────────────────────────────────────────────────


def test_features_data_mask() -> None:
    """FeaturesData.mask filters points, descriptors, colors, semantic."""
    points = np.array([[0, 0, 1, 0], [1, 1, 2, 0], [
                      2, 2, 3, 0]], dtype=np.float32)
    desc = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [
                      0, 0, 255]], dtype=np.float32)
    seg = np.array([1, 2, 3])
    sem = features.SemanticData(seg, None, [])
    fd = features.FeaturesData(points, desc, colors, sem)

    mask = np.array([True, False, True])
    masked = fd.mask(mask)
    assert masked.points.shape == (2, 4)
    assert masked.descriptors is not None
    assert masked.descriptors.shape == (2, 2)
    assert masked.colors.shape == (2, 3)
    assert np.array_equal(masked.points[0], points[0])
    assert np.array_equal(masked.points[1], points[2])


def test_features_data_mask_no_descriptors() -> None:
    """FeaturesData.mask handles None descriptors."""
    points = np.array([[0, 0, 1, 0], [1, 1, 2, 0]], dtype=np.float32)
    colors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    fd = features.FeaturesData(points, None, colors, None)

    mask = np.array([False, True])
    masked = fd.mask(mask)
    assert masked.descriptors is None
    assert masked.points.shape == (1, 4)


def test_features_data_mask_with_depths() -> None:
    """FeaturesData.mask preserves and filters depths."""
    points = np.array([[0, 0, 1, 0], [1, 1, 2, 0]], dtype=np.float32)
    desc = np.array([[1], [2]], dtype=np.float32)
    colors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    depths = np.array([10.0, 20.0])
    fd = features.FeaturesData(points, desc, colors, None, depths)

    mask = np.array([True, False])
    masked = fd.mask(mask)
    assert masked.depths is not None
    assert np.allclose(masked.depths, [10.0])


# ── FeaturesData save / from_file round-trip ─────────────────────────


def _make_config(**overrides: Any) -> Dict[str, Any]:
    """Minimal config for features I/O."""
    cfg: Dict[str, Any] = {
        "feature_type": "SIFT",
        "feature_root": False,
        "hahog_normalize_to_uchar": False,
        "akaze_descriptor": "MSURF",
        "reprojection_error_sd": 0.004,
    }
    cfg.update(overrides)
    return cfg


def test_features_data_save_load_roundtrip_v3() -> None:
    """FeaturesData v3 round-trip through BytesIO preserves arrays."""
    points = np.random.RandomState(0).rand(10, 4).astype(np.float32)
    desc = np.random.RandomState(1).rand(10, 128).astype(np.float32)
    colors = np.random.RandomState(2).randint(
        0, 256, (10, 3)).astype(np.float64)
    seg = np.random.RandomState(3).randint(0, 10, 10).astype(np.uint8)
    sem = features.SemanticData(seg, None, [])
    fd = features.FeaturesData(points, desc, colors, sem)

    config = _make_config()
    buf = BytesIO()
    fd.save(buf, config)
    buf.seek(0)
    loaded = features.FeaturesData.from_file(buf, config)

    assert np.allclose(loaded.points, points)
    assert loaded.descriptors is not None
    assert np.allclose(loaded.descriptors, desc)
    assert np.allclose(loaded.colors, colors)


def test_features_data_save_load_no_semantic() -> None:
    """Round-trip with no semantic data."""
    points = np.random.RandomState(0).rand(5, 4).astype(np.float32)
    desc = np.random.RandomState(1).rand(5, 64).astype(np.float32)
    colors = np.ones((5, 3), dtype=np.float64)
    fd = features.FeaturesData(points, desc, colors, None)

    config = _make_config()
    buf = BytesIO()
    fd.save(buf, config)
    buf.seek(0)
    loaded = features.FeaturesData.from_file(buf, config)

    assert np.allclose(loaded.points, points)
    assert loaded.semantic is None


def test_features_data_save_load_with_instances() -> None:
    """Round-trip with segmentation and instances."""
    points = np.random.RandomState(0).rand(5, 4).astype(np.float32)
    desc = np.random.RandomState(1).rand(5, 128).astype(np.float32)
    colors = np.ones((5, 3), dtype=np.float64)
    seg = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    inst = np.array([100, 200, 300, 400, 500], dtype=np.int16)
    sem = features.SemanticData(seg, inst, [])
    fd = features.FeaturesData(points, desc, colors, sem)

    config = _make_config()
    buf = BytesIO()
    fd.save(buf, config)
    buf.seek(0)
    loaded = features.FeaturesData.from_file(buf, config)

    assert loaded.semantic is not None
    assert loaded.semantic.has_instances()
    assert np.array_equal(loaded.semantic.segmentation, seg)


# ── FeaturesData get_segmentation ────────────────────────────────────


def test_get_segmentation_with_semantic() -> None:
    """get_segmentation returns segmentation array when semantic is set."""
    seg = np.array([1, 2, 3])
    sem = features.SemanticData(seg, None, [])
    fd = features.FeaturesData(np.zeros((3, 4)), None, np.zeros((3, 3)), sem)
    assert fd.get_segmentation() is not None
    assert np.array_equal(fd.get_segmentation(), seg)


def test_get_segmentation_without_semantic() -> None:
    """get_segmentation returns None when no semantic data."""
    fd = features.FeaturesData(np.zeros((3, 4)), None, np.zeros((3, 3)), None)
    assert fd.get_segmentation() is None
