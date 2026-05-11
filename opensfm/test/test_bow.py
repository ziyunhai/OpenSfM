# pyre-strict
"""Tests for opensfm.bow – BagOfWords class and helpers."""

import numpy as np
from opensfm import bow


def _make_bow(n_words: int = 100, dim: int = 128) -> bow.BagOfWords:
    """Create a BagOfWords with random words and uniform frequencies."""
    rng = np.random.RandomState(42)
    words = rng.rand(n_words, dim).astype(np.float32)
    frequencies = np.ones(n_words, dtype=np.float32) * 10
    return bow.BagOfWords(words, frequencies)


# ── histogram ────────────────────────────────────────────────────────


def test_histogram_sums_to_one() -> None:
    """Histogram is normalized to sum to 1."""
    b = _make_bow(50)
    word_ids = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
    h = b.histogram(word_ids)
    assert np.isclose(h.sum(), 1.0)


def test_histogram_length() -> None:
    """Histogram has one bin per word."""
    b = _make_bow(50)
    word_ids = np.array([0, 10, 10], dtype=np.int32)
    h = b.histogram(word_ids)
    assert h.shape == (50,)


def test_histogram_nonzero_bins() -> None:
    """Only bins corresponding to observed words are nonzero."""
    b = _make_bow(50)
    word_ids = np.array([5, 5, 5], dtype=np.int32)
    h = b.histogram(word_ids)
    # bin 5 should be nonzero, all others zero
    nonzero = np.nonzero(h)[0]
    assert list(nonzero) == [5]


# ── bow_distance ─────────────────────────────────────────────────────


def test_bow_distance_identical() -> None:
    """Distance between identical word sets is zero."""
    b = _make_bow(50)
    w = np.array([0, 1, 2, 3], dtype=np.int32)
    d = b.bow_distance(w, w)
    assert np.isclose(d, 0.0)


def test_bow_distance_different() -> None:
    """Distance between disjoint word sets is positive."""
    b = _make_bow(50)
    w1 = np.array([0, 0, 0], dtype=np.int32)
    w2 = np.array([1, 1, 1], dtype=np.int32)
    d = b.bow_distance(w1, w2)
    assert d > 0


def test_bow_distance_with_precomputed_histograms() -> None:
    """Pre-computed histograms give the same result."""
    b = _make_bow(50)
    w1 = np.array([0, 1, 2], dtype=np.int32)
    w2 = np.array([3, 4, 5], dtype=np.int32)
    h1 = b.histogram(w1)
    h2 = b.histogram(w2)
    d1 = b.bow_distance(w1, w2)
    d2 = b.bow_distance(w1, w2, h1=h1, h2=h2)
    assert np.isclose(d1, d2)


# ── map_to_words ─────────────────────────────────────────────────────


def test_map_to_words_shape() -> None:
    """map_to_words returns (n_descriptors, k) indices."""
    b = _make_bow(50, 128)
    descriptors = np.random.RandomState(0).rand(10, 128).astype(np.float32)
    idx = b.map_to_words(descriptors, k=3)
    assert idx.shape == (10, 3)
    assert idx.dtype == np.int32


def test_map_to_words_bruteforce() -> None:
    """BruteForce matcher also returns correct shape."""
    b = _make_bow(50, 128)
    descriptors = np.random.RandomState(0).rand(5, 128).astype(np.float32)
    idx = b.map_to_words(descriptors, k=2, matcher_type="BruteForce")
    assert idx.shape == (5, 2)
