# pyre-strict
"""Tests for the LAS/LAZ point-cloud writers (``pypointcloud``).

Covers attribute round-tripping, CRS embedding (OGC WKT VLR + global-encoding
WKT bit), the integer-offset that keeps projected coordinates from overflowing
the LAS int32 encoding, and the untagged topocentric default.
"""
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from opensfm import geo, pypointcloud


def _write(
    path: str,
    pos: NDArray,
    nrm: Optional[NDArray] = None,
    col: Optional[NDArray] = None,
    lbl: Optional[NDArray] = None,
    crs_wkt: str = "",
    offset: Optional[List[float]] = None,
) -> None:
    """Write a cloud through the streaming writer, setting header options."""
    h = pypointcloud.PointCloudHeader()
    h.point_count = len(pos)
    h.has_normals = nrm is not None
    h.has_colors = col is not None
    h.has_labels = lbl is not None
    if crs_wkt:
        h.crs_wkt = crs_wkt
    if offset is not None:
        h.offset = offset
    writer = pypointcloud.open_writer(path, h)
    kw: Dict[str, NDArray] = {}
    if nrm is not None:
        kw["normals"] = nrm.astype(np.float32)
    if col is not None:
        kw["colors"] = col.astype(np.uint8)
    if lbl is not None:
        kw["labels"] = lbl.astype(np.uint8)
    assert writer.write_chunk(pos.astype(np.float64), **kw)
    assert writer.finalize()


def _parse_header_vlrs(path: str) -> Tuple[int, List[Tuple[str, int, bytes]]]:
    """Return ``(global_encoding, [(user_id, record_id, payload), ...])``.

    VLRs live uncompressed in the header region for both LAS and LAZ, so the same
    little-endian walk works for either.
    """
    raw = open(path, "rb").read()
    assert raw[:4] == b"LASF"
    global_encoding = struct.unpack_from("<H", raw, 6)[0]
    num_vlrs = struct.unpack_from("<I", raw, 100)[0]
    header_size = struct.unpack_from("<H", raw, 94)[0]
    pos = header_size
    vlrs: List[Tuple[str, int, bytes]] = []
    for _ in range(num_vlrs):
        user_id = raw[pos + 2:pos + 18].split(b"\0")[0].decode("latin1")
        record_id = struct.unpack_from("<H", raw, pos + 18)[0]
        rec_len = struct.unpack_from("<H", raw, pos + 20)[0]
        payload = raw[pos + 54:pos + 54 + rec_len]
        vlrs.append((user_id, record_id, payload))
        pos += 54 + rec_len
    return global_encoding, vlrs


@pytest.mark.parametrize("fmt", ["las", "laz"])
def test_roundtrip_attributes(tmp_path, fmt: str) -> None:
    """Positions, normals, colors and labels survive a write/read cycle."""
    n = 32
    rng = np.random.RandomState(0)
    pos = rng.rand(n, 3) * 100.0
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    col = rng.randint(0, 256, (n, 3)).astype(np.uint8)
    lbl = (np.arange(n) % 8).astype(np.uint8)

    path = str(tmp_path / f"cloud.{fmt}")
    _write(path, pos, nrm, col, lbl)
    rpos, rnrm, rcol, rlbl = pypointcloud.read_point_cloud(path)

    assert len(rpos) == n
    assert np.allclose(rpos, pos, atol=1e-2)   # default 1 mm scale
    assert np.array_equal(rcol, col)           # 8-bit RGB is exact
    assert np.array_equal(rlbl, lbl)
    assert np.allclose(rnrm, nrm, atol=1e-6)   # f32 extra bytes are exact


@pytest.mark.parametrize("fmt", ["las", "laz"])
def test_crs_embedded(tmp_path, fmt: str) -> None:
    """A CRS is written as an OGC WKT VLR with the global-encoding WKT bit set."""
    wkt = geo.crs_to_wkt("EPSG:32631")
    n = 5
    pos = np.column_stack([
        448000.0 + np.arange(n), 5411000.0 + np.arange(n), 35.0 + np.arange(n),
    ]).astype(np.float64)

    path = str(tmp_path / f"cloud.{fmt}")
    _write(path, pos, crs_wkt=wkt, offset=[448000.0, 5411000.0, 35.0])

    global_encoding, vlrs = _parse_header_vlrs(path)
    assert global_encoding & 0x10                       # WKT bit
    wkt_vlrs = [v for v in vlrs if v[0] == "LASF_Projection" and v[1] == 2112]
    assert len(wkt_vlrs) == 1
    assert b"UTM zone 31N" in wkt_vlrs[0][2]

    # Projected coordinates also round-trip losslessly thanks to the offset.
    rpos, _, _, _ = pypointcloud.read_point_cloud(path)
    assert np.allclose(rpos, pos, atol=1e-2)


@pytest.mark.parametrize("fmt", ["las", "laz"])
def test_no_crs_when_topocentric(tmp_path, fmt: str) -> None:
    """Without a CRS the file carries no WKT VLR and no WKT global-encoding bit."""
    pos = np.random.RandomState(1).rand(5, 3) * 10.0
    path = str(tmp_path / f"cloud.{fmt}")
    _write(path, pos)

    global_encoding, vlrs = _parse_header_vlrs(path)
    assert not (global_encoding & 0x10)
    assert not any(record_id == 2112 for _, record_id, _ in vlrs)


@pytest.mark.parametrize("fmt", ["las", "laz"])
def test_offset_prevents_int32_overflow(tmp_path, fmt: str) -> None:
    """A large projected northing round-trips only because of the header offset.

    With the default offset 0 and 1 mm scale, ``northing / 0.001`` (~5.4e9)
    overflows the LAS int32 coordinate; the explicit offset brings it in range.
    """
    n = 4
    pos = np.column_stack([
        448000.0 + np.arange(n), 5411000.0 + np.arange(n), np.zeros(n),
    ]).astype(np.float64)
    path = str(tmp_path / f"cloud.{fmt}")
    _write(path, pos, offset=[448000.0, 5411000.0, 0.0])
    rpos, _, _, _ = pypointcloud.read_point_cloud(path)
    assert np.allclose(rpos[:, 1], pos[:, 1], atol=1e-2)


def test_offset_overflow_without_offset_is_lossy(tmp_path) -> None:
    """Sanity guard: large coords with offset 0 do NOT round-trip (justifies the
    offset).  Uses LAS so the failure is the int32 clamp, not a writer error."""
    n = 4
    pos = np.column_stack([
        448000.0 + np.arange(n), 5411000.0 + np.arange(n), np.zeros(n),
    ]).astype(np.float64)
    path = str(tmp_path / "cloud.las")
    _write(path, pos)  # no offset → all northings clamp to the int32 max
    rpos, _, _, _ = pypointcloud.read_point_cloud(path)
    assert not np.allclose(rpos[:, 1], pos[:, 1], atol=1.0)
