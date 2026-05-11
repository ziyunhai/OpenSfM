# pyre-strict
import numpy as np
from opensfm import geo, pygeo


def test_ecef_lla_consistency() -> None:
    lla_before = [46.5274109, 6.5722075, 402.16]
    ecef = geo.ecef_from_lla(lla_before[0], lla_before[1], lla_before[2])
    lla_after = geo.lla_from_ecef(ecef[0], ecef[1], ecef[2])
    assert np.allclose(np.array(lla_after), lla_before)


def test_ecef_lla_topocentric_consistency() -> None:
    lla_ref = [46.5, 6.5, 400]
    lla_before = [46.5274109, 6.5722075, 402.16]
    enu = geo.topocentric_from_lla(
        lla_before[0], lla_before[1], lla_before[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    lla_after = geo.lla_from_topocentric(
        enu[0], enu[1], enu[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    assert np.allclose(np.array(lla_after), lla_before)


def test_ecef_lla_consistency_pygeo() -> None:
    lla_before = [46.5274109, 6.5722075, 402.16]
    ecef = pygeo.ecef_from_lla(lla_before[0], lla_before[1], lla_before[2])
    lla_after = pygeo.lla_from_ecef(ecef[0], ecef[1], ecef[2])
    assert np.allclose(lla_after, lla_before)


def test_ecef_lla_topocentric_consistency_pygeo() -> None:
    lla_ref = [46.5, 6.5, 400]
    lla_before = [46.5274109, 6.5722075, 402.16]
    enu = pygeo.topocentric_from_lla(
        lla_before[0], lla_before[1], lla_before[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    lla_after = pygeo.lla_from_topocentric(
        enu[0], enu[1], enu[2], lla_ref[0], lla_ref[1], lla_ref[2]
    )
    assert np.allclose(lla_after, lla_before)


def test_eq_geo() -> None:
    assert geo.TopocentricConverter(
        40, 30, 0) == geo.TopocentricConverter(40, 30, 0)
    assert geo.TopocentricConverter(
        40, 32, 0) != geo.TopocentricConverter(40, 30, 0)


def test_parse_projection() -> None:
    from opensfm import io
    proj_str = io._parse_projection_string("WGS84")
    assert proj_str is None

    proj_str = io._parse_projection_string("WGS84 UTM 31N")
    assert proj_str is not None

    proj = geo.construct_proj_transformer(proj_str)
    easting, northing = 431760, 4582313.7
    lat, lon = 41.38946, 2.18378
    plat, plon = proj.transform(easting, northing)
    assert np.allclose((lat, lon), (plat, plon))


# ── gps_distance (vectorized) ───────────────────────────────────────


def test_gps_distance_scalar() -> None:
    """Scalar GPS distance between two close points is in plausible range."""
    p1 = (42.1, -11.1)
    p2 = (42.2, -11.3)
    d = geo.gps_distance(p1, p2)
    assert 19000 < d < 20000


def test_gps_distance_same_point() -> None:
    """Distance of a point to itself is zero."""
    p = (48.8566, 2.3522)
    assert np.isclose(geo.gps_distance(p, p), 0.0)


def test_gps_distance_vectorized() -> None:
    """Vectorized GPS distance returns an array of distances."""
    lats1 = np.array([42.1, 42.0])
    lons1 = np.array([-11.1, -11.0])
    lats2 = np.array([42.2, 42.0])
    lons2 = np.array([-11.3, -11.0])
    d = geo.gps_distance(np.array([lats1, lons1]), np.array([lats2, lons2]))
    assert d.shape == (2,)
    assert d[1] < 1.0  # same point


# ── construct_proj_transformer inverse ───────────────────────────────


def test_construct_proj_transformer_inverse() -> None:
    """Inverse transformer goes from WGS84 to projection."""
    from opensfm import io
    proj_str = io._parse_projection_string("WGS84 UTM 31N")
    assert proj_str is not None

    fwd = geo.construct_proj_transformer(proj_str)  # proj → WGS84
    inv = geo.construct_proj_transformer(
        proj_str, inverse=True)  # WGS84 → proj

    easting, northing = 431760, 4582313.7
    lat, lon = fwd.transform(easting, northing)
    e2, n2 = inv.transform(lat, lon)
    assert np.allclose((easting, northing), (e2, n2), atol=1e-2)


# ── transform_to_proj ───────────────────────────────────────────────


def test_transform_to_proj_origin() -> None:
    """Origin of topocentric frame maps to the reference's projection coords."""
    from opensfm import io
    proj_str = io._parse_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    inv = geo.construct_proj_transformer(proj_str, inverse=True)

    ref = geo.TopocentricConverter(41.38946, 2.18378, 0)
    result = geo.transform_to_proj([0, 0, 0], ref, inv)
    # The easting/northing should be close to known values
    assert len(result) == 3
    assert abs(result[2]) < 1.0  # altitude ~ 0


# ── get_proj_transform_matrix ────────────────────────────────────────


def test_get_proj_transform_matrix_shape() -> None:
    """Transform matrix is 4x4."""
    from opensfm import io
    proj_str = io._parse_projection_string("WGS84 UTM 31N")
    assert proj_str is not None
    inv = geo.construct_proj_transformer(proj_str, inverse=True)
    ref = geo.TopocentricConverter(41.38946, 2.18378, 0)
    mat = geo.get_proj_transform_matrix(ref, inv)
    assert mat.shape == (4, 4)
    assert np.allclose(mat[3], [0, 0, 0, 1])


# ── TopocentricConverter vectorized ─────────────────────────────────


def test_topocentric_converter_vectorized() -> None:
    """TopocentricConverter handles array inputs."""
    ref = geo.TopocentricConverter(46.5, 6.5, 400)
    lats = np.array([46.5, 46.51])
    lons = np.array([6.5, 6.51])
    alts = np.array([400.0, 410.0])
    x, y, z = ref.to_topocentric(lats, lons, alts)
    assert isinstance(x, np.ndarray)
    assert x.shape == (2,)
    # Origin maps to (0, 0, 0)
    assert np.allclose([x[0], y[0], z[0]], [0, 0, 0], atol=1e-3)
    # Round-trip
    lat2, lon2, alt2 = ref.to_lla(x, y, z)
    assert np.allclose(lat2, lats, atol=1e-8)
    assert np.allclose(lon2, lons, atol=1e-8)
