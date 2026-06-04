# pyre-strict
"""Test the rig module."""

import numpy as np
from opensfm import pygeometry, pymap, rig, types


def test_create_instances_with_patterns() -> None:
    # A first rig model defined as left/right/top/bottom

    # A complete instance
    instance1 = [
        "12345_left.jpg",
        "12345_bottom.jpg",
        "12345_top.jpg",
        "12345_right.jpg",
    ]

    # An incomplete one
    instance2 = [
        "1234567_left.jpg",
        "1234567_bottom.jpg",
        "1234567_top.jpg",
    ]
    patterns_12 = {
        "camera_left": "(left)",
        "camera_right": "(right)",
        "camera_top": "(top)",
        "camera_bottom": "(bottom)",
    }

    # A second one as RED/GREEN/BLUE
    instance3 = [
        "RED_SENSOR_001-12345678.jpg",
        "GREEN_SENSOR_002-12345678.jpg",
        "BLUE_SENSOR_003-12345678.jpg",
    ]
    patterns_3 = {
        "red": "(RED_SENSOR_001)",
        "green": "(GREEN_SENSOR_002)",
        "blue": "(BLUE_SENSOR_003)",
    }

    # Two single shots
    instance4 = [
        "RED_toto.jpg",
        "tata.jpg",
    ]

    # Run detection with these two rig model patterns
    rig_patterns = patterns_12
    rig_patterns.update(patterns_3)
    instances, single_shots = rig.create_instances_with_patterns(
        instance1 + instance2 + instance3 + instance4, rig_patterns
    )

    # Ensure we have 2 instance for the first rig, and 1 for the second
    assert len(instances) == 3

    # Ensure the two single shots
    assert len(single_shots) == 2

    recovered_instance1 = instances["12345_.jpg"]
    assert [x[0] for x in recovered_instance1] == instance1

    recovered_instance2 = instances["1234567_.jpg"]
    assert [x[0] for x in recovered_instance2] == instance2

    recovered_instance3 = instances["-12345678.jpg"]
    assert [x[0] for x in recovered_instance3] == instance3


def test_compute_relative_pose() -> None:
    # 4-cameras rig
    camera1 = pygeometry.Camera.create_spherical()
    camera1.id = "camera1"
    camera2 = pygeometry.Camera.create_spherical()
    camera2.id = "camera2"
    camera3 = pygeometry.Camera.create_spherical()
    camera3.id = "camera3"
    camera4 = pygeometry.Camera.create_spherical()
    camera4.id = "camera4"

    # a bit cumbersome that we need to have some reconstruction
    rec = types.Reconstruction()
    rec.add_camera(camera1)
    rec.add_camera(camera2)
    rec.add_camera(camera3)
    rec.add_camera(camera4)

    # First rig instance
    rec.create_shot(
        "shot1", "camera1", pygeometry.Pose(
            np.array([0, 0, 0]), np.array([-2, -2, 0]))
    )
    rec.create_shot(
        "shot2", "camera2", pygeometry.Pose(
            np.array([0, 0, 0]), np.array([-3, -3, 0]))
    )
    rec.create_shot(
        "shot3", "camera3", pygeometry.Pose(
            np.array([0, 0, 0]), np.array([-1, -3, 0]))
    )
    rec.create_shot(
        "shot4", "camera4", pygeometry.Pose(
            np.array([0, 0, 0]), np.array([-2, -4, 0]))
    )

    # Second rig instance (rotated by pi/2 around Z)
    pose_instance = pygeometry.Pose(np.array([0, 0, -1.5707963]))
    pose_instance.set_origin(np.array([-6, 0, 0]))
    rec.create_shot("shot5", "camera1", pose_instance)
    pose_instance.set_origin(np.array([-7, 1, 0]))
    rec.create_shot("shot6", "camera2", pose_instance)
    pose_instance.set_origin(np.array([-7, -1, 0]))
    rec.create_shot("shot7", "camera3", pose_instance)
    pose_instance.set_origin(np.array([-8, 0, 0]))
    rec.create_shot("shot8", "camera4", pose_instance)

    pose_instances = [
        [
            (
                rec.shots["shot1"],
                "camera_id_1",
            ),
            (
                rec.shots["shot2"],
                "camera_id_2",
            ),
            (
                rec.shots["shot3"],
                "camera_id_3",
            ),
            (
                rec.shots["shot4"],
                "camera_id_4",
            ),
        ],
        [
            (
                rec.shots["shot5"],
                "camera_id_1",
            ),
            (
                rec.shots["shot6"],
                "camera_id_2",
            ),
            (
                rec.shots["shot7"],
                "camera_id_3",
            ),
            (
                rec.shots["shot8"],
                "camera_id_4",
            ),
        ],
    ]

    # Compute rig cameras poses
    rig_cameras = rig.compute_relative_pose(pose_instances)

    # Note: compute_relative_pose averages origins across instances in world
    # frame (not rig-local frame), so with a rotated second instance the
    # averages are midpoints between the two instance offsets.
    assert np.allclose(
        [0.5, -0.5, 0], rig_cameras["camera_id_1"].pose.get_origin(), atol=1e-7
    )
    assert np.allclose(
        [0.5, 0.5, 0], rig_cameras["camera_id_2"].pose.get_origin(), atol=1e-7
    )
    assert np.allclose(
        [-0.5, -0.5, 0], rig_cameras["camera_id_3"].pose.get_origin(), atol=1e-7
    )
    assert np.allclose(
        [-0.5, 0.5, 0], rig_cameras["camera_id_4"].pose.get_origin(), atol=1e-7
    )


# ── find_image_rig ───────────────────────────────────────────────────


def test_find_image_rig_match() -> None:
    """Image matching a pattern returns rig_camera_id and instance_member_id."""
    patterns = {"camera_left": "(left)", "camera_right": "(right)"}
    cam_id, inst_id = rig.find_image_rig("12345_left.jpg", patterns)
    assert cam_id == "camera_left"
    assert inst_id == "12345_.jpg"


def test_find_image_rig_no_match() -> None:
    """Image matching no pattern returns (None, None)."""
    patterns = {"camera_left": "(left)"}
    cam_id, inst_id = rig.find_image_rig("12345_top.jpg", patterns)
    assert cam_id is None
    assert inst_id is None


def test_find_image_rig_full_match_skipped() -> None:
    """Pattern that consumes entire image name (empty instance_member_id) is skipped."""
    patterns = {"all": "(12345)"}
    cam_id, inst_id = rig.find_image_rig("12345", patterns)
    assert cam_id is None
    assert inst_id is None


# ── group_instances ──────────────────────────────────────────────────


def test_group_instances_single_group() -> None:
    """All instances with same camera set are in one group."""
    instances = {
        "inst1": [("im1_left", "cam_left"), ("im1_right", "cam_right")],
        "inst2": [("im2_left", "cam_left"), ("im2_right", "cam_right")],
    }
    groups = rig.group_instances(instances)
    assert len(groups) == 1
    key = list(groups.keys())[0]
    assert len(groups[key]) == 2


def test_group_instances_multiple_groups() -> None:
    """Instances with different camera sets form separate groups."""
    instances = {
        "inst1": [("im1_left", "cam_left"), ("im1_right", "cam_right")],
        "inst2": [("im2_red", "cam_red"), ("im2_green", "cam_green")],
    }
    groups = rig.group_instances(instances)
    assert len(groups) == 2


# ── rig_assignments_per_image ────────────────────────────────────────


def test_rig_assignments_per_image_basic() -> None:
    """Each image gets its instance_id, rig_camera_id, and sibling shots."""
    assignments = {
        "inst1": [("shot_a", "cam_left"), ("shot_b", "cam_right")],
    }
    result = rig.rig_assignments_per_image(assignments)
    assert "shot_a" in result
    inst_id, cam_id, siblings = result["shot_a"]
    assert inst_id == "inst1"
    assert cam_id == "cam_left"
    assert set(siblings) == {"shot_a", "shot_b"}


def test_rig_assignments_per_image_multiple_instances() -> None:
    assignments = {
        "inst1": [("s1", "c1")],
        "inst2": [("s2", "c1"), ("s3", "c2")],
    }
    result = rig.rig_assignments_per_image(assignments)
    assert len(result) == 3
    assert result["s2"][0] == "inst2"


# ── count_reconstructed_instances ────────────────────────────────────


def test_count_reconstructed_instances_all_found() -> None:
    """All instances are counted when all shots are reconstructed."""
    rec = types.Reconstruction()
    cam = pygeometry.Camera.create_perspective(0.5, 0.0, 0.0)
    cam.id = "cam"
    rec.add_camera(cam)
    for sid in ["s1", "s2", "s3", "s4"]:
        rec.create_shot(sid, "cam")

    instances = [
        [("s1", "c1"), ("s2", "c2")],
        [("s3", "c1"), ("s4", "c2")],
    ]
    assert rig.count_reconstructed_instances(instances, rec) == 2


def test_count_reconstructed_instances_partial() -> None:
    """Instances missing shots are not fully counted."""
    rec = types.Reconstruction()
    cam = pygeometry.Camera.create_perspective(0.5, 0.0, 0.0)
    cam.id = "cam"
    rec.add_camera(cam)
    rec.create_shot("s1", "cam")
    rec.create_shot("s2", "cam")
    # s3 is missing from reconstruction

    instances = [
        [("s1", "c1"), ("s2", "c2")],
        [("s3", "c1")],  # s3 missing
    ]
    # Instance 1 fully reconstructed, instance 2 not
    assert rig.count_reconstructed_instances(instances, rec) == 1


# ── default_rig_cameras ─────────────────────────────────────────────


def test_default_rig_cameras() -> None:
    """Creates identity-pose rig cameras for each camera_id."""
    result = rig.default_rig_cameras(["cam1", "cam2"])
    assert len(result) == 2
    assert "cam1" in result
    assert "cam2" in result
    for cam_id, rc in result.items():
        assert rc.id == cam_id
        assert np.allclose(rc.pose.rotation, [0, 0, 0])
