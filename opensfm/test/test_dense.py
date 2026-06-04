# pyre-strict
import numpy as np
from opensfm import dense, pygeometry, types


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
