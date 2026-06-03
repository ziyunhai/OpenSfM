# pyre-strict
import os
import tempfile
from typing import Any, Dict, List

from opensfm.actions import extract_geolocation


class FakeData:
    def __init__(self, images: List[str]) -> None:
        self._images = images
        self._exif_overrides: Dict[str, Dict[str, Any]] = {}
        self.data_path = ""

    def images(self) -> List[str]:
        return self._images

    def exif_overrides_exists(self) -> bool:
        return True

    def load_exif_overrides(self) -> Dict[str, Dict[str, Any]]:
        return self._exif_overrides

    def save_exif_overrides(self, exif_overrides: Dict[str, Dict[str, Any]]) -> None:
        self._exif_overrides = exif_overrides


def test_extract_geolocation_basic() -> None:
    # Build fake dataset and data
    data = FakeData(["img1.jpg", "img2.jpg"])

    # We write a fake CSV file
    csv_content = (
        "# Some headers\n"
        "img1.jpg,2681192.57,1250342.89,605.29,105.98,13.89,-6.94,0.03,0.05\n"
        "img2.jpg,2681212.76,1250336.03,605.42,107.44,16.74,1.25,0.03,0.05\n"
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f_name = f.name

    try:
        extract_geolocation.run_dataset(data, f_name, crs="EPSG:2056")

        # Verify img1.jpg values loaded correctly
        overrides = data.load_exif_overrides()
        assert "img1.jpg" in overrides
        assert "img2.jpg" in overrides

        img1 = overrides["img1.jpg"]
        assert "latitude" in img1
        assert "longitude" in img1
        assert "altitude" in img1
        assert abs(img1["latitude"] - 47.404) < 0.01
        assert abs(img1["longitude"] - 8.510) < 0.01
        assert abs(img1["altitude"] - 605.29) < 0.5

        assert "latitude_std" in img1
        assert img1["latitude_std"] == 0.03
        assert img1["longitude_std"] == 0.03
        assert img1["altitude_std"] == 0.05

        assert "opk" in img1
        opk1 = img1["opk"]
        assert "omega" in opk1
        assert "phi" in opk1
        assert "kappa" in opk1
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)
