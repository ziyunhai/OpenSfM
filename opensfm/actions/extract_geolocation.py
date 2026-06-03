# pyre-strict
import logging
import os
from typing import Dict, Any, List
import numpy as np

from opensfm import pymap
from opensfm.geo import opk_from_ypr
from opensfm.dataset import DataSet, DataSetBase


logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(data: DataSetBase, geotag_file: str, crs: str = "WGS84") -> None:
    """Extract geolocation and orientation from a CSV/text file and write/update exif_overrides.json.

    Args:
        data: dataset object
        geotag_file: path to the geotag CSV/text file
        crs: CRS coordinate reference system of the X Y Z coordinates in the file
    """
    path = geotag_file
    if not os.path.isfile(path):
        # Backup lookup relative to dataset directory
        path = os.path.join(data.data_path, geotag_file)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Geotag file not found: {geotag_file}")

    with open(path, "r") as f:
        content = f.read()

    images = data.images()
    # Call the C++ parser
    parsed_items = pymap.parse_geolocation_file(content, images, crs)
    if not parsed_items:
        logger.warning("No matches or geolocations found in the geotag file.")
        return

    # Load existing overrides
    exif_overrides: Dict[str, Dict[str, Any]] = {}
    if data.exif_overrides_exists():
        exif_overrides = data.load_exif_overrides()

    for item in parsed_items:
        image = item.filename
        if image not in exif_overrides:
            exif_overrides[image] = {}

        if item.has_lla:
            exif_overrides[image]["latitude"] = item.lat
            exif_overrides[image]["longitude"] = item.lon
            exif_overrides[image]["altitude"] = item.alt

        if item.has_std:
            exif_overrides[image]["latitude_std"] = item.lat_std
            exif_overrides[image]["longitude_std"] = item.lon_std
            exif_overrides[image]["altitude_std"] = item.alt_std

        if item.has_ypr and item.has_lla:
            opk = opk_from_ypr(
                item.lat, item.lon, item.alt, item.yaw, item.pitch, item.roll)
            if opk:
                exif_overrides[image]["opk"] = opk

    data.save_exif_overrides(exif_overrides)
    logger.info(
        f"Successfully processed {len(parsed_items)} geolocations and saved to exif_overrides.json.")
