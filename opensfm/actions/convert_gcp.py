# pyre-strict
import logging
from typing import Dict, List, Tuple

from opensfm import io
from opensfm.dataset import DataSet

logger: logging.Logger = logging.getLogger(__name__)


def run_dataset(data: DataSet, from_format: str) -> None:
    """Convert ground control points from gcp_list.txt to ground_control_points.json, or vice-versa.

    Args:
        data: dataset object
        from_format: either 'txt' (txt to JSON conversion) or 'json' (JSON to txt conversion)
    """
    gcp_path = data._gcp_list_file()
    json_path = data._ground_control_points_file()

    exif = {image: data.load_exif(image) for image in data.images()}
    cdn_enabled = data.config["proj_cdn_enabled"]
    grid_cache_dir = data.config["proj_grid_cache_dir"]

    if from_format == "txt":
        if not data.io_handler.isfile(gcp_path):
            raise RuntimeError(f"GCP list file '{gcp_path}' does not exist.")

        logger.info(f"Converting GCPs from {gcp_path} to {json_path}")
        with data.io_handler.open_rt(gcp_path) as fin:
            crs = io.read_gcp_projection_string(fin)
        with data.io_handler.open_rt(gcp_path) as fin:
            gcps = io.read_gcp_list(fin, exif, cdn_enabled, grid_cache_dir)

        if not crs:
            crs = "WGS84"

        if data.io_handler.isfile(json_path):
            bak_path = json_path + ".bak"
            logger.info(f"Creating backup of existing JSON file at {bak_path}")
            with data.io_handler.open_rt(json_path) as fin:
                content = fin.read()
            with data.io_handler.open_wt(bak_path) as fout:
                fout.write(content)

        with data.io_handler.open_wt(json_path) as fout:
            io.write_ground_control_points(
                gcps, fout, crs, cdn_enabled, grid_cache_dir)

    elif from_format == "json":
        if not data.io_handler.isfile(json_path):
            raise RuntimeError(
                f"Ground control points JSON file '{json_path}' does not exist.")

        logger.info(f"Converting GCPs from {json_path} to {gcp_path}")
        with data.io_handler.open_rt(json_path) as fin:
            gcps, crs = io.read_ground_control_points(
                fin, cdn_enabled, grid_cache_dir)

        if not crs:
            crs = "WGS84"

        if data.io_handler.isfile(gcp_path):
            bak_path = gcp_path + ".bak"
            logger.info(f"Creating backup of existing text file at {bak_path}")
            with data.io_handler.open_rt(gcp_path) as fin:
                content = fin.read()
            with data.io_handler.open_wt(bak_path) as fout:
                fout.write(content)

        with data.io_handler.open_wt(gcp_path) as fout:
            io.write_gcp_list(gcps, fout, crs, exif,
                              cdn_enabled, grid_cache_dir)

    else:
        raise ValueError(
            f"Invalid format '{from_format}'. Must be 'txt' or 'json'.")
