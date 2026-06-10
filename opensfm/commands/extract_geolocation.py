# pyre-strict
import argparse

from opensfm.actions import extract_geolocation
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "extract_geolocation"
    help = "Extract geolocation and orientation from CSV/TXT file and write/update exif_overrides.json"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        extract_geolocation.run_dataset(
            dataset, geotag_file=args.geotag_file, crs=args.crs)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--geotag-file",
            required=True,
            help="Path to the CVS/TXT file containing image geolocations.",
        )
        parser.add_argument(
            "--crs",
            default="WGS84",
            help="Coordinate reference system of the X Y Z coordinates in the file (e.g. 'EPSG:2056' or 'WGS84').",
        )
