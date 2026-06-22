# pyre-strict
import argparse

from opensfm.actions import fuse_depthmaps
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "fuse_depthmaps"
    help = "Dense stage 3: fuse cleaned depthmaps per cluster"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        fuse_depthmaps.run_dataset(dataset, args.subfolder)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--subfolder",
            help="undistorted subfolder where to load and store data",
            default="undistorted",
        )
