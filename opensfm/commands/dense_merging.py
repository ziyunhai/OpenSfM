# pyre-strict
import argparse

from opensfm.actions import dense_merging
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "dense_merging"
    help = "Dense stage 4: merge fused PLYs and DSM/ortho, then export"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        dense_merging.run_dataset(dataset, args.subfolder)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--subfolder",
            help="undistorted subfolder where to load and store data",
            default="undistorted",
        )
