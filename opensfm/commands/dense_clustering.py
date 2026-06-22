# pyre-strict
import argparse

from opensfm.actions import dense_clustering
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "dense_clustering"
    help = "Dense stage 1: build clusters, neighbours and depth ranges"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        dense_clustering.run_dataset(dataset, args.subfolder)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--subfolder",
            help="undistorted subfolder where to load and store data",
            default="undistorted",
        )
