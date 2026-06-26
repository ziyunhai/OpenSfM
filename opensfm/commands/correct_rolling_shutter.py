# pyre-strict
import argparse

from opensfm.actions import correct_rolling_shutter
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "correct_rolling_shutter"
    help = "Trim/correct features for rolling shutter distortion"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        correct_rolling_shutter.run_dataset(dataset, args.rolling_shutter_readout)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--rolling-shutter-readout",
            help="Rolling shutter sensor readout time (ms)",
            type=float,
            default=30.0,
        )
