# pyre-strict
import argparse

from opensfm.actions import export_report
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "export_report"
    help = "Export a nice report based on previously generated statistics"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        export_report.run_dataset(dataset, title=args.title, accent_color=args.accent_color)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--title",
            default=None,
            help="Custom report title (replaces the default OpenSfM branding)",
        )
        parser.add_argument(
            "--accent-color",
            nargs=3,
            type=int,
            default=None,
            metavar=("R", "G", "B"),
            help="Custom accent color as three RGB values (0-255)",
        )
