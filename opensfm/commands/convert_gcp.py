# pyre-strict
import argparse

from opensfm.actions import convert_gcp
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "convert_gcp"
    help = "Convert ground control points between gcp_list.txt and ground_control_points.json formats."

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        convert_gcp.run_dataset(dataset, args.from_format)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--from",
            dest="from_format",
            choices=["txt", "json"],
            required=True,
            help="Source format to convert from (txt to convert gcp_list.txt to JSON, json to convert JSON to gcp_list.txt).",
        )
