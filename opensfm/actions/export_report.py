# pyre-strict
from typing import Optional

from opensfm import report
from opensfm.dataset import DataSet


def run_dataset(data: DataSet, title: Optional[str] = None) -> None:
    """Export a nice report based on previously generated statistics

    Args:
        data: dataset object
        title: optional custom report title

    """
    pdf_report = report.Report(data, title=title)
    pdf_report.generate_report()
    pdf_report.save_report("report.pdf")
