# pyre-strict
from typing import List, Optional, Tuple

from opensfm import report
from opensfm.dataset import DataSet


def run_dataset(
    data: DataSet,
    title: Optional[str] = None,
    accent_color: Optional[List[int]] = None,
) -> None:
    """Export a nice report based on previously generated statistics

    Args:
        data: dataset object
        title: optional custom report title
        accent_color: optional RGB accent color as [R, G, B] (0-255)

    """
    color_tuple: Optional[Tuple[int, int, int]] = None
    if accent_color is not None:
        color_tuple = (accent_color[0], accent_color[1], accent_color[2])
    pdf_report = report.Report(data, title=title, accent_color=color_tuple)
    pdf_report.generate_report()
    pdf_report.save_report("report.pdf")
