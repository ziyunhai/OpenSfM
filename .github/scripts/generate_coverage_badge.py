#!/usr/bin/env python3
"""Generate a repository-local SVG badge from Cobertura XML reports."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class CoverageTotals:
    """Coverage counters extracted from a Cobertura XML report."""

    name: str
    lines_covered: int
    lines_valid: int
    branches_covered: int
    branches_valid: int

    def line_rate(self) -> Optional[float]:
        if self.lines_valid == 0:
            return None
        return (self.lines_covered / self.lines_valid) * 100.0

    def branch_rate(self) -> Optional[float]:
        if self.branches_valid == 0:
            return None
        return (self.branches_covered / self.branches_valid) * 100.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "lines_covered": self.lines_covered,
            "lines_valid": self.lines_valid,
            "line_rate": self.line_rate(),
            "branches_covered": self.branches_covered,
            "branches_valid": self.branches_valid,
            "branch_rate": self.branch_rate(),
        }


def _parse_count(attributes: Dict[str, str], key: str) -> int:
    raw_value = attributes.get(key)
    if raw_value is None:
        return 0
    return int(float(raw_value))


def _load_report(report_path: Path) -> CoverageTotals:
    root = ElementTree.parse(report_path).getroot()
    return CoverageTotals(
        name=report_path.stem,
        lines_covered=_parse_count(root.attrib, "lines-covered"),
        lines_valid=_parse_count(root.attrib, "lines-valid"),
        branches_covered=_parse_count(root.attrib, "branches-covered"),
        branches_valid=_parse_count(root.attrib, "branches-valid"),
    )


def _merge_reports(reports: List[CoverageTotals]) -> CoverageTotals:
    return CoverageTotals(
        name="total",
        lines_covered=sum(report.lines_covered for report in reports),
        lines_valid=sum(report.lines_valid for report in reports),
        branches_covered=sum(report.branches_covered for report in reports),
        branches_valid=sum(report.branches_valid for report in reports),
    )


def _format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _badge_color(rate: Optional[float]) -> str:
    if rate is None:
        return "#9f9f9f"
    if rate < 50.0:
        return "#e05d44"
    if rate < 70.0:
        return "#fe7d37"
    if rate < 85.0:
        return "#dfb317"
    if rate < 95.0:
        return "#a4a61d"
    return "#4c1"


def _segment_width(text: str) -> int:
    return max(10, len(text) * 7 + 10)


def _build_badge_svg(label: str, value: str, color: str) -> str:
    label_width = _segment_width(label)
    value_width = _segment_width(value)
    total_width = label_width + value_width
    label_center = label_width * 5
    value_center = (label_width + (value_width / 2.0)) * 10
    aria_label = escape(f"{label}: {value}")
    label_text = escape(label)
    value_text = escape(value)
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"20\" role=\"img\" aria-label=\"{aria_label}\">\n  <linearGradient id=\"s\" x2=\"0\" y2=\"100%\">\n    <stop offset=\"0\" stop-color=\"#bbb\" stop-opacity=\".1\"/>\n    <stop offset=\"1\" stop-opacity=\".1\"/>\n  </linearGradient>\n  <clipPath id=\"r\">\n    <rect width=\"{total_width}\" height=\"20\" rx=\"3\" fill=\"#fff\"/>\n  </clipPath>\n  <g clip-path=\"url(#r)\">\n    <rect width=\"{label_width}\" height=\"20\" fill=\"#555\"/>\n    <rect x=\"{label_width}\" width=\"{value_width}\" height=\"20\" fill=\"{color}\"/>\n    <rect width=\"{total_width}\" height=\"20\" fill=\"url(#s)\"/>\n  </g>\n  <g fill=\"#fff\" text-anchor=\"middle\" font-family=\"Verdana,Geneva,DejaVu Sans,sans-serif\" text-rendering=\"geometricPrecision\" font-size=\"110\">\n    <text x=\"{label_center}\" y=\"140\" transform=\"scale(.1)\" fill=\"#010101\" fill-opacity=\".3\">{label_text}</text>\n    <text x=\"{label_center}\" y=\"130\" transform=\"scale(.1)\">{label_text}</text>\n    <text x=\"{value_center}\" y=\"140\" transform=\"scale(.1)\" fill=\"#010101\" fill-opacity=\".3\">{value_text}</text>\n    <text x=\"{value_center}\" y=\"130\" transform=\"scale(.1)\">{value_text}</text>\n  </g>\n</svg>\n"""


def _build_summary(reports: List[CoverageTotals], total: CoverageTotals) -> Dict[str, object]:
    return {
        "reports": [report.to_dict() for report in reports],
        "total": total.to_dict(),
        "badge": {
            "label": "coverage",
            "message": _format_percentage(total.line_rate()),
            "color": _badge_color(total.line_rate()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path,
                        help="Cobertura XML coverage reports")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to the SVG badge to write")
    parser.add_argument(
        "--summary-json",
        required=True,
        type=Path,
        help="Path to a JSON summary file written alongside the badge",
    )
    args = parser.parse_args()

    reports = [_load_report(path) for path in args.reports]
    total = _merge_reports(reports)
    badge_value = _format_percentage(total.line_rate())
    badge_svg = _build_badge_svg(
        "coverage", badge_value, _badge_color(total.line_rate()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(badge_svg, encoding="utf-8")
    args.summary_json.write_text(
        json.dumps(_build_summary(reports, total),
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
