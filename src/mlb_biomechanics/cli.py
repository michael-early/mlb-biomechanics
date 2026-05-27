from __future__ import annotations

import argparse
import json

from .data import download_public_sources
from .pipeline import build_mvp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLB biomechanics MVP pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-data", help="Download public OpenBiomechanics CSVs")
    subparsers.add_parser("build", help="Build metrics, models, tables, and report")
    subparsers.add_parser("report", help="Alias for build; writes reports/mvp_report.html")

    args = parser.parse_args(argv)
    if args.command == "download-data":
        outputs = download_public_sources()
        print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))
        return 0
    if args.command in {"build", "report"}:
        outputs = build_mvp()
        print(json.dumps(outputs, indent=2))
        return 0
    return 2

