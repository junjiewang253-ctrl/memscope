from __future__ import annotations

import argparse
from pathlib import Path
from memscope.visualization.runtime_html import write_runtime_report_html

def main():
    parser = argparse.ArgumentParser(description="MemScope CLI")
    subparsers = parser.add_subparsers(dest="command")

    vis_parser = subparsers.add_parser(
        "visualize-runtime", 
        help="Render runtime_report.json to a standalone HTML file", 
    )
    vis_parser.add_argument(
        "--report", 
        type=str, 
        required=True, 
        help="Path to runtime_report.json", 
    )
    vis_parser.add_argument(
        "--out", 
        type=str, 
        default=None, 
        help="Output HTML path. Defaults to report_path with .html suffix", 
    )

    args = parser.parse_args()

    if args.command == "visualize-runtime":
        report_path = Path(args.report)
        outpath = Path(args.out) if args.out else report_path.with_suffix(".html")
        result = write_runtime_report_html(report_path, outpath)
        print(f"[MemScope] HTML report written to: {result}")

    parser.print_help()

if __name__ == "main":
    main()