import argparse
from pathlib import Path

from memscope.parsers.config_loader import load_config
from memscope.report.console_reporter import print_static_report
from memscope.report.json_reporter import write_json_report
from memscope.report.markdown_reporter import render_markdown_report
from memscope.static.analyzer import analyze_static

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to json/yaml config")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = analyze_static(cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "static_report.json"
    md_path = outdir / "static_report.md"

    write_json_report(report, str(json_path))
    md_text = render_markdown_report(report)
    md_path.write_text(md_text, encoding="utf-8")

    print_static_report(report)
    print(f"\nSaved JSON report to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()