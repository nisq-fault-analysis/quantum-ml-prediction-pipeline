"""CLI entry point for durable thesis milestone reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting.milestone_reports import generate_milestone_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for milestone report generation."""

    parser = argparse.ArgumentParser(
        description="Generate a durable milestone report from saved experiment artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the milestone report YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_milestone_report(config_path=args.config)
    print(f"Milestone report Markdown saved to: {outputs.markdown_path}")
    print(f"Milestone report JSON saved to: {outputs.json_path}")
    print(f"Milestone report schema saved to: {outputs.schema_path}")


if __name__ == "__main__":
    main()
