"""
CLI script to compile figures, tables, and reports from saved experiment results.
"""
import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shift_ami.config import load_config
from shift_ami.reporting.manuscript_assets import generate_all_reports
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.scripts.build_report")


def main():
    parser = argparse.ArgumentParser(description="Generate all figures, tables, and reports.")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Building all reports, tables, and figures...")

    generate_all_reports(
        results_dir=config.paths.results_dir,
        reports_dir=config.paths.reports_dir,
        processed_dir=config.paths.processed_dir
    )

    logger.info("Report building completed successfully.")


if __name__ == "__main__":
    main()
