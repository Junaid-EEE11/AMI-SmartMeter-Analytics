"""
CLI script to execute the main forecasting and conformal prediction experiment.
"""
import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shift_ami.config import load_config
from shift_ami.experiments.runner import run_experiment_pipeline
from shift_ami.reporting.manuscript_assets import generate_all_reports
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.scripts.run_experiment")


def main():
    parser = argparse.ArgumentParser(description="Run shift-ami forecasting and conformal prediction experiment.")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to YAML configuration file")
    parser.add_argument("--skip-reporting", action="store_true", help="Skip figure and table generation")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    # Run pipeline
    results = run_experiment_pipeline(config)

    # Generate figures, tables, and reports
    if not args.skip_reporting:
        generate_all_reports(
            results_dir=config.paths.results_dir,
            reports_dir=config.paths.reports_dir,
            processed_dir=config.paths.processed_dir
        )

    logger.info("Experiment run completed successfully.")


if __name__ == "__main__":
    main()
