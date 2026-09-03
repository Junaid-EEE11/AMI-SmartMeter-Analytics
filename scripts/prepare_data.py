"""
CLI script to prepare, validate, clean, and aggregate AMI load datasets.
"""
import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl
from shift_ami.config import load_config
from shift_ami.data.synthetic import generate_synthetic_ami_dataset
from shift_ami.data.ingest import load_raw_ami_data
from shift_ami.data.validate import validate_ami_data, generate_data_quality_report
from shift_ami.data.preprocess import clean_and_standardize_ami
from shift_ami.data.aggregate import aggregate_cohorts
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.scripts.prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Clean, validate, and aggregate AMI datasets.")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to config file")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic test fixture generation")
    args = parser.parse_args()

    config = load_config(args.config)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    config.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    raw_path = config.paths.raw_file

    if args.force_synthetic or not raw_path.exists():
        logger.info(
            "Generating synthetic AMI dataset (explicitly labeled: SYNTHETIC — FOR PIPELINE VALIDATION ONLY)..."
        )
        syn_path = config.paths.processed_dir / "synthetic_ami.parquet"
        syn_df = generate_synthetic_ami_dataset(
            output_path=syn_path,
            seed=config.random_seed
        )
        pl_df = pl.from_pandas(syn_df)
    else:
        logger.info(f"Loading raw AMI data from {raw_path}...")
        pl_df = load_raw_ami_data(raw_path, schema_mapping=config.schema_mapping)

    # 1. Validate data quality
    logger.info("Executing data quality validation...")
    val_stats = validate_ami_data(pl_df, config.inclusion_criteria, output_dir=config.paths.results_dir / "data_quality")
    report_path = config.paths.reports_dir / "data_quality_report.md"
    generate_data_quality_report(val_stats, report_path)
    logger.info(f"Data quality report written to: {report_path}")

    # 2. Clean and standardize
    logger.info("Cleaning and standardizing records...")
    interim_path = config.paths.interim_dir / "ami_cleaned.parquet"
    cleaned_df = clean_and_standardize_ami(pl_df, config.inclusion_criteria, output_path=interim_path)

    # 3. Aggregate cohorts
    logger.info("Constructing cohort load aggregates...")
    cohorts = aggregate_cohorts(
        cleaned_df,
        config.inclusion_criteria,
        output_dir=config.paths.processed_dir,
        seed=config.random_seed
    )

    logger.info(f"Data preparation complete! Generated {len(cohorts)} cohort series in: {config.paths.processed_dir}")


if __name__ == "__main__":
    main()
