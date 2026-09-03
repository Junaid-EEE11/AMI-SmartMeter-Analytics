"""
CLI script to inspect raw dataset schemas and print summaries.
"""
import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shift_ami.config import load_config
from shift_ami.data.ingest import load_raw_ami_data
from shift_ami.data.validate import validate_ami_data, generate_data_quality_report
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.scripts.inspect_raw_data")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw Low Carbon London AMI dataset files.")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to config file")
    parser.add_argument("--n-rows", type=int, default=100000, help="Number of rows to preview")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = config.paths.raw_file

    if not raw_path.exists():
        logger.warning(f"Raw file not found at: {raw_path}")
        logger.info(f"Check data/README.md for download instructions or run prepare_data.py to generate synthetic fixture.")
        return

    logger.info(f"Inspecting raw file: {raw_path} (previewing {args.n_rows:,} rows)...")
    df = load_raw_ami_data(raw_path, schema_mapping=config.schema_mapping, n_rows=args.n_rows)
    logger.info(f"Columns present: {df.columns}")
    logger.info(f"Sample rows:\n{df.head(5)}")

    # Audit quality
    stats = validate_ami_data(df, config.inclusion_criteria)
    report_path = config.paths.reports_dir / "data_quality_report.md"
    generate_data_quality_report(stats, report_path)
    logger.info(f"Generated data quality report at: {report_path}")


if __name__ == "__main__":
    main()
