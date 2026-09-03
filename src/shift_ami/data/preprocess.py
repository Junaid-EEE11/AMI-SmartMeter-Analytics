"""
Data preprocessing, cleaning, and standardization pipeline.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import polars as pl

from shift_ami.config import InclusionConfig
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.data.preprocess")


def clean_and_standardize_ami(
    df: pl.DataFrame,
    inclusion_cfg: InclusionConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Clean raw AMI dataframe according to explicit rules:
    1. Deduplicate by (household_id, timestamp).
    2. Filter invalid/corrupted timestamps.
    3. Filter/clip energy values to valid physical limits.
    4. Retain only households with sufficient history and acceptable missingness ratio.
    """
    logger.info(f"Starting data preprocessing on {len(df):,} records...")

    # Step 1: Remove null timestamps or household_ids
    df_clean = df.filter(
        pl.col("timestamp").is_not_null() &
        pl.col("household_id").is_not_null()
    )

    # Step 2: Deduplicate
    df_clean = df_clean.unique(subset=["household_id", "timestamp"], keep="first")

    # Step 3: Handle energy_kwh bounds
    # Nulls filled with 0.0 or forward fill in aggregate, clip negative and extreme values
    min_kwh, max_kwh = inclusion_cfg.valid_energy_range
    df_clean = df_clean.with_columns(
        pl.col("energy_kwh").fill_null(0.0).clip(min_kwh, max_kwh)
    )

    # Step 4: Household-level filtering based on history length and missingness
    # Compute duration per household
    household_stats = df_clean.group_by("household_id").agg([
        pl.col("timestamp").min().alias("min_ts"),
        pl.col("timestamp").max().alias("max_ts"),
        pl.col("energy_kwh").count().alias("count")
    ]).with_columns(
        ((pl.col("max_ts") - pl.col("min_ts")).dt.total_seconds() / 86400.0).alias("duration_days")
    )

    valid_households = household_stats.filter(
        pl.col("duration_days") >= inclusion_cfg.min_history_days
    ).select("household_id")

    df_filtered = df_clean.join(valid_households, on="household_id", how="inner")

    # Sort chronologically by household and timestamp
    df_filtered = df_filtered.sort(["household_id", "timestamp"])

    logger.info(f"Preprocessing completed: {len(df_filtered):,} clean records across {df_filtered['household_id'].n_unique():,} households.")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.write_parquet(output_path)
        logger.info(f"Saved preprocessed dataset to: {output_path}")

    return df_filtered
