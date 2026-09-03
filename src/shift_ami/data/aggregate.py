"""
Cohort and synthetic aggregate load constructor.

CRITICAL RESEARCH RULE (Section 8 of GEMINI.md):
Do NOT claim that these aggregates represent physical distribution feeders unless
actual network-topology/feeder mapping exists in the source data.
Call them cohorts or synthetic aggregates, as appropriate.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import polars as pl

from shift_ami.config import InclusionConfig
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.data.aggregate")


def aggregate_cohorts(
    df: pl.DataFrame,
    inclusion_cfg: InclusionConfig,
    output_dir: Optional[Path] = None,
    sample_subset_n: int = 50,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Construct half-hourly aggregate load series for:
    1. dToU dynamic tariff cohort aggregate
    2. Standard flat tariff cohort aggregate
    3. Total population aggregate
    4. Sampled subset of individual households for robustness experiments

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping cohort name to continuous half-hourly DataFrame indexed by timestamp.
    """
    logger.info("Building cohort aggregates...")

    # Ensure timestamp is datetime and sort
    df_sorted = df.sort("timestamp")
    
    # 1. Total population aggregate
    total_agg = (
        df_sorted.group_by("timestamp")
        .agg([
            pl.col("energy_kwh").sum().alias("load_kwh"),
            pl.col("energy_kwh").mean().alias("load_mean_kwh"),
            pl.col("energy_kwh").std().alias("load_std_kwh"),
            pl.col("household_id").n_unique().alias("active_households"),
        ])
        .sort("timestamp")
        .to_pandas()
    )
    total_agg["cohort"] = "total_population"

    # 2. dToU Cohort aggregate
    dtou_mask = pl.col("tariff_group").str.contains("(?i)tou|dynamic")
    dtou_df = df_sorted.filter(dtou_mask)
    if len(dtou_df) > 0:
        dtou_agg = (
            dtou_df.group_by("timestamp")
            .agg([
                pl.col("energy_kwh").sum().alias("load_kwh"),
                pl.col("energy_kwh").mean().alias("load_mean_kwh"),
                pl.col("energy_kwh").std().alias("load_std_kwh"),
                pl.col("household_id").n_unique().alias("active_households"),
            ])
            .sort("timestamp")
            .to_pandas()
        )
        dtou_agg["cohort"] = "dtou"
    else:
        dtou_agg = pd.DataFrame()

    # 3. Standard Tariff Cohort aggregate
    std_mask = pl.col("tariff_group").str.contains("(?i)std|flat")
    std_df = df_sorted.filter(std_mask)
    if len(std_df) > 0:
        std_agg = (
            std_df.group_by("timestamp")
            .agg([
                pl.col("energy_kwh").sum().alias("load_kwh"),
                pl.col("energy_kwh").mean().alias("load_mean_kwh"),
                pl.col("energy_kwh").std().alias("load_std_kwh"),
                pl.col("household_id").n_unique().alias("active_households"),
            ])
            .sort("timestamp")
            .to_pandas()
        )
        std_agg["cohort"] = "standard"
    else:
        std_agg = pd.DataFrame()

    # 4. Sampled individual households for robustness experiments
    all_households = df_sorted["household_id"].unique().to_list()
    rng = np.random.default_rng(seed)
    sampled_ids = rng.choice(all_households, size=min(sample_subset_n, len(all_households)), replace=False).tolist()
    sample_df = df_sorted.filter(pl.col("household_id").is_in(sampled_ids)).to_pandas()

    results = {
        "total": total_agg,
        "dtou": dtou_agg,
        "standard": std_agg,
        "household_sample": sample_df,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df_cohort in results.items():
            if not df_cohort.empty:
                out_path = output_dir / f"cohort_{name}_halfhourly.parquet"
                df_cohort.to_parquet(out_path, index=False)
                logger.info(f"Saved {name} cohort aggregate ({len(df_cohort):,} rows) to: {out_path}")

    return results
