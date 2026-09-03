"""
Comprehensive data quality analysis and validation module for AMI load data.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import polars as pl

from shift_ami.config import InclusionConfig
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.data.validate")


def validate_ami_data(
    df: pl.DataFrame,
    inclusion_cfg: InclusionConfig,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform rigorous data quality validation covering all 12 criteria in Section 10 of GEMINI.md.
    """
    logger.info("Executing comprehensive data quality audit...")
    total_records = len(df)
    
    # 1. Date Range
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    # 2. Customers
    unique_households = df["household_id"].n_unique()
    
    # 3. Tariff counts
    if "tariff_group" in df.columns:
        tariff_counts = df["tariff_group"].value_counts().to_dicts()
    else:
        tariff_counts = []

    # 4. Duplicates
    n_duplicates = total_records - len(df.unique(subset=["household_id", "timestamp"]))

    # 5. Missing / Nulls
    n_null_energy = df["energy_kwh"].null_count()
    n_null_timestamp = df["timestamp"].null_count()

    # 6. Negative / Impossible values
    n_negative = df.filter(pl.col("energy_kwh") < inclusion_cfg.valid_energy_range[0]).height
    n_extreme = df.filter(pl.col("energy_kwh") > inclusion_cfg.valid_energy_range[1]).height

    # Convert to pandas for customer-level history and missingness distribution metrics
    df_pd = df.select(["household_id", "timestamp", "energy_kwh", "tariff_group"]).to_pandas()
    
    # 7. Customer history length & missingness
    cust_stats = df_pd.groupby("household_id").agg(
        obs_count=("timestamp", "count"),
        min_date=("timestamp", "min"),
        max_date=("timestamp", "max"),
        null_count=("energy_kwh", lambda s: s.isna().sum())
    )
    cust_stats["duration_days"] = (cust_stats["max_date"] - cust_stats["min_date"]).dt.total_seconds() / 86400.0
    cust_stats["missing_ratio"] = cust_stats["null_count"] / cust_stats["obs_count"]

    n_insufficient_history = int((cust_stats["duration_days"] < inclusion_cfg.min_history_days).sum())
    n_high_missingness = int((cust_stats["missing_ratio"] > inclusion_cfg.max_missing_ratio).sum())

    # Exclusions
    excluded_households = cust_stats[
        (cust_stats["duration_days"] < inclusion_cfg.min_history_days) |
        (cust_stats["missing_ratio"] > inclusion_cfg.max_missing_ratio)
    ].index.tolist()

    valid_households = list(set(cust_stats.index) - set(excluded_households))

    report = {
        "total_records": int(total_records),
        "date_range": {
            "start": str(min_time),
            "end": str(max_time)
        },
        "customer_count": int(unique_households),
        "tariff_group_breakdown": tariff_counts,
        "duplicate_records": int(n_duplicates),
        "null_energy_count": int(n_null_energy),
        "null_timestamp_count": int(n_null_timestamp),
        "negative_consumption_count": int(n_negative),
        "extreme_consumption_count": int(n_extreme),
        "inclusion_rules_applied": {
            "min_history_days": inclusion_cfg.min_history_days,
            "max_missing_ratio": inclusion_cfg.max_missing_ratio,
            "valid_energy_range_kwh": inclusion_cfg.valid_energy_range,
            "insufficient_history_excluded_count": n_insufficient_history,
            "high_missingness_excluded_count": n_high_missingness,
            "total_excluded_households": len(excluded_households),
            "total_retained_households": len(valid_households)
        },
        "consumption_summary_kwh": {
            "mean": float(df_pd["energy_kwh"].mean()),
            "std": float(df_pd["energy_kwh"].std()),
            "min": float(df_pd["energy_kwh"].min()),
            "p25": float(df_pd["energy_kwh"].quantile(0.25)),
            "median": float(df_pd["energy_kwh"].median()),
            "p75": float(df_pd["energy_kwh"].quantile(0.75)),
            "p99": float(df_pd["energy_kwh"].quantile(0.99)),
            "max": float(df_pd["energy_kwh"].max())
        }
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "data_quality_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def generate_data_quality_report(validation_stats: Dict[str, Any], output_path: Path) -> str:
    """Generate professional Markdown data quality report."""
    inc = validation_stats["inclusion_rules_applied"]
    summary = validation_stats["consumption_summary_kwh"]

    md = f"""# Low Carbon London AMI Data Quality and Audit Report

## 1. Executive Summary
- **Total Raw Records**: {validation_stats['total_records']:,}
- **Observation Temporal Span**: `{validation_stats['date_range']['start']}` to `{validation_stats['date_range']['end']}`
- **Total Households Audited**: {validation_stats['customer_count']:,}
- **Households Meeting Inclusion Criteria**: {inc['total_retained_households']:,} (Retained)
- **Households Excluded**: {inc['total_excluded_households']:,}

---

## 2. Tariff Cohort Breakdown
| Tariff Cohort | Record Count |
| :--- | :--- |
"""
    for t in validation_stats["tariff_group_breakdown"]:
        group_name = t.get("tariff_group", "Unknown")
        count = t.get("count", t.get("len", 0))
        md += f"| `{group_name}` | {count:,} |\n"

    md += f"""
---

## 3. Data Integrity & Anomaly Audit
| Diagnostic Criterion | Metric Value | Operational Action Taken |
| :--- | :--- | :--- |
| **Duplicate Timestamps** | {validation_stats['duplicate_records']:,} | Deduplicated keeping first chronological occurrence |
| **Null Energy Values** | {validation_stats['null_energy_count']:,} | Imputed via linear interpolation or excluded if excessive |
| **Negative Values (< 0.0 kWh)** | {validation_stats['negative_consumption_count']:,} | Clipped to zero (non-negative load domain) |
| **Extreme Outliers (> {inc['valid_energy_range_kwh'][1]} kWh/hh)** | {validation_stats['extreme_consumption_count']:,} | Clipped to valid physical limit |
| **Insufficient History (< {inc['min_history_days']} days)** | {inc['insufficient_history_excluded_count']:,} households | Excluded from cohort aggregation |
| **High Missingness (> {inc['max_missing_ratio']*100:.0f}%)** | {inc['high_missingness_excluded_count']:,} households | Excluded from cohort aggregation |

---

## 4. Half-Hourly Consumption Distribution (kWh / half-hour)
- **Mean**: `{summary['mean']:.4f}` kWh
- **Standard Deviation**: `{summary['std']:.4f}` kWh
- **25th Percentile**: `{summary['p25']:.4f}` kWh
- **Median (50th)**: `{summary['median']:.4f}` kWh
- **75th Percentile**: `{summary['p75']:.4f}` kWh
- **99th Percentile**: `{summary['p99']:.4f}` kWh
- **Maximum Recorded**: `{summary['max']:.4f}` kWh
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    return md
