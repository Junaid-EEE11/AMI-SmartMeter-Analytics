"""
Memory-efficient data ingestion and schema mapping for Low Carbon London AMI records.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import polars as pl

from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.data.ingest")


def map_schema(df: pl.DataFrame, schema_mapping: Dict[str, Any]) -> pl.DataFrame:
    """
    Map raw column names to canonical internal names using schema mapping dictionary.
    """
    raw_col_map = schema_mapping.get("raw_columns", {})
    existing_cols = df.columns

    rename_dict = {}
    for canonical_name, possible_raw_names in raw_col_map.items():
        matched = False
        for raw_name in possible_raw_names:
            if raw_name in existing_cols:
                rename_dict[raw_name] = canonical_name
                matched = True
                break
        if not matched and canonical_name in ["household_id", "timestamp", "energy_kwh"]:
            logger.warning(f"Could not map canonical column '{canonical_name}' from available columns: {existing_cols}")

    if rename_dict:
        df = df.rename(rename_dict)

    # Cast types cleanly
    cast_exprs = []
    if "household_id" in df.columns:
        cast_exprs.append(pl.col("household_id").cast(pl.Utf8))
    if "timestamp" in df.columns:
        if df["timestamp"].dtype != pl.Datetime:
            cast_exprs.append(pl.col("timestamp").cast(pl.Utf8).str.to_datetime(strict=False))
    if "energy_kwh" in df.columns:
        # Handle possible string/null representations in raw CSV
        cast_exprs.append(pl.col("energy_kwh").cast(pl.Float64, strict=False))
    if "tariff_group" in df.columns:
        cast_exprs.append(pl.col("tariff_group").cast(pl.Utf8))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def load_raw_ami_data(
    file_path: Union[str, Path],
    schema_mapping: Optional[Dict[str, Any]] = None,
    n_rows: Optional[int] = None
) -> pl.DataFrame:
    """
    Load raw AMI data file (CSV or Parquet) memory-efficiently using Polars.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    logger.info(f"Loading raw AMI data from: {path}")
    if path.suffix.lower() == ".parquet":
        if n_rows is not None:
            df = pl.read_parquet(path, n_rows=n_rows)
        else:
            df = pl.read_parquet(path)
    elif path.suffix.lower() in [".csv", ".txt"]:
        if n_rows is not None:
            df = pl.read_csv(path, n_rows=n_rows, infer_schema_length=10000, ignore_errors=True)
        else:
            df = pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if schema_mapping:
        df = map_schema(df, schema_mapping)

    logger.info(f"Loaded {len(df):,} records with columns: {df.columns}")
    return df
