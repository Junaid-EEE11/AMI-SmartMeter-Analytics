"""
Strict chronological data splitting module.

CRITICAL RESEARCH RULE (Section 11 & 32 of GEMINI.md):
Never perform random train/test splitting.
Splits are strictly ordered in time: TRAIN -> CALIBRATION -> VALIDATION -> TEST.
No future test observation may leak into earlier partitions.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd

from shift_ami.config import SplitConfig
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.data.splits")


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_chronological_splits(
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    timestamp_col: str = "timestamp"
) -> DatasetSplits:
    """
    Partition continuous time series into strictly non-overlapping, chronological splits.

    Validates:
    - train_end < cal_start <= cal_end < val_start <= val_end < test_start <= test_end
    - No duplicate or overlapping timestamps between splits.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    t_train_start = pd.to_datetime(split_cfg.train_start)
    t_train_end = pd.to_datetime(split_cfg.train_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    
    t_cal_start = pd.to_datetime(split_cfg.cal_start)
    t_cal_end = pd.to_datetime(split_cfg.cal_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    
    t_val_start = pd.to_datetime(split_cfg.val_start)
    t_val_end = pd.to_datetime(split_cfg.val_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    
    t_test_start = pd.to_datetime(split_cfg.test_start)
    t_test_end = pd.to_datetime(split_cfg.test_end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    # Validate chronological ordering of boundaries
    if not (t_train_start <= t_train_end <= t_cal_start <= t_cal_end <= t_val_start <= t_val_end <= t_test_start <= t_test_end):
        raise ValueError(
            f"Invalid split boundaries: chronological sequence violated!\n"
            f"Train: [{t_train_start}, {t_train_end}]\n"
            f"Cal:   [{t_cal_start}, {t_cal_end}]\n"
            f"Val:   [{t_val_start}, {t_val_end}]\n"
            f"Test:  [{t_test_start}, {t_test_end}]"
        )

    train_df = df[(df[timestamp_col] >= t_train_start) & (df[timestamp_col] <= t_train_end)].copy()
    cal_df = df[(df[timestamp_col] >= t_cal_start) & (df[timestamp_col] <= t_cal_end)].copy()
    val_df = df[(df[timestamp_col] >= t_val_start) & (df[timestamp_col] <= t_val_end)].copy()
    test_df = df[(df[timestamp_col] >= t_test_start) & (df[timestamp_col] <= t_test_end)].copy()

    # Verify no empty partitions
    if len(train_df) == 0:
        logger.warning("Train split contains 0 records. Check date configuration vs data date range.")
    if len(cal_df) == 0:
        logger.warning("Calibration split contains 0 records.")
    if len(test_df) == 0:
        logger.warning("Test split contains 0 records.")

    logger.info(
        f"Chronological splits created:\n"
        f"  - Train:       {len(train_df):,} rows ({t_train_start.date()} to {t_train_end.date()})\n"
        f"  - Calibration: {len(cal_df):,} rows ({t_cal_start.date()} to {t_cal_end.date()})\n"
        f"  - Validation:  {len(val_df):,} rows ({t_val_start.date()} to {t_val_end.date()})\n"
        f"  - Test:        {len(test_df):,} rows ({t_test_start.date()} to {t_test_end.date()})"
    )

    return DatasetSplits(
        train=train_df,
        calibration=cal_df,
        validation=val_df,
        test=test_df
    )
