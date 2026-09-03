"""
Automated tests for chronological data splitting.

CRITICAL LEAKAGE TEST (Section 24 of GEMINI.md):
Verifies that:
1. Split boundaries are strictly chronological: Train < Cal < Val < Test.
2. No test observations enter training or calibration.
3. No duplicate or overlapping timestamps between splits.
"""
import pandas as pd
import pytest

from shift_ami.config import SplitConfig
from shift_ami.data.splits import create_chronological_splits


def test_chronological_split_ordering():
    # Generate 1 year of half-hourly dummy timestamps
    dates = pd.date_range("2012-01-01", "2013-12-31 23:30:00", freq="30min")
    df = pd.DataFrame({
        "timestamp": dates,
        "load_kwh": 10.0
    })

    cfg = SplitConfig(
        train_start="2012-01-01",
        train_end="2012-06-30",
        cal_start="2012-07-01",
        cal_end="2012-08-31",
        val_start="2012-09-01",
        val_end="2012-09-30",
        test_start="2012-10-01",
        test_end="2012-12-31"
    )

    splits = create_chronological_splits(df, cfg)

    # 1. Non-empty
    assert len(splits.train) > 0
    assert len(splits.calibration) > 0
    assert len(splits.validation) > 0
    assert len(splits.test) > 0

    # 2. Strict boundary inequalities
    assert splits.train["timestamp"].max() < splits.calibration["timestamp"].min()
    assert splits.calibration["timestamp"].max() < splits.validation["timestamp"].min()
    assert splits.validation["timestamp"].max() < splits.test["timestamp"].min()

    # 3. No timestamp overlap
    train_ts = set(splits.train["timestamp"])
    cal_ts = set(splits.calibration["timestamp"])
    val_ts = set(splits.validation["timestamp"])
    test_ts = set(splits.test["timestamp"])

    assert len(train_ts.intersection(cal_ts)) == 0
    assert len(train_ts.intersection(test_ts)) == 0
    assert len(cal_ts.intersection(test_ts)) == 0
    assert len(val_ts.intersection(test_ts)) == 0


def test_invalid_split_boundary_raises_error():
    # If train end is after calibration start, must raise ValueError
    df = pd.DataFrame({"timestamp": pd.date_range("2012-01-01", "2012-12-31", freq="1D"), "load_kwh": 1.0})
    invalid_cfg = SplitConfig(
        train_start="2012-01-01",
        train_end="2012-08-31",
        cal_start="2012-07-01",  # Overlaps with train
        cal_end="2012-09-30",
        val_start="2012-10-01",
        val_end="2012-10-31",
        test_start="2012-11-01",
        test_end="2012-12-31"
    )
    with pytest.raises(ValueError):
        create_chronological_splits(df, invalid_cfg)
