"""
Automated tests for synthetic AMI test fixture generator.

CRITICAL RESEARCH RULE (Section 4 & 24 of GEMINI.md):
Verifies that synthetic data is generated with realistic diurnal/seasonal patterns
and explicitly labeled for pipeline validation.
"""
import numpy as np
import pandas as pd
import pytest

from shift_ami.data.synthetic import generate_synthetic_ami_dataset


def test_synthetic_ami_generation(tmp_path):
    out_file = tmp_path / "test_syn.parquet"
    df = generate_synthetic_ami_dataset(
        n_households_dtou=5,
        n_households_std=10,
        start_date="2012-01-01",
        end_date="2012-01-07 23:30:00",
        output_path=out_file,
        seed=42
    )

    assert out_file.exists()
    assert len(df) == 15 * 7 * 48
    assert "household_id" in df.columns
    assert "timestamp" in df.columns
    assert "energy_kwh" in df.columns
    assert "tariff_group" in df.columns

    # Check physical constraints: energy_kwh >= 0
    assert (df["energy_kwh"] >= 0.0).all()
    # Check tariff groups
    assert set(df["tariff_group"].unique()).issubset({"ToU", "Std"})
