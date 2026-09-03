"""
Synthetic AMI Dataset Generator for Unit Testing, Integration Testing, and Pipeline Verification.

CRITICAL RESEARCH RULE (Section 25 of GEMINI.md):
Every synthetic dataset and output is explicitly labeled:
'SYNTHETIC — FOR PIPELINE VALIDATION ONLY'
Never present synthetic results as empirical findings from the Low Carbon London dataset.
"""
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_ami_dataset(
    start_date: str = "2012-01-01",
    end_date: str = "2013-12-31",
    n_households_dtou: int = 20,
    n_households_std: int = 40,
    shift_start_date: str = "2013-03-01",
    shift_magnitude_mean: float = 0.25,
    shift_magnitude_var: float = 1.40,
    shift_peak_dampening: float = -0.20,
    seed: int = 42,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate synthetic half-hourly smart-meter data with realistic diurnal cycles,
    weekly periodicity, household heterogeneity, and controllable regime/distribution shifts.

    Parameters
    ----------
    start_date : str
        Start timestamp (YYYY-MM-DD).
    end_date : str
        End timestamp (YYYY-MM-DD).
    n_households_dtou : int
        Number of synthetic Dynamic Time-of-Use cohort households.
    n_households_std : int
        Number of synthetic Standard flat tariff households.
    shift_start_date : str
        Timestamp after which behavioral and variance distribution shift is applied.
    shift_magnitude_mean : float
        Proportional baseline shift added post shift_start_date.
    shift_magnitude_var : float
        Variance multiplier applied post shift_start_date.
    shift_peak_dampening : float
        Behavioral peak-shaving response during peak hours (17:00-21:00) for dToU cohort.
    seed : int
        Deterministic random seed.
    output_path : Optional[Path]
        If provided, saves synthetic dataset to Parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['household_id', 'timestamp', 'energy_kwh', 'tariff_group', 'acorn_group', 'data_source']
    """
    rng = np.random.default_rng(seed)

    # 30-minute frequency time index
    timestamps = pd.date_range(start=start_date, end=end_date, freq="30min", inclusive="both")
    n_steps = len(timestamps)

    # Base diurnal pattern (48 half-hours)
    # Peak in evening (approx half-hour 34-42, 17:00 - 21:00), smaller peak morning (half-hour 14-18, 07:00-09:00), low overnight (00:00-05:00)
    hh_idx = timestamps.hour * 2 + timestamps.minute // 30
    dow = timestamps.dayofweek  # 0=Monday, 6=Sunday
    is_weekend = (dow >= 5).astype(float)

    # Seasonal annual cycle (winter cold higher demand in London, summer lower)
    doy = timestamps.dayofyear
    annual_cycle = 1.0 + 0.35 * np.cos(2 * np.pi * (doy - 20) / 365.25)

    # Daily diurnal base profile
    diurnal_base = (
        0.15
        + 0.12 * np.exp(-0.5 * ((hh_idx - 16) / 3.0) ** 2)   # morning peak ~8:00
        + 0.35 * np.exp(-0.5 * ((hh_idx - 38) / 4.0) ** 2)   # evening peak ~19:00
        + 0.08 * is_weekend * np.exp(-0.5 * ((hh_idx - 24) / 6.0) ** 2) # weekend midday
    )

    records = []
    shift_mask = timestamps >= pd.to_datetime(shift_start_date)

    # Generate dToU cohort
    for i in range(n_households_dtou):
        hid = f"SYN_DTOU_{i+1:04d}"
        h_scale = rng.uniform(0.7, 1.4)  # Household baseline scale
        h_noise_std = rng.uniform(0.04, 0.08)

        # Baseline series for household
        expected_load = diurnal_base * annual_cycle * h_scale

        # Shift effects for dToU:
        # 1. Overall mean/variance shift from weather/macro drift
        # 2. Tariff response: behavioral peak-shaving from 17:00 to 21:00 (hh 34 to 42) and slight off-peak rebound
        is_peak_period = (hh_idx >= 34) & (hh_idx <= 42)
        is_night_rebound = (hh_idx >= 44) | (hh_idx <= 6)

        load = expected_load.copy()
        
        # Apply structural shift post shift_start_date
        shift_effect = np.zeros(n_steps)
        shift_effect[shift_mask] = (
            shift_magnitude_mean * expected_load[shift_mask]
            + shift_peak_dampening * expected_load[shift_mask] * is_peak_period[shift_mask]
            + 0.10 * expected_load[shift_mask] * is_night_rebound[shift_mask]
        )
        load += shift_effect

        # Dynamic noise with variance shift
        noise_std = np.where(shift_mask, h_noise_std * shift_magnitude_var, h_noise_std)
        noise = rng.normal(0, noise_std, size=n_steps)
        energy_kwh = np.maximum(0.01, load + noise)

        df_h = pd.DataFrame({
            "household_id": hid,
            "timestamp": timestamps,
            "energy_kwh": energy_kwh.round(4),
            "tariff_group": "ToU",
            "acorn_group": rng.choice(["Affluent", "Comfortable", "Adversity"]),
            "data_source": "SYNTHETIC — FOR PIPELINE VALIDATION ONLY"
        })
        records.append(df_h)

    # Generate Standard Tariff cohort
    for i in range(n_households_std):
        hid = f"SYN_STD_{i+1:04d}"
        h_scale = rng.uniform(0.7, 1.4)
        h_noise_std = rng.uniform(0.04, 0.08)

        expected_load = diurnal_base * annual_cycle * h_scale
        load = expected_load.copy()

        # Standard tariff undergoes seasonal/macro drift, but NO peak-shaving tariff response
        shift_effect = np.zeros(n_steps)
        shift_effect[shift_mask] = shift_magnitude_mean * expected_load[shift_mask]
        load += shift_effect

        noise_std = np.where(shift_mask, h_noise_std * shift_magnitude_var, h_noise_std)
        noise = rng.normal(0, noise_std, size=n_steps)
        energy_kwh = np.maximum(0.01, load + noise)

        df_h = pd.DataFrame({
            "household_id": hid,
            "timestamp": timestamps,
            "energy_kwh": energy_kwh.round(4),
            "tariff_group": "Std",
            "acorn_group": rng.choice(["Affluent", "Comfortable", "Adversity"]),
            "data_source": "SYNTHETIC — FOR PIPELINE VALIDATION ONLY"
        })
        records.append(df_h)

    full_df = pd.concat(records, ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(output_path, index=False)

    return full_df
