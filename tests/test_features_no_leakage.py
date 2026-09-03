"""
Automated tests verifying strict absence of future data leakage in feature engineering.

CRITICAL LEAKAGE TEST (Section 24 of GEMINI.md):
Demonstrates that:
1. Feature at forecast origin t_0 never uses load observed after t_0.
2. Perturbing future values leaves past feature matrices 100% unaltered.
3. Scaling parameters are learned strictly from training split only.
"""
import numpy as np
import pandas as pd
import pytest

from shift_ami.features.calendar import extract_calendar_features
from shift_ami.features.lagged_load import construct_lagged_load_features
from shift_ami.features.build import FeaturePipeline


def test_lagged_features_strict_non_leakage():
    # 500 half-hours of synthetic load
    timestamps = pd.date_range("2013-01-01", periods=500, freq="30min")
    rng = np.random.default_rng(42)
    load_original = pd.Series(rng.uniform(10, 50, size=500), index=timestamps)

    horizon = 48
    lags = [48, 96, 144, 336]

    # Build features on original series
    feats_orig = construct_lagged_load_features(load_original, horizon=horizon, lags=lags)

    # Now create a perturbed copy where timestamps >= index 300 have large noise added
    cutoff_idx = 300
    load_perturbed = load_original.copy()
    load_perturbed.iloc[cutoff_idx:] += 1000.0  # Massive future perturbation

    feats_pert = construct_lagged_load_features(load_perturbed, horizon=horizon, lags=lags)

    # For any target time t < cutoff_idx, the forecast origin is t_0 = t - horizon <= cutoff_idx - 48 < 300.
    # Therefore, features for all t < cutoff_idx MUST be 100% identical between original and perturbed!
    safe_slice_orig = feats_orig.iloc[:cutoff_idx].dropna()
    safe_slice_pert = feats_pert.iloc[:cutoff_idx].dropna()

    pd.testing.assert_frame_equal(safe_slice_orig, safe_slice_pert)


def test_lag_smaller_than_horizon_raises_leakage_error():
    # If a user attempts to request lag 24 for horizon 48, it must raise a Data Leakage ValueError!
    s = pd.Series(np.arange(100.0))
    with pytest.raises(ValueError, match="Data Leakage Violation"):
        construct_lagged_load_features(s, horizon=48, lags=[24])


def test_scaler_learned_strictly_from_training():
    pipe = FeaturePipeline(scale_features=True)

    # Training features (mean ~ 10.0, std ~ 2.0)
    rng = np.random.default_rng(42)
    df_train = pd.DataFrame({"feat_a": rng.normal(10.0, 2.0, 500)})

    # Test features with different distribution (mean ~ 50.0, std ~ 10.0)
    df_test = pd.DataFrame({"feat_a": rng.normal(50.0, 10.0, 500)})

    # Fit on train
    pipe.fit(df_train)

    # Scaler mean must match training mean, NOT test
    assert np.isclose(pipe.scaler.mean_[0], df_train["feat_a"].mean(), atol=1e-3)
    assert not np.isclose(pipe.scaler.mean_[0], df_test["feat_a"].mean(), atol=1.0)
