"""
Model-independent distribution shift characterization and statistical diagnostics.

CRITICAL RESEARCH RULE (Section 19 of GEMINI.md):
Quantify distribution shift independently of model performance.
Do not equate every seasonal change with tariff-induced behavioral response.
Maintain precise terminology: temporal shift, behavioral/regime shift, tariff cohort.
"""
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from statsmodels.tsa.stattools import acf


def characterize_distribution_shift(
    pre_series: pd.Series,
    post_series: pd.Series,
    pre_label: str = "Pre-Shift",
    post_label: str = "Post-Shift"
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical shift analysis comparing pre-shift and post-shift load series.
    """
    pre = pre_series.dropna().values
    post = post_series.dropna().values

    # 1. First and second moments
    mean_pre, mean_post = float(np.mean(pre)), float(np.mean(post))
    std_pre, std_post = float(np.std(pre)), float(np.std(post))
    median_pre, median_post = float(np.median(pre)), float(np.median(post))

    # 2. Peak and tails (95th & 99th percentiles)
    p95_pre, p95_post = float(np.percentile(pre, 95)), float(np.percentile(post, 95))
    p99_pre, p99_post = float(np.percentile(pre, 99)), float(np.percentile(post, 99))

    # 3. Ramps (half-hourly delta)
    ramps_pre = np.abs(np.diff(pre))
    ramps_post = np.abs(np.diff(post))
    mean_ramp_pre, mean_ramp_post = float(np.mean(ramps_pre)), float(np.mean(ramps_post))

    # 4. Autocorrelation structure (lag 48 = 24h, lag 336 = 1 week)
    nlags = min(350, len(pre) // 2 - 1, len(post) // 2 - 1)
    if nlags >= 48:
        acf_pre = acf(pre, nlags=nlags, fft=True)
        acf_post = acf(post, nlags=nlags, fft=True)
        acf_lag48_pre = float(acf_pre[48])
        acf_lag48_post = float(acf_post[48])
        acf_lag336_pre = float(acf_pre[336]) if nlags >= 336 else np.nan
        acf_lag336_post = float(acf_post[336]) if nlags >= 336 else np.nan
    else:
        acf_lag48_pre, acf_lag48_post = np.nan, np.nan
        acf_lag336_pre, acf_lag336_post = np.nan, np.nan

    # 5. Non-parametric distribution distances & hypothesis tests
    w1_dist = float(wasserstein_distance(pre, post))
    ks_res = ks_2samp(pre, post)
    mwu_res = mannwhitneyu(pre, post, alternative="two-sided")

    return {
        "labels": {"pre": pre_label, "post": post_label},
        "sample_sizes": {"n_pre": len(pre), "n_post": len(post)},
        "load_mean": {"pre": round(mean_pre, 4), "post": round(mean_post, 4), "pct_change": round((mean_post - mean_pre)/mean_pre * 100.0, 2)},
        "load_std": {"pre": round(std_pre, 4), "post": round(std_post, 4), "pct_change": round((std_post - std_pre)/std_pre * 100.0, 2)},
        "load_median": {"pre": round(median_pre, 4), "post": round(median_post, 4)},
        "peak_95th": {"pre": round(p95_pre, 4), "post": round(p95_post, 4)},
        "peak_99th": {"pre": round(p99_pre, 4), "post": round(p99_post, 4)},
        "mean_absolute_ramp": {"pre": round(mean_ramp_pre, 4), "post": round(mean_ramp_post, 4)},
        "autocorrelation_lag48": {"pre": round(acf_lag48_pre, 4), "post": round(acf_lag48_post, 4)},
        "autocorrelation_lag336": {"pre": round(acf_lag336_pre, 4), "post": round(acf_lag336_post, 4)},
        "distribution_tests": {
            "wasserstein_1d": round(w1_dist, 4),
            "ks_statistic": round(float(ks_res.statistic), 4),
            "ks_p_value": float(ks_res.pvalue),
            "mann_whitney_p_value": float(mwu_res.pvalue),
            "statistically_significant_shift": bool(ks_res.pvalue < 0.01)
        }
    }
