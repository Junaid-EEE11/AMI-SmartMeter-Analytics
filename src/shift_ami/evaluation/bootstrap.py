"""
Day-Level Paired Block Bootstrap for Rigorous Statistical Comparison.

CRITICAL RESEARCH RULE (Section 17 of GEMINI.md):
Do not treat correlated half-hourly time steps as independent replicates.
Perform paired block resampling at the full-day block level (48 half-hours).
Estimate mean difference, 95% bootstrap confidence interval, and p-value.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from shift_ami.evaluation.probabilistic_metrics import calculate_probabilistic_metrics


@dataclass
class BootstrapResult:
    metric_name: str
    method_a_name: str
    method_b_name: str
    mean_a: float
    mean_b: float
    mean_diff: float  # Mean(A) - Mean(B)
    ci_lower: float   # 2.5% quantile of bootstrap diffs
    ci_upper: float   # 97.5% quantile of bootstrap diffs
    p_value: float    # Two-sided bootstrap p-value against null of no difference
    is_significant_05: bool
    n_resamples: int
    n_days: int


def paired_block_bootstrap_comparison(
    df_results: pd.DataFrame,
    method_a_prefix: str = "sa_acp",
    method_b_prefix: str = "aci",
    target_alpha: float = 0.10,
    n_resamples: int = 2000,
    seed: int = 42,
    timestamp_col: str = "timestamp",
    true_col: str = "y_true"
) -> Dict[str, BootstrapResult]:
    """
    Perform paired day-level block bootstrap comparison between Method A and Method B.

    Compares across primary endpoints:
    1. Absolute Coverage Error (ACE) (Primary Endpoint)
    2. Winkler Interval Score (Secondary Endpoint)
    3. Mean Prediction Interval Width (MPIW)
    4. Empirical Coverage Probability

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing ground truth and paired predictions for methods A and B.
        Columns required: [timestamp_col, true_col,
                           f"{method_a_prefix}_lower", f"{method_a_prefix}_upper",
                           f"{method_b_prefix}_lower", f"{method_b_prefix}_upper"]
    """
    df = df_results.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Assign day block identifier
    df["date_block"] = df[timestamp_col].dt.date
    unique_days = np.array(sorted(df["date_block"].unique()))
    n_days = len(unique_days)

    if n_days < 5:
        raise ValueError(f"Insufficient daily blocks ({n_days}) to perform meaningful block bootstrapping.")

    # Group dataframe into a list of daily slices for fast indexed sampling
    day_groups = [group for _, group in df.groupby("date_block", sort=True)]

    rng = np.random.default_rng(seed)

    # Pre-allocate bootstrap arrays for metrics
    metrics_to_compare = ["ace", "interval_score", "mpiw", "empirical_coverage"]
    diff_samples: Dict[str, List[float]] = {m: [] for m in metrics_to_compare}
    a_samples: Dict[str, List[float]] = {m: [] for m in metrics_to_compare}
    b_samples: Dict[str, List[float]] = {m: [] for m in metrics_to_compare}

    for _ in range(n_resamples):
        # Sample daily blocks with replacement
        sampled_indices = rng.choice(n_days, size=n_days, replace=True)
        # Concatenate sampled day blocks
        sampled_df = pd.concat([day_groups[idx] for idx in sampled_indices], ignore_index=True)

        y_t = sampled_df[true_col].values
        
        # Method A metrics
        mA = calculate_probabilistic_metrics(
            y_t,
            sampled_df[f"{method_a_prefix}_lower"].values,
            sampled_df[f"{method_a_prefix}_upper"].values,
            nominal_alpha=target_alpha
        )

        # Method B metrics
        mB = calculate_probabilistic_metrics(
            y_t,
            sampled_df[f"{method_b_prefix}_lower"].values,
            sampled_df[f"{method_b_prefix}_upper"].values,
            nominal_alpha=target_alpha
        )

        for m in metrics_to_compare:
            a_val = mA[m]
            b_val = mB[m]
            a_samples[m].append(a_val)
            b_samples[m].append(b_val)
            diff_samples[m].append(a_val - b_val)

    # Assemble summary results
    results = {}
    for m in metrics_to_compare:
        diffs = np.array(diff_samples[m])
        mean_diff = float(np.mean(diffs))
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        
        # Two-sided empirical bootstrap p-value against H0: diff = 0
        # Fraction of resamples where sign is opposite to observed mean diff
        p_val = 2.0 * min(
            float(np.mean(diffs <= 0.0)),
            float(np.mean(diffs >= 0.0))
        )
        p_val = min(1.0, max(0.0, p_val))

        # Significant if 95% CI excludes 0
        is_sig = bool((ci_lo > 0.0) or (ci_hi < 0.0))

        results[m] = BootstrapResult(
            metric_name=m,
            method_a_name=method_a_prefix,
            method_b_name=method_b_prefix,
            mean_a=round(float(np.mean(a_samples[m])), 4),
            mean_b=round(float(np.mean(b_samples[m])), 4),
            mean_diff=round(mean_diff, 4),
            ci_lower=round(float(ci_lo), 4),
            ci_upper=round(float(ci_hi), 4),
            p_value=round(p_val, 4),
            is_significant_05=is_sig,
            n_resamples=n_resamples,
            n_days=n_days
        )

    return results
