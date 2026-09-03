"""
Conditional reliability and subgroup breakdown evaluation across operational dimensions.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from shift_ami.evaluation.probabilistic_metrics import calculate_probabilistic_metrics


def assign_season(month: int) -> str:
    """Map month (1-12) to meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def evaluate_conditional_reliability(
    df_eval: pd.DataFrame,
    alpha: float = 0.10,
    true_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    timestamp_col: str = "timestamp",
    shift_flag_col: Optional[str] = "shift_flag"
) -> Dict[str, pd.DataFrame]:
    """
    Perform deep conditional reliability analysis across operational dimensions:
    1. By Forecast Horizon (h in 1..48)
    2. By Hour of Day (0..23)
    3. Peak (17:00-21:00) vs Off-Peak
    4. Weekday vs Weekend
    5. Season (Winter, Spring, Summer, Autumn)
    6. Load Magnitude Tertiles (Low, Medium, High)
    7. Ramp Magnitude Tertiles (Low, Medium, High)
    8. Shift Detected vs Normal (if shift_flag available)

    Returns
    -------
    Dict[str, pd.DataFrame] containing subgroup evaluation tables.
    """
    df = df_eval.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df["hour"] = ts.dt.hour
    df["hh_idx"] = ts.dt.hour * 2 + ts.dt.minute // 30
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).map({True: "Weekend", False: "Weekday"})
    df["month"] = ts.dt.month
    df["season"] = df["month"].apply(assign_season)
    df["is_peak"] = ((df["hour"] >= 17) & (df["hour"] <= 21)).map({True: "Peak (17-21h)", False: "Off-Peak"})

    # Compute load tertiles
    y_vals = df[true_col].values
    q33, q66 = np.quantile(y_vals, [0.333, 0.667])
    df["load_regime"] = pd.cut(
        df[true_col],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low Demand", "Medium Demand", "High Demand"]
    )

    # Compute ramp tertiles (absolute change from prior step)
    ramps = np.abs(np.diff(y_vals, prepend=y_vals[0]))
    rq33, rq66 = np.quantile(ramps, [0.333, 0.667])
    df["ramp_regime"] = pd.cut(
        ramps,
        bins=[-np.inf, rq33, rq66, np.inf],
        labels=["Low Ramp", "Medium Ramp", "High Ramp"]
    )

    dimensions = {
        "by_hour": "hour",
        "by_peak_status": "is_peak",
        "by_day_type": "is_weekend",
        "by_season": "season",
        "by_load_regime": "load_regime",
        "by_ramp_regime": "ramp_regime"
    }

    if "horizon" in df.columns:
        dimensions["by_horizon"] = "horizon"

    if shift_flag_col and shift_flag_col in df.columns:
        df["shift_status"] = df[shift_flag_col].map({True: "Shift Detected", False: "Normal / Low Shift"})
        dimensions["by_shift_status"] = "shift_status"

    subgroup_tables = {}

    for name, group_col in dimensions.items():
        records = []
        for group_val, grp in df.groupby(group_col, observed=True):
            metrics = calculate_probabilistic_metrics(
                grp[true_col].values,
                grp[lower_col].values,
                grp[upper_col].values,
                nominal_alpha=alpha
            )
            records.append({
                "subgroup": str(group_val),
                "n_obs": len(grp),
                "empirical_coverage": metrics["empirical_coverage"],
                "target_coverage": metrics["target_coverage"],
                "ace": metrics["ace"],
                "mpiw": metrics["mpiw"],
                "interval_score": metrics["interval_score"]
            })
        subgroup_tables[name] = pd.DataFrame(records)

    return subgroup_tables
