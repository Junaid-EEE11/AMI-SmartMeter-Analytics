"""
Master Experiment Pipeline Runner for shift-ami.

Coordinates end-to-end data ingestion, chronological partitioning, feature engineering,
baseline model training, conformal calibration, sequential simulation, bootstrap inference,
subgroup diagnostics, ablations, and results serialization.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from shift_ami.config import AppConfig, load_config
from shift_ami.data.splits import create_chronological_splits
from shift_ami.data.aggregate import aggregate_cohorts
from shift_ami.data.synthetic import generate_synthetic_ami_dataset
from shift_ami.features.build import FeaturePipeline
from shift_ami.models.seasonal_naive import SeasonalNaiveModel
from shift_ami.models.ridge import RidgeModel
from shift_ami.models.gradient_boosting import HistGBRModel
from shift_ami.models.quantile import QuantileGBRModel
from shift_ami.evaluation.point_metrics import calculate_point_metrics
from shift_ami.evaluation.probabilistic_metrics import calculate_probabilistic_metrics
from shift_ami.evaluation.subgroup_metrics import evaluate_conditional_reliability
from shift_ami.evaluation.bootstrap import paired_block_bootstrap_comparison
from shift_ami.evaluation.shift_analysis import characterize_distribution_shift
from shift_ami.experiments.sequential import run_sequential_evaluation
from shift_ami.experiments.ablations import run_ablation_studies
from shift_ami.utils.logging import setup_logger
from shift_ami.utils.reproducibility import set_seed, get_provenance_metadata

logger = setup_logger("shift_ami.experiments.runner")


def run_experiment_pipeline(config: AppConfig) -> Dict[str, Any]:
    """
    Execute full research pipeline pursuant to GEMINI.md specifications.
    """
    logger.info("Initializing shift-ami research experiment pipeline...")
    set_seed(config.random_seed)

    # 1. Ensure output directories exist
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load or generate cohort data
    cohort_file = config.paths.processed_dir / "cohort_total_halfhourly.parquet"
    if not cohort_file.exists():
        # Check if raw data is present or generate synthetic fixture
        if not config.paths.raw_file.exists():
            logger.warning(
                f"Raw dataset not found at {config.paths.raw_file}. "
                f"Generating synthetic AMI test fixture labeled: SYNTHETIC — FOR PIPELINE VALIDATION ONLY"
            )
            syn_df = generate_synthetic_ami_dataset(
                output_path=config.paths.processed_dir / "synthetic_ami.parquet",
                seed=config.random_seed
            )
            # Process and aggregate synthetic data
            import polars as pl
            pl_df = pl.from_pandas(syn_df)
            cohort_dict = aggregate_cohorts(pl_df, config.inclusion_criteria, output_dir=config.paths.processed_dir)
        else:
            import polars as pl
            from shift_ami.data.ingest import load_raw_ami_data
            from shift_ami.data.preprocess import clean_and_standardize_ami
            raw_pl = load_raw_ami_data(config.paths.raw_file, config.schema_mapping)
            cleaned_pl = clean_and_standardize_ami(raw_pl, config.inclusion_criteria)
            cohort_dict = aggregate_cohorts(cleaned_pl, config.inclusion_criteria, output_dir=config.paths.processed_dir)
    else:
        cohort_dict = {
            "total": pd.read_parquet(config.paths.processed_dir / "cohort_total_halfhourly.parquet"),
            "dtou": pd.read_parquet(config.paths.processed_dir / "cohort_dtou_halfhourly.parquet"),
            "standard": pd.read_parquet(
                config.paths.processed_dir / "cohort_standard_halfhourly.parquet"
                if (config.paths.processed_dir / "cohort_standard_halfhourly.parquet").exists()
                else config.paths.processed_dir / "cohort_std_halfhourly.parquet"
            ),
        }

    # Primary analysis on Total cohort aggregate (and compare dToU vs Std)
    primary_df = cohort_dict.get("total", list(cohort_dict.values())[0])
    logger.info(f"Loaded primary cohort load series with {len(primary_df):,} records.")

    # 3. Create strictly chronological splits
    splits = create_chronological_splits(primary_df, config.splits)

    # 4. Feature engineering with training-only scaling
    feature_pipe = FeaturePipeline(
        lags=config.features.get("lags", [48, 96, 144, 336]),
        include_holidays=config.features.get("calendar", {}).get("include_holidays", True),
        scale_features=True
    )

    X_train_raw, y_train, ts_train = feature_pipe.build_raw_features_for_series(splits.train)
    X_cal_raw, y_cal, ts_cal = feature_pipe.build_raw_features_for_series(splits.calibration)
    X_val_raw, y_val, ts_val = feature_pipe.build_raw_features_for_series(splits.validation)
    X_test_raw, y_test, ts_test = feature_pipe.build_raw_features_for_series(splits.test)

    # Fit scaler strictly on training split
    X_train = feature_pipe.fit_transform(X_train_raw)
    X_cal = feature_pipe.transform(X_cal_raw)
    X_val = feature_pipe.transform(X_val_raw)
    X_test = feature_pipe.transform(X_test_raw)

    # 5. Train Point Forecast Baselines (B0, B1, B2)
    logger.info("Training point forecast baseline models...")
    b0_day = SeasonalNaiveModel(season_lag=48).fit(X_train_raw, y_train)
    b0_week = SeasonalNaiveModel(season_lag=336).fit(X_train_raw, y_train)
    b1_ridge = RidgeModel().fit(X_train, y_train)
    b2_histgbr = HistGBRModel(random_state=config.random_seed).fit(X_train, y_train)

    # Evaluate point predictions on test set
    pred_b0_day = b0_day.predict(X_test_raw)
    pred_b0_week = b0_week.predict(X_test_raw)
    pred_b1_ridge = b1_ridge.predict(X_test)
    pred_b2_histgbr = b2_histgbr.predict(X_test)

    point_preds_test = {
        "seasonal_naive_day": pred_b0_day,
        "seasonal_naive_week": pred_b0_week,
        "ridge": pred_b1_ridge,
        "hist_gradient_boosting": pred_b2_histgbr
    }

    point_metrics_summary = {}
    for m_name, preds in point_preds_test.items():
        point_metrics_summary[m_name] = calculate_point_metrics(y_test.values, preds)

    # 6. Train Quantile Baseline Models (B3 / P0)
    logger.info("Training multi-quantile gradient boosting regressors...")
    nominal_alpha = config.conformal.alpha
    q_lo_level = round(nominal_alpha / 2.0, 3)
    q_hi_level = round(1.0 - nominal_alpha / 2.0, 3)
    quantiles_to_fit = [q_lo_level, 0.50, q_hi_level]

    quantile_model = QuantileGBRModel(
        quantiles=quantiles_to_fit,
        random_state=config.random_seed
    ).fit(X_train, y_train)

    # Calibration set predictions for conformalization
    cal_point_pred = b2_histgbr.predict(X_cal)
    cal_q_lo = quantile_model.predict_quantile(X_cal, q_lo_level)
    cal_q_hi = quantile_model.predict_quantile(X_cal, q_hi_level)

    # Test quantile predictions
    test_q_lo = quantile_model.predict_quantile(X_test, q_lo_level)
    test_q_hi = quantile_model.predict_quantile(X_test, q_hi_level)
    quantile_preds_test = {
        q_lo_level: test_q_lo,
        0.50: quantile_model.predict_quantile(X_test, 0.50),
        q_hi_level: test_q_hi
    }

    # 7. Execute Sequential Operational Evaluation (P0 - P5)
    sa_params = {
        "detector": config.conformal.sa_acp_detector,
        "window_length": config.conformal.sa_acp_window_length,
        "reference_length": config.conformal.sa_acp_reference_length,
        "threshold": config.conformal.sa_acp_threshold,
        "gamma_slow": config.conformal.sa_acp_gamma_slow,
        "gamma_fast": config.conformal.sa_acp_gamma_fast,
        "shift_penalty_eta": config.conformal.sa_acp_shift_penalty_eta,
        "enable_detector": True
    }

    seq_results_df = run_sequential_evaluation(
        df_test_features=X_test,
        y_test_true=y_test,
        test_timestamps=ts_test,
        point_predictions=point_preds_test,
        quantile_predictions=quantile_preds_test,
        cal_true=y_cal.values,
        cal_pred_point=cal_point_pred,
        cal_q_lo=cal_q_lo,
        cal_q_hi=cal_q_hi,
        conformal_alpha=nominal_alpha,
        aci_gamma=config.conformal.aci_gamma,
        sa_acp_params=sa_params,
        rolling_window_steps=config.conformal.rolling_window_days * 48
    )

    # 8. Compute Probabilistic Metrics for all Methods
    methods_dict = {
        "P0_Uncalibrated_Quantile": ("uncalibrated_lower", "uncalibrated_upper"),
        "P1_Static_Split_Conformal": ("static_lower", "static_upper"),
        "P2_CQR": ("cqr_lower", "cqr_upper"),
        "P3_Rolling_Conformal": ("rolling_lower", "rolling_upper"),
        "P4_ACI": ("aci_lower", "aci_upper"),
        "P5_SA_ACP": ("sa_acp_lower", "sa_acp_upper"),
    }

    prob_metrics_summary = {}
    y_test_arr = seq_results_df["y_true"].values
    for m_label, (l_col, u_col) in methods_dict.items():
        prob_metrics_summary[m_label] = calculate_probabilistic_metrics(
            y_test_arr,
            seq_results_df[l_col].values,
            seq_results_df[u_col].values,
            nominal_alpha=nominal_alpha
        )

    # 9. Paired Day-Level Block Bootstrap (Primary: SA-ACP vs ACI)
    logger.info("Executing day-level paired block bootstrap for statistical significance...")
    n_resamples = config.bootstrap.get("n_resamples", 2000)
    bootstrap_results = paired_block_bootstrap_comparison(
        seq_results_df,
        method_a_prefix="sa_acp",
        method_b_prefix="aci",
        target_alpha=nominal_alpha,
        n_resamples=n_resamples,
        seed=config.random_seed
    )

    # Also compute Static vs SA-ACP bootstrap
    bootstrap_static_vs_sa = paired_block_bootstrap_comparison(
        seq_results_df,
        method_a_prefix="sa_acp",
        method_b_prefix="static",
        target_alpha=nominal_alpha,
        n_resamples=n_resamples,
        seed=config.random_seed
    )

    # 10. Conditional Subgroup Reliability Analysis
    logger.info("Evaluating conditional reliability across subgroups...")
    subgroups_sa = evaluate_conditional_reliability(
        seq_results_df,
        alpha=nominal_alpha,
        lower_col="sa_acp_lower",
        upper_col="sa_acp_upper"
    )

    # 11. Run All Preregistered Ablations (A1 - A8)
    ablations_dict = run_ablation_studies(
        cal_true=y_cal.values,
        cal_pred=cal_point_pred,
        y_test=y_test_arr,
        y_pred_test=pred_b2_histgbr,
        default_alpha=nominal_alpha
    )

    # 12. Model-Independent Distribution Shift Analysis
    shift_stats = characterize_distribution_shift(
        pre_series=splits.train["load_kwh"],
        post_series=splits.test["load_kwh"],
        pre_label="Train_Split",
        post_label="Test_Split"
    )

    # 13. Failure Analysis Diagnostics
    # Compute worst 20 days by MAE and worst 20 by coverage error
    seq_results_df["date"] = pd.to_datetime(seq_results_df["timestamp"]).dt.date
    daily_diag = seq_results_df.groupby("date").agg(
        day_mae=("y_true", lambda y: float(np.mean(np.abs(y - seq_results_df.loc[y.index, "y_point_pred"])))),
        day_sa_cov=("y_true", lambda y: float(np.mean((y >= seq_results_df.loc[y.index, "sa_acp_lower"]) & (y <= seq_results_df.loc[y.index, "sa_acp_upper"])))),
        day_static_cov=("y_true", lambda y: float(np.mean((y >= seq_results_df.loc[y.index, "static_lower"]) & (y <= seq_results_df.loc[y.index, "static_upper"])))),
        day_shift_detected_hours=("shift_flag", lambda s: int(np.sum(s))),
    ).reset_index()
    daily_diag["day_sa_ace"] = np.abs(daily_diag["day_sa_cov"] - (1.0 - nominal_alpha))

    worst_20_mae = daily_diag.sort_values("day_mae", ascending=False).head(20).to_dict(orient="records")
    worst_20_coverage = daily_diag.sort_values("day_sa_ace", ascending=False).head(20).to_dict(orient="records")

    # 14. Save Output Artifacts
    results_dir = config.paths.results_dir
    seq_results_df.to_parquet(results_dir / "sequential_test_trajectories.parquet", index=False)

    bootstrap_serializable = {
        k: {
            "metric": v.metric_name,
            "mean_a": v.mean_a,
            "mean_b": v.mean_b,
            "mean_diff": v.mean_diff,
            "ci_95": [v.ci_lower, v.ci_upper],
            "p_value": v.p_value,
            "significant_at_05": v.is_significant_05,
            "n_resamples": v.n_resamples,
            "n_days": v.n_days
        }
        for k, v in bootstrap_results.items()
    }

    full_results_payload = {
        "provenance": get_provenance_metadata(config.raw_dict),
        "point_metrics": point_metrics_summary,
        "probabilistic_metrics": prob_metrics_summary,
        "bootstrap_sa_acp_vs_aci": bootstrap_serializable,
        "distribution_shift_analysis": shift_stats,
        "failure_diagnostics": {
            "worst_20_days_by_mae": worst_20_mae,
            "worst_20_days_by_coverage_error": worst_20_coverage
        }
    }

    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_results_payload, f, indent=2, default=str)

    # Save ablation CSVs
    for abl_name, df_abl in ablations_dict.items():
        df_abl.to_csv(results_dir / f"{abl_name}.csv", index=False)

    # Save subgroup CSVs
    for sub_name, df_sub in subgroups_sa.items():
        df_sub.to_csv(results_dir / f"subgroup_{sub_name}.csv", index=False)

    logger.info(f"Master experiment pipeline completed! Results saved to: {results_dir}")
    return full_results_payload
