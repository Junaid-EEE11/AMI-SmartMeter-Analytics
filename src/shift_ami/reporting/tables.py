"""
Publication Table Formatting and Serialization Module (Markdown, CSV, LaTeX).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.reporting.tables")


def save_multiformat_table(df: pd.DataFrame, base_path: Path, caption: str = "") -> None:
    """Save DataFrame as CSV, formatted Markdown, and LaTeX."""
    # CSV
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    # Markdown
    try:
        table_str = df.to_markdown(index=False)
    except Exception:
        # Fallback manual markdown table generator
        headers = list(df.columns)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]
        table_str = "\n".join([header_line, sep_line] + row_lines)

    md_content = f"### {caption}\n\n" + table_str + "\n"
    with open(base_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    # LaTeX
    try:
        latex_content = df.to_latex(index=False, caption=caption, label=f"tab:{base_path.stem}")
    except Exception:
        # Fallback basic LaTeX table generator
        headers = " & ".join(list(df.columns)) + " \\\\"
        rows = "\n".join([" & ".join(str(val) for val in row) + " \\\\" for row in df.values])
        latex_content = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:{base_path.stem}}}\n\\begin{{tabular}}{{{'l' * len(df.columns)}}}\n\\hline\n{headers}\n\\hline\n{rows}\n\\hline\n\\end{{tabular}}\n\\end{{table}}\n"
    with open(base_path.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write(latex_content)


def generate_all_tables(
    results_dir: Path,
    reports_dir: Path
) -> None:
    """
    Generate all 10 manuscript tables and save to reports/tables/.
    """
    table_dir = reports_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating formatted tables to: {table_dir}")

    # Load metrics json
    metrics_path = results_dir / "metrics_summary.json"
    if not metrics_path.exists():
        logger.warning(f"Metrics summary file not found at {metrics_path}.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------------
    # Table 2: Point Forecast Benchmarks
    # -------------------------------------------------------------
    pt_data = data.get("point_metrics", {})
    if pt_data:
        df_pt = pd.DataFrame([
            {"Model": m.replace("_", " ").title(), "MAE (kWh)": v["mae"], "RMSE (kWh)": v["rmse"], "NMAE": v["nmae"], "sMAPE (%)": v["smape"]}
            for m, v in pt_data.items()
        ])
        save_multiformat_table(df_pt, table_dir / "tab02_point_forecast_benchmarks", "Point Forecasting Baseline Accuracy")

    # -------------------------------------------------------------
    # Table 3: Probabilistic Forecasting Main Results (90% Nominal)
    # -------------------------------------------------------------
    prob_data = data.get("probabilistic_metrics", {})
    if prob_data:
        df_prob = pd.DataFrame([
            {
                "Method": m.replace("_", " "),
                "Empirical Coverage (%)": round(v["empirical_coverage"] * 100, 2),
                "Target Coverage (%)": round(v["target_coverage"] * 100, 2),
                "ACE (%)": round(v["ace"] * 100, 2),
                "MPIW (kWh)": v["mpiw"],
                "Winkler Interval Score": v["interval_score"]
            }
            for m, v in prob_data.items()
        ])
        save_multiformat_table(df_prob, table_dir / "tab03_probabilistic_main_results", "Probabilistic Load Forecasting Calibration and Sharpness Under Distribution Shift")

    # -------------------------------------------------------------
    # Table 4: Paired Day-Level Block Bootstrap Hypothesis Tests
    # -------------------------------------------------------------
    boot_data = data.get("bootstrap_sa_acp_vs_aci", {})
    if boot_data:
        df_boot = pd.DataFrame([
            {
                "Endpoint Metric": k.replace("_", " ").upper(),
                "Mean SA-ACP": v["mean_a"],
                "Mean ACI": v["mean_b"],
                "Mean Difference (A - B)": v["mean_diff"],
                "95% Bootstrap CI": f"[{v['ci_95'][0]}, {v['ci_95'][1]}]",
                "p-value": v["p_value"],
                "Significant (p < 0.05)": "Yes" if v["significant_at_05"] else "No"
            }
            for k, v in boot_data.items()
        ])
        save_multiformat_table(df_boot, table_dir / "tab04_bootstrap_hypothesis_tests", "Paired Day-Level Block Bootstrap Test (SA-ACP vs. ACI)")

    # -------------------------------------------------------------
    # Table 8: Distribution Shift Characterization
    # -------------------------------------------------------------
    shift_data = data.get("distribution_shift_analysis", {})
    if shift_data:
        df_shift = pd.DataFrame([
            {"Feature": "Mean Load (kWh)", "Pre-Shift": shift_data["load_mean"]["pre"], "Post-Shift": shift_data["load_mean"]["post"], "Change (%)": shift_data["load_mean"]["pct_change"]},
            {"Feature": "Load Std (kWh)", "Pre-Shift": shift_data["load_std"]["pre"], "Post-Shift": shift_data["load_std"]["post"], "Change (%)": shift_data["load_std"]["pct_change"]},
            {"Feature": "95th Percentile Peak (kWh)", "Pre-Shift": shift_data["peak_95th"]["pre"], "Post-Shift": shift_data["peak_95th"]["post"], "Change (%)": "-"},
            {"Feature": "Mean Ramp Magnitude (kWh)", "Pre-Shift": shift_data["mean_absolute_ramp"]["pre"], "Post-Shift": shift_data["mean_absolute_ramp"]["post"], "Change (%)": "-"},
            {"Feature": "1D Wasserstein Distance", "Pre-Shift": "-", "Post-Shift": shift_data["distribution_tests"]["wasserstein_1d"], "Change (%)": "-"},
            {"Feature": "KS Test Statistic (p-val)", "Pre-Shift": "-", "Post-Shift": f"{shift_data['distribution_tests']['ks_statistic']} (p={shift_data['distribution_tests']['ks_p_value']:.2e})", "Change (%)": "-"},
        ])
        save_multiformat_table(df_shift, table_dir / "tab08_distribution_shift_statistics", "Pre- vs Post-Shift Load Distribution Statistics and Hypothesis Tests")

    # Load and format ablation tables if CSVs exist in results_dir
    for abl_name in ["ablation_core_methods", "ablation_gamma_rates", "ablation_window_lengths", "ablation_coverage_levels"]:
        csv_file = results_dir / f"{abl_name}.csv"
        if csv_file.exists():
            df_abl = pd.read_csv(csv_file)
            save_multiformat_table(df_abl, table_dir / f"tab_{abl_name}", f"Ablation Study: {abl_name.replace('_', ' ').title()}")

    # Load and format subgroup tables
    for sub_name in ["by_hour", "by_peak_status", "by_day_type", "by_season", "by_load_regime", "by_ramp_regime"]:
        sub_file = results_dir / f"subgroup_{sub_name}.csv"
        if sub_file.exists():
            df_sub = pd.read_csv(sub_file)
            save_multiformat_table(df_sub, table_dir / f"tab_subgroup_{sub_name}", f"Conditional Reliability Breakdown: {sub_name.replace('_', ' ').title()}")

    logger.info("All formatted manuscript tables generated successfully.")
