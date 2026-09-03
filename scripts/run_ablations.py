"""
CLI script to run all 8 preregistered ablation experiments (A1 to A8).
"""
import argparse
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from shift_ami.config import load_config
from shift_ami.experiments.ablations import run_ablation_studies
from shift_ami.reporting.tables import save_multiformat_table
from shift_ami.utils.logging import setup_logger

logger = setup_logger("shift_ami.scripts.run_ablations")


def main():
    parser = argparse.ArgumentParser(description="Run shift-ami ablation experiments.")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config.paths.results_dir
    reports_dir = config.paths.reports_dir

    traj_path = results_dir / "sequential_test_trajectories.parquet"
    if not traj_path.exists():
        logger.error(f"Sequential test trajectory file not found at {traj_path}. Please run run_experiment.py first.")
        return

    df_traj = pd.read_parquet(traj_path)
    y_test = df_traj["y_true"].values
    y_pred = df_traj["y_point_pred"].values

    # Approximate calibration set from early test slice or dummy for standalone ablation check
    cal_true = y_test[:48*28]
    cal_pred = y_pred[:48*28]

    ablations = run_ablation_studies(
        cal_true=cal_true,
        cal_pred=cal_pred,
        y_test=y_test,
        y_pred_test=y_pred,
        default_alpha=config.conformal.alpha
    )

    table_dir = reports_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    for abl_name, df_abl in ablations.items():
        out_csv = results_dir / f"{abl_name}.csv"
        df_abl.to_csv(out_csv, index=False)
        save_multiformat_table(df_abl, table_dir / f"tab_{abl_name}", f"Ablation: {abl_name.replace('_', ' ').title()}")
        logger.info(f"Saved {abl_name} results to {out_csv}")

    logger.info("Ablation studies completed successfully.")


if __name__ == "__main__":
    main()
