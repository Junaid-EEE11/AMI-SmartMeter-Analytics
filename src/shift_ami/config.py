"""Configuration loader and schema validation for shift-ami."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from shift_ami.utils.paths import resolve_path


@dataclass
class PathConfig:
    raw_dir: Path
    raw_file: Path
    household_info_file: Path
    interim_dir: Path
    processed_dir: Path
    results_dir: Path
    reports_dir: Path


@dataclass
class InclusionConfig:
    min_history_days: int = 90
    max_missing_ratio: float = 0.10
    valid_energy_range: List[float] = field(default_factory=lambda: [0.0, 50.0])
    dtou_identifier: str = "ToU"
    standard_identifier: str = "Std"


@dataclass
class SplitConfig:
    train_start: str
    train_end: str
    cal_start: str
    cal_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass
class ConformalConfig:
    alpha: float = 0.10
    rolling_window_days: int = 28
    aci_gamma: float = 0.01
    aci_alpha_min: float = 0.01
    aci_alpha_max: float = 0.50
    sa_acp_detector: str = "wasserstein_1d"
    sa_acp_window_length: int = 336
    sa_acp_reference_length: int = 1344
    sa_acp_threshold: float = 1.25
    sa_acp_gamma_slow: float = 0.005
    sa_acp_gamma_fast: float = 0.035
    sa_acp_shift_penalty_eta: float = 0.02
    sa_acp_alpha_min: float = 0.01
    sa_acp_alpha_max: float = 0.50


@dataclass
class AppConfig:
    random_seed: int
    paths: PathConfig
    schema_mapping: Dict[str, Any]
    inclusion_criteria: InclusionConfig
    aggregation: Dict[str, Any]
    splits: SplitConfig
    forecast_horizon: int
    features: Dict[str, Any]
    models: Dict[str, List[str]]
    quantiles: Dict[str, Any]
    conformal: ConformalConfig
    bootstrap: Dict[str, Any]
    raw_dict: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load and parse YAML configuration into strongly typed AppConfig."""
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_dict = cfg.get("paths", {})
    paths = PathConfig(
        raw_dir=resolve_path(paths_dict.get("raw_dir", "data/raw")),
        raw_file=resolve_path(paths_dict.get("raw_file", "data/raw/CC_LCL-FullData.csv")),
        household_info_file=resolve_path(paths_dict.get("household_info_file", "data/raw/informations_households.csv")),
        interim_dir=resolve_path(paths_dict.get("interim_dir", "data/interim")),
        processed_dir=resolve_path(paths_dict.get("processed_dir", "data/processed")),
        results_dir=resolve_path(paths_dict.get("results_dir", "results")),
        reports_dir=resolve_path(paths_dict.get("reports_dir", "reports")),
    )

    inc_dict = cfg.get("inclusion_criteria", {})
    inclusion = InclusionConfig(
        min_history_days=inc_dict.get("min_history_days", 90),
        max_missing_ratio=inc_dict.get("max_missing_ratio", 0.10),
        valid_energy_range=inc_dict.get("valid_energy_range", [0.0, 50.0]),
        dtou_identifier=inc_dict.get("dtou_identifier", "ToU"),
        standard_identifier=inc_dict.get("standard_identifier", "Std"),
    )

    s_dict = cfg.get("splits", {})
    splits = SplitConfig(
        train_start=s_dict.get("train_start", "2012-01-01"),
        train_end=s_dict.get("train_end", "2012-10-31"),
        cal_start=s_dict.get("cal_start", "2012-11-01"),
        cal_end=s_dict.get("cal_end", "2012-12-31"),
        val_start=s_dict.get("val_start", "2013-01-01"),
        val_end=s_dict.get("val_end", "2013-02-28"),
        test_start=s_dict.get("test_start", "2013-03-01"),
        test_end=s_dict.get("test_end", "2013-12-31"),
    )

    c_dict = cfg.get("conformal", {})
    aci_dict = c_dict.get("aci", {})
    sa_dict = c_dict.get("sa_acp", {})
    conformal = ConformalConfig(
        alpha=c_dict.get("alpha", 0.10),
        rolling_window_days=c_dict.get("rolling_window_days", 28),
        aci_gamma=aci_dict.get("gamma", 0.01),
        aci_alpha_min=aci_dict.get("alpha_min", 0.01),
        aci_alpha_max=aci_dict.get("alpha_max", 0.50),
        sa_acp_detector=sa_dict.get("detector", "wasserstein_1d"),
        sa_acp_window_length=sa_dict.get("window_length", 336),
        sa_acp_reference_length=sa_dict.get("reference_length", 1344),
        sa_acp_threshold=sa_dict.get("threshold", 1.25),
        sa_acp_gamma_slow=sa_dict.get("gamma_slow", 0.005),
        sa_acp_gamma_fast=sa_dict.get("gamma_fast", 0.035),
        sa_acp_shift_penalty_eta=sa_dict.get("shift_penalty_eta", 0.02),
        sa_acp_alpha_min=sa_dict.get("alpha_min", 0.01),
        sa_acp_alpha_max=sa_dict.get("alpha_max", 0.50),
    )

    return AppConfig(
        random_seed=cfg.get("random_seed", 42),
        paths=paths,
        schema_mapping=cfg.get("schema_mapping", {}),
        inclusion_criteria=inclusion,
        aggregation=cfg.get("aggregation", {}),
        splits=splits,
        forecast_horizon=cfg.get("forecast_horizon", 48),
        features=cfg.get("features", {}),
        models=cfg.get("models", {}),
        quantiles=cfg.get("quantiles", {}),
        conformal=conformal,
        bootstrap=cfg.get("bootstrap", {}),
        raw_dict=cfg,
    )
