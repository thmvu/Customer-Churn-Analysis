from __future__ import annotations

from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "project": {
        "random_state": 42,
        "test_size": 0.2,
        "target_column": "Churn",
        "optimize_metric": "f1",
        "enable_smote": True,
        "enable_tuning": False,
    },
    "paths": {
        "raw_data": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "processed_dir": "data/processed",
        "trained_models_dir": "models/trained_models",
        "metrics_dir": "models/metrics",
        "figures_dir": "reports/figures",
    },
    "artifacts": {
        "cleaned_data": "cleaned_data.csv",
        "train_data": "train.csv",
        "test_data": "test.csv",
        "metrics_file": "model_results.csv",
        "report_file": "classification_report.txt",
        "best_model_file": "best_model.pkl",
    },
    "models": {
        "include": [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "adaboost",
            "xgboost",
            "lightgbm",
        ]
    },
}


def _deep_update(base: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_project_config(path: str | Path = "configs/config.yaml") -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)

    if not config_path.exists():
        return config

    try:
        import yaml
    except ImportError:
        return config

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return _deep_update(config, loaded)
