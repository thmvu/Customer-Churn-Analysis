from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(model, X_test, y_test) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, probabilities)
    return float("nan")


def evaluate_models(fitted_models: dict[str, object], X_test, y_test):
    rows = []
    report_map: dict[str, str] = {}

    for model_name, model in fitted_models.items():
        predictions = model.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": _safe_roc_auc(model, X_test, y_test),
            }
        )
        report_map[model_name] = classification_report(y_test, predictions, zero_division=0)

    results_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return results_df, report_map


def export_reports(
    results_df: pd.DataFrame,
    report_map: dict[str, str],
    metrics_path: Path,
    report_path: Path,
    *,
    tuning_summary: dict | None = None,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(metrics_path, index=False)

    lines = []
    if tuning_summary:
        lines.append("Tuning summary")
        for model_name, summary in tuning_summary.items():
            lines.append(f"- {model_name}: best_score={summary['best_score']:.4f}, best_params={summary['best_params']}")
        lines.append("")

    for model_name, report in report_map.items():
        lines.append(f"=== {model_name} ===")
        lines.append(report)
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
