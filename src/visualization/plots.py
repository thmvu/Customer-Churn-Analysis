from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


sns.set_theme(style="whitegrid")


def _save_current_figure(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_churn_distribution(df: pd.DataFrame, output_path: str | Path, target_column: str = "Churn") -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=target_column, hue=target_column, palette="Set2")
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title("Customer Churn Distribution")
    _save_current_figure(output_path)


def plot_contract_vs_churn(df: pd.DataFrame, output_path: str | Path, target_column: str = "Churn") -> None:
    if "Contract" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="Contract", hue=target_column, palette="Set2")
    plt.title("Contract Type by Churn")
    plt.xticks(rotation=15)
    _save_current_figure(output_path)


def plot_monthly_charges_by_churn(df: pd.DataFrame, output_path: str | Path, target_column: str = "Churn") -> None:
    if "MonthlyCharges" not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    ax = sns.boxplot(data=df, x=target_column, y="MonthlyCharges", hue=target_column, palette="Set2")
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.title("Monthly Charges by Churn")
    _save_current_figure(output_path)


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    _save_current_figure(output_path)


def plot_roc_curve(model, X_test, y_test, output_path: str | Path, title: str) -> None:
    if not hasattr(model, "predict_proba"):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    _save_current_figure(output_path)


def plot_feature_importance(model, feature_names: list[str], output_path: str | Path, title: str) -> None:
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef)

    if values is None:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(12)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title(title)
    _save_current_figure(output_path)
