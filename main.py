from pathlib import Path

import joblib
import pandas as pd

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_pipeline
from src.models.evaluate import evaluate_models, export_reports
from src.models.train import train_all_models
from src.utils.config import load_project_config
from src.utils.helpers import ensure_directories
from src.utils.logger import get_logger
from src.visualization.plots import (
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_contract_vs_churn,
    plot_feature_importance,
    plot_monthly_charges_by_churn,
    plot_roc_curve,
)


def main() -> None:
    config = load_project_config()
    logger = get_logger("customer_churn")

    paths = config["paths"]
    artifacts = config["artifacts"]
    ensure_directories(
        [
            paths["processed_dir"],
            paths["trained_models_dir"],
            paths["metrics_dir"],
            paths["figures_dir"],
        ]
    )

    logger.info("Loading raw dataset...")
    raw_df = load_dataset(paths["raw_data"])

    logger.info("Running preprocessing pipeline...")
    bundle = preprocess_pipeline(raw_df, config)

    processed_dir = Path(paths["processed_dir"])
    bundle.cleaned_df.to_csv(processed_dir / artifacts["cleaned_data"], index=False)
    pd.concat([bundle.X_train, bundle.y_train.rename(bundle.target_column)], axis=1).to_csv(
        processed_dir / artifacts["train_data"],
        index=False,
    )
    pd.concat([bundle.X_test, bundle.y_test.rename(bundle.target_column)], axis=1).to_csv(
        processed_dir / artifacts["test_data"],
        index=False,
    )

    logger.info("Saving EDA figures...")
    figures_dir = Path(paths["figures_dir"])
    plot_churn_distribution(bundle.cleaned_df, figures_dir / "churn_distribution.png", bundle.target_column)
    plot_contract_vs_churn(bundle.cleaned_df, figures_dir / "contract_vs_churn.png", bundle.target_column)
    plot_monthly_charges_by_churn(
        bundle.cleaned_df,
        figures_dir / "monthly_charges_by_churn.png",
        bundle.target_column,
    )

    logger.info("Training models...")
    trained_models, tuning_summary = train_all_models(bundle.X_train, bundle.y_train, config)

    logger.info("Evaluating models...")
    results_df, report_map = evaluate_models(trained_models, bundle.X_test, bundle.y_test)
    export_reports(
        results_df,
        report_map,
        Path(paths["metrics_dir"]) / artifacts["metrics_file"],
        Path(paths["metrics_dir"]) / artifacts["report_file"],
        tuning_summary=tuning_summary,
    )

    best_row = results_df.sort_values(config["project"]["optimize_metric"], ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]
    best_model_path = Path(paths["trained_models_dir"]) / artifacts["best_model_file"]
    joblib.dump(best_model, best_model_path)

    logger.info("Creating evaluation figures for best model: %s", best_model_name)
    y_pred = best_model.predict(bundle.X_test)
    plot_confusion_matrix(
        bundle.y_test,
        y_pred,
        figures_dir / f"{best_model_name}_confusion_matrix.png",
        title=f"Confusion Matrix - {best_model_name}",
    )
    plot_roc_curve(
        best_model,
        bundle.X_test,
        bundle.y_test,
        figures_dir / f"{best_model_name}_roc_curve.png",
        title=f"ROC Curve - {best_model_name}",
    )
    plot_feature_importance(
        best_model,
        bundle.feature_names,
        figures_dir / f"{best_model_name}_feature_importance.png",
        title=f"Feature Importance - {best_model_name}",
    )

    logger.info("Done. Best model: %s", best_model_name)
    logger.info("Results saved to %s", Path(paths["metrics_dir"]) / artifacts["metrics_file"])
    logger.info("Model saved to %s", best_model_path)


if __name__ == "__main__":
    main()
