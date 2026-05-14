from __future__ import annotations

from copy import deepcopy

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


def build_model_registry(config: dict) -> dict[str, object]:
    random_state = config["project"]["random_state"]
    include = set(config.get("models", {}).get("include", []))

    registry: dict[str, object] = {}

    if "logistic_regression" in include:
        registry["logistic_regression"] = LogisticRegression(max_iter=2000, random_state=random_state)
    if "decision_tree" in include:
        registry["decision_tree"] = DecisionTreeClassifier(max_depth=6, random_state=random_state)
    if "random_forest" in include:
        registry["random_forest"] = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
    if "gradient_boosting" in include:
        registry["gradient_boosting"] = GradientBoostingClassifier(random_state=random_state)
    if "adaboost" in include:
        registry["adaboost"] = AdaBoostClassifier(random_state=random_state)
    if "xgboost" in include and XGBClassifier is not None:
        registry["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if "lightgbm" in include and LGBMClassifier is not None:
        registry["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        )

    return registry


def tune_selected_models(models: dict[str, object], X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    if not config["project"].get("enable_tuning", False):
        return models, {}

    tuned_models = deepcopy(models)
    tuning_summary: dict[str, dict] = {}

    parameter_grids = {
        "random_forest": {
            "n_estimators": [200, 300],
            "max_depth": [8, 10, None],
            "min_samples_split": [2, 4],
        },
        "gradient_boosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    }

    for model_name, param_grid in parameter_grids.items():
        if model_name not in tuned_models:
            continue

        grid = GridSearchCV(
            estimator=tuned_models[model_name],
            param_grid=param_grid,
            cv=3,
            scoring=config["project"]["optimize_metric"],
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        tuned_models[model_name] = grid.best_estimator_
        tuning_summary[model_name] = {
            "best_score": grid.best_score_,
            "best_params": grid.best_params_,
        }

    return tuned_models, tuning_summary


def build_notebook_style_ensemble(models: dict[str, object], random_state: int) -> VotingClassifier:
    estimators = []
    for name in ("gradient_boosting", "logistic_regression", "adaboost"):
        if name in models:
            estimators.append((name, clone(models[name])))

    if len(estimators) < 2:
        fallback = [("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=random_state))]
        estimators.extend(fallback)

    return VotingClassifier(estimators=estimators, voting="soft")


def build_teacher_style_ensemble(models: dict[str, object], random_state: int) -> VotingClassifier:
    preferred_names = ("random_forest", "xgboost", "lightgbm", "gradient_boosting")
    estimators = [(name, clone(models[name])) for name in preferred_names if name in models]

    if len(estimators) < 2:
        for name in ("logistic_regression", "adaboost"):
            if name in models:
                estimators.append((name, clone(models[name])))

    if len(estimators) < 2:
        estimators.append(("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=random_state)))

    return VotingClassifier(estimators=estimators, voting="soft")


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    random_state = config["project"]["random_state"]
    models = build_model_registry(config)
    models, tuning_summary = tune_selected_models(models, X_train, y_train, config)

    models["notebook_voting_ensemble"] = build_notebook_style_ensemble(models, random_state)
    models["teacher_voting_ensemble"] = build_teacher_style_ensemble(models, random_state)

    fitted_models: dict[str, object] = {}
    for model_name, estimator in models.items():
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(X_train, y_train)
        fitted_models[model_name] = fitted_estimator

    return fitted_models, tuning_summary
