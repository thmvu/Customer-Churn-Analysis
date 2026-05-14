from src.models.train import build_model_registry, build_teacher_style_ensemble
from src.utils.config import DEFAULT_CONFIG


def test_model_registry_contains_core_models():
    models = build_model_registry(DEFAULT_CONFIG)
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models


def test_teacher_ensemble_has_multiple_estimators():
    models = build_model_registry(DEFAULT_CONFIG)
    ensemble = build_teacher_style_ensemble(models, DEFAULT_CONFIG["project"]["random_state"])
    assert len(ensemble.estimators) >= 2
