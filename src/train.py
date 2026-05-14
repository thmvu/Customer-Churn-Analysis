from src.models.train import build_teacher_style_ensemble, build_model_registry
from src.utils.config import DEFAULT_CONFIG


def train_optimized_ensemble(X_train, y_train):
    models = build_model_registry(DEFAULT_CONFIG)
    ensemble = build_teacher_style_ensemble(models, DEFAULT_CONFIG["project"]["random_state"])
    ensemble.fit(X_train, y_train)
    return ensemble
