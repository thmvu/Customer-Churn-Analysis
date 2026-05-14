from src.data.preprocess import preprocess_pipeline
from src.utils.config import DEFAULT_CONFIG


def clean_and_split(df):
    bundle = preprocess_pipeline(df, DEFAULT_CONFIG)
    return bundle.X_train, bundle.X_test, bundle.y_train, bundle.y_test
