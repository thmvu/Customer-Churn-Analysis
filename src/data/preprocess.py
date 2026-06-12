from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocessBundle:
    cleaned_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    encoders: dict[str, LabelEncoder]
    scaler: StandardScaler


def clean_dataframe(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    cleaned = df.copy()

    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())

    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    if target_column not in cleaned.columns:
        raise KeyError(f"Target column '{target_column}' is missing from dataset.")

    return cleaned


def encode_dataframe(
    df: pd.DataFrame,
    target_column: str = "Churn",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], list[str], list[str]]:
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    numeric_columns = encoded.select_dtypes(include="number").columns.tolist()
    numeric_columns = [column for column in numeric_columns if column != target_column]

    categorical_columns = [
        column
        for column in encoded.columns
        if column not in numeric_columns and column != target_column
    ]

    columns_to_encode = categorical_columns + [target_column]
    for column in columns_to_encode:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder

    return encoded, encoders, numeric_columns, categorical_columns


def split_scale_balance(
    encoded_df: pd.DataFrame,
    *,
    target_column: str,
    test_size: float,
    random_state: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
    enable_smote: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    X = encoded_df.drop(columns=[target_column])
    y = encoded_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if enable_smote:
        minority_count = y_train.value_counts().min()
        if minority_count <= 1:
            enable_smote = False

    if enable_smote:
        categorical_indices = [X_train.columns.get_loc(column) for column in categorical_columns]
        k_neighbors = max(1, min(5, minority_count - 1))
        if categorical_indices:
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                random_state=random_state,
                k_neighbors=k_neighbors,
            )
        else:
            sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train = pd.Series(y_train_resampled, name=target_column)
    else:
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = StandardScaler()
    if numeric_columns:
        X_train = X_train.copy()
        X_test = X_test.copy()
        cast_map = {column: float for column in numeric_columns}
        X_train = X_train.astype(cast_map)
        X_test = X_test.astype(cast_map)
        X_train.loc[:, numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test.loc[:, numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train, X_test, y_train, y_test, scaler


def preprocess_pipeline(df: pd.DataFrame, config: dict) -> PreprocessBundle:
    project_config = config["project"]
    target_column = project_config["target_column"]

    cleaned_df = clean_dataframe(df, target_column=target_column)
    encoded_df, encoders, numeric_columns, categorical_columns = encode_dataframe(
        cleaned_df,
        target_column=target_column,
    )
    X_train, X_test, y_train, y_test, scaler = split_scale_balance(
        encoded_df,
        target_column=target_column,
        test_size=project_config["test_size"],
        random_state=project_config["random_state"],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        enable_smote=project_config["enable_smote"],
    )

    return PreprocessBundle(
        cleaned_df=cleaned_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X_train.columns.tolist(),
        target_column=target_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        encoders=encoders,
        scaler=scaler,
    )


def transform_new_customers(customers: pd.DataFrame, bundle: PreprocessBundle) -> pd.DataFrame:
    """Apply the fitted preprocessing objects to new customer rows."""
    transformed = customers.copy()

    for column in transformed.select_dtypes(include="object").columns:
        transformed[column] = transformed[column].astype(str).str.strip()

    if "customerID" in transformed.columns:
        transformed = transformed.drop(columns=["customerID"])

    if "TotalCharges" in transformed.columns:
        transformed["TotalCharges"] = pd.to_numeric(transformed["TotalCharges"], errors="coerce")
        fill_value = bundle.cleaned_df["TotalCharges"].median()
        transformed["TotalCharges"] = transformed["TotalCharges"].fillna(fill_value)

    required_columns = bundle.numeric_columns + bundle.categorical_columns
    missing_columns = [column for column in required_columns if column not in transformed.columns]
    if missing_columns:
        raise KeyError(f"Missing required customer columns: {missing_columns}")

    for column in bundle.categorical_columns:
        encoder = bundle.encoders[column]
        unknown_values = sorted(set(transformed[column].astype(str)) - set(encoder.classes_))
        if unknown_values:
            raise ValueError(
                f"Column '{column}' has unseen categories {unknown_values}. "
                f"Known categories: {encoder.classes_.tolist()}"
            )
        transformed[column] = encoder.transform(transformed[column].astype(str))

    if bundle.numeric_columns:
        transformed = transformed.astype({column: float for column in bundle.numeric_columns})
        transformed.loc[:, bundle.numeric_columns] = bundle.scaler.transform(transformed[bundle.numeric_columns])

    return transformed.loc[:, bundle.feature_names]
