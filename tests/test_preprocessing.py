import pandas as pd

from src.data.preprocess import clean_dataframe, preprocess_pipeline
from src.utils.config import DEFAULT_CONFIG


def test_clean_dataframe_handles_total_charges_and_customer_id():
    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Male", "Female"],
            "tenure": [0, 12],
            "MonthlyCharges": [70.0, 80.0],
            "TotalCharges": [" ", "960.0"],
            "Churn": ["Yes", "No"],
        }
    )

    cleaned = clean_dataframe(df)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_preprocess_pipeline_returns_expected_shapes():
    rows = []
    for i in range(12):
        rows.append(
            {
                "customerID": f"id-{i}",
                "gender": "Male" if i % 2 == 0 else "Female",
                "Partner": "Yes" if i % 3 == 0 else "No",
                "tenure": i + 1,
                "MonthlyCharges": 50 + i,
                "TotalCharges": str((50 + i) * (i + 1)),
                "Contract": "Month-to-month" if i % 2 == 0 else "Two year",
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )

    df = pd.DataFrame(rows)
    bundle = preprocess_pipeline(df, DEFAULT_CONFIG)

    assert not bundle.X_train.empty
    assert not bundle.X_test.empty
    assert bundle.target_column == "Churn"
