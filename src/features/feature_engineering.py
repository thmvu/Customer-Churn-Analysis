import pandas as pd


def add_charges_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    if {"TotalCharges", "tenure"}.issubset(engineered.columns):
        engineered["charges_per_tenure"] = engineered["TotalCharges"] / engineered["tenure"].replace(0, 1)
    return engineered
