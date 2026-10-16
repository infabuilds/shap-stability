"""Loading and cleaning of the IBM Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/WA_Fn-UseC_-_Telco_Customer_Churn.csv")


def load_telco(path: str | Path = DEFAULT_PATH) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the IBM Telco Customer Churn CSV "
            "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it there."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # 11 blank values
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    return X, y
