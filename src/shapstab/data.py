"""Dataset loading. Three public tabular binary-classification datasets.

* ``adult``         UCI Adult (census income), via OpenML  -> positive class: income > 50K
* ``german_credit`` German Credit (credit-g), via OpenML   -> positive class: bad credit
* ``telco``         IBM Telco Customer Churn, local CSV    -> positive class: churned

OpenML datasets are downloaded once and cached under ``data/openml``. The Telco CSV is not
redistributed: put it at ``data/WA_Fn-UseC_-_Telco_Customer_Churn.csv`` to include it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = ("adult", "german_credit", "telco")
DEFAULT_PATH = Path("data/WA_Fn-UseC_-_Telco_Customer_Churn.csv")
OPENML_HOME = Path("data/openml")


def _openml(name: str, version: int, positive: str) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name, version=version, as_frame=True, data_home=str(OPENML_HOME), parser="auto")
    return d.data.copy(), (d.target.astype(str) == positive).astype(int)


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
    return df.drop(columns=["Churn", "customerID"]), y


def load_dataset(name: str, max_rows: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset; optionally take a stratified subsample of at most ``max_rows`` rows."""
    if name == "adult":
        X, y = _openml("adult", 2, ">50K")
    elif name == "german_credit":
        X, y = _openml("credit-g", 1, "bad")
    elif name == "telco":
        X, y = load_telco()
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X.reset_index(drop=True), y.reset_index(drop=True)
