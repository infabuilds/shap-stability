"""Preprocessing and the two models compared in the experiments."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = ("logreg", "random_forest")


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def make_classifier(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def encoded_groups(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    """Names of the encoded columns and, for each, the original feature it came from."""
    names = preprocessor.get_feature_names_out().tolist()
    groups = []
    for n in names:
        _, rest = n.split("__", 1)
        if n.startswith("cat__"):
            for orig in preprocessor.transformers_[1][2]:
                if rest == orig or rest.startswith(orig + "_"):
                    groups.append(orig)
                    break
            else:
                raise ValueError(f"cannot map encoded column {n!r}")
        else:
            groups.append(rest)
    return names, groups
