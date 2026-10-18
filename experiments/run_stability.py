"""Experiment: how stable is the SHAP feature ranking when only the training data changes?

Design
------
* For each dataset, a fixed holdout is set aside once (20%, stratified). The same holdout
  rows are explained in every run, so any change in the explanations comes from the
  training data / model seed, not from different explained rows.
* For each model and training fraction, ``--repeats`` runs train on a bootstrap sample of the
  remaining pool (size = fraction x pool size) and compute mean |SHAP| per original feature
  on (up to ``--explain-rows`` rows of) the holdout.
* Raw per-run importances are saved in long format; all statistics are computed afterwards by
  ``experiments/analyze.py`` so they can be re-run without retraining.

Usage:  python experiments/run_stability.py --repeats 30
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shapstab.data import DATASETS, load_dataset  # noqa: E402
from shapstab.models import MODELS, encoded_groups, make_classifier, make_preprocessor  # noqa: E402
from shapstab.stability import aggregate_importance  # noqa: E402

warnings.filterwarnings("ignore", message="Background dataset has")


def explain(model_name, clf, X_train_enc, X_explain_enc, seed):
    import shap  # imported lazily: heavy

    rng = np.random.default_rng(seed)
    if model_name == "logreg":
        bg = X_train_enc[rng.choice(len(X_train_enc), size=min(100, len(X_train_enc)), replace=False)]
        values = shap.LinearExplainer(clf, bg).shap_values(X_explain_enc)
    else:
        values = shap.TreeExplainer(clf).shap_values(X_explain_enc)
    values = np.asarray(values[1]) if isinstance(values, list) else np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes) in recent shap versions
        values = values[:, :, 1]
    return values


def feature_max_correlation(X_enc: np.ndarray, groups: list[str]) -> dict[str, float]:
    """For each original feature: the highest |correlation| between one of its encoded columns
    and any encoded column of a *different* feature."""
    corr = np.abs(np.corrcoef(X_enc, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    g = np.array(groups)
    out = {}
    for name in dict.fromkeys(groups):
        mine, other = g == name, g != name
        out[name] = float(corr[np.ix_(mine, other)].max()) if other.any() else 0.0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["adult", "german_credit", "telco"], choices=DATASETS)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--explain-rows", type=int, default=500)
    p.add_argument("--max-rows", type=int, default=10000, help="cap on dataset size (stratified subsample)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows, corr_rows, t0 = [], [], time.time()

    for ds in args.datasets:
        try:
            X, y = load_dataset(ds, max_rows=args.max_rows)
        except FileNotFoundError as e:
            print(f"[skip] {ds}: {e}")
            continue
        pool_X, hold_X, pool_y, hold_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        explain_X = hold_X.iloc[: args.explain_rows]
        print(f"== {ds}: {len(X)} rows, {X.shape[1]} features, positive rate {y.mean():.3f}", flush=True)

        pre_full = make_preprocessor(X).fit(pool_X)
        _, groups_full = encoded_groups(pre_full)
        for f, c in feature_max_correlation(pre_full.transform(pool_X), groups_full).items():
            corr_rows.append({"dataset": ds, "feature": f, "max_abs_corr": c})

        for model_name in args.models:
            for frac in args.fractions:
                for r in range(args.repeats):
                    rng = np.random.default_rng(1000 * r + int(frac * 100))
                    idx = rng.choice(len(pool_X), size=max(int(frac * len(pool_X)), 50), replace=True)
                    Xt, yt = pool_X.iloc[idx], pool_y.iloc[idx]
                    pre = make_preprocessor(X)
                    Xt_enc = pre.fit_transform(Xt)
                    _, groups = encoded_groups(pre)
                    clf = make_classifier(model_name, seed=r).fit(Xt_enc, yt)
                    auc = float(roc_auc_score(hold_y, clf.predict_proba(pre.transform(hold_X))[:, 1]))
                    shap_vals = explain(model_name, clf, Xt_enc, pre.transform(explain_X), r)
                    imp = aggregate_importance(shap_vals, pre.get_feature_names_out().tolist(), groups)
                    for feat, val in imp.items():
                        rows.append({"dataset": ds, "model": model_name, "fraction": frac, "run": r,
                                     "feature": feat, "importance": val, "holdout_auc": auc})
                print(f"  {ds:14s} {model_name:14s} frac={frac:<5} done  [{time.time() - t0:6.0f}s]", flush=True)
                pd.DataFrame(rows).to_csv(out / "importances_long.csv", index=False)  # checkpoint

    pd.DataFrame(rows).to_csv(out / "importances_long.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "feature_correlation.csv", index=False)
    print(f"saved results to {out}/  (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
