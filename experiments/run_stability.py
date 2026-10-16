"""Experiment: how stable is the SHAP feature ranking when only the training data changes?

Design
------
* A fixed explanation set is held out once (same rows in every run), so any change in the
  explanations comes from the training data / model seed, not from different explained rows.
* For each training fraction and each model, ``--repeats`` runs train on a bootstrap sample
  of the remaining pool and compute mean |SHAP| per original feature on the explanation set.
* Stability = mean pairwise Spearman correlation and top-k Jaccard overlap across runs.

Usage:  python experiments/run_stability.py --repeats 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shapstab.data import DEFAULT_PATH, load_telco  # noqa: E402
from shapstab.models import MODELS, encoded_groups, make_classifier, make_preprocessor  # noqa: E402
from shapstab.stability import aggregate_importance, bootstrap_ci, pairwise_stability  # noqa: E402


def explain(model_name, clf, X_train_enc, X_explain_enc, seed):
    import shap  # imported lazily: heavy

    rng = np.random.default_rng(seed)
    if model_name == "logreg":
        bg = X_train_enc[rng.choice(len(X_train_enc), size=min(100, len(X_train_enc)), replace=False)]
        values = shap.LinearExplainer(clf, bg).shap_values(X_explain_enc)
    else:
        values = shap.TreeExplainer(clf).shap_values(X_explain_enc)
    values = np.asarray(values) if not isinstance(values, list) else np.asarray(values[1])
    if values.ndim == 3:  # (samples, features, classes) in recent shap versions
        values = values[:, :, 1]
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_PATH))
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--explain-rows", type=int, default=500)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    X, y = load_telco(args.data)
    pool_X, explain_X, pool_y, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    explain_X = explain_X.iloc[: args.explain_rows]
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    summary, all_imp = [], []
    for model_name in args.models:
        for frac in args.fractions:
            runs, features = [], None
            for r in range(args.repeats):
                rng = np.random.default_rng(1000 * r + int(frac * 100))
                idx = rng.choice(len(pool_X), size=int(frac * len(pool_X)), replace=True)
                Xt, yt = pool_X.iloc[idx], pool_y.iloc[idx]
                pre = make_preprocessor(X)
                Xt_enc = pre.fit_transform(Xt)
                Xe_enc = pre.transform(explain_X)
                _, groups = encoded_groups(pre)
                clf = make_classifier(model_name, seed=r).fit(Xt_enc, yt)
                imp = aggregate_importance(explain(model_name, clf, Xt_enc, Xe_enc, r),
                                           pre.get_feature_names_out().tolist(), groups)
                features = list(imp)
                runs.append([imp[f] for f in features])
                all_imp.append({"model": model_name, "fraction": frac, "run": r, **imp})
            arr = np.array(runs)
            st = pairwise_stability(arr, k=args.top_k)
            per_run_rank_spread = arr.std(axis=0) / (arr.mean(axis=0) + 1e-12)
            lo, hi = bootstrap_ci(per_run_rank_spread)
            summary.append({"model": model_name, "fraction": frac, "repeats": args.repeats,
                            **st, "cv_mean": float(per_run_rank_spread.mean()), "cv_ci_low": lo, "cv_ci_high": hi})
            print(f"{model_name:14s} frac={frac:<5} spearman={st['spearman_mean']:.3f} "
                  f"top{args.top_k}-jaccard={st['topk_jaccard_mean']:.3f}")

    pd.DataFrame(summary).to_csv(out / "stability_summary.csv", index=False)
    pd.DataFrame(all_imp).to_csv(out / "importances_per_run.csv", index=False)
    print(f"saved results to {out}/")


if __name__ == "__main__":
    main()
