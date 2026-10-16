"""Metrics to quantify how stable a feature-importance ranking is across repeated runs."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def aggregate_importance(shap_values: np.ndarray, columns: list[str], groups: list[str]) -> dict[str, float]:
    """Mean |SHAP| per original feature.

    ``shap_values`` has shape (n_samples, n_encoded_columns). ``groups[i]`` is the original
    feature that encoded column ``i`` came from. One-hot columns of the same feature are
    summed *per sample* first (signed), so opposite-sign contributions can cancel as they
    do in the model, then the absolute value is averaged over samples.
    """
    if shap_values.shape[1] != len(groups):
        raise ValueError("groups must have one entry per encoded column")
    out: dict[str, float] = {}
    for name in dict.fromkeys(groups):
        idx = [i for i, g in enumerate(groups) if g == name]
        out[name] = float(np.abs(shap_values[:, idx].sum(axis=1)).mean())
    return out


def top_k(importance: np.ndarray, k: int) -> set[int]:
    """Indices of the k most important features."""
    return set(np.argsort(-importance, kind="stable")[:k].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def pairwise_stability(importances: np.ndarray, k: int = 5) -> dict[str, float]:
    """Stability of rankings across repeats.

    ``importances`` has shape (n_runs, n_features). Returns the mean Spearman correlation
    and the mean top-k Jaccard overlap over all pairs of runs.
    """
    n_runs, n_features = importances.shape
    if n_runs < 2:
        raise ValueError("need at least two runs")
    k = min(k, n_features)
    spearmans, jaccards = [], []
    for i, j in combinations(range(n_runs), 2):
        a, b = importances[i], importances[j]
        if np.ptp(a) == 0 or np.ptp(b) == 0:  # constant ranking: correlation undefined
            continue
        rho = spearmanr(a, b).statistic
        spearmans.append(rho)
        jaccards.append(jaccard(top_k(a, k), top_k(b, k)))
    return {
        "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "topk_jaccard_mean": float(np.mean(jaccards)) if jaccards else float("nan"),
        "n_pairs": len(spearmans),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
