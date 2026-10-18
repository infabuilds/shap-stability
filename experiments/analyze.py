"""Turn the raw per-run importances into statistics, tables and figures.

Reads  results/importances_long.csv  and  results/feature_correlation.csv
Writes results/stability_summary.csv, results/feature_rank_stats.csv, results/corr_vs_instability.csv,
       results/summary.md and figures in results/figures/.

Usage:  python experiments/analyze.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shapstab.stability import mean_offdiag_ci, pairwise_matrices, rank_matrix  # noqa: E402

COLORS = {"logreg": "#2563EB", "random_forest": "#D97706"}
LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest"}
TITLES = {"adult": "Adult (census income)", "german_credit": "German Credit", "telco": "Telco churn"}


def matrix_for(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    wide = df.pivot_table(index="run", columns="feature", values="importance")
    return wide.to_numpy(), list(wide.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(res / "importances_long.csv")
    corr = pd.read_csv(res / "feature_correlation.csv")

    summary, feat_rows = [], []
    for (ds, model, frac), g in long.groupby(["dataset", "model", "fraction"]):
        imp, feats = matrix_for(g)
        S, J = pairwise_matrices(imp, k=args.top_k)
        s_mean, s_lo, s_hi = mean_offdiag_ci(S)
        j_mean, j_lo, j_hi = mean_offdiag_ci(J)
        auc = g.drop_duplicates("run")["holdout_auc"]
        summary.append({"dataset": ds, "model": model, "fraction": frac, "n_runs": len(imp),
                        "n_features": len(feats), "spearman": s_mean, "spearman_lo": s_lo, "spearman_hi": s_hi,
                        "topk_jaccard": j_mean, "jaccard_lo": j_lo, "jaccard_hi": j_hi,
                        "holdout_auc_mean": auc.mean(), "holdout_auc_sd": auc.std()})
        ranks = rank_matrix(imp)
        for i, f in enumerate(feats):
            feat_rows.append({"dataset": ds, "model": model, "fraction": frac, "feature": f,
                              "rank_mean": ranks[:, i].mean(), "rank_sd": ranks[:, i].std(ddof=1),
                              "importance_mean": imp[:, i].mean(),
                              "importance_cv": imp[:, i].std(ddof=1) / (imp[:, i].mean() + 1e-12)})
    summary = pd.DataFrame(summary).sort_values(["dataset", "model", "fraction"])
    feats_df = pd.DataFrame(feat_rows).merge(corr, on=["dataset", "feature"], how="left")
    summary.to_csv(res / "stability_summary.csv", index=False)
    feats_df.to_csv(res / "feature_rank_stats.csv", index=False)

    # Relation between feature correlation and rank variability (hypothesis 2)
    rel = []
    for (ds, model, frac), g in feats_df.groupby(["dataset", "model", "fraction"]):
        if g["max_abs_corr"].nunique() > 2 and g["rank_sd"].nunique() > 2:
            rho, p = spearmanr(g["max_abs_corr"], g["rank_sd"])
            rel.append({"dataset": ds, "model": model, "fraction": frac, "n_features": len(g),
                        "spearman_corr_vs_rank_sd": rho, "p_value": p})
    rel = pd.DataFrame(rel)
    rel.to_csv(res / "corr_vs_instability.csv", index=False)

    # ---- Figures ----
    datasets = list(summary["dataset"].unique())
    for metric, lo, hi, ylabel, fname in [
        ("spearman", "spearman_lo", "spearman_hi", "Mean Spearman correlation\nbetween rankings of two runs", "stability_spearman.png"),
        ("topk_jaccard", "jaccard_lo", "jaccard_hi", f"Mean top-{args.top_k} overlap (Jaccard)\nbetween two runs", "stability_topk.png"),
    ]:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ds in zip(axes, datasets):
            for model in ("logreg", "random_forest"):
                s = summary[(summary.dataset == ds) & (summary.model == model)]
                if s.empty:
                    continue
                ax.errorbar(s["fraction"] * 100, s[metric], yerr=[s[metric] - s[lo], s[hi] - s[metric]],
                            marker="o", capsize=3, lw=2, color=COLORS[model], label=LABELS[model])
            ax.set_title(TITLES.get(ds, ds))
            ax.set_xlabel("Training data used (% of pool)")
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100])
            ax.set_xticklabels(["10", "25", "50", "100"])
            ax.set_ylim(min(0.0, summary[lo].min() - 0.05), 1.02)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False, loc="lower right")
        fig.suptitle("How stable is the SHAP feature ranking when only the training data changes?\n"
                     "(points: mean over run pairs, bars: 95% bootstrap CI over runs)", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)

    full = feats_df[feats_df.fraction == feats_df.fraction.max()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        for model in ("logreg", "random_forest"):
            g = full[(full.dataset == ds) & (full.model == model)]
            ax.scatter(g["max_abs_corr"], g["rank_sd"], s=45, alpha=0.8, color=COLORS[model], label=LABELS[model])
        ax.set_title(TITLES.get(ds, ds))
        ax.set_xlabel("Max |correlation| with another feature")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("SD of the feature's rank across runs")
    axes[0].legend(frameon=False)
    fig.suptitle("Do correlated features have less stable ranks? (full training pool)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_vs_rank_sd.png", dpi=160)
    plt.close(fig)

    # ---- Markdown summary ----
    def fmt(r):
        return (f"| {TITLES.get(r.dataset, r.dataset)} | {LABELS[r.model]} | {int(r.fraction * 100)}% | "
                f"{r.spearman:.3f} [{r.spearman_lo:.3f}, {r.spearman_hi:.3f}] | "
                f"{r.topk_jaccard:.3f} [{r.jaccard_lo:.3f}, {r.jaccard_hi:.3f}] | {r.holdout_auc_mean:.3f} |")

    lines = ["| Dataset | Model | Training data | Spearman (95% CI) | Top-5 Jaccard (95% CI) | Holdout AUC |",
             "|---|---|---|---|---|---|"] + [fmt(r) for r in summary.itertuples()]
    if not rel.empty:
        lines += ["", "Correlation between a feature's max |correlation| with other features and the SD of its rank:", "",
                  "| Dataset | Model | Training data | Spearman rho | p-value | n features |", "|---|---|---|---|---|---|"]
        lines += [f"| {TITLES.get(r.dataset, r.dataset)} | {LABELS[r.model]} | {int(r.fraction * 100)}% | "
                  f"{r.spearman_corr_vs_rank_sd:.2f} | {r.p_value:.3f} | {r.n_features} |" for r in rel.itertuples()]
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
