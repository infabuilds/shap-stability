import numpy as np
import pytest

from shapstab.stability import aggregate_importance, bootstrap_ci, jaccard, pairwise_stability, top_k


def test_identical_runs_are_perfectly_stable():
    base = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    res = pairwise_stability(np.vstack([base, base, base]), k=3)
    assert res["spearman_mean"] == pytest.approx(1.0)
    assert res["topk_jaccard_mean"] == pytest.approx(1.0)
    assert res["n_pairs"] == 3


def test_reversed_ranking_is_perfectly_unstable():
    base = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    res = pairwise_stability(np.vstack([base, base[::-1]]), k=2)
    assert res["spearman_mean"] == pytest.approx(-1.0)
    assert res["topk_jaccard_mean"] == pytest.approx(0.0)


def test_top_k_and_jaccard():
    assert top_k(np.array([0.1, 0.9, 0.5]), 2) == {1, 2}
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_needs_two_runs():
    with pytest.raises(ValueError):
        pairwise_stability(np.array([[1.0, 2.0, 3.0]]))


def test_aggregate_groups_one_hot_columns_per_sample():
    shap_values = np.array([[1.0, -1.0, 2.0], [1.0, 1.0, -2.0]])
    imp = aggregate_importance(shap_values, ["a_x", "a_y", "b"], ["a", "a", "b"])
    # sample 1: a = 0 (cancels), sample 2: a = 2  -> mean |.| = 1 ; b -> 2
    assert imp["a"] == pytest.approx(1.0)
    assert imp["b"] == pytest.approx(2.0)


def test_aggregate_rejects_mismatched_groups():
    with pytest.raises(ValueError):
        aggregate_importance(np.zeros((2, 3)), ["a", "b", "c"], ["a", "b"])


def test_bootstrap_ci_contains_the_mean():
    values = np.random.default_rng(0).normal(10, 1, size=200)
    lo, hi = bootstrap_ci(values, n_boot=300)
    assert lo < values.mean() < hi


from shapstab.stability import mean_offdiag_ci, pairwise_matrices, rank_matrix  # noqa: E402


def test_rank_matrix_orders_most_important_first():
    r = rank_matrix(np.array([[0.1, 0.9, 0.5]]))
    assert r.tolist() == [[3.0, 1.0, 2.0]]


def test_pairwise_matrices_identical_and_reversed_runs():
    base = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    S, J = pairwise_matrices(np.vstack([base, base, base[::-1]]), k=2)
    assert S[0, 1] == pytest.approx(1.0) and S[0, 2] == pytest.approx(-1.0)
    assert J[0, 1] == pytest.approx(1.0) and J[0, 2] == pytest.approx(0.0)


def test_mean_offdiag_ci_ignores_self_pairs_and_brackets_mean():
    rng = np.random.default_rng(0)
    imp = rng.random((20, 6))
    S, _ = pairwise_matrices(imp)
    mean, lo, hi = mean_offdiag_ci(S, n_boot=200)
    assert lo <= mean <= hi
    assert hi < 1.0  # would be pulled towards 1 if identical-run pairs were counted
