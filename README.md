# shap-stability

**How stable are SHAP explanations of a tabular churn model when only the training data changes?**

A small, reproducible study on the IBM Telco Customer Churn dataset. It compares a
logistic regression and a random forest and measures how much the *feature ranking*
produced by SHAP changes across repeated training runs.

> **Status:** work in progress. The experiment code is in place; results have not been
> generated yet, so this README intentionally reports no numbers.

## Research question

When the training sample changes (bootstrap resamples, different sizes, different random
seeds), do the SHAP feature rankings stay the same? Is the answer different for a linear
model and a tree ensemble?

## Hypotheses (to be tested, not assumed)

1. Logistic regression gives very stable rankings.
2. Random forest is less stable, especially among correlated features
   (`tenure`, `MonthlyCharges`, `TotalCharges`).
3. Stability grows with the size of the training set.

## Method

- A fixed explanation set is held out once, so explained rows are identical in every run.
- For each model and training fraction (10%, 25%, 50%, 100% of the pool), the model is
  trained on a bootstrap sample for N repeats; mean |SHAP| is computed per original feature
  (one-hot columns are summed per sample before taking the absolute value).
- Stability metrics across all pairs of runs: Spearman rank correlation and top-k Jaccard
  overlap; per-feature variability with bootstrap confidence intervals.

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# put the dataset at data/WA_Fn-UseC_-_Telco_Customer_Churn.csv  (see below)
pytest -q
python experiments/run_stability.py --repeats 30
```

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
It is not redistributed in this repo.

## Limitations (planned)

- One dataset and one explanation method (SHAP). Results may not transfer to other data.
- Stability of rankings is not the same as correctness of explanations.
- Correlated features can legitimately share credit differently between runs.

## Structure

```
src/shapstab/      data loading, models, stability metrics
experiments/       run_stability.py
tests/             unit tests for the metrics
results/           generated CSVs (created by the experiment)
```

## License

MIT
