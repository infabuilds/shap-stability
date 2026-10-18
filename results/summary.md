| Dataset | Model | Training data | Spearman (95% CI) | Top-5 Jaccard (95% CI) | Holdout AUC |
|---|---|---|---|---|---|
| Adult (census income) | Logistic regression | 10% | 0.777 [0.742, 0.808] | 0.614 [0.565, 0.670] | 0.877 |
| Adult (census income) | Logistic regression | 25% | 0.873 [0.855, 0.891] | 0.714 [0.678, 0.759] | 0.884 |
| Adult (census income) | Logistic regression | 50% | 0.905 [0.895, 0.913] | 0.688 [0.655, 0.726] | 0.887 |
| Adult (census income) | Logistic regression | 100% | 0.923 [0.912, 0.933] | 0.777 [0.737, 0.824] | 0.889 |
| Adult (census income) | Random forest | 10% | 0.918 [0.905, 0.929] | 0.761 [0.724, 0.797] | 0.879 |
| Adult (census income) | Random forest | 25% | 0.948 [0.941, 0.955] | 0.826 [0.779, 0.881] | 0.886 |
| Adult (census income) | Random forest | 50% | 0.959 [0.952, 0.965] | 0.838 [0.790, 0.891] | 0.890 |
| Adult (census income) | Random forest | 100% | 0.975 [0.971, 0.980] | 0.843 [0.791, 0.901] | 0.893 |
| German Credit | Logistic regression | 10% | 0.336 [0.301, 0.369] | 0.274 [0.248, 0.297] | 0.707 |
| German Credit | Logistic regression | 25% | 0.421 [0.390, 0.457] | 0.333 [0.303, 0.370] | 0.727 |
| German Credit | Logistic regression | 50% | 0.537 [0.502, 0.567] | 0.465 [0.413, 0.520] | 0.754 |
| German Credit | Logistic regression | 100% | 0.681 [0.651, 0.710] | 0.613 [0.563, 0.671] | 0.763 |
| German Credit | Random forest | 10% | 0.547 [0.517, 0.577] | 0.325 [0.291, 0.365] | 0.715 |
| German Credit | Random forest | 25% | 0.638 [0.602, 0.677] | 0.394 [0.358, 0.435] | 0.741 |
| German Credit | Random forest | 50% | 0.760 [0.738, 0.781] | 0.454 [0.421, 0.493] | 0.769 |
| German Credit | Random forest | 100% | 0.861 [0.844, 0.880] | 0.602 [0.554, 0.655] | 0.779 |
| Telco churn | Logistic regression | 10% | 0.491 [0.448, 0.529] | 0.440 [0.404, 0.475] | 0.836 |
| Telco churn | Logistic regression | 25% | 0.596 [0.556, 0.632] | 0.553 [0.514, 0.591] | 0.844 |
| Telco churn | Logistic regression | 50% | 0.736 [0.703, 0.767] | 0.633 [0.587, 0.685] | 0.848 |
| Telco churn | Logistic regression | 100% | 0.810 [0.787, 0.834] | 0.753 [0.702, 0.820] | 0.849 |
| Telco churn | Random forest | 10% | 0.816 [0.795, 0.837] | 0.616 [0.566, 0.672] | 0.836 |
| Telco churn | Random forest | 25% | 0.894 [0.882, 0.906] | 0.742 [0.687, 0.798] | 0.840 |
| Telco churn | Random forest | 50% | 0.934 [0.927, 0.942] | 0.808 [0.757, 0.861] | 0.841 |
| Telco churn | Random forest | 100% | 0.957 [0.950, 0.963] | 0.834 [0.782, 0.890] | 0.841 |

Correlation between a feature's max |correlation| with other features and the SD of its rank:

| Dataset | Model | Training data | Spearman rho | p-value | n features |
|---|---|---|---|---|---|
| Adult (census income) | Logistic regression | 10% | -0.17 | 0.552 | 14 |
| Adult (census income) | Logistic regression | 25% | 0.24 | 0.413 | 14 |
| Adult (census income) | Logistic regression | 50% | 0.10 | 0.736 | 14 |
| Adult (census income) | Logistic regression | 100% | 0.14 | 0.636 | 14 |
| Adult (census income) | Random forest | 10% | -0.35 | 0.223 | 14 |
| Adult (census income) | Random forest | 25% | -0.45 | 0.107 | 14 |
| Adult (census income) | Random forest | 50% | -0.38 | 0.178 | 14 |
| Adult (census income) | Random forest | 100% | -0.32 | 0.261 | 14 |
| German Credit | Logistic regression | 10% | 0.38 | 0.094 | 20 |
| German Credit | Logistic regression | 25% | 0.55 | 0.012 | 20 |
| German Credit | Logistic regression | 50% | 0.44 | 0.054 | 20 |
| German Credit | Logistic regression | 100% | 0.45 | 0.044 | 20 |
| German Credit | Random forest | 10% | 0.24 | 0.312 | 20 |
| German Credit | Random forest | 25% | 0.23 | 0.340 | 20 |
| German Credit | Random forest | 50% | 0.24 | 0.312 | 20 |
| German Credit | Random forest | 100% | 0.05 | 0.820 | 20 |
| Telco churn | Logistic regression | 10% | -0.09 | 0.705 | 19 |
| Telco churn | Logistic regression | 25% | -0.10 | 0.680 | 19 |
| Telco churn | Logistic regression | 50% | 0.07 | 0.763 | 19 |
| Telco churn | Logistic regression | 100% | 0.21 | 0.387 | 19 |
| Telco churn | Random forest | 10% | -0.22 | 0.363 | 19 |
| Telco churn | Random forest | 25% | 0.06 | 0.815 | 19 |
| Telco churn | Random forest | 50% | -0.11 | 0.661 | 19 |
| Telco churn | Random forest | 100% | -0.22 | 0.365 | 19 |
