# Model Card — Purchase Intent Prediction

## Intended use
Predict whether an online shopping session will end in a purchase.
Use the probability + value-based decision threshold to plan retention interventions.

## Selected model
`CatBoost`

## Decision threshold & business value

- Decision threshold: **0.334** (chosen by `value_based_threshold`, not 0.5)
- Expected value at threshold: **1542.0**
- Precision @ threshold: **0.595**
- Recall @ threshold: **0.947**

## Held-out test set metrics — sorted by PR-AUC (PRD asks for PR-AUC, not accuracy)

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 | CV PR-AUC (mean ± std) |
|---|---|---|---|---|---|---|
| CatBoost | 0.862 | 0.971 | 0.915 | 0.657 | 0.765 | 0.845 ± 0.009 |
| RandomForest | 0.849 | 0.969 | 0.714 | 0.820 | 0.763 | 0.835 ± 0.012 |
| XGBoost | 0.847 | 0.969 | 0.761 | 0.769 | 0.765 | 0.837 ± 0.007 |
| LightGBM | 0.839 | 0.967 | 0.857 | 0.705 | 0.774 | 0.838 ± 0.006 |
| LogisticRegression | 0.803 | 0.965 | 0.907 | 0.656 | 0.762 | 0.804 ± 0.022 |
| DecisionTree | 0.792 | 0.936 | 0.897 | 0.604 | 0.721 | 0.742 ± 0.011 |

## Feature importance — lens #1 (SHAP, top 10)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | PageValues | 2.9292 |
| 2 | PageValuePerProduct | 1.4856 |
| 3 | ProductRelated | 0.3897 |
| 4 | EngagementScore | 0.3112 |
| 5 | ExitRates | 0.2482 |
| 6 | BounceRates | 0.2127 |
| 7 | BounceExitRatio | 0.1750 |
| 8 | TotalDuration | 0.1702 |
| 9 | ProductRelated_Duration | 0.1360 |
| 10 | Administrative_Duration | 0.1237 |

## Feature importance — lens #2 (scikit-learn permutation, top 10)

| Rank | Feature | Importance | Std |
|---|---|---|---|
| 1 | PageValues | 0.5610 | 0.0190 |
| 2 | PageValuePerProduct | 0.0446 | 0.0074 |
| 3 | ProductRelated | 0.0295 | 0.0108 |
| 4 | ExitRates | 0.0252 | 0.0060 |
| 5 | BounceRates | 0.0166 | 0.0033 |
| 6 | AvgProductDuration | 0.0150 | 0.0028 |
| 7 | Administrative_Duration | 0.0126 | 0.0038 |
| 8 | EngagementScore | 0.0108 | 0.0074 |
| 9 | TotalDuration | 0.0080 | 0.0031 |
| 10 | BounceExitRatio | 0.0076 | 0.0037 |

## PageValues leakage audit (PRD §2.2)
- Dropped features: ['PageValues', 'PageValuePerProduct']
- Selected model still trained WITHOUT PageValues. PR-AUC: 0.543, ROC-AUC: 0.841, Recall: 0.687
- Honest take: PageValues is a deployment-time data hazard (only known after the funnel renders). See `docs/pagevalues_ablation.md` for the full write-up.

## Responsible use
- Intended for aggregate funnel analysis and for choosing which sessions may get a low-cost nudge (for example a discount banner).
- Do NOT use it to identify or profile a named person, to deny service, to set an individual's price, or for any decision with legal or financial consequences for a person.
- The data is anonymised session behaviour. Do not join it with personally identifying fields.
- PageValues may not be known when a visitor is first seen. Do not score early in a session without re-checking the PageValues audit above.

## Limitations
- Trained on a single ~12k-session dataset. Re-check distribution shift before any production deploy.
- PageValues is a strong correlative signal; treat it as correlation, not cause.
- Class imbalance is ~15.7% positives; the primary metric on this dataset must remain PR-AUC, not accuracy.