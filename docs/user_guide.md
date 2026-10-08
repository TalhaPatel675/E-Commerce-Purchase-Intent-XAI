# User guide

## Audience

Internship student / reviewer / analyst.

## What the system does

Predicts whether each e-commerce session ends with a purchase and exposes it as:
* a probability (0–1),
* a binary decision using a value-based threshold (not naive 0.5),
* a confidence indicator (low/medium/high),
* the top contributing behaviours (local SHAP, served by `/predict`).

## Headline metrics (held-out 20% test set)

Macro view lives in `reports\MODEL_CARD.md`. Re-generated every time you retrain.

| Top model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| CatBoost (typically) | ≈ 0.86 | ≈ 0.97 | ≈ 0.91 | ≈ 0.65 | ≈ 0.76 |

The primary key is **PR-AUC** because the data is imbalanced (≈ 16% positives). Accuracy is shown but not used as a tiebreaker.

## How to interpret a `/predict` response

```json
{
  "prediction": 1,
  "conversion_probability": 0.73,
  "confidence": "high",
  "decision_threshold": 0.3335,
  "top_contributors": [
    {"feature": "PageValues", "direction": "increases_purchase_likelihood", "contribution": 0.31, "value": "PageValues=36.0"}
  ],
  "model_name": "CatBoost"
}
```

* `prediction=1` means the probability crossed the business threshold.
* `confidence` = distance of the probability from 0.5 (low/medium/high from `config.yaml`).
* `top_contributors` are SHAP local values, mapped back to the raw feature prefix.
* `decision_threshold` was chosen by `value_based_threshold` (intervention cost vs purchase value), not 0.5.

## Feature engineering rationale

| Feature | Rationale |
|---|---|
| `TotalDuration` | Engagement across page types — buyers linger longer. |
| `AvgProductDuration` | Dwell per product page; controls "glanced at many vs studied few". |
| `BounceExitRatio` | Captures how fast sessions are leaving the funnel. |
| `PageValuePerProduct` | Normalises the strongest purchase signal (PageValues) by product count. |
| `EngagementScore` | Cheap sum of page types for tree models to split on. |

Drop-in at `src/features/engineer.py`; the API applies the same engineering at `/predict` time — no train/serve skew.

## Important caveat — PageValues leakage

PageValues is a strong correlative signal but is only known **after the page renders**. The full audit lives in `docs/pagevalues_ablation.md`; the headline is:

* with PageValues — PR-AUC ≈ 0.86
* without PageValues — PR-AUC ≈ 0.83

In production you should treat it as a deployment-leakage candidate: drop it for online scoring, or use it only post-fact for cohort analysis.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 422 on `/predict` | Pydantic range violation | Use values within PRD-specified ranges |
| 503 on `/predict` | Artifacts missing | Run `scripts\train.bat` |
| Dashboard "Artifacts missing" | Training not yet run | Run `scripts\train.bat` |
| `lightgbm` pip fails on Windows | Need VC++ Redist 2015-22 | Install from the link in `docs/install_windows.md` |
