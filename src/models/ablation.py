"""PageValues deployment-leakage ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import get_logger
from src.utils.metrics import full_metric_suite

log = get_logger(__name__)


def run_pagevalues_ablation(
    best_name: str,
    best_config: Any,
    pipeline_factory,
    X_train_df,
    y_train,
    X_test_df,
    y_test,
    categorical_columns: list[str],
    numeric_columns: list[str],
    feature_columns_drop=("PageValues", "PageValuePerProduct"),
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Refit the winning model without PageValues or features derived from it.

    Preprocessing is fitted inside each CV fold so the ablation is directly
    comparable to the main leakage-safe benchmark.
    """
    drops = list(feature_columns_drop)
    log.info("PageValues ablation: dropping %s", drops)
    X_train_drop = X_train_df.drop(columns=drops, errors="ignore").reset_index(drop=True)
    X_test_drop = X_test_df.drop(columns=drops, errors="ignore").reset_index(drop=True)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    cat_cols = [c for c in categorical_columns if c in X_train_drop.columns]
    num_cols = [c for c in numeric_columns if c not in drops and c in X_train_drop.columns]

    def make_pipe():
        return pipeline_factory(
            categorical_columns=cat_cols,
            numeric_columns=num_cols,
            scale_numeric="robust",
        )

    pipe = make_pipe()
    Xt_train = pipe.fit_transform(X_train_drop)
    Xt_test = pipe.transform(X_test_drop)
    model = best_config.factory()
    model.fit(Xt_train, y_train)
    test_proba = model.predict_proba(Xt_test)[:, 1]
    test_metrics = full_metric_suite(y_test, (test_proba >= 0.5).astype(int), test_proba)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train_drop, y_train), start=1):
        fold_pipe = make_pipe()
        Xtr = fold_pipe.fit_transform(X_train_drop.iloc[tr_idx])
        Xva = fold_pipe.transform(X_train_drop.iloc[va_idx])
        fold_model = best_config.factory()
        fold_model.fit(Xtr, y_train[tr_idx])
        fold_proba = fold_model.predict_proba(Xva)[:, 1]
        fold_scores.append(average_precision_score(y_train[va_idx], fold_proba))
        log.info("PageValues ablation CV fold %d/%d complete", fold, cv_folds)

    test_metrics["cv_pr_auc_mean"] = float(np.mean(fold_scores))
    test_metrics["cv_pr_auc_std"] = float(np.std(fold_scores))
    return {
        "dropped_features": drops,
        "selected_model": best_name,
        "metrics": test_metrics,
        "comparison_note": "PageValues and the derived PageValuePerProduct feature were both removed.",
    }
