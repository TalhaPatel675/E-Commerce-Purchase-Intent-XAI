"""Trainer factories — six algorithms through one common interface (PRD §4.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ModelConfig:
    name: str
    factory: Any  # zero-arg callable returning a fresh estimator


def default_model_factory(
    random_seed: int, class_weight: str | None = "balanced"
) -> dict[str, ModelConfig]:
    """Return 6 algorithms. class_weight='balanced' is the chosen imbalance strategy
    (no SMOTE pre-resample → no leakage; cf. PRD §4.2)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    cw = class_weight if class_weight == "balanced" else None
    factories: dict[str, ModelConfig] = {}

    def lr():
        return LogisticRegression(max_iter=2000, class_weight=cw, C=1.0, random_state=random_seed)

    def dt():
        return DecisionTreeClassifier(max_depth=8, class_weight=cw, random_state=random_seed)

    def rf():
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight=cw, random_state=random_seed, n_jobs=1
        )

    factories["LogisticRegression"] = ModelConfig("LogisticRegression", lr)
    factories["DecisionTree"] = ModelConfig("DecisionTree", dt)
    factories["RandomForest"] = ModelConfig("RandomForest", rf)
    factories.update(boosting_factory(random_seed))
    return factories


def boosting_factory(random_seed: int) -> dict[str, ModelConfig]:
    factories: dict[str, ModelConfig] = {}
    try:
        from xgboost import XGBClassifier

        factories["XGBoost"] = ModelConfig(
            "XGBoost",
            lambda: XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=random_seed,
            ),
        )
    except Exception as e:
        log.warning("XGBoost unavailable: %s", e)
    try:
        from lightgbm import LGBMClassifier

        factories["LightGBM"] = ModelConfig(
            "LightGBM",
            lambda: LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                class_weight="balanced",
                random_state=random_seed,
                verbose=-1,
            ),
        )
    except Exception as e:
        log.warning("LightGBM unavailable: %s", e)
    try:
        from catboost import CatBoostClassifier

        factories["CatBoost"] = ModelConfig(
            "CatBoost",
            lambda: CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="PRAUC",
                random_seed=random_seed,
                auto_class_weights="Balanced",
                verbose=False,
            ),
        )
    except Exception as e:
        log.warning("CatBoost unavailable: %s", e)
    return factories


class Trainer:
    """Common trainer interface."""

    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.model = model

    def fit(self, X, y) -> Trainer:
        log.info(
            "Training %s on X=%s y=%s",
            self.name,
            getattr(X, "shape", len(X)),
            getattr(y, "shape", len(y)),
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
