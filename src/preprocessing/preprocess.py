"""ColumnTransformer for mixed numeric+categorical data (PRD §4.2).

Encoding/scaling happens per-fold (i.e. fitted on TRAIN only), never on the full
data before the split. The fitted transformer is persisted as a joblib so the API
loads the EXACT same transformation at serving time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ..utils.logger import get_logger

log = get_logger(__name__)


class PreprocessingPipeline:
    def __init__(
        self,
        categorical_columns: list[str],
        numeric_columns: list[str],
        scale_numeric: str = "robust",
    ) -> None:
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.scale_numeric = scale_numeric
        self.transformer_: ColumnTransformer | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame) -> PreprocessingPipeline:
        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        steps = [("impute", SimpleImputer(strategy="median"))]
        if self.scale_numeric == "robust":
            steps.append(("scale", RobustScaler()))
        elif self.scale_numeric == "standard":
            from sklearn.preprocessing import StandardScaler

            steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(steps=steps)

        self.transformer_ = ColumnTransformer(
            transformers=[
                ("cat", cat_pipe, self.categorical_columns),
                ("num", num_pipe, self.numeric_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        # Defensive: avoid SettingWithCopyWarning by coercing categoricals to str
        X_clean = X.copy()
        for col in self.categorical_columns:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].astype(str)
        self.transformer_.fit(X_clean)
        try:
            self._feature_names = list(self.transformer_.get_feature_names_out())
        except Exception:
            self._feature_names = []
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("PreprocessingPipeline not fitted")
        X_clean = X.copy()
        for col in self.categorical_columns:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].astype(str)
        return self.transformer_.transform(X_clean)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self) -> list[str]:
        if self._feature_names:
            return self._feature_names
        if self.transformer_ is None:
            return []
        try:
            return list(self.transformer_.get_feature_names_out())
        except Exception:
            return []

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer_, path)
        log.info("Saved preprocessing pipeline to %s", path)

    @classmethod
    def load(cls, path: Path) -> PreprocessingPipeline:
        transformer = joblib.load(path)
        inst = cls(categorical_columns=[], numeric_columns=[])
        inst.transformer_ = transformer
        try:
            inst._feature_names = list(transformer.get_feature_names_out())
        except Exception:
            inst._feature_names = []
        return inst
