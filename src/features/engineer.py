"""Behavioural feature engineering (PRD §4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Five engineered features, each with a documented rationale in
    `docs/user_guide.md` and `notebooks/02_features.ipynb`."""

    ENGINEERED = [
        "TotalDuration",
        "AvgProductDuration",
        "BounceExitRatio",
        "PageValuePerProduct",
        "EngagementScore",
    ]

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        self.feature_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["TotalDuration"] = (
            df.get("Administrative_Duration", 0).fillna(0)
            + df.get("Informational_Duration", 0).fillna(0)
            + df.get("ProductRelated_Duration", 0).fillna(0)
        )

        prod_pages = df.get("ProductRelated", 0).replace(0, np.nan)
        df["AvgProductDuration"] = (
            df.get("ProductRelated_Duration", 0).fillna(0) / prod_pages
        ).fillna(0.0)
        # Cap extreme values (1-second-per-1-product dwell is usually a glitch)
        df["AvgProductDuration"] = df["AvgProductDuration"].clip(lower=0.0, upper=20000.0)

        exit = df.get("ExitRates", 0).replace(0, np.nan)
        df["BounceExitRatio"] = (
            (df.get("BounceRates", 0).fillna(0) / exit).fillna(0).clip(upper=5.0)
        )

        df["PageValuePerProduct"] = (
            (df.get("PageValues", 0).fillna(0) / prod_pages).fillna(0).clip(lower=0, upper=200)
        )

        df["EngagementScore"] = (
            df.get("ProductRelated", 0) + df.get("Informational", 0) + df.get("Administrative", 0)
        )
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
