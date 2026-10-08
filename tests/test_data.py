"""Tests for data loading and feature engineering (PRD §5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataLoader, DataValidationError


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Administrative": [1, 2, 0],
            "Administrative_Duration": [10.0, 20.0, 0.0],
            "Informational": [0, 0, 1],
            "Informational_Duration": [0.0, 0.0, 5.0],
            "ProductRelated": [10, 20, 5],
            "ProductRelated_Duration": [100.0, 200.0, 50.0],
            "BounceRates": [0.01, 0.02, 0.03],
            "ExitRates": [0.05, 0.04, 0.06],
            "PageValues": [10.0, 30.0, 0.0],
            "SpecialDay": [0.0, 0.0, 0.6],
            "Month": ["May", "June", "Mar"],
            "OperatingSystems": [1, 2, 1],
            "Browser": [2, 1, 3],
            "Region": [1, 1, 2],
            "TrafficType": [1, 2, 3],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor"],
            "Weekend": [False, True, False],
            "Converted": [1, 0, 0],
        }
    )


def test_split_xy_returns_features_target():
    df = _sample()
    feature_cols = [c for c in df.columns if c != "Converted"]
    loader = DataLoader(Path("ignored.csv"), "Converted", feature_cols)
    X, y = loader.split_xy(df)
    assert X.shape == (3, 17)
    assert list(y) == [1, 0, 0]


def test_validation_missing_target():
    feature_cols = ["A", "B"]
    loader = DataLoader(Path("ignored.csv"), "Converted", feature_cols)
    bad = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(DataValidationError):
        loader.split_xy(bad)


def test_feature_engineer_adds_columns():
    from src.features.engineer import FeatureEngineer

    eng = FeatureEngineer()
    df = _sample().drop(columns=["Converted"])
    out = eng.fit_transform(df)
    for c in FeatureEngineer.ENGINEERED:
        assert c in out.columns, f"Missing engineered feature {c}"
    expected = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    assert (out["EngagementScore"].values == expected.values).all()


def test_loader_loads_real_dataset():
    repo = Path(__file__).resolve().parents[1]
    loader = DataLoader(
        repo / "dataset" / "ecommerce_sessions.csv",
        "Converted",
        [
            c
            for c in pd.read_csv(repo / "dataset" / "ecommerce_sessions.csv", nrows=1).columns
            if c != "Converted"
        ],
    )
    df = loader.load()
    assert df.shape == (12000, 18)
    assert set(df["Converted"].unique()).issubset({0, 1})
