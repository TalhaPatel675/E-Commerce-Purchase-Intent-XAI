"""Tests for the preprocessing + prediction pipeline (PRD §5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _df():
    return pd.DataFrame(
        {
            "Administrative": [1, 2, 0, 4, 1],
            "Informational": [0, 1, 1, 0, 0],
            "ProductRelated": [10, 20, 5, 19, 6],
            "BounceRates": [0.01, 0.02, 0.03, 0.04, 0.005],
            "ExitRates": [0.05, 0.04, 0.06, 0.07, 0.01],
            "PageValues": [10.0, 30.0, 0.0, 60.0, 0.0],
            "SpecialDay": [0.0, 0.0, 0.6, 0.0, 0.4],
            "Month": ["May", "June", "Mar", "Nov", "Dec"],
            "OperatingSystems": [1, 2, 1, 3, 2],
            "Browser": [2, 1, 3, 2, 1],
            "Region": [1, 1, 2, 1, 1],
            "TrafficType": [1, 2, 3, 4, 5],
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
                "Returning_Visitor",
                "Returning_Visitor",
                "Other",
            ],
            "Weekend": [False, True, False, True, False],
        }
    )


def test_pipeline_shape_matches_columns():
    from src.preprocessing.preprocess import PreprocessingPipeline

    df = _df()
    pipe = PreprocessingPipeline(
        categorical_columns=[
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ],
        numeric_columns=[
            "Administrative",
            "Informational",
            "ProductRelated",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
        ],
    )
    pipe.fit(df)
    out = pipe.transform(df.iloc[[0]])
    assert out.shape == (1, len(pipe.get_feature_names()))


def test_unknown_category_handled():
    from src.preprocessing.preprocess import PreprocessingPipeline

    pipe = PreprocessingPipeline(
        categorical_columns=[
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
            "Weekend",
        ],
        numeric_columns=[
            "Administrative",
            "Informational",
            "ProductRelated",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "SpecialDay",
        ],
    )
    pipe.fit(_df())
    df_new = _df().iloc[[0]].copy()
    df_new["Month"] = "NotAMonth"
    df_new["OperatingSystems"] = 99
    df_new["VisitorType"] = "AlienVisitor"
    out = pipe.transform(df_new)
    assert out.shape[0] == 1
    assert np.isfinite(out).all()


def test_api_health_smoke(monkeypatch):
    """Start-up smoke: when artifacts are present the GET /health route returns ok."""
    repo = Path(__file__).resolve().parents[1]
    model_p = repo / "models" / "best_model.joblib"
    pipe_p = repo / "models" / "preprocessing_pipeline.joblib"
    thr_p = repo / "models" / "decision_threshold.json"
    if not (model_p.exists() and pipe_p.exists() and thr_p.exists()):
        pytest.skip("Artifacts missing — run scripts\\train.bat first.")
    import sys

    from fastapi.testclient import TestClient

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["pipeline_loaded"] is True


def test_api_predict_smoke():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "models" / "best_model.joblib").exists():
        pytest.skip("Artifacts missing — run scripts\\train.bat first.")
    import sys

    from fastapi.testclient import TestClient

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.api.main import app

    sample = {
        "Administrative": 3,
        "Administrative_Duration": 80.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 27,
        "ProductRelated_Duration": 640.0,
        "BounceRates": 0.015,
        "ExitRates": 0.035,
        "PageValues": 36.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    r = TestClient(app).post("/predict", json={"session": sample})
    assert r.status_code == 200, r.text
    out = r.json()
    assert "prediction" in out
    assert "conversion_probability" in out
    assert "top_contributors" in out
    assert isinstance(out["top_contributors"], list)


def test_confidence_probability_bands():
    from src.api.main import _confidence

    assert _confidence(0.50) == "low"
    assert _confidence(0.70) == "medium"
    assert _confidence(0.95) == "high"
    assert _confidence(0.05) == "high"
