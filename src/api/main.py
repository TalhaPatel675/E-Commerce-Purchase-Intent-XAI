"""FastAPI real-time prediction service (PRD §4.7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.schemas import (
    Contributor,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.config.settings import resolve_config
from src.features.engineer import FeatureEngineer
from src.preprocessing.preprocess import PreprocessingPipeline
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)
configure_logging(level="INFO")

API_VERSION = "1.0.0"
app = FastAPI(
    title="Purchase Intent Prediction API",
    description="Predicts whether an e-commerce session will convert (purchase). Returns prediction, probability, confidence and SHAP top contributors.",
    version=API_VERSION,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load once at startup
cfg = resolve_config(ROOT)
MODEL = None
PIPELINE: PreprocessingPipeline | None = None
THRESHOLD: float = 0.5
MODEL_NAME = "unknown"


def _load_artifacts() -> None:
    global MODEL, PIPELINE, THRESHOLD, MODEL_NAME
    model_path = Path(cfg.paths.model_path)
    pipeline_path = Path(cfg.paths.pipeline_path)
    threshold_path = Path(cfg.paths.threshold_path)
    if not model_path.exists() or not pipeline_path.exists() or not threshold_path.exists():
        log.error("Required artifact(s) missing. Run scripts\\train.bat first.")
        return
    MODEL = joblib.load(model_path)
    PIPELINE = PreprocessingPipeline.load(pipeline_path)
    info = json.loads(threshold_path.read_text(encoding="utf-8"))
    THRESHOLD = float(info["threshold"])
    MODEL_NAME = str(info.get("selected_model", "unknown"))
    log.info("Loaded model=%s threshold=%.3f", MODEL_NAME, THRESHOLD)


_load_artifacts()


def _confidence(p: float) -> str:
    """Map probability to confidence using symmetric probability bands.

    Config values are probability cut-offs (e.g. 0.60 / 0.85), so confidence
    is high near either endpoint and low around the decision boundary.
    """
    high = cfg.api.high_confidence_threshold
    low = cfg.api.low_confidence_threshold
    distance = max(p, 1.0 - p)
    if distance >= high:
        return "high"
    if distance >= low:
        return "medium"
    return "low"


def _explain_row(X_row: np.ndarray, feature_names: list[str]) -> list[float]:
    """SHAP local explanation for one row.

    Uses TreeExplainer when the loaded estimator is tree-based (PRD §4.6 hint)."""
    try:
        import shap

        cls = type(MODEL).__name__
        if any(n in cls for n in ("XGB", "LGBM", "CatBoost", "RandomForest", "DecisionTree")):
            explainer = shap.TreeExplainer(MODEL)
            sv = explainer.shap_values(X_row)
        else:
            explainer = shap.Explainer(MODEL, feature_names=feature_names)
            sv = explainer.shap_values(X_row)

        if isinstance(sv, list):
            arr = np.asarray(sv[1])
        else:
            arr = np.asarray(sv)
        if arr.ndim == 2:
            return arr[0].tolist()
        return arr.tolist()
    except Exception as e:
        log.warning("SHAP local explanation failed: %s", e)
        return [0.0] * len(feature_names)


def _top_contributors(
    features: dict[str, Any], shap_row: np.ndarray, feature_names: list[str], top_k: int
) -> list[Contributor]:
    pairs = list(zip(feature_names, shap_row.tolist()))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    raw_lookup = {k.lower(): v for k, v in features.items()}
    out: list[Contributor] = []
    for fname, sv in pairs[:top_k]:
        base_name = fname.split("_", 1)[0]
        user_value = None
        for k, v in raw_lookup.items():
            if base_name.lower().startswith(k.split("_")[0].lower()):
                user_value = f"{k}={v}"
                break
        out.append(
            Contributor(
                feature=fname,
                direction=(
                    "increases_purchase_likelihood" if sv > 0 else "decreases_purchase_likelihood"
                ),
                contribution=float(sv),
                value=user_value,
            )
        )
    return out


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        pipeline_loaded=PIPELINE is not None,
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None or PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    session = req.session.model_dump()
    df = pd.DataFrame([session])

    eng = FeatureEngineer()
    df = eng.fit_transform(df)

    num_cols = [
        c for c in (cfg.data.numeric_columns + FeatureEngineer.ENGINEERED) if c in df.columns
    ]
    cat_cols = [c for c in cfg.data.categorical_columns if c in df.columns]

    ordered = pd.DataFrame({col: df[col] for col in (cat_cols + num_cols)})
    X = PIPELINE.transform(ordered)
    proba = float(MODEL.predict_proba(X)[:, 1][0])
    decision = int(proba >= THRESHOLD)

    feature_names = PIPELINE.get_feature_names()
    shap_row = _explain_row(X, feature_names=feature_names)
    contributors = _top_contributors(
        session,
        np.asarray(shap_row),
        feature_names,
        cfg.api.top_k_contributors,
    )
    log.info(
        "predict proba=%.3f decision=%d contributors=%d model=%s",
        proba,
        decision,
        len(contributors),
        MODEL_NAME,
    )
    return PredictResponse(
        prediction=decision,
        conversion_probability=proba,
        confidence=_confidence(proba),
        decision_threshold=THRESHOLD,
        top_contributors=contributors,
        model_name=MODEL_NAME,
    )


@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {
        "service": "purchase-intent-prediction",
        "version": API_VERSION,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "openapi_docs": "GET /docs",
            "openapi_spec": "GET /openapi.json",
        },
    }
