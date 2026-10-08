"""Pydantic request/response schemas (PRD §4.7)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionFeatures(BaseModel):
    Administrative: int = Field(..., ge=0)
    Administrative_Duration: float = Field(..., ge=0.0)
    Informational: int = Field(..., ge=0)
    Informational_Duration: float = Field(..., ge=0.0)
    ProductRelated: int = Field(..., ge=0)
    ProductRelated_Duration: float = Field(..., ge=0.0)
    BounceRates: float = Field(..., ge=0.0, le=1.0)
    ExitRates: float = Field(..., ge=0.0, le=1.0)
    PageValues: float = Field(..., ge=0.0)
    SpecialDay: float = Field(..., ge=0.0, le=1.0)
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


class PredictRequest(BaseModel):
    session: SessionFeatures


class Contributor(BaseModel):
    feature: str
    direction: str
    contribution: float
    value: str | None = None


class PredictResponse(BaseModel):
    prediction: int
    conversion_probability: float
    confidence: str
    decision_threshold: float
    top_contributors: list[Contributor]
    model_name: str = ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_loaded: bool
    model_name: str
    threshold: float
    version: str
