"""FastAPI application for the FMN Inventory Attention Engine."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from api.models import (
    AttentionResponse,
    ForecastValidationResponse,
    HealthResponse,
    RiskValidationResponse,
    SkuAssessmentResponse,
    SkuProjectionResponse,
    ValidationResponse,
)
from api.services import ArtifactStore, get_attention, get_projection, get_sku

app = FastAPI(
    title="FMN Inventory Attention Engine API",
    description=(
        "Deterministic inventory assessment services for the FMN Project 1 "
        "decision support application. Analytics decide risk; the API exposes "
        "evidence for the planner and downstream UI."
    ),
    version="0.1.0",
)

store = ArtifactStore()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return service health and the assessment data date."""
    assessments = store.assessments()
    data_as_of = str(assessments["date"].iloc[0]).split(" ")[0]
    return HealthResponse(status="ok", service="fmn-inventory-attention-api", data_as_of=data_as_of)


@app.get("/api/v1/summary", response_model=dict, tags=["attention"])
def summary() -> dict:
    """Return current SKU risk counts."""
    return store.summary()


@app.get("/api/v1/attention", response_model=AttentionResponse, tags=["attention"])
def attention(
    risk_state: Literal["Critical", "Watch", "Healthy", "Overstock"] | None = Query(default=None),
    category: str | None = Query(default=None),
    sku_type: str | None = Query(default=None),
) -> AttentionResponse:
    """Return the ranked attention queue with optional filters."""
    as_of_date, summary_data, items = get_attention(store, risk_state, category, sku_type)
    return AttentionResponse(as_of_date=as_of_date, summary=summary_data, items=items)


@app.get("/api/v1/skus/{sku_id}", response_model=SkuAssessmentResponse, tags=["skus"])
def sku_details(sku_id: str) -> SkuAssessmentResponse:
    """Return the current deterministic assessment for one SKU."""
    assessment = get_sku(store, sku_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"SKU {sku_id} was not found.")
    return SkuAssessmentResponse.model_validate(assessment)


@app.get("/api/v1/skus/{sku_id}/projection", response_model=SkuProjectionResponse, tags=["skus"])
def sku_projection(sku_id: str) -> SkuProjectionResponse:
    """Return the projected inventory trajectory for one SKU."""
    projection = get_projection(store, sku_id)
    if projection is None:
        raise HTTPException(status_code=404, detail=f"SKU {sku_id} was not found.")
    return SkuProjectionResponse.model_validate(projection)


@app.get("/api/v1/validation", response_model=ValidationResponse, tags=["validation"])
def validation() -> ValidationResponse:
    """Return forecast and risk validation evidence."""
    forecast = store.forecast_validation().where(lambda frame: frame.notna(), None)
    risk = store.risk_validation().where(lambda frame: frame.notna(), None)
    return ValidationResponse(
        forecast=ForecastValidationResponse(results=forecast.to_dict(orient="records")),
        risk=RiskValidationResponse(results=risk.to_dict(orient="records")),
    )
