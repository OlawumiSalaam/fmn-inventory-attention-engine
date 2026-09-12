"""Pydantic response contracts for the FMN Inventory Attention Engine API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskState = Literal["Critical", "Watch", "Healthy", "Overstock"]
AttentionPriority = Literal["P1", "P2", "P3", "P4"]


class RiskDriverResponse(BaseModel):
    """Structured evidence explaining why an SKU received its assessment."""

    model_config = ConfigDict(extra="forbid")

    driver_type: str
    label: str
    value: float | str | None = None
    unit: str | None = None
    evidence: str | None = None


class SkuAssessmentResponse(BaseModel):
    """Public API representation of a deterministic SKU assessment."""

    model_config = ConfigDict(extra="ignore")

    sku_id: str
    date: str
    category: str
    sku_type: str
    current_stock: float
    lead_time_days: int
    planning_horizon_days: int
    lead_time_range_days: str | None = None
    forecast_daily_demand: float
    forecast_lead_time_demand: float
    uncertainty_daily: float
    safety_buffer: float
    coverage_days: float
    expected_delivery_date: str | None = None
    expected_receipt_qty: float = 0.0
    cadence_days: float | None = None
    risk_state: RiskState
    days_to_projected_stockout: float | None = None
    projected_unmet_units: float
    minimum_projected_stock: float
    drivers: list[RiskDriverResponse] = Field(default_factory=list)
    recommendation: str
    attention_priority: AttentionPriority
    attention_score: float
    data_quality_flags: list[str] = Field(default_factory=list)
    forecast_method: str
    attention_rank: int | None = None


class AttentionSummaryResponse(BaseModel):
    """Current counts used by the Attention Center header."""

    total_skus: int
    critical: int
    watch: int
    overstock: int
    healthy: int
    attention_skus: int


class AttentionResponse(BaseModel):
    """Ranked attention queue response."""

    as_of_date: str
    summary: AttentionSummaryResponse
    items: list[SkuAssessmentResponse]


class ProjectionPoint(BaseModel):
    """One day in the deterministic projected inventory trajectory."""

    date: str
    projected_stock: float
    expected_receipt: float


class SkuProjectionResponse(BaseModel):
    """Inventory projection returned for SKU drilldown."""

    sku_id: str
    as_of_date: str
    horizon_days: int
    points: list[ProjectionPoint]


class ForecastValidationResponse(BaseModel):
    """Forecast model comparison evidence."""

    results: list[dict[str, Any]]


class RiskValidationResponse(BaseModel):
    """Risk model comparison evidence."""

    results: list[dict[str, Any]]


class ValidationResponse(BaseModel):
    """Validation evidence exposed to the product validation page."""

    forecast: ForecastValidationResponse
    risk: RiskValidationResponse


class HealthResponse(BaseModel):
    """API health response."""

    status: Literal["ok"]
    service: str
    data_as_of: str
