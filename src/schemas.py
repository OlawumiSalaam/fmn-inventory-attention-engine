"""Shared data contracts for the FMN Inventory Attention Engine.

The contracts in this module define the stable interface between deterministic
analytics, the application, and the grounded AI layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RiskState = Literal["Critical", "Watch", "Healthy", "Overstock"]
AttentionPriority = Literal["P1", "P2", "P3", "P4"]


@dataclass(frozen=True)
class RiskDriver:
    """A structured explanation driver produced by deterministic analytics."""

    driver_type: str
    label: str
    value: float | str | None = None
    unit: str | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class SkuAssessment:
    """Canonical SKU level assessment consumed by UI and AI components."""

    sku_id: str
    date: str
    category: str
    sku_type: str
    current_stock: float
    lead_time_days: int
    planning_horizon_days: int
    lead_time_range_days: str | None
    forecast_daily_demand: float
    forecast_lead_time_demand: float
    uncertainty_daily: float
    safety_buffer: float
    coverage_days: float
    expected_delivery_date: str | None
    expected_receipt_qty: float
    cadence_days: float | None
    risk_state: RiskState
    days_to_projected_stockout: float | None
    projected_unmet_units: float
    minimum_projected_stock: float
    drivers: list[RiskDriver] = field(default_factory=list)
    recommendation: str = ""
    attention_priority: AttentionPriority = "P4"
    attention_score: float = 0.0
    data_quality_flags: list[str] = field(default_factory=list)
    forecast_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON friendly dictionary representation."""
        return asdict(self)
