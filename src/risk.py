"""Inventory risk and attention engine for FMN Project 1.

The module keeps inventory risk deterministic and separates analytical risk from
planner attention. Forecasts and expected replenishment are supplied to the
engine; the engine does not make autonomous purchasing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RiskState = Literal["Critical", "Watch", "Healthy", "Overstock"]


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for projected inventory risk assessment."""

    review_window_days: int = 3
    uncertainty_multiplier: float = 0.5
    overstock_multiplier: float = 1.5
    persistence_days: int = 1


def robust_daily_uncertainty(history: pd.Series, window: int = 28) -> float:
    """Estimate recent daily demand uncertainty using rolling standard deviation."""
    values = history.dropna().astype(float).tail(window)
    if len(values) < 7:
        return 0.0
    return float(values.std(ddof=1))


def expected_replenishment(
    history: pd.DataFrame,
    origin_date: pd.Timestamp,
) -> tuple[pd.Timestamp | None, float, float]:
    """Estimate the next delivery date, receipt quantity, and cadence from past receipts only."""
    receipts = history.loc[
        (history["date"] <= origin_date) & (history["units_received"] > 0),
        ["date", "units_received"],
    ].sort_values("date")

    if receipts.empty:
        return None, 0.0, float("nan")

    receipt_dates = receipts["date"].drop_duplicates().sort_values()
    gaps = receipt_dates.diff().dt.days.dropna()
    cadence = float(gaps.median()) if not gaps.empty else float("nan")
    quantity = float(receipts["units_received"].tail(5).median())

    if np.isnan(cadence) or cadence <= 0:
        return None, quantity, cadence

    last_receipt = receipt_dates.iloc[-1]
    next_date = last_receipt + pd.Timedelta(days=int(round(cadence)))
    while next_date <= origin_date:
        next_date += pd.Timedelta(days=int(round(cadence)))

    return next_date, quantity, cadence


def project_inventory(
    current_stock: float,
    daily_forecast: float,
    horizon_days: int,
    expected_delivery_date: pd.Timestamp | None,
    expected_receipt_qty: float,
    uncertainty_daily: float = 0.0,
    uncertainty_multiplier: float = 0.0,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Project daily inventory using forecast demand and expected replenishment."""
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()

    buffer_per_day = uncertainty_multiplier * uncertainty_daily
    rows: list[dict] = []
    stock = float(current_stock)

    for day in range(1, horizon_days + 1):
        date = start_date + pd.Timedelta(days=day)
        receipt = expected_receipt_qty if expected_delivery_date == date else 0.0
        stock = stock + receipt - (daily_forecast + buffer_per_day)
        rows.append(
            {
                "date": date,
                "projected_stock": stock,
                "expected_receipt": receipt,
            }
        )

    return pd.DataFrame(rows)


def classify_risk(
    current_stock: float,
    projected: pd.DataFrame,
    safety_buffer: float,
    cycle_demand: float,
    overstock_multiplier: float = 1.5,
    review_window_days: int = 3,
) -> RiskState:
    """Assign a risk state using both stockout timing and delivery protection.

    A stockout before an expected delivery is not automatically Critical. If the
    expected receipt arrives on or before the projected shortage point and brings
    inventory back above zero, the delivery protects the SKU and the near miss is
    classified as Watch. Lead-time coverage risk is also classified as Watch when
    current stock is below expected demand over the replenishment lead time, even
    when scheduled replenishment prevents a stockout. Critical is reserved for
    zero current stock or a shortage that remains after the expected replenishment,
    or for an unprotected stockout when no delivery is available.
    """
    if current_stock <= 0:
        return "Critical"

    if projected.empty:
        return "Healthy"

    receipt_mask = projected["expected_receipt"] > 0
    first_receipt_idx = projected.index[receipt_mask][0] if receipt_mask.any() else None

    if first_receipt_idx is not None:
        receipt_position = projected.index.get_loc(first_receipt_idx)
        pre_delivery = projected.iloc[:receipt_position]
        on_delivery = projected.loc[first_receipt_idx, "projected_stock"]

        if not pre_delivery.empty and (pre_delivery["projected_stock"] <= 0).any():
            # The key distinction is whether the expected receipt actually protects
            # the SKU for the remainder of the planning horizon. A receipt that
            # restores stock above zero and prevents a later shortage turns the
            # near miss into Watch; an insufficient receipt remains Critical.
            post_delivery = projected.iloc[receipt_position:]
            if on_delivery <= 0 or (post_delivery["projected_stock"] <= 0).any():
                return "Critical"
            return "Watch"
    elif (projected["projected_stock"] <= 0).any():
        return "Critical"

    # Lead-time coverage is a forward risk signal even when the scheduled receipt
    # prevents an actual stockout. It should surface the SKU as Watch rather than
    # allowing replenishment protection to make the underlying coverage risk appear
    # Healthy. Critical has already been handled above for unprotected shortages.
    if current_stock < cycle_demand:
        return "Watch"

    review = projected.head(min(len(projected), review_window_days))
    if (review["projected_stock"] <= safety_buffer).any():
        return "Watch"

    if projected.iloc[-1]["projected_stock"] > overstock_multiplier * cycle_demand + safety_buffer:
        return "Overstock"

    return "Healthy"


def assess_sku(
    history: pd.DataFrame,
    origin_date: pd.Timestamp,
    forecast_daily_demand: float,
    config: RiskConfig,
) -> dict:
    """Build a deterministic SKU risk assessment from information available at origin."""
    row = history.loc[history["date"] == origin_date].iloc[-1]
    current_stock = float(row["closing_stock"])
    lead_time = int(round(row["lead_time_days"]))
    uncertainty = robust_daily_uncertainty(history.loc[history["date"] <= origin_date, "units_sold"])
    expected_date, expected_qty, cadence = expected_replenishment(history, origin_date)

    cycle_demand = forecast_daily_demand * lead_time
    safety_buffer = config.uncertainty_multiplier * uncertainty * np.sqrt(max(lead_time, 1))
    horizon = max(lead_time + config.review_window_days, 7)

    projection = project_inventory(
        current_stock=current_stock,
        daily_forecast=forecast_daily_demand,
        horizon_days=horizon,
        expected_delivery_date=expected_date,
        expected_receipt_qty=expected_qty,
        uncertainty_daily=uncertainty,
        uncertainty_multiplier=config.uncertainty_multiplier,
        start_date=origin_date,
    )
    state = classify_risk(
        current_stock=current_stock,
        projected=projection,
        safety_buffer=safety_buffer,
        cycle_demand=cycle_demand,
        overstock_multiplier=config.overstock_multiplier,
        review_window_days=config.review_window_days,
    )

    positive = projection.loc[projection["projected_stock"] <= 0]
    days_to_stockout = int((positive.iloc[0]["date"] - origin_date).days) if not positive.empty else np.nan
    min_projected = float(projection["projected_stock"].min())

    # A negative balance before an expected receipt is a replenishment timing gap,
    # not necessarily residual unmet demand. The receipt is applied on its delivery
    # date in project_inventory, so shortage exposure is measured from the projected
    # inventory after all scheduled receipts have been applied.
    projected_unmet = max(0.0, -min_projected)
    if expected_date is not None:
        receipt_date = expected_date
        after_receipt = projection.loc[projection["date"] >= receipt_date, "projected_stock"]
        if not after_receipt.empty:
            projected_unmet = max(0.0, -float(after_receipt.min()))
    coverage_days = current_stock / forecast_daily_demand if forecast_daily_demand > 0 else np.inf

    drivers: list[str] = []
    if current_stock <= 0:
        drivers.append("Low stock")
    if coverage_days < lead_time:
        drivers.append("Short stock coverage")
    if forecast_daily_demand > history.loc[history["date"] <= origin_date, "units_sold"].tail(28).mean():
        drivers.append("High forecast demand")
    if uncertainty > 0 and safety_buffer > 0:
        drivers.append("Forecast uncertainty")
    if expected_date is not None:
        days_to_delivery = int((expected_date - origin_date).days)
        if days_to_stockout == days_to_stockout and days_to_stockout < days_to_delivery:
            drivers.append("Expected delivery timing")
    if state == "Overstock":
        drivers.append("Excess projected stock")
    if history.loc[history["date"] <= origin_date, "units_sold"].tail(7).mean() > history.loc[history["date"] <= origin_date, "units_sold"].tail(28).mean() * 1.10:
        drivers.append("Increasing demand")

    if state == "Critical":
        recommendation = "Review replenishment and consider expediting."
    elif state == "Watch":
        recommendation = "Monitor closely and investigate replenishment timing."
    elif state == "Overstock":
        recommendation = "Investigate excess inventory before the next replenishment."
    else:
        recommendation = "No immediate action indicated."

    return {
        "sku_id": row["sku_id"],
        "date": origin_date,
        "current_stock": current_stock,
        "lead_time_days": lead_time,
        "forecast_daily_demand": forecast_daily_demand,
        "forecast_lead_time_demand": cycle_demand,
        "uncertainty_daily": uncertainty,
        "safety_buffer": safety_buffer,
        "coverage_days": coverage_days,
        "expected_delivery_date": expected_date,
        "expected_receipt_qty": expected_qty,
        "cadence_days": cadence,
        "risk_state": state,
        "days_to_projected_stockout": days_to_stockout,
        "projected_unmet_units": projected_unmet,
        "minimum_projected_stock": min_projected,
        "drivers": drivers,
        "recommendation": recommendation,
    }
