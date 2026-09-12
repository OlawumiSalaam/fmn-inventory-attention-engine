"""Build canonical SKU assessments and rank planner attention.

This module converts deterministic risk outputs into a stable product contract.
It does not create forecasts, make purchasing decisions, or call an LLM.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import pandas as pd

from .schemas import RiskDriver, SkuAssessment


PRIORITY_BY_STATE = {
    "Critical": "P1",
    "Watch": "P2",
    "Overstock": "P3",
    "Healthy": "P4",
}


def _parse_drivers(value: Any) -> list[str]:
    """Parse the stored driver representation into a list of labels."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(value)] if str(value).strip() else []


def _priority_score(row: pd.Series) -> float:
    """Calculate a deterministic attention score within the risk state."""
    state = str(row["risk_state"])
    if state == "Critical":
        unmet = max(float(row.get("projected_unmet_units", 0) or 0), 0)
        days = row.get("days_to_projected_stockout")
        urgency = 1.0 if pd.isna(days) else 1.0 / max(float(days), 1.0)
        return unmet * 10.0 + urgency * 100.0
    if state == "Watch":
        days = row.get("days_to_projected_stockout")
        return 100.0 / max(float(days), 1.0) if not pd.isna(days) else 0.0
    if state == "Overstock":
        excess = max(float(row.get("minimum_projected_stock", 0) or 0), 0)
        return excess
    return 0.0


def _driver_objects(row: pd.Series) -> list[RiskDriver]:
    """Convert driver labels into structured driver objects with concrete evidence."""
    objects: list[RiskDriver] = []
    for label in _parse_drivers(row.get("drivers")):
        value = None
        unit = None
        evidence = None

        if label == "Low stock":
            value, unit = float(row["current_stock"]), "units"
            evidence = f"{value:,.0f} units in stock"
        elif label == "Projected shortage":
            value, unit = float(row["projected_unmet_units"]), "units"
            evidence = f"{value:,.0f} units short"
        elif label == "Short stock coverage":
            value, unit = float(row["coverage_days"]), "days"
            evidence = f"{value:.1f}d cover vs {float(row['lead_time_days']):.0f}d lead time"
        elif label == "High forecast demand":
            value, unit = float(row["forecast_daily_demand"]), "units/day"
            evidence = f"Forecast demand {value:,.1f} units/day"
        elif label == "Forecast uncertainty":
            value, unit = float(row["uncertainty_daily"]), "units/day"
            evidence = f"Demand variability {value:,.1f} units/day"
        elif label == "Expected delivery timing":
            value = row.get("expected_delivery_date")
            date_label = pd.Timestamp(value).strftime("%d %b") if pd.notna(value) else "unknown date"
            if str(row.get("risk_state")) == "Watch":
                evidence = f"Delivery {date_label} protects the projected stockout gap"
            else:
                evidence = f"Delivery {date_label} does not fully protect projected demand"
        elif label == "Increasing demand":
            evidence = "Recent demand is rising vs the trailing 28d level"
        elif label == "Excess projected stock":
            value, unit = float(row.get("minimum_projected_stock", 0)), "units"
            coverage = float(row.get("coverage_days", 0))
            lead = float(row.get("lead_time_days", 0))
            evidence = f"{coverage:.1f}d cover vs {lead:.0f}d lead time; {value:,.0f} units projected at minimum"
        elif label == "Lead time inconsistency":
            value = row.get("lead_time_range_days")
            unit = "range"
            evidence = f"Observed lead time range is {value}."
        elif label == "Limited SKU history":
            value = row.get("history_days")
            unit = "days" if value is not None and not pd.isna(value) else None
            evidence = (
                f"Only {float(value):.0f} days of SKU history are available."
                if value is not None and not pd.isna(value)
                else "The SKU has limited history and therefore higher uncertainty."
            )
        else:
            evidence = label

        objects.append(
            RiskDriver(
                label.lower().replace(" ", "_"),
                label,
                value,
                unit,
                evidence,
            )
        )
    return objects


def build_assessments(
    risk_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical assessment table from risk outputs and prepared data."""
    latest = prepared_df.sort_values("date").groupby("sku_id", as_index=False).tail(1)
    latest = latest[["sku_id", "category", "sku_status", "data_quality_flags"]]

    df = risk_df.merge(latest, on="sku_id", how="left")
    df["attention_priority"] = df["risk_state"].map(PRIORITY_BY_STATE).fillna("P4")
    df["attention_score"] = df.apply(_priority_score, axis=1)
    df["drivers_structured"] = df.apply(_driver_objects, axis=1)

    assessments: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        flags = _parse_drivers(row.get("data_quality_flags"))
        assessments.append(
            SkuAssessment(
                sku_id=str(row["sku_id"]),
                date=str(row["date"]),
                category=str(row["category"]),
                sku_type=str(row["sku_status"]),
                current_stock=float(row["current_stock"]),
                lead_time_days=int(row["lead_time_days"]),
                planning_horizon_days=int(row.get("planning_horizon_days", row["lead_time_days"])),
                lead_time_range_days=None if pd.isna(row.get("lead_time_range_days", pd.NA)) else str(row.get("lead_time_range_days")),
                forecast_daily_demand=float(row["forecast_daily_demand"]),
                forecast_lead_time_demand=float(row["forecast_lead_time_demand"]),
                uncertainty_daily=float(row["uncertainty_daily"]),
                safety_buffer=float(row["safety_buffer"]),
                coverage_days=float(row["coverage_days"]),
                expected_delivery_date=None if pd.isna(row["expected_delivery_date"]) else str(row["expected_delivery_date"]),
                expected_receipt_qty=float(row["expected_receipt_qty"]),
                cadence_days=None if pd.isna(row["cadence_days"]) else float(row["cadence_days"]),
                risk_state=str(row["risk_state"]),
                days_to_projected_stockout=None if pd.isna(row["days_to_projected_stockout"]) else float(row["days_to_projected_stockout"]),
                projected_unmet_units=float(row["projected_unmet_units"]),
                minimum_projected_stock=float(row["minimum_projected_stock"]),
                drivers=row["drivers_structured"],
                recommendation=str(row["recommendation"]),
                attention_priority=str(row["attention_priority"]),
                attention_score=float(row["attention_score"]),
                data_quality_flags=flags,
                forecast_method=str(row.get("forecast_method", "")),
            ).to_dict()
        )

    result = pd.DataFrame(assessments)
    result["drivers"] = result["drivers"].apply(lambda items: [item for item in items])
    return result


def rank_attention(assessments: pd.DataFrame) -> pd.DataFrame:
    """Return exceptions first, ranked by business urgency rather than anomaly size."""
    ranked = assessments.copy()
    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    ranked["_priority_order"] = ranked["attention_priority"].map(priority_order).fillna(9)
    ranked = ranked.sort_values(
        ["_priority_order", "attention_score", "sku_id"],
        ascending=[True, False, True],
    ).drop(columns="_priority_order")
    ranked["attention_rank"] = range(1, len(ranked) + 1)
    return ranked


def attention_summary(ranked: pd.DataFrame) -> dict[str, Any]:
    """Summarise the current attention queue for the application header."""
    counts = ranked["risk_state"].value_counts().to_dict()
    return {
        "total_skus": int(len(ranked)),
        "critical": int(counts.get("Critical", 0)),
        "watch": int(counts.get("Watch", 0)),
        "overstock": int(counts.get("Overstock", 0)),
        "healthy": int(counts.get("Healthy", 0)),
        "attention_skus": int((ranked["risk_state"] != "Healthy").sum()),
    }
