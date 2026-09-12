"""Tests for the shared assessment contract and attention ranking."""

import pandas as pd

from src.assess import attention_summary, rank_attention


def test_critical_items_rank_before_healthy_items():
    """Critical exceptions should appear before Healthy SKUs."""
    df = pd.DataFrame(
        {
            "sku_id": ["SKU-2", "SKU-1", "SKU-3"],
            "risk_state": ["Healthy", "Critical", "Overstock"],
            "attention_priority": ["P4", "P1", "P3"],
            "attention_score": [0.0, 1000.0, 500.0],
        }
    )
    ranked = rank_attention(df)
    assert ranked.iloc[0]["sku_id"] == "SKU-1"
    assert ranked.iloc[-1]["sku_id"] == "SKU-2"


def test_attention_summary_counts_states():
    """The summary should expose planner facing exception counts."""
    df = pd.DataFrame(
        {
            "sku_id": ["1", "2", "3", "4"],
            "risk_state": ["Critical", "Watch", "Overstock", "Healthy"],
        }
    )
    summary = attention_summary(df)
    assert summary["attention_skus"] == 3
    assert summary["critical"] == 1
    assert summary["watch"] == 1
    assert summary["overstock"] == 1
    assert summary["healthy"] == 1


def test_structured_drivers_include_concrete_evidence() -> None:
    """Driver construction should expose planner usable evidence, not labels alone."""
    from src.assess import _driver_objects

    row = pd.Series(
        {
            "drivers": [
                "Low stock",
                "Limited SKU history",
                "Lead time inconsistency",
                "Projected shortage",
            ],
            "current_stock": 0,
            "history_days": 12,
            "lead_time_range_days": "3–14 days",
            "projected_unmet_units": 4320,
            "coverage_days": 0,
            "lead_time_days": 14,
            "forecast_daily_demand": 283.25,
            "uncertainty_daily": 50.6,
        }
    )
    drivers = _driver_objects(row)
    by_label = {driver.label: driver for driver in drivers}

    assert by_label["Low stock"].value == 0
    assert by_label["Limited SKU history"].value == 12
    assert by_label["Lead time inconsistency"].value == "3–14 days"
    assert by_label["Projected shortage"].value == 4320
    assert all(driver.evidence for driver in drivers)
