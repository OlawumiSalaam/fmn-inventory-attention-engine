"""Tests for delivery aware inventory risk classification."""

import pandas as pd

from src.risk import classify_risk


def _projection(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build a minimal inventory projection for risk classification tests."""
    return pd.DataFrame(rows, columns=["date", "projected_stock", "expected_receipt"]).assign(
        date=lambda frame: pd.to_datetime(frame["date"])
    )


def test_protected_stockout_is_watch() -> None:
    """A stockout before a sufficient expected receipt is a Watch, not Critical."""
    projection = _projection(
        [
            ("2026-07-01", -10.0, 0.0),
            ("2026-07-02", 150.0, 200.0),
        ]
    )
    assert classify_risk(100.0, projection, 20.0, 300.0) == "Watch"


def test_insufficient_receipt_remains_critical() -> None:
    """A receipt that still leaves inventory negative remains Critical."""
    projection = _projection(
        [
            ("2026-07-01", -100.0, 0.0),
            ("2026-07-02", -50.0, 25.0),
        ]
    )
    assert classify_risk(100.0, projection, 20.0, 300.0) == "Critical"


def test_unprotected_stockout_is_critical() -> None:
    """A projected stockout without an expected receipt is Critical."""
    projection = _projection(
        [
            ("2026-07-01", -10.0, 0.0),
            ("2026-07-02", -50.0, 0.0),
        ]
    )
    assert classify_risk(100.0, projection, 20.0, 300.0) == "Critical"


def test_zero_current_stock_is_critical() -> None:
    """Zero current inventory is Critical regardless of future receipts."""
    projection = _projection(
        [("2026-07-01", 100.0, 200.0)]
    )
    assert classify_risk(0.0, projection, 20.0, 300.0) == "Critical"

def test_insufficient_lead_time_coverage_is_watch_when_delivery_protects() -> None:
    """Below lead-time demand coverage is Watch when the planned receipt protects stock."""
    projection = _projection(
        [
            ("2026-07-01", 20.0, 0.0),
            ("2026-07-02", 100.0, 100.0),
            ("2026-07-03", 60.0, 0.0),
        ]
    )
    # Current stock is below the 3-day cycle demand, but the projection remains
    # positive because the scheduled receipt protects the inventory position.
    assert classify_risk(100.0, projection, 10.0, 120.0) == "Watch"


def test_insufficient_lead_time_coverage_is_watch_without_projected_stockout() -> None:
    """Coverage risk remains visible as Watch even when no stockout is projected yet."""
    projection = _projection(
        [
            ("2026-07-01", 80.0, 0.0),
            ("2026-07-02", 60.0, 0.0),
            ("2026-07-03", 40.0, 0.0),
        ]
    )
    assert classify_risk(100.0, projection, 10.0, 120.0) == "Watch"

