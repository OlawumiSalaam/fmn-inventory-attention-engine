"""Application services for the FMN Inventory Attention Engine API.

The API layer reads the canonical analytical artifacts produced by the
pipeline. It does not duplicate forecasting or risk business logic.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.risk import project_inventory


class ArtifactStore:
    """Load and cache the small, deterministic artifacts used by the API."""

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialise the artifact store from the repository root."""
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.evaluation_dir = self.project_root / "artifacts" / "evaluation"
        self._assessments: pd.DataFrame | None = None

    def assessments(self) -> pd.DataFrame:
        """Return the canonical assessment table."""
        if self._assessments is None:
            path = self.evaluation_dir / "sku_assessments.csv"
            self._assessments = pd.read_csv(path)
        return self._assessments.copy()

    def summary(self) -> dict[str, Any]:
        """Return the persisted current attention summary."""
        path = self.evaluation_dir / "attention_summary.json"
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def forecast_validation(self) -> pd.DataFrame:
        """Return forecast model comparison results."""
        return pd.read_csv(self.evaluation_dir / "forecast_model_comparison.csv")

    def risk_validation(self) -> pd.DataFrame:
        """Return risk model comparison results."""
        return pd.read_csv(self.evaluation_dir / "risk_model_comparison.csv")


def _parse_list(value: Any) -> list[Any]:
    """Parse list-like values stored in CSV cells."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            return [text]


def _normalise_assessment(row: pd.Series) -> dict[str, Any]:
    """Convert a CSV assessment row into JSON-safe API data."""
    result = row.to_dict()
    result["drivers"] = _parse_list(result.get("drivers"))
    result["data_quality_flags"] = [str(x) for x in _parse_list(result.get("data_quality_flags"))]

    for key in result:
        if pd.isna(result[key]) if not isinstance(result[key], list) else False:
            result[key] = None

    if result.get("attention_rank") is not None:
        result["attention_rank"] = int(result["attention_rank"])

    return result


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    """Return assessments in the canonical planner attention order."""
    priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    ranked = df.copy()
    ranked["_priority_order"] = ranked["attention_priority"].map(priority_order).fillna(9)
    return ranked.sort_values(
        ["_priority_order", "attention_score", "sku_id"],
        ascending=[True, False, True],
    ).drop(columns="_priority_order")


def get_attention(
    store: ArtifactStore,
    risk_state: str | None = None,
    category: str | None = None,
    sku_type: str | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Return the filtered and ranked planner attention queue."""
    df = _ranked(store.assessments())
    if risk_state:
        df = df.loc[df["risk_state"].eq(risk_state)]
    if category:
        df = df.loc[df["category"].eq(category)]
    if sku_type:
        df = df.loc[df["sku_type"].eq(sku_type)]

    summary = store.summary()
    as_of_date = str(store.assessments()["date"].iloc[0]).split(" ")[0]
    return as_of_date, summary, [_normalise_assessment(row) for _, row in df.iterrows()]


def get_sku(store: ArtifactStore, sku_id: str) -> dict[str, Any] | None:
    """Return one SKU assessment by identifier."""
    df = store.assessments()
    matches = df.loc[df["sku_id"].eq(sku_id)]
    if matches.empty:
        return None
    return _normalise_assessment(matches.iloc[0])


def get_projection(store: ArtifactStore, sku_id: str) -> dict[str, Any] | None:
    """Build the deterministic projected inventory trajectory for a SKU."""
    assessment = get_sku(store, sku_id)
    if assessment is None:
        return None

    origin = pd.Timestamp(assessment["date"])
    expected_delivery = (
        pd.Timestamp(assessment["expected_delivery_date"])
        if assessment["expected_delivery_date"]
        else None
    )
    horizon = int(assessment["planning_horizon_days"])
    projection = project_inventory(
        current_stock=float(assessment["current_stock"]),
        daily_forecast=float(assessment["forecast_daily_demand"]),
        horizon_days=horizon,
        expected_delivery_date=expected_delivery,
        expected_receipt_qty=float(assessment["expected_receipt_qty"]),
        uncertainty_daily=float(assessment["uncertainty_daily"]),
        uncertainty_multiplier=0.5,
        start_date=origin,
    )
    points = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "projected_stock": float(row.projected_stock),
            "expected_receipt": float(row.expected_receipt),
        }
        for row in projection.itertuples(index=False)
    ]
    return {
        "sku_id": sku_id,
        "as_of_date": origin.strftime("%Y-%m-%d"),
        "horizon_days": horizon,
        "points": points,
    }
