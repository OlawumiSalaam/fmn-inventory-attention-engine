"""Structured tools exposed to the FMN Inventory Attention Engine LLM.

The tools in this module provide bounded access to authoritative analytical
artifacts. They do not calculate forecasts or inventory risk. The existing
deterministic services remain the source of truth for those decisions.

The LLM can request evidence through these tools, but it cannot execute
arbitrary dataframe queries or modify analytical state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from api.services import ArtifactStore, get_attention, get_sku


def _clean_value(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into JSON-safe Python values."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_clean_value(item) for item in value]

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def _assessment_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded fields useful for planner questions."""
    fields = [
        "sku_id",
        "date",
        "category",
        "sku_type",
        "current_stock",
        "lead_time_days",
        "planning_horizon_days",
        "lead_time_range_days",
        "forecast_daily_demand",
        "forecast_lead_time_demand",
        "uncertainty_daily",
        "safety_buffer",
        "coverage_days",
        "expected_delivery_date",
        "expected_receipt_qty",
        "cadence_days",
        "risk_state",
        "days_to_projected_stockout",
        "projected_unmet_units",
        "minimum_projected_stock",
        "drivers",
        "recommendation",
        "attention_priority",
        "attention_score",
        "data_quality_flags",
        "forecast_method",
        "attention_rank",
    ]
    return {
        field: _clean_value(item.get(field))
        for field in fields
        if field in item
    }


def list_attention_skus(
    store: ArtifactStore,
    risk_state: str | None = None,
    category: str | None = None,
    sku_type: str | None = None,
) -> dict[str, Any]:
    """Return the ranked SKU attention queue with optional filters.

    Args:
        store: Application artifact store.
        risk_state: Optional risk state filter.
        category: Optional logical category filter.
        sku_type: Optional SKU type filter.

    Returns:
        Structured attention queue data suitable for LLM grounding.
    """
    as_of_date, summary, items = get_attention(
        store,
        risk_state=risk_state,
        category=category,
        sku_type=sku_type,
    )

    return {
        "as_of_date": as_of_date,
        "summary": _clean_value(summary),
        "items": [_assessment_summary(item) for item in items],
    }


def get_sku_details(
    store: ArtifactStore,
    sku_id: str,
) -> dict[str, Any]:
    """Return the authoritative deterministic assessment for one SKU.

    Args:
        store: Application artifact store.
        sku_id: SKU identifier.

    Returns:
        Structured SKU assessment or a not-found error.
    """
    assessment = get_sku(store, sku_id)

    if assessment is None:
        return {
            "sku_id": sku_id,
            "error": f"SKU {sku_id} was not found.",
        }

    return _assessment_summary(assessment)


def rank_skus(
    store: ArtifactStore,
    metric: str = "attention_score",
    limit: int = 10,
    descending: bool = True,
) -> dict[str, Any]:
    """Rank SKUs using one of the bounded analytical metrics.

    Supported metrics are:
        attention_score
        projected_unmet_units
        coverage_days
        current_stock
        forecast_daily_demand

    Args:
        store: Application artifact store.
        metric: Analytical field used for ranking.
        limit: Maximum number of SKUs returned.
        descending: Whether to rank highest values first.

    Returns:
        Ranked structured SKU records.
    """
    supported_metrics = {
        "attention_score",
        "projected_unmet_units",
        "coverage_days",
        "current_stock",
        "forecast_daily_demand",
    }

    if metric not in supported_metrics:
        return {
            "error": (
                f"Unsupported ranking metric '{metric}'. "
                f"Supported metrics: {sorted(supported_metrics)}."
            )
        }

    limit = max(1, min(int(limit), 28))

    df = store.assessments().copy()
    df = df.sort_values(
        metric,
        ascending=not descending,
        kind="stable",
    ).head(limit)

    fields = [
        "sku_id",
        "category",
        "sku_type",
        "risk_state",
        "attention_priority",
        "attention_score",
        "current_stock",
        "coverage_days",
        "forecast_daily_demand",
        "forecast_lead_time_demand",
        "projected_unmet_units",
        "minimum_projected_stock",
        "lead_time_days",
        "expected_delivery_date",
        "data_quality_flags",
    ]

    records = []
    for _, row in df[fields].iterrows():
        record = {
            field: _clean_value(row[field])
            for field in fields
        }
        records.append(record)

    return {
        "metric": metric,
        "descending": descending,
        "limit": limit,
        "items": records,
    }


def get_sku_history(
    store: ArtifactStore,
    sku_id: str,
    days: int = 14,
) -> dict[str, Any]:
    """Return recent observed demand and stock history for one SKU.

    The history is read from the canonical raw dataset and is limited to a
    small window so the LLM receives relevant evidence rather than the full
    dataset.

    Args:
        store: Application artifact store.
        sku_id: SKU identifier.
        days: Number of most recent observations to return.

    Returns:
        Recent SKU observations and basic history metadata.
    """
    days = max(1, min(int(days), 30))

    data_path = store.project_root / "data" / "raw" / "project1_supply_chain_demand.csv"

    if not data_path.exists():
        return {
            "sku_id": sku_id,
            "error": "The canonical raw dataset was not found.",
        }

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    sku_df = (
        df.loc[df["sku_id"].eq(sku_id)]
        .sort_values("date")
        .drop_duplicates()
        .tail(days)
    )

    if sku_df.empty:
        return {
            "sku_id": sku_id,
            "error": f"SKU {sku_id} was not found.",
        }

    columns = [
        "date",
        "sku_id",
        "category",
        "units_sold",
        "units_received",
        "closing_stock",
        "lead_time_days",
    ]

    records = []
    for _, row in sku_df[columns].iterrows():
        record = {
            field: _clean_value(
                row[field].strftime("%Y-%m-%d")
                if field == "date" and pd.notna(row[field])
                else row[field]
            )
            for field in columns
        }
        records.append(record)

    return {
        "sku_id": sku_id,
        "observations_returned": len(records),
        "history_start": records[0]["date"],
        "history_end": records[-1]["date"],
        "items": records,
    }


def get_data_quality(store: ArtifactStore) -> dict[str, Any]:
    """Return known data quality facts from the assessment dataset.

    This tool intentionally exposes known assessment limitations rather than
    asking the LLM to infer data quality from raw records.
    """
    df = store.assessments()

    quality_flags = df["data_quality_flags"].dropna().astype(str)

    flag_counts: dict[str, int] = {}

    for value in quality_flags:
        for flag in str(value).split("|"):
            flag = flag.strip()
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "assessment_skus": int(len(df)),
        "data_as_of": str(df["date"].iloc[0]).split(" ")[0],
        "sku_types": sorted(
            str(value)
            for value in df["sku_type"].dropna().unique()
        ),
        "data_quality_flag_counts": flag_counts,
        "known_limitations": [
            "The assessment uses the provided project dataset.",
            "The dataset covers approximately six months.",
            "New SKUs have limited history.",
            "Observed lead time can vary for some SKUs.",
            "The assessment does not contain unit cost, so overstock impact is expressed in units rather than currency.",
            "The dataset has controlled or simulated characteristics, so real-world sales may be censored during stockouts.",
        ],
    }


def get_model_performance(store: ArtifactStore) -> dict[str, Any]:
    """Return persisted forecast and risk validation evidence."""
    forecast = store.forecast_validation()
    risk = store.risk_validation()

    return {
        "forecast_validation": [
            _clean_value(record)
            for record in forecast.to_dict(orient="records")
        ],
        "risk_validation": [
            _clean_value(record)
            for record in risk.to_dict(orient="records")
        ],
    }


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI compatible function schemas for the bounded tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_attention_skus",
                "description": (
                    "List SKUs in the planner attention queue. Use this when "
                    "the user asks which SKUs need attention, are critical, "
                    "are at risk, or belong to a particular category or SKU type."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "risk_state": {
                            "type": "string",
                            "enum": [
                                "Critical",
                                "Watch",
                                "Healthy",
                                "Overstock",
                            ],
                        },
                        "category": {
                            "type": "string",
                        },
                        "sku_type": {
                            "type": "string",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_sku_details",
                "description": (
                    "Get the authoritative current assessment for one SKU, "
                    "including stock, coverage, forecast, risk, drivers, "
                    "expected delivery and recommendation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku_id": {
                            "type": "string",
                            "description": "SKU identifier such as SKU-1010.",
                        },
                    },
                    "required": ["sku_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rank_skus",
                "description": (
                    "Rank SKUs using a bounded analytical metric. Use this "
                    "when the user asks for highest, lowest, worst or best SKUs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": [
                                "attention_score",
                                "projected_unmet_units",
                                "coverage_days",
                                "current_stock",
                                "forecast_daily_demand",
                            ],
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 28,
                        },
                        "descending": {
                            "type": "boolean",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_sku_history",
                "description": (
                    "Return recent observed demand, receipts, closing stock "
                    "and lead time history for a specific SKU."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku_id": {
                            "type": "string",
                        },
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                        },
                    },
                    "required": ["sku_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_data_quality",
                "description": (
                    "Return known data quality facts and limitations of the "
                    "current assessment dataset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_model_performance",
                "description": (
                    "Return persisted forecast and risk model validation "
                    "results. Use when the user asks about model performance, "
                    "validation, recall, precision or forecast accuracy."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_tool_functions(
    store: ArtifactStore,
) -> dict[str, Any]:
    """Return tool names mapped to callable functions for the LLM."""
    return {
        "list_attention_skus": lambda **kwargs: list_attention_skus(
            store,
            **kwargs,
        ),
        "get_sku_details": lambda **kwargs: get_sku_details(
            store,
            **kwargs,
        ),
        "rank_skus": lambda **kwargs: rank_skus(
            store,
            **kwargs,
        ),
        "get_sku_history": lambda **kwargs: get_sku_history(
            store,
            **kwargs,
        ),
        "get_data_quality": lambda **kwargs: get_data_quality(
            store,
            **kwargs,
        ),
        "get_model_performance": lambda **kwargs: get_model_performance(
            store,
            **kwargs,
        ),
    }