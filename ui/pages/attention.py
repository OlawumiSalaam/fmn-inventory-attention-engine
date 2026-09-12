"""Attention Center page for the FMN Inventory Attention Engine."""

from __future__ import annotations

from typing import Any

import dash_ag_grid as dag
import requests
from dash import Input, Output, callback, dcc, html

from ui.api_client import api_client


RISK_OPTIONS = ["Needs attention", "Critical", "Watch", "Overstock", "All SKUs"]
SKU_TYPE_OPTIONS = ["All", "new", "established"]


def _option_list(values: list[str]) -> list[dict[str, str]]:
    """Convert filter values to Dash dropdown options."""
    return [{"label": value, "value": value} for value in values]


def _metric_card(label: str, value: str | int, key: str) -> html.Div:
    """Build one compact summary metric card."""
    return html.Div(
        [html.Div(label, className="metric-label"), html.Div(str(value), className="metric-value")],
        className=f"metric-card metric-{key}",
    )


def _format_stock(value: Any) -> str:
    """Format stock quantities for planner friendly display."""
    if value is None:
        return "—"
    return f"{float(value):,.0f}"


def _format_days(value: Any) -> str:
    """Format coverage and stockout day values."""
    if value is None:
        return "—"
    return f"{float(value):.1f}d"


def _format_shortage(value: Any) -> str:
    """Format projected shortage quantities."""
    if value is None or float(value) <= 0:
        return "—"
    return f"{float(value):,.0f}"


def _delivery(row: dict[str, Any]) -> str:
    """Return a concise expected delivery label."""
    value = row.get("expected_delivery_date")
    return value[:10] if value else "Not confirmed"


def _driver_evidence(driver: dict[str, Any]) -> str:
    """Return concise planner facing evidence from one structured risk driver."""
    evidence = driver.get("evidence")
    if evidence:
        return str(evidence).rstrip(".")
    label = driver.get("label")
    return str(label or "Review required")


def _reason(row: dict[str, Any]) -> str:
    """Build a concise row level explanation from structured assessment evidence."""
    risk = str(row.get("risk_state") or "")
    drivers = row.get("drivers") or []

    if risk == "Overstock":
        coverage = row.get("coverage_days")
        lead = row.get("lead_time_days")
        minimum = row.get("minimum_projected_stock")
        if coverage is not None and lead is not None and minimum is not None:
            return f"{float(coverage):.1f}d cover vs {float(lead):.0f}d lead time; {float(minimum):,.0f} units projected at minimum"

    by_label = {str(driver.get("label") or ""): _driver_evidence(driver) for driver in drivers}

    if risk == "Critical":
        labels = ["Projected shortage", "Low stock", "Short stock coverage", "Expected delivery timing", "Limited SKU history"]
    elif risk == "Watch":
        labels = ["Short stock coverage", "Expected delivery timing", "Limited SKU history"]
    else:
        labels = ["Low stock", "Limited SKU history", "Lead time inconsistency"]

    preferred = [by_label[label] for label in labels if label in by_label]
    if preferred:
        return "; ".join(preferred[:2])

    return "Review required"


def _prepare_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map API assessments into the compact Attention Center table contract."""
    rows: list[dict[str, Any]] = []
    for row in items:
        rows.append(
            {
                "priority": row.get("attention_priority", "—"),
                "sku_id": row.get("sku_id", "—"),
                "category": row.get("category", "—"),
                "sku_type": row.get("sku_type", "—"),
                "risk_state": row.get("risk_state", "—"),
                "current_stock_display": _format_stock(row.get("current_stock")),
                "coverage_display": (
                    f"{_format_days(row.get('coverage_days'))} / "
                    f"{row.get('lead_time_days', '—')}d"
                ),
                "delivery": _delivery(row),
                "expected_delivery_date": row.get("expected_delivery_date"),
                "shortage_display": (
                    f"{float(row.get('projected_unmet_units', 0)):,.0f} short"
                    if row.get("risk_state") in {"Critical", "Watch"} and float(row.get("projected_unmet_units", 0) or 0) > 0
                    else (
                        "0 stock"
                        if float(row.get("current_stock", 0) or 0) <= 0
                        else (
                            "Protected"
                            if row.get("risk_state") == "Watch" and row.get("expected_delivery_date")
                            else (
                                f"{float(row.get('minimum_projected_stock', 0)):,.0f} excess"
                                if row.get("risk_state") == "Overstock" and float(row.get("minimum_projected_stock", 0) or 0) > 0
                                else "—"
                            )
                        )
                    )
                ),
                "coverage_days": row.get("coverage_days"),
                "lead_time_days": row.get("lead_time_days"),
                "minimum_projected_stock": row.get("minimum_projected_stock"),
                "reason": _reason(row),
            }
        )
    return rows


COLUMN_DEFS = [
    {"field": "priority", "headerName": "Priority", "width": 82, "pinned": "left"},
    {"field": "sku_id", "headerName": "SKU", "width": 105, "pinned": "left"},
    {"field": "category", "headerName": "Category", "width": 115},
    {"field": "risk_state", "headerName": "Risk", "width": 105},
    {"field": "current_stock_display", "headerName": "Stock", "width": 100, "type": "rightAligned"},
    {
        "field": "coverage_display",
        "headerName": "Cover / lead",
        "width": 125,
        "headerTooltip": "Current stock cover / lead time",
    },
    {"field": "delivery", "headerName": "Delivery", "width": 125},
   {
    "field": "shortage_display",
    "headerName": "Projected impact",
    "width": 135,
    "type": "rightAligned",
    "headerTooltip": "Projected shortage, protection, zero stock, or excess inventory.",
    },
    {
        "field": "reason",
        "headerName": "Why",
        "minWidth": 360,
        "flex": 1,
        "wrapText": True,
        "autoHeight": True,
        "cellClass": "why-cell",
    },
]


def layout() -> html.Div:
    """Build the Attention Center layout."""
    return html.Div(
        [
            dcc.Store(id="attention-data"),
            dcc.Store(id="selected-attention-row"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Inventory Attention", className="page-eyebrow"),
                            html.H1("Attention Center", className="page-title"),
                            html.P(
                                "Which SKUs require attention before replenishment can arrive?",
                                className="page-subtitle",
                            ),
                        ],
                        className="page-heading",
                    ),
                    html.Div(id="data-as-of", className="data-as-of"),
                ],
                className="page-header",
            ),
            html.Div(id="metric-strip", className="metric-strip"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Attention Queue", className="section-title"),
                            html.P("Prioritised exceptions requiring planner review.", className="section-subtitle"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.Label("Risk", className="filter-label"), dcc.Dropdown(id="risk-filter", options=_option_list(RISK_OPTIONS), value="Needs attention", clearable=False)],
                                className="filter-control",
                            ),
                            html.Div(
                                [html.Label("Category", className="filter-label"), dcc.Dropdown(id="category-filter", options=[{"label": "All", "value": "All"}], value="All", clearable=False)],
                                className="filter-control",
                            ),
                            html.Div(
                                [html.Label("SKU type", className="filter-label"), dcc.Dropdown(id="sku-type-filter", options=_option_list(SKU_TYPE_OPTIONS), value="All", clearable=False)],
                                className="filter-control",
                            ),
                            html.Div(
                                [html.Label("Search SKU", className="filter-label"), dcc.Input(id="sku-search", type="search", placeholder="e.g. SKU-2001", className="search-input")],
                                className="filter-control filter-search",
                            ),
                        ],
                        className="filter-row",
                    ),
                ],
                className="queue-header",
            ),
            html.Div(id="queue-status", className="queue-status"),
            dag.AgGrid(
                id="attention-grid",
                rowData=[],
                columnDefs=COLUMN_DEFS,
                defaultColDef={
                    "sortable": True,
                    "resizable": True,
                    "filter": True,
                    "minWidth": 90,
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 10,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "rowSelection": "single",
                    "getRowId": {"function": "params.data.sku_id"},
                },
                className="attention-grid",
            ),
            html.Div(id="selection-panel", className="selection-panel"),
        ],
        className="page-content",
    )


@callback(
    Output("metric-strip", "children"),
    Output("data-as-of", "children"),
    Output("category-filter", "options"),
    Output("attention-grid", "rowData"),
    Output("queue-status", "children"),
    Input("risk-filter", "value"),
    Input("category-filter", "value"),
    Input("sku-type-filter", "value"),
    Input("sku-search", "value"),
)
def update_attention_queue(
    risk: str,
    category: str,
    sku_type: str,
    search: str | None,
) -> tuple[list[html.Div], str, list[dict[str, str]], list[dict[str, Any]], str]:
    """Fetch the attention queue and update filters and table rows."""
    try:
        payload = api_client.get_attention(
            risk_state=risk if risk in {"Critical", "Watch", "Overstock"} else None,
            category=None if category in (None, "All") else category,
            sku_type=None if sku_type in (None, "All") else sku_type,
        )
    except requests.RequestException as exc:
        return [], "API unavailable", [], [], f"Could not reach FastAPI: {exc}"

    summary = payload.get("summary", {})
    items = payload.get("items", [])

    # The default queue is the attention population, while "All SKUs"
    # deliberately exposes the complete 28 SKU fleet.
    if risk == "Needs attention":
        items = [item for item in items if item.get("risk_state") != "Healthy"]

    categories = sorted({str(item.get("category")) for item in items if item.get("category")})
    options = [{"label": "All", "value": "All"}] + [{"label": value, "value": value} for value in categories]

    if search:
        needle = search.strip().lower()
        items = [item for item in items if needle in str(item.get("sku_id", "")).lower()]

    rows = _prepare_rows(items)
    cards = [
        _metric_card("Total SKUs", summary.get("total_skus", 0), "total"),
        _metric_card("Critical", summary.get("critical", 0), "critical"),
        _metric_card("Watch", summary.get("watch", 0), "watch"),
        _metric_card("Overstock", summary.get("overstock", 0), "overstock"),
        _metric_card("Need Attention", summary.get("attention_skus", 0), "attention"),
    ]
    if risk == "Needs attention":
        status = f"Showing {len(rows)} SKU{'s' if len(rows) != 1 else ''} requiring attention."
    elif risk == "All SKUs":
        status = f"Showing all {len(rows)} SKU{'s' if len(rows) != 1 else ''} in the current view."
    else:
        status = f"Showing {len(rows)} SKU{'s' if len(rows) != 1 else ''} in the current view."
    return cards, f"Data as of {payload.get('as_of_date', 'unknown')}", options, rows, status


@callback(
    Output("selection-panel", "children"),
    Input("attention-grid", "selectedRows"),
)
def show_selected_sku(rows: list[dict[str, Any]] | None) -> html.Div:
    """Load and display the full deterministic assessment for the selected SKU."""
    if not rows:
        return html.Div(
            [
                html.Strong("Select a SKU to investigate"),
                html.Span(
                    " Click a row in the attention queue to inspect its evidence.",
                    className="selection-help",
                ),
            ],
            className="selection-empty",
        )

    row = rows[0]
    sku_id = str(row.get("sku_id", ""))

    if not sku_id:
        return html.Div(
            [
                html.Strong("Unable to load SKU"),
                html.Span(
                    " The selected row does not contain a valid SKU identifier.",
                    className="selection-help",
                ),
            ],
            className="selection-empty",
        )

    try:
        assessment = api_client.get_sku(sku_id)
    except requests.RequestException:
        return html.Div(
            [
                html.Strong(f"{sku_id}"),
                html.P(
                    "The SKU assessment could not be loaded. "
                    "Please check that the API is running and try again."
                ),
            ],
            className="selection-card selection-error",
        )

    risk = str(assessment.get("risk_state") or "Unknown")
    priority = str(assessment.get("attention_priority") or "—")
    sku_type = str(assessment.get("sku_type") or "—")

    current_stock = _format_stock(assessment.get("current_stock"))
    coverage = _format_days(assessment.get("coverage_days"))
    lead_time = assessment.get("lead_time_days")
    lead_display = f"{float(lead_time):.0f}d" if lead_time is not None else "—"

    delivery = _delivery(assessment)

    projected_shortage = assessment.get("projected_unmet_units")
    if projected_shortage is not None and float(projected_shortage) > 0:
        impact = f"{float(projected_shortage):,.0f} units short"
    elif risk == "Overstock":
        minimum_stock = assessment.get("minimum_projected_stock")
        impact = (
            f"{float(minimum_stock):,.0f} units excess"
            if minimum_stock is not None and float(minimum_stock) > 0
            else "Excess inventory"
        )
    elif risk == "Watch":
        impact = "Projected gap protected"
    elif float(assessment.get("current_stock", 0) or 0) <= 0:
        impact = "0 stock"
    else:
        impact = "—"

    drivers = assessment.get("drivers") or []

    reason_items = []
    for driver in drivers[:3]:
        label = str(driver.get("label") or "Review required")
        evidence = _driver_evidence(driver)

        reason_items.append(
            html.Li(
                [
                    html.Strong(f"{label}: "),
                    html.Span(evidence),
                ]
            )
        )

    if not reason_items:
        reason_items = [html.Li("Review required")]

    recommendation = str(
        assessment.get("recommendation")
        or "Review the SKU assessment and underlying replenishment assumptions."
    )

    quality_flags = assessment.get("data_quality_flags") or []

    badges = [
        html.Span(
            sku_type.replace("_", " ").title(),
            className="selection-badge",
        )
    ]

    if quality_flags:
        badges.append(
            html.Span(
                "Data quality flag",
                className="selection-badge selection-badge-warning",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Selected SKU", className="selection-eyebrow"),
                    html.H3(sku_id, className="selection-title"),
                    html.Div(badges, className="selection-badges"),
                ],
                className="selection-heading",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Risk", className="selection-label"),
                            html.Strong(risk),
                        ],
                        className="selection-item",
                    ),
                    html.Div(
                        [
                            html.Span("Priority", className="selection-label"),
                            html.Strong(priority),
                        ],
                        className="selection-item",
                    ),
                    html.Div(
                        [
                            html.Span("Stock", className="selection-label"),
                            html.Strong(current_stock),
                        ],
                        className="selection-item",
                    ),
                    html.Div(
                        [
                            html.Span("Coverage", className="selection-label"),
                            html.Strong(coverage),
                        ],
                        className="selection-item",
                    ),
                    html.Div(
                        [
                            html.Span("Lead time", className="selection-label"),
                            html.Strong(lead_display),
                        ],
                        className="selection-item",
                    ),
                    html.Div(
                        [
                            html.Span("Expected delivery", className="selection-label"),
                            html.Strong(delivery),
                        ],
                        className="selection-item",
                    ),
                ],
                className="selection-grid",
            ),

            html.Div(
                [
                    html.Strong("Projected impact"),
                    html.P(impact),
                ],
                className="selection-reason",
            ),

            html.Div(
                [
                    html.Strong("Why this needs attention"),
                    html.Ul(reason_items),
                ],
                className="selection-reason",
            ),

            html.Div(
                [
                    html.Strong("What to investigate"),
                    html.P(recommendation),
                ],
                className="selection-reason selection-recommendation",
            ),
        ],
        className="selection-card",
    )