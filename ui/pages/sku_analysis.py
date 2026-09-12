"""SKU Analysis page for the FMN Inventory Attention Engine."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import requests
from dash import Input, Output, callback, dcc, html

from ui.api_client import api_client


def _format_number(value: Any) -> str:
    """Format a numeric value for planner friendly display."""
    if value is None:
        return "—"
    return f"{float(value):,.0f}"


def _format_days(value: Any) -> str:
    """Format a duration in days."""
    if value is None:
        return "—"
    return f"{float(value):.1f}d"


def _format_lead_time(value: Any) -> str:
    """Format lead time as a whole number of days."""
    if value is None:
        return "—"
    return f"{float(value):.0f}d"


def _format_date(value: Any) -> str:
    """Format an ISO date for compact display."""
    if not value:
        return "Not confirmed"
    return str(value)[:10]


def _metric(label: str, value: str, key: str) -> html.Div:
    """Build one SKU analysis metric card."""
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
        className=f"metric-card metric-{key}",
    )


def _sku_options() -> list[dict[str, str]]:
    """Return all available SKU selector options."""
    try:
        payload = api_client.get_attention()
    except requests.RequestException:
        return []

    items = payload.get("items") or []

    sku_ids = sorted(
        {
            str(item.get("sku_id"))
            for item in items
            if item.get("sku_id")
        }
    )

    return [
        {
            "label": sku_id,
            "value": sku_id,
        }
        for sku_id in sku_ids
    ]


def _build_projection_chart(
    projection: dict[str, Any],
) -> go.Figure:
    """Build the projected inventory chart from deterministic API output."""
    points = projection.get("points") or []

    if not points:
        figure = go.Figure()

        figure.update_layout(
            title="Projected Inventory",
            xaxis_title="Date",
            yaxis_title="Projected stock",
            height=360,
            margin={"l": 55, "r": 25, "t": 55, "b": 45},
        )

        return figure

    dates = [
        str(point.get("date", ""))[:10]
        for point in points
    ]

    stock = [
        point.get("projected_stock")
        for point in points
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=stock,
            mode="lines+markers",
            name="Projected stock",
            hovertemplate=(
                "%{x}<br>"
                "Projected stock: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
        line_dash="dash",
        annotation_text="Stockout threshold",
        annotation_position="top left",
    )

    receipt_points = [
        point
        for point in points
        if point.get("expected_receipt")
    ]

    if receipt_points:
        receipt_dates = [
            str(point.get("date", ""))[:10]
            for point in receipt_points
        ]

        receipt_values = [
            point.get("projected_stock")
            for point in receipt_points
        ]

        figure.add_trace(
            go.Scatter(
                x=receipt_dates,
                y=receipt_values,
                mode="markers",
                name="Expected receipt",
                marker={
                    "size": 10,
                    "symbol": "diamond",
                },
                hovertemplate=(
                    "%{x}<br>"
                    "Expected receipt"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title="Projected Inventory",
        xaxis_title="Date",
        yaxis_title="Projected stock",
        hovermode="x unified",
        height=360,
        margin={
            "l": 55,
            "r": 25,
            "t": 55,
            "b": 45,
        },
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
    )

    return figure


def _driver_list(
    drivers: list[dict[str, Any]],
) -> html.Ul:
    """Build a planner facing list of deterministic risk drivers."""
    if not drivers:
        return html.Ul(
            [
                html.Li(
                    "No specific risk drivers were returned."
                )
            ]
        )

    items = []

    for driver in drivers[:4]:
        label = str(
            driver.get("label")
            or "Review required"
        )

        evidence = str(
            driver.get("evidence")
            or driver.get("description")
            or "Evidence unavailable"
        ).rstrip(".")

        items.append(
            html.Li(
                [
                    html.Strong(f"{label}: "),
                    html.Span(evidence),
                ]
            )
        )

    return html.Ul(items)


def _impact(
    assessment: dict[str, Any],
) -> str:
    """Format the projected inventory impact."""
    risk = str(
        assessment.get("risk_state")
        or ""
    )

    shortage = assessment.get(
        "projected_unmet_units"
    )

    if shortage is not None and float(shortage) > 0:
        return f"{float(shortage):,.0f} units short"

    if risk == "Overstock":
        minimum_stock = assessment.get(
            "minimum_projected_stock"
        )

        if (
            minimum_stock is not None
            and float(minimum_stock) > 0
        ):
            return f"{float(minimum_stock):,.0f} units excess"

        return "Excess inventory"

    if risk == "Watch":
        return "Projected gap protected"

    if float(
        assessment.get("current_stock", 0) or 0
    ) <= 0:
        return "0 stock"

    return "No projected shortage"


def _ai_explanation_card(
    explanation: dict[str, Any] | None,
) -> html.Div:
    """Build the planner facing AI explanation card."""
    if not explanation:
        return html.Div(
            [
                html.H3(
                    "AI explanation",
                    className="section-title",
                ),
                html.P(
                    (
                        "The plain language explanation is "
                        "currently unavailable."
                    )
                ),
            ],
            className="analysis-detail-card ai-explanation-card",
        )

    text = explanation.get("explanation")

    if not text:
        return html.Div(
            [
                html.H3(
                    "AI explanation",
                    className="section-title",
                ),
                html.P(
                    (
                        "The plain language explanation is "
                        "currently unavailable."
                    )
                ),
            ],
            className="analysis-detail-card ai-explanation-card",
        )

    grounding_passed = bool(
        explanation.get("grounding_passed")
    )

    grounding_label = (
        "Grounded in assessment data"
        if grounding_passed
        else "Grounding check unavailable"
    )

    grounding_class = (
        "ai-grounding-badge ai-grounding-passed"
        if grounding_passed
        else "ai-grounding-badge ai-grounding-warning"
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        "AI explanation",
                        className="section-title",
                    ),
                    html.Span(
                        grounding_label,
                        className=grounding_class,
                    ),
                ],
                className="ai-explanation-header",
            ),
            dcc.Markdown(
                str(text),
                className="ai-explanation-text",
            ),
            html.P(
                (
                    "AI explanation is generated from the "
                    "inventory assessment. The risk decision "
                    "comes from the underlying analytics."
                ),
                className="ai-explanation-note",
            ),
        ],
        className="analysis-detail-card ai-explanation-card",
    )


def _ai_loading_card() -> html.Div:
    """Build the temporary AI explanation loading state."""
    return html.Div(
        [
            html.H3(
                "AI explanation",
                className="section-title",
            ),
            html.P(
                "Generating a plain language explanation..."
            ),
        ],
        className="analysis-detail-card ai-explanation-card",
    )


def layout() -> html.Div:
    """Build the SKU Analysis page."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Inventory Investigation",
                                className="page-eyebrow",
                            ),
                            html.H1(
                                "SKU Analysis",
                                className="page-title",
                            ),
                            html.P(
                                (
                                    "Understand the inventory position, "
                                    "projected risk, and evidence for a SKU."
                                ),
                                className="page-subtitle",
                            ),
                        ],
                        className="page-heading",
                    ),
                ],
                className="page-header",
            ),
            html.Div(
                [
                    html.Label(
                        "Select SKU",
                        className="filter-label",
                    ),
                    dcc.Dropdown(
                        id="sku-analysis-selector",
                        options=_sku_options(),
                        placeholder="Select a SKU",
                        clearable=False,
                    ),
                ],
                className="sku-selector",
            ),
            html.Div(
                id="sku-analysis-content",
                children=html.Div(
                    [
                        html.Strong(
                            "Select a SKU to investigate"
                        ),
                        html.P(
                            (
                                "Choose a SKU above to view its "
                                "current position, projected "
                                "inventory path, and risk evidence."
                            )
                        ),
                    ],
                    className="selection-empty",
                ),
            ),
        ],
        className="page-content",
    )


@callback(
    Output(
        "sku-analysis-content",
        "children",
    ),
    Input(
        "sku-analysis-selector",
        "value",
    ),
)
def update_sku_analysis(
    sku_id: str | None,
) -> html.Div:
    """Render deterministic SKU assessment and projection."""
    if not sku_id:
        return html.Div(
            [
                html.Strong(
                    "Select a SKU to investigate"
                ),
                html.P(
                    (
                        "Choose a SKU above to view its "
                        "current position, projected inventory "
                        "path, and risk evidence."
                    )
                ),
            ],
            className="selection-empty",
        )

    try:
        assessment = api_client.get_sku(
            sku_id
        )

        projection = api_client.get_projection(
            sku_id,
            horizon_days=14,
        )

    except requests.RequestException as exc:
        return html.Div(
            [
                html.Strong(sku_id),
                html.P(
                    f"Unable to load SKU analysis: {exc}"
                ),
            ],
            className="selection-card selection-error",
        )

    risk = str(
        assessment.get("risk_state")
        or "Unknown"
    )

    priority = str(
        assessment.get("attention_priority")
        or "—"
    )

    sku_type = str(
        assessment.get("sku_type")
        or "—"
    ).replace(
        "_",
        " ",
    ).title()

    impact = _impact(assessment)

    drivers = assessment.get(
        "drivers"
    ) or []

    recommendation = str(
        assessment.get("recommendation")
        or (
            "Review the SKU assessment and "
            "replenishment assumptions."
        )
    )

    chart = _build_projection_chart(
        projection
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "SKU assessment",
                                className="selection-eyebrow",
                            ),
                            html.H2(
                                sku_id,
                                className="selection-title",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        f"{risk} · {priority}",
                                        className="selection-badge",
                                    ),
                                    html.Span(
                                        sku_type,
                                        className="selection-badge",
                                    ),
                                ],
                                className="selection-badges",
                            ),
                        ],
                        className="selection-heading",
                    ),
                    html.Div(
                        [
                            _metric(
                                "Current stock",
                                _format_number(
                                    assessment.get(
                                        "current_stock"
                                    )
                                ),
                                "total",
                            ),
                            _metric(
                                "Coverage",
                                _format_days(
                                    assessment.get(
                                        "coverage_days"
                                    )
                                ),
                                "attention",
                            ),
                            _metric(
                                "Lead time",
                                _format_lead_time(
                                    assessment.get(
                                        "lead_time_days"
                                    )
                                ),
                                "critical",
                            ),
                            _metric(
                                "Expected delivery",
                                _format_date(
                                    assessment.get(
                                        "expected_delivery_date"
                                    )
                                ),
                                "total",
                            ),
                            _metric(
                                "Forecast / day",
                                _format_number(
                                    assessment.get(
                                        "forecast_daily_demand"
                                    )
                                ),
                                "total",
                            ),
                            _metric(
                                "Projected impact",
                                impact,
                                (
                                    "critical"
                                    if risk == "Critical"
                                    else "total"
                                ),
                            ),
                        ],
                        className="metric-strip",
                    ),
                ],
                className="selection-card",
            ),

            html.Div(
                id="sku-analysis-ai",
                children=_ai_loading_card(),
            ),

            html.Div(
                [
                    dcc.Graph(
                        figure=chart,
                        config={
                            "displayModeBar": False
                        },
                    )
                ],
                className="analysis-chart-card",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Why this SKU is flagged",
                                className="section-title",
                            ),
                            _driver_list(drivers),
                        ],
                        className="analysis-detail-card",
                    ),
                    html.Div(
                        [
                            html.H3(
                                "What to investigate",
                                className="section-title",
                            ),
                            html.P(
                                recommendation
                            ),
                        ],
                        className="analysis-detail-card",
                    ),
                ],
                className="analysis-detail-grid",
            ),
        ]
    )


@callback(
    Output(
        "sku-analysis-ai",
        "children",
    ),
    Input(
        "sku-analysis-selector",
        "value",
    ),
)
def update_sku_ai_explanation(
    sku_id: str | None,
) -> html.Div:
    """Load the AI explanation independently from deterministic analysis."""
    if not sku_id:
        return html.Div()

    try:
        return _ai_explanation_card(
            api_client.get_sku_explanation(
                sku_id
            )
        )

    except requests.RequestException:
        return _ai_explanation_card(None)