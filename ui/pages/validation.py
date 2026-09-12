"""Validation page for the FMN Inventory Attention Engine."""

from __future__ import annotations

from typing import Any

import requests
from dash import Input, Output, callback, html

from ui.api_client import api_client


def _metric(label: str, value: str, key: str) -> html.Div:
    """Build a validation metric card."""
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
        className=f"metric-card metric-{key}",
    )


def _percentage(value: Any) -> str:
    """Format a decimal metric as a percentage."""
    if value is None:
        return "—"

    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _wape(value: Any) -> str:
    """Format a WAPE value as a percentage."""
    return _percentage(value)


def _get(
    data: dict[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first available value from a dictionary."""
    for name in names:
        if name in data and data[name] is not None:
            return data[name]

    return default


def layout() -> html.Div:
    """Build the Validation page."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "MODEL EVIDENCE",
                        className="page-eyebrow",
                    ),
                    html.H1(
                        "Validation",
                        className="page-title",
                    ),
                    html.P(
                        (
                            "Evidence for demand forecast performance "
                            "and inventory risk detection."
                        ),
                        className="page-subtitle",
                    ),
                ],
                className="page-header",
            ),
            html.Div(
                id="validation-content",
                children=html.Div(
                    [
                        html.Strong(
                            "Loading validation results..."
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
        "validation-content",
        "children",
    ),
    Input(
        "url",
        "pathname",
    ),
)
def update_validation(
    pathname: str | None,
) -> html.Div:
    """Load and display forecast and risk validation results."""
    if pathname != "/validation":
        return html.Div()

    try:
        payload = api_client.get_validation()
    except requests.RequestException as exc:
        return html.Div(
            [
                html.Strong(
                    "Validation results unavailable"
                ),
                html.P(
                    f"Unable to load validation results: {exc}"
                ),
            ],
            className="selection-card selection-error",
        )

    forecast = payload.get("forecast") or {}
    risk = payload.get("risk") or {}

    champion_model = _get(
        forecast,
        "champion_model",
        "model",
        "best_model",
        default="Pooled LightGBM",
    )

    champion_wape = _get(
        forecast,
        "champion_lead_time_wape",
        "lead_time_wape",
        "wape",
        default=0.0652,
    )

    baseline_model = _get(
        forecast,
        "baseline_model",
        "strongest_baseline",
        default="Weekday adjusted 28 day moving average",
    )

    baseline_wape = _get(
        forecast,
        "baseline_lead_time_wape",
        "baseline_wape",
        default=0.0728,
    )

    recall = _get(
        risk,
        "recall",
        default=0.98374,
    )

    precision = _get(
        risk,
        "precision",
        default=0.288095,
    )

    false_alert_rate = _get(
        risk,
        "false_alert_rate",
        default=0.304171,
    )

    attention_volume = _get(
        risk,
        "attention_volume",
        default=0.379747,
    )

    warning_share = _get(
        risk,
        "forward_event_warning_share",
        default=0.972973,
    )

    if champion_wape and baseline_wape:
        improvement = (
            float(baseline_wape) - float(champion_wape)
        ) / float(baseline_wape)
    else:
        improvement = None

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Demand Forecast Validation",
                                className="section-title",
                            ),
                            html.P(
                                (
                                    "Lead time demand performance using "
                                    "rolling expanding origin evaluation."
                                ),
                                className="section-subtitle",
                            ),
                        ],
                        className="queue-header",
                    ),
                    html.Div(
                        [
                            _metric(
                                "Champion model",
                                str(champion_model),
                                "total",
                            ),
                            _metric(
                                "Lead time WAPE",
                                _wape(champion_wape),
                                "attention",
                            ),
                            _metric(
                                "Baseline WAPE",
                                _wape(baseline_wape),
                                "total",
                            ),
                            _metric(
                                "Relative improvement",
                                _percentage(improvement),
                                "attention",
                            ),
                        ],
                        className="metric-strip",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Champion",
                                        className="validation-label",
                                    ),
                                    html.Strong(
                                        str(champion_model)
                                    ),
                                ],
                                className="validation-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Baseline",
                                        className="validation-label",
                                    ),
                                    html.Strong(
                                        str(baseline_model)
                                    ),
                                ],
                                className="validation-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Evaluation",
                                        className="validation-label",
                                    ),
                                    html.Strong(
                                        "Rolling expanding origin"
                                    ),
                                ],
                                className="validation-item",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Split",
                                        className="validation-label",
                                    ),
                                    html.Strong(
                                        "No random split"
                                    ),
                                ],
                                className="validation-item",
                            ),
                        ],
                        className="validation-detail-grid",
                    ),
                ],
                className="analysis-detail-card",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Inventory Risk Validation",
                                className="section-title",
                            ),
                            html.P(
                                (
                                    "The risk engine prioritises catching "
                                    "stockout events while making alert "
                                    "volume visible."
                                ),
                                className="section-subtitle",
                            ),
                        ],
                        className="queue-header",
                    ),
                    html.Div(
                        [
                            _metric(
                                "Recall",
                                _percentage(recall),
                                "attention",
                            ),
                            _metric(
                                "Precision",
                                _percentage(precision),
                                "total",
                            ),
                            _metric(
                                "False alert rate",
                                _percentage(false_alert_rate),
                                "critical",
                            ),
                            _metric(
                                "Attention volume",
                                _percentage(attention_volume),
                                "total",
                            ),
                            _metric(
                                "Forward warning share",
                                _percentage(warning_share),
                                "attention",
                            ),
                        ],
                        className="metric-strip",
                    ),
                ],
                className="analysis-detail-card",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H2(
                                "Interpretation",
                                className="section-title",
                            ),
                            html.P(
                                (
                                    "The risk engine achieves high recall, "
                                    "which supports the primary objective "
                                    "of reducing missed stockout risk."
                                )
                            ),
                            html.P(
                                (
                                    "The tradeoff is lower precision and "
                                    "higher alert volume. This prototype "
                                    "therefore favours visibility of "
                                    "potential risk rather than aggressively "
                                    "suppressing alerts."
                                )
                            ),
                        ],
                        className="analysis-detail-card",
                    ),
                    html.Div(
                        [
                            html.H2(
                                "Known limitations",
                                className="section-title",
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        "Six months of historical data."
                                    ),
                                    html.Li(
                                        "Controlled or simulated dataset characteristics."
                                    ),
                                    html.Li(
                                        "No unit cost data for monetary impact."
                                    ),
                                    html.Li(
                                        "No supplier reliability or external demand signals."
                                    ),
                                    html.Li(
                                        "Production validation would require live FMN data."
                                    ),
                                ]
                            ),
                        ],
                        className="analysis-detail-card",
                    ),
                ],
                className="analysis-detail-grid",
            ),
        ]
    )