"""Ask the Data page for grounded Supply Chain questions."""

from __future__ import annotations

import requests
from dash import ALL, Input, Output, State, callback, dcc, html
from ui.api_client import api_client


EXAMPLE_QUESTIONS = [
    "Which SKUs currently need the most attention?",
    "Which SKUs are currently Critical?",
    "Which SKUs are overstocked?",
    "Why is SKU-1000 Critical?",
    "What are the main data quality concerns?",
]


def _example_buttons() -> html.Div:
    """Build clickable example question buttons."""
    return html.Div(
        [
            html.Div(
                "Try asking:",
                className="question-examples-label",
            ),
            html.Div(
                [
                    html.Button(
                        question,
                        id={
                            "type": "qa-example",
                            "index": index,
                        },
                        className="question-example-button",
                    )
                    for index, question in enumerate(
                        EXAMPLE_QUESTIONS
                    )
                ],
                className="question-examples",
            ),
        ],
        className="question-examples-container",
    )


def _answer_card(
    response: dict,
) -> html.Div:
    """Build the grounded answer card."""
    answer = response.get("answer")

    if not answer:
        return html.Div(
            [
                html.H3(
                    "Answer",
                    className="section-title",
                ),
                html.P(
                    (
                        "No answer was returned. "
                        "Please try another question."
                    )
                ),
            ],
            className="qa-answer-card",
        )

    grounding_passed = bool(
        response.get("grounding_passed")
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

    tool_calls = response.get("tool_calls") or []

    tool_names = [
        str(tool.get("name"))
        for tool in tool_calls
        if tool.get("name")
    ]

    children = [
        html.Div(
            [
                html.H3(
                    "Answer",
                    className="section-title",
                ),
                html.Span(
                    grounding_label,
                    className=grounding_class,
                ),
            ],
            className="ai-explanation-header",
        ),
        html.P(
            str(answer),
            className="qa-answer-text",
        ),
    ]

    if tool_names:
        children.append(
            html.Div(
                [
                    html.Span(
                        "Data used: ",
                        className="qa-tool-label",
                    ),
                    html.Span(
                        ", ".join(tool_names),
                        className="qa-tool-value",
                    ),
                ],
                className="qa-tools-used",
            )
        )

    children.append(
        html.P(
            (
                "Answers are generated from approved inventory "
                "assessment tools. The underlying analytics remain "
                "the source of the decision."
            ),
            className="ai-explanation-note",
        )
    )

    return html.Div(
        children,
        className="qa-answer-card",
    )


def _loading_card() -> html.Div:
    """Build the temporary answer loading state."""
    return html.Div(
        [
            html.H3(
                "Answer",
                className="section-title",
            ),
            html.P(
                "Checking the inventory assessment..."
            ),
        ],
        className="qa-answer-card",
    )


def layout() -> html.Div:
    """Build the Ask the Data page."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "Decision Support",
                        className="page-eyebrow",
                    ),
                    html.H1(
                        "Ask the Data",
                        className="page-title",
                    ),
                    html.P(
                        (
                            "Ask questions about inventory risk, "
                            "demand, and which SKUs need attention. "
                            "Answers are grounded in the assessment data."
                        ),
                        className="page-subtitle",
                    ),
                ],
                className="page-heading",
            ),
            html.Div(
                [
                    html.Label(
                        "Your question",
                        className="filter-label",
                    ),
                    dcc.Textarea(
                        id="qa-question",
                        placeholder=(
                            "For example: Which SKUs currently "
                            "need the most attention?"
                        ),
                        className="qa-input",
                        maxLength=1000,
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Ask",
                                id="qa-submit",
                                className="qa-submit-button",
                                n_clicks=0,
                            )
                        ],
                        className="qa-submit-row",
                    ),
                    _example_buttons(),
                ],
                className="qa-question-card",
            ),
            dcc.Loading(
                id="qa-loading",
                type="default",
                children=html.Div(
                    id="qa-answer",
                ),
            ),
        ],
        className="page-content",
    )


@callback(
    Output(
        "qa-question",
        "value",
    ),
    Input(
        {
            "type": "qa-example",
            "index": ALL,
        },
        "n_clicks",
    ),
    State(
        {
            "type": "qa-example",
            "index": ALL,
        },
        "children",
    ),
    prevent_initial_call=True,
)
def select_example_question(
    clicks: list[int],
    questions: list[str],
) -> str:
    """Populate the question box from a selected example."""
    from dash import ctx

    triggered = ctx.triggered_id

    if not triggered:
        return ""

    index = triggered.get("index")

    if index is None or index >= len(questions):
        return ""

    return str(questions[index])


@callback(
    Output(
        "qa-answer",
        "children",
    ),
    Input(
        "qa-submit",
        "n_clicks",
    ),
    State(
        "qa-question",
        "value",
    ),
    prevent_initial_call=True,
)
def answer_question(
    n_clicks: int,
    question: str | None,
) -> html.Div:
    """Submit a question and render the grounded answer."""
    if not n_clicks or not question or not question.strip():
        return html.Div()

    try:
        response = api_client.ask_data(
            question.strip()
        )

    except requests.RequestException as exc:
        return html.Div(
            [
                html.H3(
                    "Answer",
                    className="section-title",
                ),
                html.P(
                    (
                        "The question could not be answered "
                        f"because the data service is unavailable: {exc}"
                    )
                ),
            ],
            className="qa-answer-card qa-error",
        )

    return _answer_card(response)