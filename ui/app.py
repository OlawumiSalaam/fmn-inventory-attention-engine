"""Dash application entry point for the FMN Inventory Attention Engine."""

from __future__ import annotations
import os

from dash import (
    Dash,
    Input,
    Output,
    dcc,
    html,
)

from ui.layout import app_shell
from ui.pages.ask_data import layout as ask_data_layout
from ui.pages.attention import layout as attention_layout
from ui.pages.sku_analysis import layout as sku_analysis_layout
from ui.pages.validation import layout as validation_layout


app = Dash(
    __name__,
    title="FMN Inventory Attention Engine",
    suppress_callback_exceptions=True,
)


def placeholder_page(
    title: str,
    description: str,
) -> html.Div:
    """Build a temporary page placeholder."""
    return html.Div(
        [
            html.Div(
                title,
                className="page-eyebrow",
            ),
            html.H1(
                title,
                className="page-title",
            ),
            html.P(
                description,
                className="page-subtitle",
            ),
            html.Div(
                "This workspace will be implemented in the next milestone.",
                className="placeholder-card",
            ),
        ],
        className="page-content",
    )


app.layout = html.Div(
    [
        dcc.Location(
            id="url",
            refresh=False,
        ),
        html.Div(
            id="page-container",
        ),
    ]
)


@app.callback(
    Output(
        "page-container",
        "children",
    ),
    Input(
        "url",
        "pathname",
    ),
)
def render_page(
    pathname: str | None,
) -> html.Div:
    """Render the requested workspace inside the application shell."""
    if pathname in (
        None,
        "/",
        "",
    ):
        page = attention_layout()

    elif pathname == "/sku":
        page = sku_analysis_layout()

    elif pathname == "/validation":
        page = validation_layout()

    elif pathname == "/ask":
        page = ask_data_layout()

    else:
        page = placeholder_page(
            "Page not found",
            "The requested workspace does not exist.",
        )

    return app_shell(page)


server = app.server


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8050")),
    )