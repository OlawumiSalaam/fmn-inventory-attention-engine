"""Dash application entry point for the FMN Inventory Attention Engine."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from ui.layout import app_shell
from ui.pages.attention import layout as attention_layout

app = Dash(
    __name__,
    title="FMN Inventory Attention Engine",
    suppress_callback_exceptions=True,
)


def placeholder_page(title: str, description: str) -> html.Div:
    """Build a temporary page placeholder for later implementation milestones."""
    return html.Div(
        [
            html.Div("FMN Inventory Attention Engine", className="page-eyebrow"),
            html.H1(title, className="page-title"),
            html.P(description, className="page-subtitle"),
            html.Div("This workspace will be implemented in the next milestone.", className="placeholder-card"),
        ],
        className="page-content",
    )


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-container"),
    ]
)


@app.callback(
    Output("page-container", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str | None) -> html.Div:
    """Render the requested workspace inside the persistent application shell."""
    if pathname in (None, "/", ""):
        page = attention_layout()
    elif pathname == "/sku":
        page = placeholder_page("SKU Analysis", "Investigate the evidence behind an individual SKU assessment.")
    elif pathname == "/validation":
        page = placeholder_page("Validation", "Review forecast and risk decision validation evidence.")
    elif pathname == "/ask":
        page = placeholder_page("Ask the Data", "Ask grounded questions about the current inventory position.")
    else:
        page = placeholder_page("Page not found", "The requested workspace does not exist.")
    return app_shell(page)


server = app.server


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
