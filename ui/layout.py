"""Shared application shell for the FMN Inventory Attention Engine."""

from __future__ import annotations

from dash import dcc, html


def app_shell(page_content: html.Div) -> html.Div:
    """Build the persistent navigation shell around the active page."""
    return html.Div(
        [
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div("FMN", className="brand-mark"),
                            html.Div("Inventory Attention Engine", className="brand-name"),
                        ],
                        className="brand",
                    ),
                    html.Nav(
                        [
                            dcc.Link("Attention Center", href="/", className="nav-link active"),
                            dcc.Link("SKU Analysis", href="/sku", className="nav-link"),
                            dcc.Link("Validation", href="/validation", className="nav-link"),
                            dcc.Link("Ask the Data", href="/ask", className="nav-link"),
                        ],
                        className="nav",
                    ),
                    html.Div("Prototype v0.1", className="sidebar-footer"),
                ],
                className="sidebar",
            ),
            html.Main(page_content, className="main-content"),
        ],
        className="app-shell",
    )
