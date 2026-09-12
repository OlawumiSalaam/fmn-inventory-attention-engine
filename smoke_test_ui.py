"""Smoke test for the Dash Attention Center implementation."""

from ui.app import app


def main() -> None:
    """Validate that the Dash application imports and exposes a server."""
    assert app.server is not None
    print("Dash UI import OK")
    print(f"Routes: {len(app.server.url_map._rules)}")


if __name__ == "__main__":
    main()
