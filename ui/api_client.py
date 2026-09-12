"""HTTP client for the FMN Inventory Attention Engine FastAPI service."""

from __future__ import annotations

import os
from typing import Any

import requests


class ApiClient:
    """Small synchronous client for the FMN FastAPI service."""

    def __init__(self, base_url: str | None = None, timeout: int = 60) -> None:
        """Initialise the client with the configured API base URL."""
        self.base_url = (
            base_url or os.getenv("FMN_API_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request and return the decoded JSON response."""
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform a POST request and return the decoded JSON response."""
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_summary(self) -> dict[str, Any]:
        """Return the current SKU risk summary."""
        return self._get("/api/v1/summary")

    def get_attention(
        self,
        risk_state: str | None = None,
        category: str | None = None,
        sku_type: str | None = None,
    ) -> dict[str, Any]:
        """Return the ranked attention queue with optional filters."""
        params = {
            key: value
            for key, value in {
                "risk_state": risk_state,
                "category": category,
                "sku_type": sku_type,
            }.items()
            if value
        }

        return self._get("/api/v1/attention", params=params)

    def get_sku(self, sku_id: str) -> dict[str, Any]:
        """Fetch the complete deterministic assessment for one SKU."""
        return self._get(f"/api/v1/skus/{sku_id}")

    def get_projection(
        self,
        sku_id: str,
        horizon_days: int = 14,
    ) -> dict[str, Any]:
        """Fetch the deterministic projected inventory path for one SKU."""
        return self._get(
            f"/api/v1/skus/{sku_id}/projection",
            params={"horizon_days": horizon_days},
        )

    def get_sku_explanation(self, sku_id: str) -> dict[str, Any]:
        """Fetch a grounded plain language explanation for one SKU."""
        return self._get(f"/api/v1/skus/{sku_id}/explanation")

    def ask_data(self, question: str) -> dict[str, Any]:
        """Ask a grounded question about the inventory assessment data."""
        return self._post(
            "/api/v1/ask",
            {"question": question},
        )

    def get_validation(self) -> dict[str, Any]:
        """Fetch forecast and risk validation results."""
        return self._get("/api/v1/validation")


api_client = ApiClient()