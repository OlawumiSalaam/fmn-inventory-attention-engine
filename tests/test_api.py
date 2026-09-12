"""Tests for the FMN Inventory Attention Engine API contracts."""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health() -> None:
    """The health endpoint should report a healthy service."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_attention_returns_all_skus_without_filter() -> None:
    """The attention endpoint should expose the complete canonical assessment set."""
    response = client.get("/api/v1/attention")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_skus"] == 28
    assert len(body["items"]) == 28


def test_attention_filter() -> None:
    """Risk state filtering should return only matching assessments."""
    response = client.get("/api/v1/attention", params={"risk_state": "Critical"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 6
    assert all(item["risk_state"] == "Critical" for item in items)


def test_sku_details() -> None:
    """A known SKU should return its canonical assessment."""
    response = client.get("/api/v1/skus/SKU-1017")
    assert response.status_code == 200
    assert response.json()["sku_id"] == "SKU-1017"


def test_unknown_sku_returns_404() -> None:
    """An unknown SKU should produce a clear not found response."""
    response = client.get("/api/v1/skus/SKU-9999")
    assert response.status_code == 404


def test_projection() -> None:
    """A known SKU should return a deterministic projection trajectory."""
    response = client.get("/api/v1/skus/SKU-1017/projection")
    assert response.status_code == 200
    body = response.json()
    assert body["sku_id"] == "SKU-1017"
    assert len(body["points"]) == body["horizon_days"]
