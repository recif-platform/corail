"""Tests for request ID middleware."""

from httpx import ASGITransport, AsyncClient

from corail.main import app


async def test_response_includes_request_id() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/healthz")
    assert "x-request-id" in response.headers
    assert response.headers["x-request-id"].startswith("req_")


async def test_custom_request_id_propagated() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/healthz", headers={"X-Request-ID": "custom-id-123"})
    assert response.headers["x-request-id"] == "custom-id-123"
