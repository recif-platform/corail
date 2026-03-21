"""Tests for the REST chat endpoint with SSE streaming."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from corail.api.errors import register_error_status
from corail.api.rest import AgentNotFoundError
from corail.cache.memory import MemoryCache
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig
from corail.core.pipeline import Pipeline
from corail.main import app
from corail.models.stub import StubModel
from corail.strategies.simple import SimpleStrategy


@pytest.fixture(autouse=True)
async def _setup_app() -> None:
    """Set up app state for tests (ASGITransport skips lifespan)."""
    model = StubModel()
    strategy = SimpleStrategy(model=model, system_prompt="You are a test assistant.")
    app.state.pipeline = Pipeline(strategy)
    register_error_status(AgentNotFoundError, 404, "agent-not-found")

    cache = MemoryCache()
    agent_cache = AgentConfigCache(cache)
    app.state.agent_cache = agent_cache
    app.state.cache = cache

    await agent_cache.set_agent(
        AgentConfig(
            id="ag_TESTAGENTSTUB00000000000",
            name="Test Agent",
            framework="adk",
            system_prompt="You are a test assistant.",
            model="stub-model",
            llm_provider="stub",
        )
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestChatEndpoint:
    async def test_chat_returns_sse_stream(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/agents/ag_TESTAGENTSTUB00000000000/chat",
            json={"input": "Hello world"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        assert "event: start" in body
        assert "event: complete" in body
        assert "Echo: Hello world" in body

    async def test_chat_unknown_agent_returns_rfc7807_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/agents/ag_NONEXISTENT000000000000/chat",
            json={"input": "Hello"},
        )
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/problem+json"
        body = response.json()
        assert body["status"] == 404
        assert "agent-not-found" in body["type"]

    async def test_chat_empty_input_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/agents/ag_TESTAGENTSTUB00000000000/chat",
            json={"input": ""},
        )
        assert response.status_code == 422


class TestHealthEndpoints:
    async def test_healthz(self, client: AsyncClient) -> None:
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_readyz_degraded_without_recif(self, client: AsyncClient) -> None:
        response = await client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["cache"] == "active"

    async def test_openapi_docs(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        assert "paths" in response.json()
