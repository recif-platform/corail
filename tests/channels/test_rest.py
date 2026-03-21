"""Tests for RestChannel — FastAPI REST API with conversation history."""

from unittest.mock import AsyncMock

import httpx
import pytest

from corail.channels.rest import RestChannel
from corail.config import Settings
from corail.core.pipeline import Pipeline
from corail.storage.memory import MemoryStorage


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def mock_pipeline() -> AsyncMock:
    pipeline = AsyncMock(spec=Pipeline)
    pipeline.execute = AsyncMock(return_value="bot response")

    async def _stream(*_args, **_kwargs):
        for token in ["Hello", " ", "world"]:
            yield token

    pipeline.execute_stream = _stream

    # RestChannel accesses pipeline._strategy.model for the suggestion provider
    mock_strategy = AsyncMock()
    mock_strategy.model = AsyncMock()
    pipeline._strategy = mock_strategy
    pipeline.memory = None
    return pipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(storage="memory")


@pytest.fixture
def channel(mock_pipeline: AsyncMock, settings: Settings, storage: MemoryStorage) -> RestChannel:
    return RestChannel(pipeline=mock_pipeline, settings=settings, storage=storage)


@pytest.fixture
async def client(channel: RestChannel) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=channel.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthz:
    async def test_returns_200(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChat:
    async def test_returns_response_and_conversation_id(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json={"input": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["output"] == "bot response"
        assert "conversation_id" in data

    async def test_uses_provided_conversation_id(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json={"input": "hello", "conversation_id": "my-conv"})
        data = resp.json()
        assert data["conversation_id"] == "my-conv"

    async def test_stores_messages(self, client: httpx.AsyncClient, storage: MemoryStorage) -> None:
        resp = await client.post("/chat", json={"input": "hello"})
        cid = resp.json()["conversation_id"]
        messages = await storage.get_messages(cid)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1] == {"role": "assistant", "content": "bot response"}

    async def test_passes_history_on_followup(self, client: httpx.AsyncClient, mock_pipeline: AsyncMock) -> None:
        resp1 = await client.post("/chat", json={"input": "hello"})
        cid = resp1.json()["conversation_id"]

        await client.post("/chat", json={"input": "followup", "conversation_id": cid})

        # Second call should include history from first exchange
        second_call = mock_pipeline.execute.call_args_list[1]
        history = (
            second_call.kwargs.get("history") or second_call.args[1]
            if len(second_call.args) > 1
            else second_call.kwargs.get("history")
        )
        assert history is not None
        assert len(history) == 2  # user + assistant from first exchange

    async def test_sets_title_on_first_message(self, client: httpx.AsyncClient, storage: MemoryStorage) -> None:
        resp = await client.post("/chat", json={"input": "Tell me about Python"})
        cid = resp.json()["conversation_id"]
        convos = await storage.list_conversations()
        matching = [c for c in convos if c["id"] == cid]
        assert matching[0]["title"] != ""

    async def test_empty_input_rejected(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json={"input": ""})
        assert resp.status_code == 422


class TestListConversations:
    async def test_empty_initially(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/conversations")
        assert resp.status_code == 200
        assert resp.json() == {"conversations": []}

    async def test_populated_after_chat(self, client: httpx.AsyncClient) -> None:
        await client.post("/chat", json={"input": "hello"})
        resp = await client.get("/conversations")
        conversations = resp.json()["conversations"]
        assert len(conversations) == 1
        assert conversations[0]["message_count"] == 2


class TestGetConversation:
    async def test_returns_messages(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/chat", json={"input": "hello"})
        cid = resp.json()["conversation_id"]

        resp = await client.get(f"/conversations/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == cid
        assert len(data["messages"]) == 2

    async def test_nonexistent_returns_404(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/conversations/no-such-id")
        assert resp.status_code == 404
        assert "error" in resp.json()
