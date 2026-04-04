"""Tests for the Récif control plane bridge."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from corail.control.bridge import RecifBridge
from corail.events.bus import EventBus
from corail.events.types import Event, EventType


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def bridge(event_bus: EventBus) -> RecifBridge:
    return RecifBridge(event_bus, agent_id="test-agent-001")


@pytest.fixture
def app(bridge: RecifBridge) -> FastAPI:
    app = FastAPI()
    bridge.mount(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestStatusEndpoint:
    def test_returns_agent_info(self, client: TestClient) -> None:
        resp = client.get("/control/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "test-agent-001"
        assert data["phase"] == "running"
        assert "event_subscribers" in data

    def test_reflects_phase_change(self, client: TestClient) -> None:
        client.post("/control/pause")
        resp = client.get("/control/status")
        assert resp.json()["phase"] == "paused"

        client.post("/control/resume")
        resp = client.get("/control/status")
        assert resp.json()["phase"] == "running"


class TestConfigUpdate:
    def test_update_config_success(self, client: TestClient) -> None:
        resp = client.post("/control/config", json={"config": {"model_id": "gpt-4"}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "1 keys" in body["message"]

    def test_update_config_emits_event(self, client: TestClient, event_bus: EventBus) -> None:
        client.post("/control/config", json={"config": {"key": "value"}})
        config_events = [e for e in event_bus.history if e.type == EventType.CONFIG_UPDATED]
        assert len(config_events) == 1
        assert config_events[0].data["config"]["key"] == "value"

    def test_update_config_empty_rejected(self, client: TestClient) -> None:
        resp = client.post("/control/config", json={"config": {}})
        assert resp.status_code == 422


class TestReload:
    def test_reload_tools(self, client: TestClient, event_bus: EventBus) -> None:
        resp = client.post("/control/reload", json={"target": "tools", "reason": "deployment"})
        assert resp.status_code == 200
        reload_events = [e for e in event_bus.history if e.type == EventType.TOOLS_RELOAD_REQUESTED]
        assert len(reload_events) == 1
        assert reload_events[0].data["reason"] == "deployment"

    def test_reload_knowledge_bases(self, client: TestClient, event_bus: EventBus) -> None:
        resp = client.post("/control/reload", json={"target": "knowledge_bases"})
        assert resp.status_code == 200
        reload_events = [e for e in event_bus.history if e.type == EventType.KBS_RELOAD_REQUESTED]
        assert len(reload_events) == 1

    def test_reload_invalid_target_rejected(self, client: TestClient) -> None:
        resp = client.post("/control/reload", json={"target": "invalid"})
        assert resp.status_code == 422


class TestPauseResume:
    def test_pause(self, client: TestClient, event_bus: EventBus) -> None:
        resp = client.post("/control/pause")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        pause_events = [e for e in event_bus.history if e.type == EventType.AGENT_PAUSED]
        assert len(pause_events) == 1

    def test_resume(self, client: TestClient, event_bus: EventBus) -> None:
        client.post("/control/pause")
        resp = client.post("/control/resume")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        resume_events = [e for e in event_bus.history if e.type == EventType.AGENT_RESUMED]
        assert len(resume_events) == 1


class TestEventSSE:
    async def test_fan_out_delivers_to_queues(self, bridge: RecifBridge, event_bus: EventBus) -> None:
        """Verify that events emitted on the bus are delivered to SSE queues."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        bridge._event_queues.add(queue)

        await event_bus.emit(Event(
            type=EventType.TOOL_CALLED,
            agent_id="test-agent-001",
            data={"name": "calculator"},
        ))

        event = queue.get_nowait()
        assert event is not None
        assert event.type == EventType.TOOL_CALLED
        assert event.data["name"] == "calculator"
        bridge._event_queues.discard(queue)

    async def test_fan_out_ignores_full_queues(self, bridge: RecifBridge, event_bus: EventBus) -> None:
        """Full queues are discarded rather than blocking."""
        full_queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=1)
        full_queue.put_nowait(Event(type=EventType.AGENT_STARTED))  # Fill it
        bridge._event_queues.add(full_queue)

        # This should not raise — the full queue is just dropped
        await event_bus.emit(Event(type=EventType.TOOL_CALLED))
        assert full_queue not in bridge._event_queues


class TestCommandRegistry:
    async def test_custom_command(self, bridge: RecifBridge) -> None:
        """External code can register custom control commands."""
        async def custom_handler(payload: dict) -> dict:
            return {"success": True, "message": f"Custom: {payload.get('key')}"}

        bridge.register_command("custom_op", custom_handler)
        result = await bridge._commands.execute("custom_op", {"key": "val"})
        assert result["success"] is True
        assert "val" in result["message"]

    async def test_unknown_command(self, bridge: RecifBridge) -> None:
        result = await bridge._commands.execute("nonexistent", {})
        assert result["success"] is False
        assert "Unknown command" in result["message"]
