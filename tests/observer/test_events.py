"""Tests for Observer event types."""

from corail.observer.events import (
    AGENT_RESPONSE,
    AGENT_THINKING,
    AGENT_TOOL_CALL,
    ObserverEvent,
    response_event,
    thinking_event,
    tool_call_event,
)


class TestObserverEvents:
    def test_observer_event_serialization(self) -> None:
        event = ObserverEvent(event="agent.thinking", data={"content": "test"})
        data = event.model_dump()
        assert data["event"] == "agent.thinking"
        assert "timestamp" in data
        assert "trace_id" in data

    def test_thinking_event(self) -> None:
        event = thinking_event("Analyzing...")
        assert event.event == AGENT_THINKING
        assert event.data["content"] == "Analyzing..."

    def test_tool_call_event(self) -> None:
        event = tool_call_event("search", {"query": "test"})
        assert event.event == AGENT_TOOL_CALL
        assert event.data["tool_name"] == "search"

    def test_response_event(self) -> None:
        event = response_event("Hello!")
        assert event.event == AGENT_RESPONSE
        assert event.data["content"] == "Hello!"
