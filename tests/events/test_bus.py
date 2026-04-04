"""Tests for EventBus."""

import pytest
from corail.events.bus import EventBus
from corail.events.types import Event, EventType


class TestEventBus:
    async def test_emit_calls_subscriber(self):
        bus = EventBus()
        received = []
        async def handler(event: Event):
            received.append(event)
        bus.subscribe(EventType.TOOL_CALLED, handler)
        await bus.emit(Event(type=EventType.TOOL_CALLED, data={"name": "calc"}))
        assert len(received) == 1
        assert received[0].data["name"] == "calc"

    async def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        async def handler(event: Event):
            received.append(event.type)
        bus.subscribe("*", handler)
        await bus.emit(Event(type=EventType.TOOL_CALLED))
        await bus.emit(Event(type=EventType.GUARD_BLOCKED))
        assert len(received) == 2

    async def test_no_subscribers(self):
        bus = EventBus()
        await bus.emit(Event(type=EventType.TOOL_CALLED))  # No crash

    async def test_handler_error_doesnt_crash(self):
        bus = EventBus()
        async def bad_handler(event: Event):
            raise RuntimeError("boom")
        bus.subscribe(EventType.TOOL_CALLED, bad_handler)
        await bus.emit(Event(type=EventType.TOOL_CALLED))  # Should not raise

    async def test_history(self):
        bus = EventBus()
        await bus.emit(Event(type=EventType.TOOL_CALLED))
        await bus.emit(Event(type=EventType.TOOL_RESULT))
        assert len(bus.history) == 2
        assert bus.history[0].type == EventType.TOOL_RESULT  # Most recent first

    async def test_unsubscribe(self):
        bus = EventBus()
        received = []
        async def handler(event: Event):
            received.append(event)
        bus.subscribe(EventType.TOOL_CALLED, handler)
        bus.unsubscribe(EventType.TOOL_CALLED, handler)
        await bus.emit(Event(type=EventType.TOOL_CALLED))
        assert len(received) == 0

    async def test_subscriber_count(self):
        bus = EventBus()
        async def h1(e: Event): pass
        async def h2(e: Event): pass
        bus.subscribe(EventType.TOOL_CALLED, h1)
        bus.subscribe("*", h2)
        assert bus.subscriber_count == 2
