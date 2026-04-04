"""Tests for EventEmitter."""

import asyncio

from corail.observer.emitter import EventEmitter
from corail.observer.events import thinking_event


class TestEventEmitter:
    async def test_emit_to_subscriber(self) -> None:
        emitter = EventEmitter()
        sub = emitter.subscribe()

        event = thinking_event("test")
        await emitter.emit(event)

        received = await asyncio.wait_for(sub._queue.get(), timeout=1.0)
        assert received.event == "agent.thinking"

    async def test_subscriber_count(self) -> None:
        emitter = EventEmitter()
        assert emitter.subscriber_count == 0

        sub1 = emitter.subscribe()
        assert emitter.subscriber_count == 1

        sub2 = emitter.subscribe()
        assert emitter.subscriber_count == 2

        await sub1.close()
        assert emitter.subscriber_count == 1

        await sub2.close()
        assert emitter.subscriber_count == 0

    async def test_multiple_subscribers_receive_event(self) -> None:
        emitter = EventEmitter()
        sub1 = emitter.subscribe()
        sub2 = emitter.subscribe()

        event = thinking_event("broadcast")
        await emitter.emit(event)

        r1 = await asyncio.wait_for(sub1._queue.get(), timeout=1.0)
        r2 = await asyncio.wait_for(sub2._queue.get(), timeout=1.0)

        assert r1.data["content"] == "broadcast"
        assert r2.data["content"] == "broadcast"
