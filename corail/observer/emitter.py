"""Event emitter — broadcasts Observer events to subscribers."""

import asyncio
from collections.abc import AsyncIterator

import structlog

from corail.observer.events import ObserverEvent

logger = structlog.get_logger()


class EventEmitter:
    """Broadcasts observer events to all connected subscribers."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[ObserverEvent]] = []

    def subscribe(self) -> "EventSubscription":
        """Create a new subscription. Returns an async iterator of events."""
        queue: asyncio.Queue[ObserverEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return EventSubscription(queue, self)

    def unsubscribe(self, queue: asyncio.Queue[ObserverEvent]) -> None:
        """Remove a subscriber queue."""
        self._subscribers = [q for q in self._subscribers if q is not queue]

    async def emit(self, event: ObserverEvent) -> None:
        """Broadcast an event to all subscribers."""
        await logger.adebug("event_emitted", event_type=event.event, trace_id=event.trace_id)
        for queue in self._subscribers:
            await queue.put(event)

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)


class EventSubscription:
    """An async iterator over observer events from a subscription."""

    def __init__(self, queue: asyncio.Queue[ObserverEvent], emitter: EventEmitter) -> None:
        self._queue = queue
        self._emitter = emitter

    async def __aiter__(self) -> AsyncIterator[ObserverEvent]:
        """Yield events as they arrive."""
        try:
            while True:
                event = await self._queue.get()
                yield event
        finally:
            self._emitter.unsubscribe(self._queue)

    async def close(self) -> None:
        """Unsubscribe from events."""
        self._emitter.unsubscribe(self._queue)
