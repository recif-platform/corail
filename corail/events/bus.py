"""EventBus — in-process event bus for the Corail runtime.

All components emit events here. Subscribers react asynchronously.
This is NOT Kafka — it's a lightweight in-process pub/sub for a single agent pod.

Usage:
    bus = EventBus()
    bus.subscribe(EventType.TOOL_CALLED, my_handler)
    bus.subscribe("*", audit_handler)  # Wildcard: receives ALL events
    await bus.emit(Event(type=EventType.TOOL_CALLED, data={"name": "calculator"}))
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from corail.events.types import Event, EventType

logger = logging.getLogger(__name__)

# Handler signature: async (event: Event) -> None
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]

# Wildcard key for subscribers that want ALL events
_WILDCARD = "*"


class EventBus:
    """In-process async event bus. Thread-safe via asyncio."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type. Use "*" for all events."""
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        self._subscribers[key].append(handler)
        logger.debug("Subscribed %s to %s", handler.__name__, key)

    def unsubscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """Remove a handler from an event type."""
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        handlers = self._subscribers.get(key, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all matching subscribers + wildcard subscribers.

        Handlers run concurrently. Errors in handlers are logged, not raised.
        """
        self._record(event)

        # Collect matching handlers
        handlers: list[EventHandler] = []
        handlers.extend(self._subscribers.get(event.type.value, []))
        handlers.extend(self._subscribers.get(_WILDCARD, []))

        if not handlers:
            return

        # Run all handlers concurrently
        results = await asyncio.gather(
            *(self._safe_call(handler, event) for handler in handlers),
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Event handler %s failed for %s: %s",
                    handlers[i].__name__,
                    event.type.value,
                    result,
                )

    def emit_sync(self, event: Event) -> None:
        """Fire-and-forget emit from sync code. Creates a task on the running loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event))
        except RuntimeError:
            # No running loop — just record
            self._record(event)

    @property
    def history(self) -> list[Event]:
        """Recent event history (most recent first)."""
        return list(reversed(self._history))

    @property
    def subscriber_count(self) -> int:
        return sum(len(h) for h in self._subscribers.values())

    def clear_history(self) -> None:
        self._history.clear()

    def _record(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @staticmethod
    async def _safe_call(handler: EventHandler, event: Event) -> None:
        """Call handler, catching any exception."""
        await handler(event)
