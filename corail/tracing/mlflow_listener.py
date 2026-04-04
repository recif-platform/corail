"""MLflow tracing listener — collects pipeline events for trace enrichment.

Instead of creating separate spans (which end up as separate traces),
this listener collects events during execution. The channel layer
then passes them to _log_chat_trace which creates child spans inside
the @mlflow.trace decorated function where context is established.
"""

import logging
import contextvars
from typing import Any

from corail.events.types import Event, EventType

logger = logging.getLogger(__name__)

# Collected events during a request (reset per request)
_collected_events: contextvars.ContextVar[list[dict]] = contextvars.ContextVar("_collected_events", default=[])


def reset_events() -> None:
    """Reset collected events for a new request."""
    _collected_events.set([])


def get_collected_events() -> list[dict]:
    """Get all events collected during this request."""
    return _collected_events.get([])


class MLflowTracingListener:
    """Collects Corail EventBus events for MLflow trace enrichment."""

    def register(self, bus: Any) -> None:
        """Subscribe to relevant events on the bus."""
        bus.subscribe("*", self._handle_event)
        logger.info("MLflow tracing listener registered on EventBus")

    async def _handle_event(self, event: Event) -> None:
        """Collect event data for later trace enrichment."""
        try:
            etype = event.type

            if etype == EventType.LLM_CALL_STARTED:
                events = _collected_events.get([])
                events.append({
                    "type": "llm_call_started",
                    "round": event.data.get("round", 0),
                })
                _collected_events.set(events)

            elif etype == EventType.LLM_CALL_COMPLETED:
                events = _collected_events.get([])
                events.append({
                    "type": "llm_call_completed",
                    "round": event.data.get("round", 0),
                    "stop_reason": event.data.get("stop_reason", ""),
                })
                _collected_events.set(events)

            elif etype == EventType.MEMORY_RECALLED:
                events = _collected_events.get([])
                events.append({
                    "type": "memory_recalled",
                    "query": event.data.get("query", ""),
                })
                _collected_events.set(events)

            elif etype == EventType.TOOL_CALLED:
                events = _collected_events.get([])
                events.append({
                    "type": "tool_call",
                    "name": event.data.get("name", "unknown"),
                    "args": event.data.get("args", {}),
                })
                _collected_events.set(events)

            elif etype == EventType.TOOL_RESULT:
                events = _collected_events.get([])
                evt_data: dict = {
                    "type": "tool_result",
                    "name": event.data.get("name", "unknown"),
                    "output": str(event.data.get("output", "")),
                    "success": True,
                }
                sources = event.data.get("sources")
                if sources:
                    evt_data["sources"] = sources
                events.append(evt_data)
                _collected_events.set(events)

            elif etype == EventType.TOOL_ERROR:
                events = _collected_events.get([])
                events.append({
                    "type": "tool_error",
                    "name": event.data.get("name", "unknown"),
                    "error": str(event.data.get("error", "")),
                })
                _collected_events.set(events)

            elif etype == EventType.RETRIEVAL_SEARCHED:
                events = _collected_events.get([])
                events.append({
                    "type": "rag_search",
                    "query": event.data.get("query", ""),
                })
                _collected_events.set(events)

            elif etype == EventType.RETRIEVAL_RESULTS:
                events = _collected_events.get([])
                events.append({
                    "type": "rag_results",
                    "count": event.data.get("count", 0),
                })
                _collected_events.set(events)

            elif etype == EventType.GUARD_BLOCKED:
                events = _collected_events.get([])
                events.append({
                    "type": "guard_blocked",
                    "direction": event.data.get("direction", ""),
                    "reason": event.data.get("reason", ""),
                })
                _collected_events.set(events)

        except Exception:
            pass
