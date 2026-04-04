"""Observer event types for real-time streaming."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


class ObserverEvent(BaseModel):
    """Structured event emitted during agent execution."""

    event: str = Field(..., description="Event type: domain.action (e.g., agent.thinking)")
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    trace_id: str = Field(default_factory=lambda: str(ULID()))
    data: dict[str, Any] = Field(default_factory=dict)


# Standard event types
AGENT_THINKING = "agent.thinking"
AGENT_TOOL_CALL = "agent.tool_call"
AGENT_TOOL_RESULT = "agent.tool_result"
AGENT_RESPONSE = "agent.response"
EVAL_SCORE_COMPUTED = "eval.score_computed"
GUARD_INPUT_BLOCKED = "guard.input_blocked"
GUARD_OUTPUT_BLOCKED = "guard.output_blocked"


def thinking_event(content: str, trace_id: str = "") -> ObserverEvent:
    """Create an agent.thinking event."""
    return ObserverEvent(
        event=AGENT_THINKING,
        trace_id=trace_id or str(ULID()),
        data={"content": content},
    )


def tool_call_event(tool_name: str, parameters: dict[str, Any], trace_id: str = "") -> ObserverEvent:
    """Create an agent.tool_call event."""
    return ObserverEvent(
        event=AGENT_TOOL_CALL,
        trace_id=trace_id or str(ULID()),
        data={"tool_name": tool_name, "parameters": parameters},
    )


def response_event(content: str, trace_id: str = "") -> ObserverEvent:
    """Create an agent.response event."""
    return ObserverEvent(
        event=AGENT_RESPONSE,
        trace_id=trace_id or str(ULID()),
        data={"content": content},
    )
