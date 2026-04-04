"""Stream events — structured blocks emitted by strategies during streaming.

Strategies yield either plain `str` tokens or `StreamEvent` objects.
The channel layer serializes StreamEvents into the appropriate wire format (SSE, WebSocket, etc.).
This keeps strategies transport-agnostic while enabling rich structured content.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """Base class for all structured stream events.

    Each subclass defines a `type` and a `to_sse_data` method that returns
    the dict to be JSON-serialized as an SSE `data:` payload.
    Polymorphic dispatch — no type-switching needed in the channel layer.
    """

    def to_sse_data(self) -> dict[str, Any]:
        """Serialize this event to a dict suitable for SSE JSON payload."""
        raise NotImplementedError


@dataclass
class ConfirmEvent(StreamEvent):
    """Request human confirmation before executing a tool."""

    call_id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_sse_data(self) -> dict[str, Any]:
        return {
            "type": "confirm",
            "confirm": {
                "id": self.call_id,
                "tool": self.tool,
                "args": self.args,
                "message": self.message,
            },
        }


@dataclass
class ComponentEvent(StreamEvent):
    """Render a structured UI component (table, chart, code, react, etc.)."""

    component: str  # "table", "chart", "code", "json", or a React component name
    props: dict[str, Any] = field(default_factory=dict)

    def to_sse_data(self) -> dict[str, Any]:
        return {
            "type": "component",
            "component": self.component,
            "props": self.props,
        }


@dataclass
class ToolStartEvent(StreamEvent):
    """Signal that a tool execution has started."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_sse_data(self) -> dict[str, Any]:
        return {
            "type": "tool_start",
            "tool": self.tool,
            "args": self.args,
            "call_id": self.call_id,
        }


@dataclass
class ToolEndEvent(StreamEvent):
    """Signal that a tool execution has completed."""

    tool: str
    output: str = ""
    success: bool = True
    call_id: str = ""

    def to_sse_data(self) -> dict[str, Any]:
        return {
            "type": "tool_end",
            "tool": self.tool,
            "output": self.output,
            "success": self.success,
            "call_id": self.call_id,
        }


@dataclass
class PlanEvent(StreamEvent):
    """Signal a plan step status change during streaming."""

    plan_goal: str = ""
    step_description: str = ""
    step_status: str = ""  # created, in_progress, completed, failed
    step_index: int = 0
    total_steps: int = 0

    def to_sse_data(self) -> dict[str, Any]:
        return {
            "type": "plan",
            "plan": json.dumps({
                "goal": self.plan_goal,
                "step": self.step_description,
                "status": self.step_status,
                "index": self.step_index,
                "total": self.total_steps,
            }),
        }


@dataclass
class SourcesEvent(StreamEvent):
    """RAG source attribution — emitted before the first LLM token so the
    dashboard can show which documents the answer is grounded in."""

    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_sse_data(self) -> dict[str, Any]:
        return {"type": "sources", "sources": self.sources}


# Union type for everything a strategy can yield during streaming.
StreamToken = str | StreamEvent
