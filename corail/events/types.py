"""Event types — all events in the Corail runtime."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All event types emitted by the Corail runtime."""

    # Message lifecycle
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_RESPONSE = "message.response"

    # Thinking / LLM
    THINKING_STARTED = "thinking.started"
    THINKING_COMPLETED = "thinking.completed"
    LLM_CALL_STARTED = "llm.call.started"
    LLM_CALL_COMPLETED = "llm.call.completed"
    LLM_TOKEN = "llm.token"

    # Tool execution
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Guards / Security
    GUARD_INPUT_CHECKED = "guard.input.checked"
    GUARD_OUTPUT_CHECKED = "guard.output.checked"
    GUARD_BLOCKED = "guard.blocked"

    # Memory
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_UPDATED = "memory.updated"

    # Retrieval / RAG
    RETRIEVAL_SEARCHED = "retrieval.searched"
    RETRIEVAL_RESULTS = "retrieval.results"

    # Budget
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"

    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"

    # Turn lifecycle (emitted around each round of the agent loop)
    TURN_STARTED = "turn.started"
    TURN_ENDED = "turn.ended"

    # Session
    SESSION_CREATED = "session.created"
    SESSION_ENDED = "session.ended"

    # Planning
    PLAN_CREATED = "plan.created"
    PLAN_STEP_STARTED = "plan.step.started"
    PLAN_STEP_COMPLETED = "plan.step.completed"
    PLAN_STEP_FAILED = "plan.step.failed"
    PLAN_COMPLETED = "plan.completed"

    # Memory (extended)
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"

    # Self-correction
    CORRECTION_ATTEMPTED = "correction.attempted"

    # Control plane
    CONFIG_UPDATED = "control.config.updated"
    TOOLS_RELOAD_REQUESTED = "control.tools.reload"
    KBS_RELOAD_REQUESTED = "control.kbs.reload"
    AGENT_PAUSED = "control.agent.paused"
    AGENT_RESUMED = "control.agent.resumed"


@dataclass
class Event:
    """An event emitted by the runtime. Immutable once created."""

    type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_id: str = ""
    user_id: str = ""
    session_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
        }
