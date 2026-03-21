"""Shared helpers for agent strategies.

Extracted from the former monolithic `agent.py` so that multiple strategy
variants can reuse:

- `_EventEmitter` — fire-and-forget wrapper around the optional EventBus
- `_GuardRunner` — wrapper around the optional GuardPipeline (no-op when absent)
- `StopReason` — enum describing why an agent loop terminated
- `build_system_prompt` — central place that injects grounding rules, memory
  context, and skill directives into the system prompt. Every strategy shares
  the same anti-hallucination block by default.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from corail.events.bus import EventBus
from corail.events.types import Event, EventType
from corail.guards.base import GuardResult
from corail.guards.pipeline import GuardPipeline
from corail.memory.manager import MemoryManager
from corail.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Budget defaults — shared across all strategies
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS = 10
_DEFAULT_MAX_TOKENS = 100_000
_MAX_TOOL_RETRIES = 2
_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Stop reason — explicit loop termination codes
# ---------------------------------------------------------------------------


class StopReason(str, Enum):
    """Why did the agent loop stop?

    Emitted on the TURN_ENDED event and attached to the final StreamEvent
    so frontends and MLflow traces can show *why* a response ended instead
    of silently truncating.
    """

    END_TURN = "end_turn"  # LLM produced a final answer with no tool calls
    MAX_ROUNDS = "max_rounds"  # Reached _max_rounds without converging
    TOKEN_BUDGET = "token_budget"  # Accumulated output exceeded _max_tokens
    TOOL_ERROR_UNRECOVERABLE = "tool_error"  # Tool error the loop could not recover from
    GUARD_BLOCKED = "guard_blocked"  # Input or output guard rejected the content
    USER_ABORTED = "user_aborted"  # Client cancelled the stream


# ---------------------------------------------------------------------------
# Event emitter + guard runner — optional-dependency wrappers
# ---------------------------------------------------------------------------


class _EventEmitter:
    """Fire-and-forget event emission. No-ops when bus is absent."""

    __slots__ = ("_bus",)

    def __init__(self, bus: EventBus | None) -> None:
        self._bus = bus

    async def emit(self, event_type: EventType, **data: Any) -> None:
        if self._bus is not None:
            await self._bus.emit(Event(type=event_type, data=data))


class _GuardRunner:
    """Run guard checks. Returns allow-all when pipeline is absent."""

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: GuardPipeline | None) -> None:
        self._pipeline = pipeline

    async def check_input(self, content: str) -> GuardResult:
        if self._pipeline is None:
            return GuardResult(allowed=True)
        return await self._pipeline.check_input(content)

    async def check_output(self, content: str) -> GuardResult:
        if self._pipeline is None:
            return GuardResult(allowed=True)
        return await self._pipeline.check_output(content)


# ---------------------------------------------------------------------------
# System prompt builder — central grounding-rules injection point
# ---------------------------------------------------------------------------

# Applied to every agent by default. Addresses the tr-06b1992a hallucination
# bug where a model invented a stock price and a source not present in the
# web_search output. Can be disabled via grounding_strict=False at the
# strategy constructor or via the CORAIL_GROUNDING_STRICT env var.
GROUNDING_RULES = (
    "\n\nGROUNDING RULES (STRICT):\n"
    "- Every factual claim in your answer MUST be traceable to a tool result "
    "in this conversation or to information the user explicitly provided.\n"
    "- If a tool returned only titles and URLs without the data the user asked "
    'for, say so explicitly: "I could not find the specific data in the search '
    'results. Here are sources that may contain it: [list URLs]". Do not guess.\n'
    "- NEVER invent dates, numbers, prices, quotes, statistics, or sources "
    "that are not present in tool outputs.\n"
    '- NEVER fabricate a source name (for example "According to Boursorama...") '
    "unless that exact source appears in a tool result.\n"
    "- If you are unsure, say so and either ask a clarifying question or "
    "propose to run a specific tool call, rather than guessing.\n"
    "\n"
    "ACTION RULES (STRICT):\n"
    "- If you decide to call a tool, EMIT THE TOOL CALL IMMEDIATELY. Do not "
    "write 'I will search for X', 'Let me try Y', 'Je vais essayer Z', or any "
    "other announcement of a future tool call as text. Either issue the "
    "function call now or do not mention it at all.\n"
    "- If a fetch_url call fails (403, timeout, empty content, consent wall) "
    "retry with a different URL from the previous search results in the SAME "
    "turn. Do not end the turn by announcing a retry you have not made.\n"
    "- If you have partial data and want more, CALL THE TOOL AGAIN (web_search "
    "with a new query, or fetch_url on another URL). Do not write 'I will try "
    "another search' and then end the turn — either the next tool call happens "
    "in this turn or you finalise with what you have.\n"
    "- Your final message to the user must describe what you actually did and "
    "found, not what you plan to do next. If the answer is incomplete, say so "
    'explicitly ("I could only find X, not Y") rather than promising a future '
    "retry."
)


KB_PRIORITY_RULES = (
    "\n\nKNOWLEDGE BASE RULES (STRICT):\n"
    "- You have knowledge base search tools (search_*). ALWAYS try them FIRST "
    "before using web_search when the question could be answered by your document collections.\n"
    "- Only use web_search for topics clearly outside your knowledge bases "
    "(e.g. live news, weather, topics unrelated to your document collections).\n"
    "- After searching a knowledge base, ALWAYS cite the source document by name, "
    'e.g. "According to [filename.pdf], ..." or "(Source: filename.pdf)".\n'
    "- If a user asks about a specific document or topic, search your knowledge bases "
    "even if you think you already know the answer.\n"
    "- If KB search returns no results, THEN try web_search as a fallback."
)


_DEFAULT_BASE_PROMPT = "You are a helpful assistant. Respond in the user's language."


def build_system_prompt(
    user_prompt: str,
    *,
    memory: MemoryManager | None,
    memory_context: str,
    skills: SkillRegistry | None,
    channel: str = "rest",
    grounding_strict: bool = True,
    has_kb_tools: bool = False,
) -> str:
    """Assemble the final system prompt the LLM will see.

    Layering (top to bottom):
      1. User-supplied prompt (or default base prompt if empty)
      2. KB priority rules (when knowledge base tools are registered)
      3. Grounding rules (unless grounding_strict is False)
      4. Memory context if MemoryManager is present
      5. Skill directives for the current channel
    """
    base = user_prompt or _DEFAULT_BASE_PROMPT
    if has_kb_tools:
        base += KB_PRIORITY_RULES
    if grounding_strict:
        base += GROUNDING_RULES
    if memory is not None:
        if memory_context:
            base += memory_context
        else:
            base += (
                "\n\nYou have a persistent memory system. You can remember "
                "facts from previous conversations. Currently no memories "
                "are stored yet — they will be extracted automatically as "
                "you interact with the user."
            )
    if skills is not None:
        skills_prompt = skills.build_prompt(channel=channel)
        if skills_prompt:
            base += skills_prompt
    return base
