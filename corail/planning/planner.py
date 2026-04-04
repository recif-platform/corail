"""Planner — LLM-driven task decomposition into executable steps."""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field

from corail.models.base import Model

logger = logging.getLogger(__name__)

# Heuristics for needs_planning
_MIN_WORDS_FOR_PLANNING = 15
_ACTION_VERBS = frozenset({
    "create", "build", "implement", "write", "develop", "design", "set up",
    "configure", "deploy", "migrate", "refactor", "analyze", "compare",
    "integrate", "generate", "transform", "convert", "extract", "process",
    "install", "update", "fix", "debug", "test", "optimize", "benchmark",
})

_PLANNING_PROMPT = """\
You are a task planner. Decompose the following goal into concrete steps.

Available tools: {tools}

Goal: {goal}

Respond ONLY with a JSON array of step descriptions. Each step should be a single actionable sentence.
Example: ["Search for relevant files", "Read the configuration", "Update the settings"]

Steps:"""


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    result: str = ""

    def mark_started(self) -> None:
        self.status = "in_progress"

    def mark_completed(self, result: str = "") -> None:
        self.status = "completed"
        self.result = result

    def mark_failed(self, error: str = "") -> None:
        self.status = "failed"
        self.result = error

    def mark_skipped(self, reason: str = "") -> None:
        self.status = "skipped"
        self.result = reason


@dataclass
class Plan:
    """An ordered sequence of steps toward a goal."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def current_step(self) -> PlanStep | None:
        """Return the first pending step, or None if all are done."""
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    @property
    def is_complete(self) -> bool:
        """True when no pending steps remain."""
        return all(s.status in ("completed", "failed", "skipped") for s in self.steps)

    @property
    def progress(self) -> str:
        """Human-readable progress summary."""
        done = sum(1 for s in self.steps if s.status == "completed")
        return f"{done}/{len(self.steps)} steps completed"

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "completed"]

    @property
    def failed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "failed"]


class Planner:
    """Decomposes complex goals into actionable steps using an LLM."""

    def __init__(self, model: Model) -> None:
        self._model = model

    async def create_plan(self, goal: str, available_tools: list[str]) -> Plan:
        """Ask the LLM to decompose a goal into steps."""
        prompt = _PLANNING_PROMPT.format(
            tools=", ".join(available_tools) if available_tools else "(none)",
            goal=goal,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = await self._model.generate(messages=messages)
        steps = self._parse_steps(raw)
        return Plan(goal=goal, steps=steps)

    @staticmethod
    def needs_planning(user_input: str) -> bool:
        """Heuristic: complex multi-part tasks need planning; simple questions do not."""
        stripped = user_input.strip()

        # Single question → no planning
        if stripped.endswith("?") and stripped.count("?") == 1:
            return False

        words = stripped.split()
        # Short input → no planning
        if len(words) < _MIN_WORDS_FOR_PLANNING:
            return False

        # Contains action verbs + enough complexity → needs planning
        lower = stripped.lower()
        has_action = any(verb in lower for verb in _ACTION_VERBS)
        has_conjunction = any(word in lower for word in ("and", "then", "also", "after", "before", "while"))

        return has_action and has_conjunction

    @staticmethod
    def _parse_steps(raw: str) -> list[PlanStep]:
        """Parse LLM output into PlanStep objects. Handles JSON array or numbered list."""
        # Try JSON first
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group())
                if isinstance(items, list):
                    return [
                        PlanStep(id=str(uuid.uuid4())[:8], description=str(item))
                        for item in items
                        if str(item).strip()
                    ]
            except json.JSONDecodeError:
                pass

        # Fallback: numbered list (1. ..., 2. ...)
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        steps = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip("- ").strip()
            if cleaned:
                steps.append(PlanStep(id=str(uuid.uuid4())[:8], description=cleaned))

        return steps
