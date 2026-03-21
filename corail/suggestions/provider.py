"""Suggestion providers — registry-based pattern for generating follow-up suggestions.

Providers implement the SuggestionProvider protocol and are resolved via a registry.
No if/elif chains: add new providers by registering them.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from corail.models.base import Model

logger = logging.getLogger(__name__)

_SUGGESTION_PROMPT = (
    "Based on the conversation, suggest 2-3 short follow-up questions the user might ask next.\n"
    "Return ONLY a JSON array of strings, nothing else.\n"
    "Conversation: {last_response}"
)


class SuggestionProvider(Protocol):
    """Protocol for suggestion generation — any object with an async `generate` method."""

    async def generate(self, last_response: str, history: list[dict[str, str]] | None = None) -> list[str]:
        """Return a list of suggestion strings."""
        ...


class StaticSuggestionProvider:
    """Returns a fixed list of suggestions from configuration."""

    def __init__(self, suggestions: list[str]) -> None:
        self._suggestions = suggestions

    async def generate(self, last_response: str, history: list[dict[str, str]] | None = None) -> list[str]:
        return list(self._suggestions)


class LLMSuggestionProvider:
    """Calls the agent's model with a lightweight prompt to generate follow-up suggestions."""

    def __init__(self, model: Model, max_tokens: int = 100, temperature: float = 0.8) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate(self, last_response: str, history: list[dict[str, str]] | None = None) -> list[str]:
        prompt = _SUGGESTION_PROMPT.format(last_response=last_response[:500])
        try:
            raw = await self._model.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return _parse_suggestions(raw)
        except Exception:
            logger.debug("LLM suggestion generation failed", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "static": StaticSuggestionProvider,
    "llm": LLMSuggestionProvider,
}


def register_provider(name: str, cls: type) -> None:
    """Register a custom suggestion provider class."""
    _REGISTRY[name] = cls


def available_providers() -> list[str]:
    """Return names of all registered providers."""
    return sorted(_REGISTRY.keys())


def get_provider_class(name: str) -> type:
    """Resolve a provider class by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        avail = ", ".join(available_providers())
        msg = f"Unknown suggestion provider: {name}. Available: {avail}"
        raise ValueError(msg)
    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_suggestions(raw: str) -> list[str]:
    """Best-effort extraction of a JSON string array from LLM output."""
    text = raw.strip()
    # Try to find a JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if isinstance(s, str) and s.strip()][:3]
        except (json.JSONDecodeError, ValueError):
            pass
    return []
