"""LLMAdapter protocol — the port for LLM provider integrations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM provider adapters."""

    async def generate(
        self, messages: list[dict[str, str]], model: str, temperature: float = 0.7
    ) -> str:
        """Send messages to the LLM and return the completion text."""
        ...

    async def generate_with_usage(
        self, messages: list[dict[str, str]], model: str, temperature: float = 0.7
    ) -> tuple[str, dict[str, Any]]:
        """Send messages and return (completion_text, usage_stats)."""
        ...
