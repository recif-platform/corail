"""Stub LLM adapter for testing — echoes input without API calls."""

from typing import Any


class StubLLMAdapter:
    """Deterministic LLM adapter that echoes user input. No API key needed."""

    async def generate(self, messages: list[dict[str, str]], model: str, temperature: float = 0.7) -> str:
        """Return an echo of the last user message."""
        last_message = messages[-1]["content"] if messages else ""
        return f"Echo: {last_message}"

    async def generate_with_usage(
        self, messages: list[dict[str, str]], model: str, temperature: float = 0.7
    ) -> tuple[str, dict[str, Any]]:
        """Return echo response with fake usage stats."""
        text = await self.generate(messages, model, temperature)
        usage = {
            "prompt_tokens": sum(len(m["content"]) for m in messages),
            "completion_tokens": len(text),
            "total_tokens": sum(len(m["content"]) for m in messages) + len(text),
            "model": model,
        }
        return text, usage
