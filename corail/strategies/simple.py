"""SimpleStrategy — system_prompt + conversation history + user_input → LLM → response."""

from collections.abc import AsyncIterator

from corail.strategies.base import AgentStrategy


class SimpleStrategy(AgentStrategy):
    """Simple strategy with conversation history support and streaming."""

    def _build_messages(self, user_input: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        """Build message list: system_prompt + history + current input."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        """Execute with full conversation history."""
        return await self.model.generate(messages=self._build_messages(user_input, history))

    async def execute_stream(self, user_input: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[str]:
        """Stream response token by token."""
        async for token in self.model.generate_stream(messages=self._build_messages(user_input, history)):
            yield token
