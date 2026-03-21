"""Core execution pipeline — channel-agnostic."""

from collections.abc import AsyncIterator

from corail.core.stream import StreamToken
from corail.strategies.base import AgentStrategy


class Pipeline:
    """Orchestrates agent execution: receives input, returns output.

    Pipeline is channel-agnostic. The Channel calls Pipeline, not the other way.
    """

    def __init__(self, strategy: AgentStrategy) -> None:
        self._strategy = strategy

    @property
    def memory(self) -> "MemoryManager | None":
        """Expose the strategy's memory manager (if any) for REST API access."""
        return getattr(self._strategy, "_memory", None)

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object) -> str:
        """Execute the agent strategy with user input and optional conversation history."""
        return await self._strategy.execute(user_input, history=history, **kwargs)

    async def execute_stream(
        self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object
    ) -> AsyncIterator[StreamToken]:
        """Stream the agent response as tokens and structured events."""
        async for token in self._strategy.execute_stream(user_input, history=history, **kwargs):
            yield token
