"""AgentStrategy abstract base class."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from corail.models.base import Model


class AgentStrategy(ABC):
    """Base class for agent execution strategies."""

    def __init__(self, model: Model, system_prompt: str = "") -> None:
        self.model = model
        self.system_prompt = system_prompt

    @abstractmethod
    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        """Execute the strategy with user input and return the full response."""
        ...

    async def execute_stream(self, user_input: str, history: list[dict[str, str]] | None = None) -> AsyncIterator[str]:
        """Stream the response token by token. Default: yield full response at once."""
        result = await self.execute(user_input, history=history)
        yield result
