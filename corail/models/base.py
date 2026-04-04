"""Model abstract base class — LLM provider interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelResponse:
    """Response from a model that may include tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""  # "end_turn", "tool_use"


class Model(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Generate a full response from messages."""
        ...

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response token by token. Default: yield full response at once."""
        result = await self.generate(messages, **kwargs)
        yield result

    @property
    def supports_tool_use(self) -> bool:
        """Whether this model supports native tool calling."""
        return False

    async def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs: object
    ) -> ModelResponse:
        """Generate with native tool calling. Override in models that support it."""
        msg = f"{type(self).__name__} does not support native tool calling"
        raise NotImplementedError(msg)
