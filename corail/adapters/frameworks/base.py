"""FrameworkAdapter protocol — the port for agent framework integrations."""

from typing import Protocol, runtime_checkable

from corail.adapters.llms.base import LLMAdapter
from corail.core.agent_config import AgentConfig


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol for agent framework adapters.

    Implementing a new framework adapter requires only:
    1. Implement this protocol
    2. Register in AdapterRegistry
    """

    async def execute(self, config: AgentConfig, input_text: str, llm: LLMAdapter) -> str:
        """Execute an agent with the given config and return the response text."""
        ...

    def supports(self, framework: str) -> bool:
        """Return True if this adapter handles the given framework name."""
        ...
