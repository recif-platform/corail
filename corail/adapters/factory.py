"""Adapter registry and factory — resolves framework and LLM adapters by name."""

from corail.adapters.frameworks.base import FrameworkAdapter
from corail.adapters.llms.base import LLMAdapter
from corail.core.errors import AdapterNotFoundError


class AdapterRegistry:
    """Central registry for framework and LLM adapters."""

    def __init__(self) -> None:
        self._frameworks: list[FrameworkAdapter] = []
        self._llms: dict[str, LLMAdapter] = {}

    def register_framework(self, adapter: FrameworkAdapter) -> None:
        """Register a framework adapter."""
        self._frameworks.append(adapter)

    def get_framework(self, framework: str) -> FrameworkAdapter:
        """Resolve a framework adapter by name."""
        for adapter in self._frameworks:
            if adapter.supports(framework):
                return adapter
        raise AdapterNotFoundError("framework", framework)

    def register_llm(self, name: str, adapter: LLMAdapter) -> None:
        """Register an LLM provider adapter."""
        self._llms[name] = adapter

    def get_llm(self, name: str) -> LLMAdapter:
        """Resolve an LLM adapter by provider name."""
        if name not in self._llms:
            raise AdapterNotFoundError("llm", name)
        return self._llms[name]


def create_default_registry() -> AdapterRegistry:
    """Create a registry with default adapters (ADK + StubLLM)."""
    from corail.adapters.frameworks.adk import ADKAdapter
    from corail.adapters.llms.stub import StubLLMAdapter

    registry = AdapterRegistry()
    registry.register_framework(ADKAdapter())
    registry.register_llm("stub", StubLLMAdapter())
    return registry
