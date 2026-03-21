"""StrategyFactory — registry-based strategy resolution."""

import importlib

from corail.models.base import Model
from corail.strategies.base import AgentStrategy

# Registry: strategy_name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "agent-react": ("corail.strategies.agent", "UnifiedAgentStrategy"),
}


def register_strategy(name: str, module_path: str, class_name: str) -> None:
    """Register a new strategy. Allows external plugins to add strategies."""
    _REGISTRY[name] = (module_path, class_name)


class StrategyFactory:
    """Creates strategy instances via registry lookup."""

    @staticmethod
    def create(
        name: str, model: Model, system_prompt: str = "You are a helpful assistant.", **kwargs: object
    ) -> AgentStrategy:
        """Create a strategy by name. Extra kwargs are passed to the constructor."""
        entry = _REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown strategy: {name}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(model=model, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered strategy types."""
        return sorted(_REGISTRY.keys())
