"""Registry-based factory for evaluation providers."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corail.evaluation.base import EvaluationProvider

_REGISTRY: dict[str, tuple[str, str]] = {
    "mlflow": ("corail.evaluation.mlflow_provider", "MLflowProvider"),
    "phoenix": ("corail.evaluation.phoenix_provider", "PhoenixProvider"),
    "memory": ("corail.evaluation.memory_provider", "InMemoryProvider"),
}


def register_provider(name: str, module_path: str, class_name: str) -> None:
    """Register a custom evaluation provider."""
    _REGISTRY[name] = (module_path, class_name)


def create_provider(name: str = "memory", **kwargs: object) -> "EvaluationProvider":
    """Create an evaluation provider by name from the registry."""
    entry = _REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown evaluation provider: {name}. Available: {available}")
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def available_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_REGISTRY.keys())
