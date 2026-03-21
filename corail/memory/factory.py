"""Memory storage factory — registry-based backend resolution."""

import importlib
from typing import Any

# Registry: backend_name -> (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "in_memory": ("corail.memory.in_memory", "InMemoryStorage"),
    "pgvector": ("corail.memory.pgvector", "PgVectorMemoryStorage"),
}


def register_memory_backend(name: str, module_path: str, class_name: str) -> None:
    """Register a memory storage backend. Allows external plugins."""
    _REGISTRY[name] = (module_path, class_name)


def create_memory_storage(backend: str = "in_memory", **kwargs: Any) -> "MemoryStorage":
    """Create a memory storage instance by backend name."""
    entry = _REGISTRY.get(backend)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        msg = f"Unknown memory backend: {backend}. Available: {available}"
        raise ValueError(msg)

    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def available_backends() -> list[str]:
    """Return sorted list of registered memory backends."""
    return sorted(_REGISTRY.keys())
