"""StorageFactory — registry-based storage resolution."""

import importlib

from corail.storage.port import StoragePort

# Registry: storage_type → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "memory": ("corail.storage.memory", "MemoryStorage"),
    "postgresql": ("corail.storage.postgresql", "PostgreSQLStorage"),
}


def register_storage(name: str, module_path: str, class_name: str) -> None:
    """Register a new storage backend. Allows plugins (cassandra, redis, s3, etc.)."""
    _REGISTRY[name] = (module_path, class_name)


class StorageFactory:
    """Creates storage instances via registry lookup."""

    @staticmethod
    def create(name: str) -> StoragePort:
        """Create a storage backend by name."""
        entry = _REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown storage: {name}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    @staticmethod
    def available() -> list[str]:
        """Return list of registered storage types."""
        return sorted(_REGISTRY.keys())
