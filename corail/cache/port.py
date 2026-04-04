"""CachePort protocol — port for cache implementations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CachePort(Protocol):
    """Protocol for cache implementations (in-memory, Redis, etc.)."""

    async def get(self, key: str) -> Any | None:
        """Get a value by key. Returns None if missing or expired."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL in seconds."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        ...
