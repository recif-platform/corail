"""MemoryStorage — abstract interface for persistent agent memory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryEntry:
    """A single memory stored by the agent."""

    id: str
    content: str
    category: str  # fact, preference, instruction, observation
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    relevance: float = 1.0


class MemoryStorage(ABC):
    """Pluggable backend for storing and searching memories."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Persist a memory entry."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memories by relevance to query."""
        ...

    @abstractmethod
    async def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        """Return the most recent memories."""
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Delete a memory by ID."""
        ...
