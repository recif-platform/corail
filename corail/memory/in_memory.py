"""InMemoryStorage — simple keyword-based memory storage for dev/testing."""

from corail.memory.base import MemoryEntry, MemoryStorage


class InMemoryStorage(MemoryStorage):
    """In-process memory storage with keyword-based search.

    Not persistent across restarts. Useful for development and testing.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    async def store(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple keyword overlap scoring."""
        query_words = set(query.lower().split())
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries.values():
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((score * entry.relevance, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        entries = sorted(self._entries.values(), key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    async def delete(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    @property
    def count(self) -> int:
        return len(self._entries)
