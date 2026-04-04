"""Tests for memory module — InMemoryStorage, MemoryManager, factory."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from corail.memory.base import MemoryEntry, MemoryStorage
from corail.memory.factory import available_backends, create_memory_storage
from corail.memory.in_memory import InMemoryStorage
from corail.memory.manager import MemoryManager
from corail.models.base import Model


# --- Mock model ---

class _MockModel(Model):
    def __init__(self) -> None:
        self._generate = AsyncMock()

    async def generate(self, messages, **kwargs):
        return await self._generate(messages, **kwargs)


@pytest.fixture
def mock_model() -> _MockModel:
    return _MockModel()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


# --- InMemoryStorage tests ---

class TestInMemoryStorage:
    async def test_store_and_count(self, storage):
        entry = MemoryEntry(id="m1", content="Python is great", category="fact")
        await storage.store(entry)
        assert storage.count == 1

    async def test_search_by_keyword(self, storage):
        await storage.store(MemoryEntry(id="m1", content="The user prefers dark mode", category="preference"))
        await storage.store(MemoryEntry(id="m2", content="The project uses Python 3.13", category="fact"))
        await storage.store(MemoryEntry(id="m3", content="Deploy to Kubernetes cluster", category="instruction"))

        results = await storage.search("dark mode preference", top_k=5)
        assert len(results) >= 1
        # The preference entry should match best
        assert any("dark mode" in r.content for r in results)

    async def test_search_returns_empty_for_no_match(self, storage):
        await storage.store(MemoryEntry(id="m1", content="cats are cute", category="fact"))
        results = await storage.search("quantum physics", top_k=5)
        assert results == []

    async def test_search_respects_top_k(self, storage):
        for i in range(10):
            await storage.store(MemoryEntry(id=f"m{i}", content=f"memory about topic {i}", category="fact"))

        results = await storage.search("memory topic", top_k=3)
        assert len(results) <= 3

    async def test_list_recent(self, storage):
        for i in range(5):
            await storage.store(MemoryEntry(
                id=f"m{i}",
                content=f"Memory {i}",
                category="fact",
                timestamp=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            ))

        recent = await storage.list_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].id == "m4"

    async def test_delete(self, storage):
        await storage.store(MemoryEntry(id="m1", content="to delete", category="fact"))
        assert storage.count == 1
        await storage.delete("m1")
        assert storage.count == 0

    async def test_delete_nonexistent_is_noop(self, storage):
        await storage.delete("nonexistent")
        assert storage.count == 0

    async def test_relevance_affects_search_ranking(self, storage):
        await storage.store(MemoryEntry(
            id="m1", content="important fact about topic",
            category="fact", relevance=0.1,
        ))
        await storage.store(MemoryEntry(
            id="m2", content="another fact about topic",
            category="fact", relevance=10.0,
        ))

        results = await storage.search("fact about topic", top_k=2)
        assert len(results) == 2
        # Higher relevance should rank first
        assert results[0].id == "m2"


# --- MemoryManager tests ---

class TestMemoryManager:
    async def test_remember_and_recall(self, storage):
        manager = MemoryManager(storage=storage)
        await manager.remember("User likes Python", category="preference", source="test")

        results = await manager.recall("Python preference")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_build_context_with_memories(self, storage):
        manager = MemoryManager(storage=storage)
        await manager.remember("User prefers dark mode", category="preference")
        await manager.remember("Project uses FastAPI", category="fact")

        context = await manager.build_context("dark mode settings")
        assert "Memories from previous sessions" in context
        assert "dark mode" in context

    async def test_build_context_empty_when_no_memories(self, storage):
        manager = MemoryManager(storage=storage)
        context = await manager.build_context("anything")
        assert context == ""

    async def test_build_context_empty_when_no_match(self, storage):
        manager = MemoryManager(storage=storage)
        await manager.remember("cats are adorable", category="fact")
        context = await manager.build_context("quantum physics equations")
        assert context == ""

    async def test_extract_from_conversation(self, storage, mock_model):
        mock_model._generate.return_value = json.dumps([
            {"content": "User prefers Frutiger Aero aesthetic", "category": "preference"},
            {"content": "Project is named Corail", "category": "fact"},
        ])
        manager = MemoryManager(storage=storage, model=mock_model)

        messages = [
            {"role": "user", "content": "I really like the Frutiger Aero aesthetic"},
            {"role": "assistant", "content": "Great choice! I will keep that in mind for Corail."},
        ]
        await manager.extract_from_conversation(messages)

        assert storage.count == 2
        all_entries = await storage.list_recent(limit=10)
        contents = [e.content for e in all_entries]
        assert any("Frutiger Aero" in c for c in contents)

    async def test_extract_without_model_is_noop(self, storage):
        manager = MemoryManager(storage=storage, model=None)
        await manager.extract_from_conversation([{"role": "user", "content": "hello"}])
        assert storage.count == 0

    async def test_extract_handles_invalid_response(self, storage, mock_model):
        mock_model._generate.return_value = "Not valid JSON at all"
        manager = MemoryManager(storage=storage, model=mock_model)
        await manager.extract_from_conversation([{"role": "user", "content": "test"}])
        assert storage.count == 0


# --- Factory tests ---

class TestMemoryFactory:
    def test_create_in_memory_storage(self):
        storage = create_memory_storage("in_memory")
        assert isinstance(storage, InMemoryStorage)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown memory backend"):
            create_memory_storage("nonexistent")

    def test_available_backends(self):
        backends = available_backends()
        assert "in_memory" in backends
        assert "pgvector" in backends

    def test_create_pgvector_storage(self):
        from unittest.mock import MagicMock
        from corail.memory.pgvector import PgVectorMemoryStorage
        mock_embedding = MagicMock()
        mock_embedding.dimension = 768
        storage = create_memory_storage("pgvector", embedding_provider=mock_embedding)
        assert isinstance(storage, PgVectorMemoryStorage)
