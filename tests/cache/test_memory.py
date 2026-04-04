"""Tests for in-memory cache."""

import asyncio

from corail.cache.memory import MemoryCache


class TestMemoryCache:
    async def test_set_and_get(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

    async def test_get_missing_returns_none(self) -> None:
        cache = MemoryCache()
        assert await cache.get("nonexistent") is None

    async def test_delete_removes_key(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1")
        await cache.delete("key1")
        assert await cache.get("key1") is None

    async def test_exists_returns_true_for_present(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

    async def test_exists_returns_false_for_missing(self) -> None:
        cache = MemoryCache()
        assert await cache.exists("nonexistent") is False

    async def test_ttl_expiration(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1", ttl=0)
        await asyncio.sleep(0.01)
        assert await cache.get("key1") is None

    async def test_no_ttl_persists(self) -> None:
        cache = MemoryCache()
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"
