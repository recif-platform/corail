"""Tests for StorageFactory — registry-based storage resolution."""

import pytest

from corail.storage.factory import StorageFactory
from corail.storage.memory import MemoryStorage
from corail.storage.port import StoragePort


class TestStorageFactoryRegistry:
    def test_memory_returns_memory_storage(self) -> None:
        instance = StorageFactory.create("memory")
        assert isinstance(instance, MemoryStorage)
        assert isinstance(instance, StoragePort)

    def test_postgresql_is_registered(self) -> None:
        assert "postgresql" in StorageFactory.available()

    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage: nope"):
            StorageFactory.create("nope")

    def test_error_message_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available:"):
            StorageFactory.create("nonexistent")

    def test_available_returns_sorted_list(self) -> None:
        available = StorageFactory.available()
        assert available == sorted(available)
        assert "memory" in available

    def test_create_returns_new_instance_each_call(self) -> None:
        a = StorageFactory.create("memory")
        b = StorageFactory.create("memory")
        assert a is not b
