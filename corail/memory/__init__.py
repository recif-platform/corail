"""Memory — persistent agent memory with pluggable storage backends."""

from corail.memory.base import MemoryEntry, MemoryStorage
from corail.memory.factory import create_memory_storage
from corail.memory.manager import MemoryManager

__all__ = ["MemoryEntry", "MemoryManager", "MemoryStorage", "create_memory_storage"]
