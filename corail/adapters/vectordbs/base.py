"""VectorDBAdapter protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorDBAdapter(Protocol):
    """Protocol for vector database adapters."""
