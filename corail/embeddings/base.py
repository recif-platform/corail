"""EmbeddingProvider — pluggable interface for text embedding."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text → vector embedding. Implementations are swappable via registry."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential calls."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
