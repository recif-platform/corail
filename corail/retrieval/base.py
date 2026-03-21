"""Retriever abstract base class and result dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single chunk returned from vector search."""

    content: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


class Retriever(ABC):
    """Base class for retrieval backends."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search for relevant chunks given a query string."""
        ...
