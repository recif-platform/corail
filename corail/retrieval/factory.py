"""RetrieverFactory — registry-based retriever resolution."""

import importlib

from corail.retrieval.base import Retriever

# Registry: retriever_type → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "pgvector": ("corail.retrieval.pgvector", "PgVectorRetriever"),
}


def register_retriever(name: str, module_path: str, class_name: str) -> None:
    """Register a new retriever type. Allows external plugins to add retrievers."""
    _REGISTRY[name] = (module_path, class_name)


class RetrieverFactory:
    """Creates retriever instances via registry lookup. Lazy imports for minimal startup cost."""

    @staticmethod
    def create(retriever_type: str, **kwargs: object) -> Retriever:
        """Create a retriever by type. Extra kwargs are passed to the constructor."""
        entry = _REGISTRY.get(retriever_type)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown retriever type: {retriever_type}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered retriever types."""
        return sorted(_REGISTRY.keys())
