"""EmbeddingProviderFactory — registry-based resolution."""

import importlib

from corail.embeddings.base import EmbeddingProvider

_REGISTRY: dict[str, tuple[str, str]] = {
    "ollama": ("corail.embeddings.ollama", "OllamaEmbeddingProvider"),
    "vertex-ai": ("corail.embeddings.vertex", "VertexAIEmbeddingProvider"),
}


def register_embedding_provider(name: str, module_path: str, class_name: str) -> None:
    """Register a new embedding provider."""
    _REGISTRY[name] = (module_path, class_name)


class EmbeddingProviderFactory:
    """Creates embedding providers via registry lookup."""

    @staticmethod
    def create(name: str, **kwargs: object) -> EmbeddingProvider:
        entry = _REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown embedding provider: {name}. Available: {available}"
            raise ValueError(msg)
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def available() -> list[str]:
        return sorted(_REGISTRY.keys())
