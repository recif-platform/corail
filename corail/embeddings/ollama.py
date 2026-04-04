"""Ollama embedding provider."""

import os

import httpx

from corail.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via the Ollama /api/embed endpoint."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "") -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self._dimension: int | None = None

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
            )
            resp.raise_for_status()
            vec = resp.json()["embeddings"][0]
            if self._dimension is None:
                self._dimension = len(vec)
            return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            vecs = resp.json()["embeddings"]
            if self._dimension is None and vecs:
                self._dimension = len(vecs[0])
            return vecs

    @property
    def dimension(self) -> int:
        return self._dimension or 768
