"""Vertex AI embedding provider — text-embedding-005 via REST API.

Reuses the same GCP auth path as ``corail.models.vertex`` (service account
JWT or ADC), so no extra credentials setup is needed beyond what the
Vertex LLM adapter already requires.
"""

import json
import logging
import os
import time
from typing import Any

import httpx

from corail.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-005"
_DEFAULT_LOCATION = "us-central1"
_DEFAULT_DIM = 768


class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Produces embeddings via the Vertex AI ``text-embedding-*`` models."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        project: str = "",
        location: str = "",
    ) -> None:
        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", _DEFAULT_LOCATION)
        self._cached_token: str = ""
        self._token_expiry: float = 0.0

        if not self._project:
            self._project = self._detect_project()

    @staticmethod
    def _detect_project() -> str:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f).get("project_id", "")
            except (json.JSONDecodeError, OSError):
                pass
        return ""

    def _base_url(self) -> str:
        return (
            f"https://{self._location}-aiplatform.googleapis.com/v1"
            f"/projects/{self._project}/locations/{self._location}"
            f"/publishers/google/models/{self._model}"
        )

    async def _get_token(self) -> str:
        if self._cached_token and time.time() < self._token_expiry:
            return self._cached_token

        # Delegate to VertexAIModel's auth (SA JWT, ADC, metadata server).
        # The constructor is lightweight (just field assignment, no I/O).
        from corail.models.vertex import VertexAIModel

        adapter = VertexAIModel(project=self._project, location=self._location)
        token = await adapter._fetch_access_token()
        self._cached_token = token
        self._token_expiry = time.time() + 3300  # 55 min
        if not self._project and adapter.project:
            self._project = adapter.project
        return token

    @property
    def dimension(self) -> int:
        return _DEFAULT_DIM

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self._project:
            raise ValueError("GOOGLE_CLOUD_PROJECT not set for Vertex AI embeddings")

        token = await self._get_token()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url()}:predict",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"instances": [{"content": t} for t in texts]},
            )
            resp.raise_for_status()
            return [
                pred["embeddings"]["values"]
                for pred in resp.json()["predictions"]
            ]
