"""PgVector retriever — similarity search via pgvector + pluggable embeddings."""

import json

import asyncpg

from corail.embeddings.base import EmbeddingProvider
from corail.retrieval.base import RetrievalResult, Retriever


class PgVectorRetriever(Retriever):
    """Retriever backed by pgvector. Uses any EmbeddingProvider for query embedding."""

    def __init__(
        self,
        connection_url: str,
        embedding_provider: EmbeddingProvider,
        kb_id: str = "default",
        table: str = "chunks",
    ) -> None:
        self.connection_url = connection_url
        self.embedding_provider = embedding_provider
        self.kb_id = kb_id
        self.table = table
        self._pool: asyncpg.Pool | None = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_url)
        return self._pool

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        embedding = await self.embedding_provider.embed(query)
        pool = await self._ensure_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT content, 1 - (embedding <=> $1::vector) as score, metadata
                FROM chunks WHERE kb_id = $2
                ORDER BY embedding <=> $1::vector LIMIT $3
                """,
                str(embedding),
                self.kb_id,
                top_k,
            )
            return [
                RetrievalResult(
                    content=r["content"],
                    score=r["score"],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                )
                for r in rows
            ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
