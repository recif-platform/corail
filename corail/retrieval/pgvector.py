"""PgVector retriever — hybrid BM25 + semantic search via pgvector + pluggable embeddings."""

import json

import asyncpg

from corail.embeddings.base import EmbeddingProvider
from corail.retrieval.base import RetrievalResult, Retriever

_SEMANTIC_SQL = """
SELECT id, content, 1 - (embedding <=> $1::vector) AS score, metadata
FROM chunks WHERE kb_id = $2
ORDER BY embedding <=> $1::vector LIMIT $3
"""

_BM25_SQL = """
SELECT id, content, ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score, metadata
FROM chunks WHERE kb_id = $2 AND tsv @@ plainto_tsquery('english', $1)
ORDER BY score DESC LIMIT $3
"""

# Reciprocal Rank Fusion constant (standard value)
_RRF_K = 60


def _format_embedding(embedding: list[float]) -> str:
    """Convert a Python list of floats to pgvector literal format."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


class PgVectorRetriever(Retriever):
    """Retriever backed by pgvector with hybrid BM25 + semantic search.

    Uses any EmbeddingProvider for query embedding and Reciprocal Rank Fusion
    to merge semantic and keyword results.
    """

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
        """Hybrid search: combines semantic (pgvector) and BM25 (tsvector) via RRF."""
        embedding = await self.embedding_provider.embed(query)
        pool = await self._ensure_pool()
        fetch_k = top_k * 2  # over-fetch for better fusion

        embedding_str = _format_embedding(embedding)

        async with pool.acquire() as conn:
            semantic_rows = await conn.fetch(
                _SEMANTIC_SQL,
                embedding_str,
                self.kb_id,
                fetch_k,
            )
            bm25_rows = await conn.fetch(
                _BM25_SQL,
                query,
                self.kb_id,
                fetch_k,
            )

        return self._rrf_merge(semantic_rows, bm25_rows, top_k)

    @staticmethod
    def _rrf_merge(
        semantic_rows: list[asyncpg.Record],
        bm25_rows: list[asyncpg.Record],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        rows_by_id: dict[str, asyncpg.Record] = {}

        for rank, row in enumerate(semantic_rows, start=1):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            rows_by_id[rid] = row

        for rank, row in enumerate(bm25_rows, start=1):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            rows_by_id.setdefault(rid, row)

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

        return [
            RetrievalResult(
                content=rows_by_id[cid]["content"],
                score=scores[cid],
                metadata=json.loads(rows_by_id[cid]["metadata"]) if rows_by_id[cid]["metadata"] else {},
            )
            for cid in ranked_ids
        ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
