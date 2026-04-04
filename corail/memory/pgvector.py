"""PgVector memory storage — persistent semantic memory with pgvector embeddings."""

import json
import logging
import os

from corail.embeddings.base import EmbeddingProvider
from corail.memory.base import MemoryEntry, MemoryStorage

try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger(__name__)


class PgVectorMemoryStorage(MemoryStorage):
    """PostgreSQL + pgvector memory storage with semantic search.

    Memories are embedded and stored as vectors for similarity search.
    Survives Pod restarts and scales across replicas.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        connection_url: str = "",
        agent_id: str = "default",
    ) -> None:
        self._embedding = embedding_provider
        self._dsn = connection_url or os.environ.get("CORAIL_DATABASE_URL", "")
        self._agent_id = agent_id
        self._pool: "asyncpg.Pool | None" = None

    async def _ensure_pool(self) -> "asyncpg.Pool":
        if self._pool is None:
            if not HAS_ASYNCPG:
                msg = "asyncpg not installed. Install with: uv add asyncpg"
                raise RuntimeError(msg)
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            await self._init_tables()
        return self._pool

    async def _init_tables(self) -> None:
        pool = self._pool
        if pool is None:
            return
        dim = self._embedding.dimension
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS agent_memories (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'observation',
                    source TEXT DEFAULT '',
                    relevance REAL DEFAULT 1.0,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_id
                ON agent_memories(agent_id)
            """)
            # HNSW index for fast cosine similarity
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_agent_memories_embedding
                ON agent_memories USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            logger.info("agent_memories table ready (dim=%d, agent=%s)", dim, self._agent_id)

    async def store(self, entry: MemoryEntry) -> None:
        pool = await self._ensure_pool()
        embedding = await self._embedding.embed(entry.content)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memories (id, agent_id, content, category, source, relevance, embedding, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    relevance = EXCLUDED.relevance,
                    embedding = EXCLUDED.embedding
                """,
                entry.id,
                self._agent_id,
                entry.content,
                entry.category,
                entry.source,
                entry.relevance,
                str(embedding),
                entry.timestamp,
            )

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        pool = await self._ensure_pool()
        embedding = await self._embedding.embed(query)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, category, source, relevance, created_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM agent_memories
                WHERE agent_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                str(embedding),
                self._agent_id,
                top_k,
            )
            return [
                MemoryEntry(
                    id=r["id"],
                    content=r["content"],
                    category=r["category"],
                    source=r["source"] or "",
                    relevance=r["score"],
                    timestamp=r["created_at"],
                )
                for r in rows
            ]

    async def list_recent(self, limit: int = 20) -> list[MemoryEntry]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, category, source, relevance, created_at
                FROM agent_memories
                WHERE agent_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                self._agent_id,
                limit,
            )
            return [
                MemoryEntry(
                    id=r["id"],
                    content=r["content"],
                    category=r["category"],
                    source=r["source"] or "",
                    relevance=r["relevance"],
                    timestamp=r["created_at"],
                )
                for r in rows
            ]

    async def delete(self, entry_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM agent_memories WHERE id = $1 AND agent_id = $2",
                entry_id,
                self._agent_id,
            )

    async def count(self) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM agent_memories WHERE agent_id = $1",
                self._agent_id,
            )
            return row["cnt"] if row else 0

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
