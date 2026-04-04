"""PostgreSQL storage — persistent conversations across Pod restarts."""

import json
import os

from corail.storage.port import StoragePort

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False


class PostgreSQLStorage(StoragePort):
    """PostgreSQL-backed conversation storage. Survives Pod restarts and scales."""

    def __init__(self, model_id: str = "") -> None:
        self._dsn = os.environ.get("CORAIL_DATABASE_URL", "")
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
        """Create tables if they don't exist (auto-migration for storage)."""
        pool = self._pool
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)
            """)

    async def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY id",
                conversation_id,
            )
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def append_message(self, conversation_id: str, role: str, content: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
                conversation_id, role, content,
            )

    async def create_conversation(self, conversation_id: str, metadata: dict | None = None) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (id, metadata) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
                conversation_id, json.dumps(metadata or {}),
            )

    async def conversation_exists(self, conversation_id: str) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM conversations WHERE id = $1", conversation_id,
            )
            return row is not None

    async def list_conversations(self) -> list[dict]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.id, c.title, c.created_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            """)
            return [
                {
                    "id": row["id"],
                    "title": row["title"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                    "message_count": row["message_count"],
                }
                for row in rows
            ]

    async def update_title(self, conversation_id: str, title: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversations SET title = $1 WHERE id = $2",
                title, conversation_id,
            )

    async def delete_conversation(self, conversation_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            # Messages are CASCADE deleted via FK
            await conn.execute("DELETE FROM conversations WHERE id = $1", conversation_id)
