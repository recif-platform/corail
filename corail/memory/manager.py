"""MemoryManager — high-level memory operations for the agent."""

import json
import logging
import uuid
from datetime import datetime, timezone

from corail.memory.base import MemoryEntry, MemoryStorage
from corail.models.base import Model

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Review this conversation and extract key facts worth remembering for future sessions.
Focus on: user preferences, important decisions, learned facts, task outcomes.

Conversation:
{conversation}

Respond ONLY with a JSON array of objects: [{{"content": "...", "category": "fact|preference|instruction|observation"}}]
If nothing worth remembering, respond with: []

Memories:"""


class MemoryManager:
    """Orchestrates memory storage, recall, and extraction."""

    def __init__(self, storage: MemoryStorage, model: Model | None = None) -> None:
        self._storage = storage
        self._model = model

    async def remember(self, content: str, category: str, source: str = "") -> None:
        """Store a new memory."""
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],
            content=content,
            category=category,
            timestamp=datetime.now(timezone.utc),
            source=source,
        )
        await self._storage.store(entry)
        logger.debug("Stored memory: %s [%s]", content[:50], category)

    async def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memories relevant to a query."""
        return await self._storage.search(query, top_k=top_k)

    async def build_context(self, query: str) -> str:
        """Build memory context string for system prompt injection."""
        memories = await self.recall(query)
        if not memories:
            return ""
        lines = "\n".join(f"- {m.content}" for m in memories)
        return f"\n\nMemories from previous sessions:\n{lines}"

    async def extract_from_conversation(self, messages: list[dict]) -> None:
        """At end of conversation, extract and store memorable facts via LLM."""
        if self._model is None:
            return

        # Build conversation text from messages
        conversation_lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                conversation_lines.append(f"{role}: {content}")

        if not conversation_lines:
            return

        conversation_text = "\n".join(conversation_lines[-20:])  # Last 20 messages max
        prompt = _EXTRACT_PROMPT.format(conversation=conversation_text)

        try:
            raw = await self._model.generate(messages=[{"role": "user", "content": prompt}])
            entries = self._parse_extracted(raw)
            for entry_data in entries:
                await self.remember(
                    content=entry_data["content"],
                    category=entry_data.get("category", "observation"),
                    source="conversation_extraction",
                )
        except Exception:
            logger.warning("Failed to extract memories from conversation", exc_info=True)

    @staticmethod
    def _parse_extracted(raw: str) -> list[dict]:
        """Parse LLM extraction output into memory dicts."""
        try:
            # Find JSON array in response
            start = raw.index("[")
            end = raw.rindex("]") + 1
            items = json.loads(raw[start:end])
            return [
                item for item in items
                if isinstance(item, dict) and "content" in item
            ]
        except (ValueError, json.JSONDecodeError):
            return []
