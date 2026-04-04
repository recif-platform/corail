"""In-memory storage — default, no persistence across Pod restarts."""

from datetime import datetime, timezone

from corail.storage.port import StoragePort


class MemoryStorage(StoragePort):
    """In-memory conversation storage. Fast but ephemeral."""

    def __init__(self) -> None:
        self._conversations: dict[str, dict] = {}  # id → {messages, title, created_at}

    def _ensure(self, conversation_id: str) -> dict:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                "messages": [],
                "title": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._conversations[conversation_id]

    async def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
        conv = self._conversations.get(conversation_id)
        return list(conv["messages"]) if conv else []

    async def append_message(self, conversation_id: str, role: str, content: str) -> None:
        conv = self._ensure(conversation_id)
        conv["messages"].append({"role": role, "content": content})

    async def create_conversation(self, conversation_id: str, metadata: dict | None = None) -> None:
        self._ensure(conversation_id)

    async def conversation_exists(self, conversation_id: str) -> bool:
        return conversation_id in self._conversations

    async def list_conversations(self) -> list[dict]:
        result = []
        for cid, conv in self._conversations.items():
            result.append({
                "id": cid,
                "title": conv.get("title", ""),
                "created_at": conv.get("created_at", ""),
                "message_count": len(conv["messages"]),
            })
        result.sort(key=lambda c: c["created_at"], reverse=True)
        return result

    async def update_title(self, conversation_id: str, title: str) -> None:
        if conversation_id in self._conversations:
            self._conversations[conversation_id]["title"] = title

    async def delete_conversation(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
