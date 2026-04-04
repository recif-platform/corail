"""StoragePort — pluggable interface for conversation persistence."""

from abc import ABC, abstractmethod


class StoragePort(ABC):
    """Abstract interface for conversation storage. Implementations are swappable via registry."""

    @abstractmethod
    async def get_messages(self, conversation_id: str) -> list[dict[str, str]]:
        """Retrieve all messages for a conversation."""
        ...

    @abstractmethod
    async def append_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to a conversation."""
        ...

    @abstractmethod
    async def create_conversation(self, conversation_id: str, metadata: dict | None = None) -> None:
        """Create a new conversation entry."""
        ...

    @abstractmethod
    async def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists."""
        ...

    @abstractmethod
    async def list_conversations(self) -> list[dict]:
        """List all conversations with id, title, created_at, message_count."""
        ...

    @abstractmethod
    async def update_title(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation."""
        ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages."""
        ...
