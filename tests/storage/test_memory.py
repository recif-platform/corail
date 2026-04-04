"""Tests for MemoryStorage — in-memory StoragePort implementation."""

import asyncio

import pytest

from corail.storage.memory import MemoryStorage


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


class TestCreateConversation:
    async def test_creates_conversation(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        assert await storage.conversation_exists("conv-1")

    async def test_create_is_idempotent(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.create_conversation("conv-1")
        assert await storage.conversation_exists("conv-1")

    async def test_create_with_metadata_does_not_crash(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1", metadata={"key": "value"})
        assert await storage.conversation_exists("conv-1")


class TestConversationExists:
    async def test_nonexistent_returns_false(self, storage: MemoryStorage) -> None:
        assert not await storage.conversation_exists("no-such-id")

    async def test_existing_returns_true(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        assert await storage.conversation_exists("conv-1")


class TestGetMessages:
    async def test_nonexistent_conversation_returns_empty(self, storage: MemoryStorage) -> None:
        messages = await storage.get_messages("no-such-id")
        assert messages == []

    async def test_new_conversation_has_no_messages(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        messages = await storage.get_messages("conv-1")
        assert messages == []

    async def test_returns_appended_messages(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.append_message("conv-1", "user", "hello")
        await storage.append_message("conv-1", "assistant", "hi there")

        messages = await storage.get_messages("conv-1")
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1] == {"role": "assistant", "content": "hi there"}

    async def test_returns_copy_not_reference(self, storage: MemoryStorage) -> None:
        await storage.append_message("conv-1", "user", "hello")
        msgs_a = await storage.get_messages("conv-1")
        msgs_b = await storage.get_messages("conv-1")
        assert msgs_a is not msgs_b


class TestAppendMessage:
    async def test_auto_creates_conversation(self, storage: MemoryStorage) -> None:
        await storage.append_message("auto-conv", "user", "hello")
        assert await storage.conversation_exists("auto-conv")
        messages = await storage.get_messages("auto-conv")
        assert len(messages) == 1

    async def test_preserves_order(self, storage: MemoryStorage) -> None:
        for i in range(5):
            await storage.append_message("conv-1", "user", f"msg-{i}")
        messages = await storage.get_messages("conv-1")
        assert [m["content"] for m in messages] == [f"msg-{i}" for i in range(5)]


class TestListConversations:
    async def test_empty_state(self, storage: MemoryStorage) -> None:
        result = await storage.list_conversations()
        assert result == []

    async def test_lists_created_conversations(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.create_conversation("conv-2")
        result = await storage.list_conversations()
        assert len(result) == 2
        ids = {c["id"] for c in result}
        assert ids == {"conv-1", "conv-2"}

    async def test_includes_message_count(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.append_message("conv-1", "user", "hello")
        await storage.append_message("conv-1", "assistant", "hi")
        result = await storage.list_conversations()
        assert result[0]["message_count"] == 2

    async def test_newest_first_ordering(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-old")
        # Small delay so timestamps differ
        await asyncio.sleep(0.01)
        await storage.create_conversation("conv-new")

        result = await storage.list_conversations()
        assert result[0]["id"] == "conv-new"
        assert result[1]["id"] == "conv-old"

    async def test_includes_title(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.update_title("conv-1", "My Chat")
        result = await storage.list_conversations()
        assert result[0]["title"] == "My Chat"


class TestUpdateTitle:
    async def test_updates_title(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.update_title("conv-1", "New Title")
        result = await storage.list_conversations()
        assert result[0]["title"] == "New Title"

    async def test_nonexistent_conversation_no_error(self, storage: MemoryStorage) -> None:
        # update_title silently ignores nonexistent conversations
        await storage.update_title("no-such-id", "Title")

    async def test_overwrite_title(self, storage: MemoryStorage) -> None:
        await storage.create_conversation("conv-1")
        await storage.update_title("conv-1", "First")
        await storage.update_title("conv-1", "Second")
        result = await storage.list_conversations()
        assert result[0]["title"] == "Second"
