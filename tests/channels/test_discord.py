"""Tests for DiscordChannel."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corail.channels.discord import DiscordChannel
from corail.config import Settings
from corail.core.pipeline import Pipeline
from corail.core.stream import StreamEvent
from corail.storage.memory import MemoryStorage


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=Pipeline)

    async def _stream(*_args, **_kwargs):
        yield "Hello"
        yield " world"

    pipeline.execute_stream = _stream
    return pipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(storage="memory")


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def channel(mock_pipeline: MagicMock, settings: Settings) -> DiscordChannel:
    with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "fake-token"}):
        with patch("discord.Client"), patch("discord.app_commands.CommandTree"):
            ch = DiscordChannel(pipeline=mock_pipeline, settings=settings)
            ch._storage = MemoryStorage()
            return ch


class TestHandleChat:
    async def test_calls_log_chat_trace_after_stream(self, channel: DiscordChannel) -> None:
        interaction = MagicMock()
        interaction.user.id = 42
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock(return_value=MagicMock())

        with patch.object(channel, "log_chat_trace", new_callable=AsyncMock) as mock_trace:
            with patch("corail.channels.discord.reset_events"):
                with patch("corail.channels.discord.get_collected_events", return_value=[]):
                    await channel._handle_chat(interaction, "hello")

        mock_trace.assert_awaited_once()
        args = mock_trace.call_args
        assert args[0][0] == "hello"  # user_input
        assert "Hello world" in args[0][2]  # full_response

    async def test_persists_message_to_storage(self, channel: DiscordChannel) -> None:
        interaction = MagicMock()
        interaction.user.id = 99
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock(return_value=MagicMock())

        with patch.object(channel, "log_chat_trace", new_callable=AsyncMock):
            with patch("corail.channels.discord.reset_events"):
                with patch("corail.channels.discord.get_collected_events", return_value=[]):
                    await channel._handle_chat(interaction, "test message")

        messages = await channel.storage.get_messages("discord_99")
        assert any(m["role"] == "user" and m["content"] == "test message" for m in messages)
        assert any(m["role"] == "assistant" for m in messages)

    async def test_handles_stream_error_gracefully(self, channel: DiscordChannel, mock_pipeline: MagicMock) -> None:
        async def _error_stream(*_args, **_kwargs):
            raise RuntimeError("LLM down")
            yield  # make it a generator

        mock_pipeline.execute_stream = _error_stream

        interaction = MagicMock()
        interaction.user.id = 1
        interaction.response.defer = AsyncMock()
        interaction.followup.send = AsyncMock(return_value=MagicMock())

        with patch.object(channel, "log_chat_trace", new_callable=AsyncMock):
            with patch("corail.channels.discord.reset_events"):
                with patch("corail.channels.discord.get_collected_events", return_value=[]):
                    await channel._handle_chat(interaction, "hello")

        # Should send an error message, not crash
        interaction.followup.send.assert_awaited()


class TestHandleClear:
    async def test_clears_conversation(self, channel: DiscordChannel) -> None:
        cid = "discord_7"
        await channel.storage.create_conversation(cid)
        await channel.storage.append_message(cid, "user", "hi")

        interaction = MagicMock()
        interaction.user.id = 7
        interaction.response.send_message = AsyncMock()

        await channel._handle_clear(interaction)

        assert not await channel.storage.conversation_exists(cid)


class TestStartRequiresToken:
    def test_raises_if_no_token(self, channel: DiscordChannel) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove DISCORD_BOT_TOKEN from env
            import os
            os.environ.pop("DISCORD_BOT_TOKEN", None)
            with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
                channel.start()
