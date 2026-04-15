"""Tests for Channel base class — especially log_chat_trace."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from corail.channels.base import Channel
from corail.config import Settings
from corail.core.pipeline import Pipeline


class ConcreteChannel(Channel):
    """Minimal concrete channel for testing the base class."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def channel() -> ConcreteChannel:
    pipeline = MagicMock(spec=Pipeline)
    settings = Settings()
    return ConcreteChannel(pipeline=pipeline, settings=settings)


class TestLogChatTrace:
    async def test_returns_none_when_mlflow_unavailable(self, channel: ConcreteChannel) -> None:
        with patch("corail.channels.base._HAS_MLFLOW", False):
            result = await channel.log_chat_trace("hello", "conv-1", "response")
        assert result is None

    async def test_returns_none_on_exception(self, channel: ConcreteChannel) -> None:
        with patch("corail.channels.base._HAS_MLFLOW", True):
            with patch("asyncio.to_thread", side_effect=RuntimeError("mlflow down")):
                result = await channel.log_chat_trace("hello", "conv-1", "response")
        assert result is None

    async def test_accepts_empty_events(self, channel: ConcreteChannel) -> None:
        with patch("corail.channels.base._HAS_MLFLOW", False):
            result = await channel.log_chat_trace("hello", "conv-1", "response", [])
        assert result is None

    async def test_times_out_gracefully(self, channel: ConcreteChannel) -> None:
        import asyncio

        async def _slow(*_args):
            await asyncio.sleep(10)

        with patch("corail.channels.base._HAS_MLFLOW", True):
            with patch("asyncio.to_thread", side_effect=_slow):
                result = await channel.log_chat_trace("hello", "conv-1", "response")
        assert result is None

    def test_sync_log_returns_none_when_mlflow_unavailable(self, channel: ConcreteChannel) -> None:
        with patch("corail.channels.base._HAS_MLFLOW", False):
            result = channel._sync_log_chat_trace("hello", "conv-1", "response")
        assert result is None
