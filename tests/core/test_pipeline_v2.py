"""Tests for Pipeline V2 — channel-agnostic execution orchestrator."""

from unittest.mock import AsyncMock

import pytest

from corail.core.pipeline import Pipeline


@pytest.fixture
def mock_strategy() -> AsyncMock:
    strategy = AsyncMock()
    strategy.execute = AsyncMock(return_value="strategy output")

    async def _stream(*_args, **_kwargs):
        for token in ["tok1", "tok2", "tok3"]:
            yield token

    strategy.execute_stream = _stream
    return strategy


@pytest.fixture
def pipeline(mock_strategy: AsyncMock) -> Pipeline:
    return Pipeline(strategy=mock_strategy)


class TestPipelineExecute:
    async def test_returns_strategy_output(self, pipeline: Pipeline) -> None:
        result = await pipeline.execute("hello")
        assert result == "strategy output"

    async def test_passes_user_input(self, pipeline: Pipeline, mock_strategy: AsyncMock) -> None:
        await pipeline.execute("hello")
        call_args = mock_strategy.execute.call_args
        assert call_args.args[0] == "hello"

    async def test_passes_history(self, pipeline: Pipeline, mock_strategy: AsyncMock) -> None:
        history = [{"role": "user", "content": "prev"}]
        await pipeline.execute("now", history=history)
        call_kwargs = mock_strategy.execute.call_args.kwargs
        assert call_kwargs["history"] == history

    async def test_default_history_is_none(self, pipeline: Pipeline, mock_strategy: AsyncMock) -> None:
        await pipeline.execute("hello")
        call_kwargs = mock_strategy.execute.call_args.kwargs
        assert call_kwargs["history"] is None


class TestPipelineExecuteStream:
    async def test_yields_tokens(self, pipeline: Pipeline) -> None:
        tokens = []
        async for token in pipeline.execute_stream("hello"):
            tokens.append(token)
        assert tokens == ["tok1", "tok2", "tok3"]

    async def test_passes_history_to_strategy(self, pipeline: Pipeline, mock_strategy: AsyncMock) -> None:
        received_kwargs = {}

        async def _capture_stream(*args, **kwargs):
            received_kwargs.update(kwargs)
            yield "ok"

        mock_strategy.execute_stream = _capture_stream

        tokens = []
        async for token in pipeline.execute_stream("now", history=[{"role": "user", "content": "prev"}]):
            tokens.append(token)

        assert received_kwargs["history"] == [{"role": "user", "content": "prev"}]

    async def test_empty_stream(self, pipeline: Pipeline, mock_strategy: AsyncMock) -> None:
        async def _empty_stream(*_args, **_kwargs):
            return
            yield

        mock_strategy.execute_stream = _empty_stream

        tokens = []
        async for token in pipeline.execute_stream("hello"):
            tokens.append(token)
        assert tokens == []
