"""Tests for SimpleStrategy — system_prompt + history + user_input → LLM → response."""

from unittest.mock import AsyncMock

import pytest

from corail.strategies.simple import SimpleStrategy


@pytest.fixture
def mock_model() -> AsyncMock:
    model = AsyncMock()
    model.generate = AsyncMock(return_value="model response")

    async def _stream(*_args, **_kwargs):
        for token in ["Hello", " ", "world"]:
            yield token

    model.generate_stream = _stream
    return model


@pytest.fixture
def strategy(mock_model: AsyncMock) -> SimpleStrategy:
    return SimpleStrategy(model=mock_model, system_prompt="You are a test assistant.")


class TestBuildMessages:
    def test_without_history(self, strategy: SimpleStrategy) -> None:
        messages = strategy._build_messages("hello")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are a test assistant."}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_with_history(self, strategy: SimpleStrategy) -> None:
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
        ]
        messages = strategy._build_messages("second", history=history)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "first"}
        assert messages[2] == {"role": "assistant", "content": "response"}
        assert messages[3] == {"role": "user", "content": "second"}

    def test_with_empty_history(self, strategy: SimpleStrategy) -> None:
        messages = strategy._build_messages("hello", history=[])
        assert len(messages) == 2

    def test_with_none_history(self, strategy: SimpleStrategy) -> None:
        messages = strategy._build_messages("hello", history=None)
        assert len(messages) == 2

    def test_empty_system_prompt(self, mock_model: AsyncMock) -> None:
        strat = SimpleStrategy(model=mock_model, system_prompt="")
        messages = strat._build_messages("hello")
        assert messages[0] == {"role": "system", "content": ""}


class TestExecute:
    async def test_calls_model_generate(self, strategy: SimpleStrategy, mock_model: AsyncMock) -> None:
        result = await strategy.execute("hello")
        assert result == "model response"
        mock_model.generate.assert_awaited_once()

    async def test_passes_correct_messages(self, strategy: SimpleStrategy, mock_model: AsyncMock) -> None:
        await strategy.execute("hello")
        call_kwargs = mock_model.generate.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": "hello"}

    async def test_passes_history(self, strategy: SimpleStrategy, mock_model: AsyncMock) -> None:
        history = [{"role": "user", "content": "prev"}]
        await strategy.execute("now", history=history)
        call_kwargs = mock_model.generate.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 3
        assert messages[1] == {"role": "user", "content": "prev"}


class TestExecuteStream:
    async def test_yields_tokens(self, strategy: SimpleStrategy) -> None:
        tokens = []
        async for token in strategy.execute_stream("hello"):
            tokens.append(token)
        assert tokens == ["Hello", " ", "world"]

    async def test_passes_history_to_stream(self, strategy: SimpleStrategy, mock_model: AsyncMock) -> None:
        history = [{"role": "user", "content": "prev"}]
        # Capture what generate_stream receives
        received_messages = []

        async def _capture_stream(messages, **_kwargs):
            received_messages.extend(messages)
            yield "ok"

        mock_model.generate_stream = _capture_stream

        tokens = []
        async for token in strategy.execute_stream("now", history=history):
            tokens.append(token)

        assert len(received_messages) == 3
        assert received_messages[1] == {"role": "user", "content": "prev"}
