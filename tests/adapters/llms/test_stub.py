"""Tests for StubLLM adapter."""

from corail.adapters.llms.stub import StubLLMAdapter


class TestStubLLMAdapter:
    async def test_generate_echoes_input(self) -> None:
        llm = StubLLMAdapter()
        result = await llm.generate(
            [{"role": "user", "content": "Hello world"}],
            model="test-model",
        )
        assert result == "Echo: Hello world"

    async def test_generate_with_usage_returns_stats(self) -> None:
        llm = StubLLMAdapter()
        text, usage = await llm.generate_with_usage(
            [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}],
            model="test-model",
        )
        assert text == "Echo: Hi"
        assert usage["model"] == "test-model"
        assert usage["total_tokens"] > 0

    async def test_generate_empty_messages(self) -> None:
        llm = StubLLMAdapter()
        result = await llm.generate([], model="test-model")
        assert result == "Echo: "
