"""Tests for the core execution pipeline."""

from corail.core.pipeline import Pipeline
from corail.models.stub import StubModel
from corail.strategies.simple import SimpleStrategy


def _make_pipeline() -> Pipeline:
    model = StubModel()
    strategy = SimpleStrategy(model=model, system_prompt="You are helpful.")
    return Pipeline(strategy)


class TestPipeline:
    async def test_execute_returns_echo(self) -> None:
        pipeline = _make_pipeline()
        result = await pipeline.execute("Hello pipeline")
        assert result == "Echo: Hello pipeline"

    async def test_execute_with_history(self) -> None:
        pipeline = _make_pipeline()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = await pipeline.execute("Follow up", history=history)
        assert result == "Echo: Follow up"

    async def test_execute_stream_yields_tokens(self) -> None:
        pipeline = _make_pipeline()
        tokens: list[str] = []
        async for token in pipeline.execute_stream("Hello stream"):
            tokens.append(str(token))
        assert any("Echo: Hello stream" in t for t in tokens)

    async def test_memory_property_returns_none_for_simple_strategy(self) -> None:
        pipeline = _make_pipeline()
        assert pipeline.memory is None
