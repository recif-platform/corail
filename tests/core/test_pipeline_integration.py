"""Integration test — full pipeline execution flow."""

from corail.core.pipeline import Pipeline
from corail.models.stub import StubModel
from corail.strategies.simple import SimpleStrategy


async def test_full_pipeline_execution() -> None:
    """End-to-end: user input -> Pipeline(Strategy(StubModel)) -> echo response."""
    model = StubModel()
    strategy = SimpleStrategy(model=model, system_prompt="You are a test assistant.")
    pipeline = Pipeline(strategy)

    response = await pipeline.execute("Integration test input")
    assert response == "Echo: Integration test input"
