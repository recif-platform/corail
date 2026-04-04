"""Integration test — full pipeline execution flow."""

from corail.adapters.factory import create_default_registry
from corail.core.agent_config import AgentConfig, ExecutionRequest
from corail.core.pipeline import Pipeline


async def test_full_pipeline_execution() -> None:
    """End-to-end: AgentConfig -> Pipeline -> ExecutionResponse via StubLLM."""
    config = AgentConfig(
        id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        name="Integration Test Agent",
        framework="adk",
        system_prompt="You are a test assistant.",
        model="test-model",
        llm_provider="stub",
        temperature=0.5,
    )
    request = ExecutionRequest(agent_config=config, input="Integration test input")

    registry = create_default_registry()
    pipeline = Pipeline(registry)

    response = await pipeline.execute(request)

    assert response.output == "Echo: Integration test input"
    assert response.agent_id == config.id
    assert response.execution_id.startswith("ex_")
    assert response.framework == "adk"
    assert response.model == "test-model"
