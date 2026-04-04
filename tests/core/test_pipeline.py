"""Tests for the core execution pipeline."""

import pytest

from corail.adapters.factory import AdapterRegistry, create_default_registry
from corail.core.agent_config import AgentConfig, ExecutionRequest
from corail.core.errors import AdapterNotFoundError
from corail.core.pipeline import Pipeline


def _make_request(framework: str = "adk", llm_provider: str = "stub") -> ExecutionRequest:
    config = AgentConfig(
        id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        name="Test Agent",
        framework=framework,
        system_prompt="You are helpful.",
        model="gpt-4",
        llm_provider=llm_provider,
    )
    return ExecutionRequest(agent_config=config, input="Hello pipeline")


class TestPipeline:
    async def test_execute_returns_response(self) -> None:
        registry = create_default_registry()
        pipeline = Pipeline(registry)
        request = _make_request()

        response = await pipeline.execute(request)

        assert response.output == "Echo: Hello pipeline"
        assert response.agent_id == "ag_01ARZ3NDEKTSV4RRFFQ69G5FAV"
        assert response.framework == "adk"
        assert response.model == "gpt-4"
        assert response.execution_id.startswith("ex_")

    async def test_unknown_framework_raises_error(self) -> None:
        registry = create_default_registry()
        pipeline = Pipeline(registry)
        request = _make_request(framework="unknown_framework")

        with pytest.raises(AdapterNotFoundError):
            await pipeline.execute(request)

    async def test_unknown_llm_raises_error(self) -> None:
        registry = create_default_registry()
        pipeline = Pipeline(registry)
        request = _make_request(llm_provider="nonexistent")

        with pytest.raises(AdapterNotFoundError):
            await pipeline.execute(request)

    async def test_empty_registry_raises_error(self) -> None:
        registry = AdapterRegistry()
        pipeline = Pipeline(registry)
        request = _make_request()

        with pytest.raises(AdapterNotFoundError):
            await pipeline.execute(request)
