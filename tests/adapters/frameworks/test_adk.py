"""Tests for ADK framework adapter."""

import pytest

from corail.adapters.frameworks.adk import ADKAdapter
from corail.adapters.llms.stub import StubLLMAdapter
from corail.core.agent_config import AgentConfig
from corail.core.errors import LLMError


def _make_config() -> AgentConfig:
    return AgentConfig(
        id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        name="Test Agent",
        framework="adk",
        system_prompt="You are a helpful assistant.",
        model="gpt-4",
        llm_provider="stub",
    )


class TestADKAdapter:
    def test_supports_adk(self) -> None:
        adapter = ADKAdapter()
        assert adapter.supports("adk") is True

    def test_does_not_support_other_frameworks(self) -> None:
        adapter = ADKAdapter()
        assert adapter.supports("langchain") is False
        assert adapter.supports("crewai") is False

    async def test_execute_returns_llm_response(self) -> None:
        adapter = ADKAdapter()
        llm = StubLLMAdapter()
        config = _make_config()

        result = await adapter.execute(config, "Hello world", llm)

        assert result == "Echo: Hello world"

    async def test_execute_wraps_llm_errors(self) -> None:
        adapter = ADKAdapter()
        config = _make_config()

        class FailingLLM:
            async def generate(self, messages: list[dict[str, str]], model: str, temperature: float = 0.7) -> str:
                raise RuntimeError("API key invalid")

            async def generate_with_usage(
                self, messages: list[dict[str, str]], model: str, temperature: float = 0.7
            ) -> tuple[str, dict]:
                raise RuntimeError("API key invalid")

        with pytest.raises(LLMError, match="LLM call failed"):
            await adapter.execute(config, "Hello", FailingLLM())
