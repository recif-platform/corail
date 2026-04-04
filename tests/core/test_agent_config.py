"""Tests for agent configuration Pydantic models."""

import pytest
from pydantic import ValidationError

from corail.core.agent_config import AgentConfig, ExecutionRequest, ExecutionResponse


class TestAgentConfig:
    def test_valid_config(self) -> None:
        config = AgentConfig(
            id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            name="Test Agent",
            framework="adk",
            system_prompt="You are a helpful assistant.",
            model="gpt-4",
            llm_provider="stub",
        )
        assert config.framework == "adk"
        assert config.temperature == 0.7

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(id="ag_TEST", name="Test")  # type: ignore[call-arg]

    def test_invalid_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                id="bad-id",
                name="Test",
                framework="adk",
                system_prompt="test",
                model="gpt-4",
                llm_provider="stub",
            )

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                name="Test",
                framework="adk",
                system_prompt="test",
                model="gpt-4",
                llm_provider="stub",
                temperature=3.0,
            )

    def test_default_values(self) -> None:
        config = AgentConfig(
            id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            name="Test",
            framework="adk",
            system_prompt="test",
            model="gpt-4",
            llm_provider="stub",
        )
        assert config.tools == []
        assert config.config == {}
        assert config.temperature == 0.7


class TestExecutionRequest:
    def test_valid_request(self) -> None:
        config = AgentConfig(
            id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            name="Test",
            framework="adk",
            system_prompt="test",
            model="gpt-4",
            llm_provider="stub",
        )
        request = ExecutionRequest(agent_config=config, input="Hello")
        assert request.input == "Hello"
        assert request.conversation_id is None

    def test_empty_input_rejected(self) -> None:
        config = AgentConfig(
            id="ag_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            name="Test",
            framework="adk",
            system_prompt="test",
            model="gpt-4",
            llm_provider="stub",
        )
        with pytest.raises(ValidationError):
            ExecutionRequest(agent_config=config, input="")


class TestExecutionResponse:
    def test_serialization(self) -> None:
        response = ExecutionResponse(
            output="Hello!",
            agent_id="ag_TEST",
            execution_id="ex_TEST",
            framework="adk",
            model="gpt-4",
        )
        data = response.model_dump()
        assert data["output"] == "Hello!"
        assert data["usage"] == {}
