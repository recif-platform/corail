"""Agent configuration and execution models (Pydantic v2)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    """Agent configuration loaded from YAML or API."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., min_length=3, max_length=30, pattern=r"^ag_[A-Z0-9]+$")
    name: str = Field(..., min_length=1, max_length=255)
    framework: str = Field(..., min_length=1, max_length=50)
    system_prompt: str = Field(..., min_length=1, max_length=32000)
    model: str = Field(..., min_length=1, max_length=100)
    llm_provider: str = Field(..., min_length=1, max_length=50)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tools: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class ExecutionRequest(BaseModel):
    """Request to execute an agent."""

    model_config = ConfigDict(strict=True)

    agent_config: AgentConfig
    input: str = Field(..., min_length=1)
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResponse(BaseModel):
    """Response from agent execution."""

    output: str
    agent_id: str
    execution_id: str
    framework: str
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
