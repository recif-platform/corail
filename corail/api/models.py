"""API request/response models."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    input: str = Field(..., min_length=1, max_length=32000)
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
