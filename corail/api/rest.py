"""REST API router — chat endpoint with SSE streaming."""

import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from corail.api.models import ChatRequest
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig, ExecutionRequest
from corail.core.errors import CoreError

logger = structlog.get_logger()

router = APIRouter()


class AgentNotFoundError(CoreError):
    """Raised when an agent ID is not found in the cache."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(
            message=f"Agent '{agent_id}' not found",
            code="AGENT_NOT_FOUND",
            details={"agent_id": agent_id},
        )


async def _get_agent_config(agent_id: str, cache: AgentConfigCache) -> AgentConfig:
    """Look up agent config from cache."""
    config = await cache.get_agent(agent_id)
    if config is None:
        raise AgentNotFoundError(agent_id)
    return config


@router.post("/agents/{agent_id}/chat")
async def chat(agent_id: str, body: ChatRequest, request: Request) -> StreamingResponse:
    """Chat with an agent. Returns SSE stream with execution events."""
    request_id = getattr(request.state, "request_id", "")
    await logger.ainfo("chat_request_received", agent_id=agent_id, request_id=request_id)

    agent_cache: AgentConfigCache = request.app.state.agent_cache
    agent_config = await _get_agent_config(agent_id, agent_cache)
    pipeline = request.app.state.pipeline

    execution_request = ExecutionRequest(
        agent_config=agent_config,
        input=body.input,
        conversation_id=body.conversation_id,
        metadata=body.metadata,
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield f"event: start\ndata: {json.dumps({'agent_id': agent_id})}\n\n"
            full_output = ""
            async for token in pipeline.execute_stream(body.input, history=body.metadata.get("history")):
                chunk = str(token)
                full_output += chunk
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            yield f"event: complete\ndata: {json.dumps({'output': full_output})}\n\n"
        except Exception as exc:
            error_data = json.dumps(
                {
                    "code": "STREAM_ERROR",
                    "message": str(exc),
                    "request_id": request_id,
                }
            )
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
