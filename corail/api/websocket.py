"""WebSocket endpoint for real-time agent communication — hot path."""

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import ExecutionRequest
from corail.core.pipeline import Pipeline
from corail.observer.events import response_event, thinking_event

logger = structlog.get_logger()

ws_router = APIRouter()


@ws_router.websocket("/ws/agents/{agent_id}/chat")
async def websocket_chat(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket endpoint for real-time agent chat. Streams Observer events."""
    await websocket.accept()

    agent_cache: AgentConfigCache = websocket.app.state.agent_cache
    pipeline: Pipeline = websocket.app.state.pipeline

    config = await agent_cache.get_agent(agent_id)
    if config is None:
        await websocket.send_json({"event": "error", "data": {"message": f"Agent {agent_id} not found"}})
        await websocket.close()
        return

    await logger.ainfo("ws_connected", agent_id=agent_id)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            user_input = message.get("input", "")

            if not user_input:
                await websocket.send_json({"event": "error", "data": {"message": "Empty input"}})
                continue

            # Emit thinking event
            await websocket.send_json(thinking_event("Processing request...", trace_id="").model_dump())

            # Execute through pipeline
            request = ExecutionRequest(agent_config=config, input=user_input)
            try:
                result = await pipeline.execute(request)
                await websocket.send_json(response_event(result.output).model_dump())
            except Exception as exc:
                await websocket.send_json(
                    {
                        "event": "error",
                        "data": {"message": str(exc)},
                    }
                )

    except WebSocketDisconnect:
        await logger.ainfo("ws_disconnected", agent_id=agent_id)
    except Exception as exc:
        await logger.aerror("ws_error", agent_id=agent_id, error=str(exc))
