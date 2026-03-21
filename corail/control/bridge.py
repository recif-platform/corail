"""Récif Bridge — control plane endpoints + event streaming.

Exposes control endpoints on the existing FastAPI app.  No separate server,
no new port, no extra dependency.

Endpoints:
  POST /control/config   — update agent config
  POST /control/reload   — reload tools or knowledge bases
  POST /control/pause    — pause the agent
  POST /control/resume   — resume the agent
  GET  /control/status   — get agent status
  GET  /control/events   — SSE stream of all events (Récif subscribes)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from corail.events.types import Event, EventType

if TYPE_CHECKING:
    from corail.events.bus import EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    """Payload for POST /control/config."""

    config: dict[str, str] = Field(..., min_length=1)


class ReloadRequest(BaseModel):
    """Payload for POST /control/reload."""

    target: str = Field(..., pattern=r"^(tools|knowledge_bases)$")
    reason: str = ""


# ---------------------------------------------------------------------------
# Command handler registry
# ---------------------------------------------------------------------------

# Signature: async (payload: dict[str, Any]) -> dict[str, Any]
CommandHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_RELOAD_TARGET_TO_EVENT: dict[str, EventType] = {
    "tools": EventType.TOOLS_RELOAD_REQUESTED,
    "knowledge_bases": EventType.KBS_RELOAD_REQUESTED,
}


class _CommandRegistry:
    """Maps command names to async handlers. No if/elif chains."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, name: str, handler: CommandHandler) -> None:
        self._handlers[name] = handler

    async def execute(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"success": False, "message": f"Unknown command: {name}"}
        return await handler(payload)

    @property
    def names(self) -> list[str]:
        return list(self._handlers.keys())


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class RecifBridge:
    """Control plane bridge between Récif (Go) and Corail (Python).

    Wired into the existing FastAPI app — subscribes to the EventBus wildcard
    and exposes an SSE endpoint so Récif can consume all events.
    """

    def __init__(self, event_bus: EventBus, *, agent_id: str = "") -> None:
        self._bus = event_bus
        self._agent_id = agent_id
        self._phase = "running"
        self._commands = _CommandRegistry()
        self._event_queues: set[asyncio.Queue[Event | None]] = set()

        # Register built-in commands
        self._commands.register("update_config", self._handle_update_config)
        self._commands.register("reload", self._handle_reload)
        self._commands.register("pause", self._handle_pause)
        self._commands.register("resume", self._handle_resume)

        # Subscribe to all events for SSE fan-out
        self._bus.subscribe("*", self._fan_out)

    # -- Public API --------------------------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    def register_command(self, name: str, handler: CommandHandler) -> None:
        """Allow external code to register additional control commands."""
        self._commands.register(name, handler)

    def mount(self, app: FastAPI) -> None:
        """Register control routes on the FastAPI app."""
        router = self._build_router()
        app.include_router(router, prefix="/control", tags=["control"])
        logger.info("Control plane mounted at /control")

    # -- Router ------------------------------------------------------------

    def _build_router(self) -> APIRouter:
        router = APIRouter()
        router.get("/status")(self._status)
        router.get("/events")(self._events_sse)
        router.post("/config")(self._update_config)
        router.post("/reload")(self._reload)
        router.post("/pause")(self._pause)
        router.post("/resume")(self._resume)
        return router

    # -- Endpoint handlers -------------------------------------------------

    async def _status(self) -> JSONResponse:
        return JSONResponse(
            {
                "agent_id": self._agent_id,
                "phase": self._phase,
                "active_sessions": 0,
                "tools_count": 0,
                "kbs_count": 0,
                "loaded_tools": [],
                "loaded_kbs": [],
                "event_subscribers": self._bus.subscriber_count,
                "event_history_size": len(self._bus.history),
            }
        )

    async def _events_sse(self) -> StreamingResponse:
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._event_queues.add(queue)

        async def stream() -> AsyncGenerator[str]:
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    data = json.dumps(event.to_dict())
                    yield f"data: {data}\n\n"
            finally:
                self._event_queues.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def _update_config(self, body: ConfigUpdateRequest) -> JSONResponse:
        result = await self._commands.execute("update_config", {"config": body.config})
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)

    async def _reload(self, body: ReloadRequest) -> JSONResponse:
        result = await self._commands.execute("reload", {"target": body.target, "reason": body.reason})
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)

    async def _pause(self) -> JSONResponse:
        result = await self._commands.execute("pause", {})
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)

    async def _resume(self) -> JSONResponse:
        result = await self._commands.execute("resume", {})
        status = 200 if result.get("success") else 400
        return JSONResponse(result, status_code=status)

    # -- Command handlers --------------------------------------------------

    async def _handle_update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = payload.get("config", {})
        await self._bus.emit(
            Event(
                type=EventType.CONFIG_UPDATED,
                agent_id=self._agent_id,
                data={"config": config},
            )
        )
        return {"success": True, "message": f"Config updated ({len(config)} keys)"}

    async def _handle_reload(self, payload: dict[str, Any]) -> dict[str, Any]:
        target = payload.get("target", "")
        event_type = _RELOAD_TARGET_TO_EVENT.get(target)
        if event_type is None:
            return {"success": False, "message": f"Unknown reload target: {target}"}
        await self._bus.emit(
            Event(
                type=event_type,
                agent_id=self._agent_id,
                data={"reason": payload.get("reason", "")},
            )
        )
        return {"success": True, "message": f"Reload requested: {target}"}

    async def _handle_pause(self, _payload: dict[str, Any]) -> dict[str, Any]:
        self._phase = "paused"
        await self._bus.emit(
            Event(
                type=EventType.AGENT_PAUSED,
                agent_id=self._agent_id,
            )
        )
        return {"success": True, "message": "Agent paused"}

    async def _handle_resume(self, _payload: dict[str, Any]) -> dict[str, Any]:
        self._phase = "running"
        await self._bus.emit(
            Event(
                type=EventType.AGENT_RESUMED,
                agent_id=self._agent_id,
            )
        )
        return {"success": True, "message": "Agent resumed"}

    # -- Internal ----------------------------------------------------------

    async def _fan_out(self, event: Event) -> None:
        """Forward every event to all connected SSE clients."""
        dead: set[asyncio.Queue[Event | None]] = set()
        for queue in self._event_queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
        self._event_queues -= dead
