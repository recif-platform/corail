"""Standalone control plane server -- Recif's dedicated door to the agent.

Runs on port 8001 independently of the agent's channel (which runs on 8000).
This ensures Recif can always communicate with the agent regardless of channel type.

The server exposes two transports on the same port:
  - HTTP/REST (FastAPI + uvicorn) for backward compatibility and browser access
  - gRPC (grpcio) for Recif <-> Corail control plane communication

The gRPC server runs in a background thread alongside the FastAPI server.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from corail import __version__
from corail.control.bridge import RecifBridge
from corail.control.endpoints import (
    delete_conversation,
    delete_memory_entry,
    get_conversation,
    get_generation_status,
    handle_chat,
    handle_chat_stream,
    handle_evaluate,
    list_conversations,
    list_memories,
    memory_status,
    search_memories,
    store_memory,
)
from corail.control.grpc_server import GrpcControlServer
from corail.events.bus import EventBus

if TYPE_CHECKING:
    from corail.config import Settings
    from corail.core.pipeline import Pipeline
    from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)

# gRPC port offset from the HTTP control port (HTTP 8001 -> gRPC 9001).
_GRPC_PORT_OFFSET = 1000


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    input: str = Field(..., min_length=1)
    conversation_id: str | None = None
    options: dict = Field(default_factory=dict)


class MemoryStoreRequest(BaseModel):
    """Request body for storing a memory."""

    content: str = Field(..., min_length=1)
    category: str = "observation"
    source: str = ""


class MemorySearchRequest(BaseModel):
    """Request body for searching memories."""

    query: str = Field(..., min_length=1)
    top_k: int = 10


class EvalCaseRequest(BaseModel):
    """A single evaluation test case."""

    input: str = Field(..., min_length=1)
    expected_output: str = ""


class EvaluateRequest(BaseModel):
    """Request body for running evaluation against the agent pipeline."""

    dataset: list[EvalCaseRequest] = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    agent_version: str = "unknown"
    scorers: dict[str, dict] = Field(default_factory=dict)
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_profile: str = "standard"
    callback_url: str | None = None


# ---------------------------------------------------------------------------
# Control Server
# ---------------------------------------------------------------------------


class ControlServer:
    """Standalone control plane server -- Recif's dedicated door to the agent.

    Runs on port 8001 independently of the agent's channel (which runs on 8000).
    This ensures Recif can always communicate with the agent regardless of channel type.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        settings: Settings,
        storage: StoragePort | None = None,
        bridge: RecifBridge | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._storage = storage
        self._bridge = bridge or RecifBridge(EventBus())
        self._bg_tasks: set[asyncio.Task] = set()
        self._active_generations: dict[str, str] = {}
        self._grpc_server: GrpcControlServer | None = None
        self._grpc_thread: threading.Thread | None = None

        self.app = FastAPI(title="Corail Control Plane", version=__version__)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Health
        self.app.get("/healthz")(self._healthz)

        # Chat
        self.app.post("/control/chat")(self._chat)
        self.app.post("/control/chat/stream")(self._chat_stream)

        # Conversations
        self.app.get("/control/conversations")(self._list_conversations)
        self.app.get("/control/conversations/{conversation_id}")(self._get_conversation)
        self.app.delete("/control/conversations/{conversation_id}")(self._delete_conversation)
        self.app.get("/control/conversations/{conversation_id}/status")(self._get_generation_status)

        # Memory
        self.app.get("/control/memory")(self._list_memories)
        self.app.get("/control/memory/status")(self._memory_status)
        self.app.post("/control/memory")(self._store_memory)
        self.app.post("/control/memory/search")(self._search_memories)
        self.app.delete("/control/memory/{entry_id}")(self._delete_memory)

        # Evaluation
        self.app.post("/control/evaluate")(self._evaluate)

        # Bridge (status, events, config, reload, pause, resume)
        self._bridge.mount(self.app)

    # -- Storage accessor ---------------------------------------------------

    @property
    def storage(self) -> StoragePort:
        """Lazy-init storage if not injected."""
        if self._storage is None:
            from corail.storage.factory import StorageFactory

            self._storage = StorageFactory.create(self._settings.storage)
        return self._storage

    @property
    def _memory(self):
        """Access the pipeline's memory manager."""
        return self._pipeline.memory

    # -- Health -------------------------------------------------------------

    async def _healthz(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # -- Chat ---------------------------------------------------------------

    async def _chat(self, request: ChatRequest) -> JSONResponse:
        result = await handle_chat(
            self._pipeline,
            self.storage,
            request.input,
            request.conversation_id,
            request.options,
        )
        return JSONResponse(result)

    async def _chat_stream(self, request: ChatRequest) -> StreamingResponse:
        _cid, generator = await handle_chat_stream(
            self._pipeline,
            self.storage,
            request.input,
            request.conversation_id,
            request.options,
            active_generations=self._active_generations,
            bg_tasks=self._bg_tasks,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    # -- Conversations ------------------------------------------------------

    async def _list_conversations(self) -> JSONResponse:
        convos = await list_conversations(self.storage)
        return JSONResponse({"conversations": convos})

    async def _get_conversation(self, conversation_id: str) -> JSONResponse:
        data = await get_conversation(self.storage, conversation_id)
        if data is None:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        return JSONResponse(data)

    async def _delete_conversation(self, conversation_id: str) -> JSONResponse:
        deleted = await delete_conversation(self.storage, conversation_id)
        if not deleted:
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        return JSONResponse({"deleted": True, "conversation_id": conversation_id})

    async def _get_generation_status(self, conversation_id: str) -> JSONResponse:
        status = await get_generation_status(self._active_generations, conversation_id)
        return JSONResponse(status)

    # -- Memory -------------------------------------------------------------

    async def _list_memories(self, limit: int = 50) -> JSONResponse:
        data = await list_memories(self._memory, limit=limit)
        return JSONResponse(data)

    async def _memory_status(self) -> JSONResponse:
        data = await memory_status(self._memory)
        return JSONResponse(data)

    async def _store_memory(self, request: MemoryStoreRequest) -> JSONResponse:
        result = await store_memory(self._memory, request.content, request.category, request.source)
        if "error" in result:
            return JSONResponse(result, status_code=503)
        return JSONResponse(result)

    async def _search_memories(self, request: MemorySearchRequest) -> JSONResponse:
        entries = await search_memories(self._memory, request.query, request.top_k)
        return JSONResponse({"memories": entries})

    async def _delete_memory(self, entry_id: str) -> JSONResponse:
        result = await delete_memory_entry(self._memory, entry_id)
        if "error" in result:
            return JSONResponse(result, status_code=503)
        return JSONResponse(result)

    # -- Evaluation ---------------------------------------------------------

    async def _evaluate(self, request: EvaluateRequest) -> JSONResponse:
        dataset = [{"input": c.input, "expected_output": c.expected_output} for c in request.dataset]
        result = await handle_evaluate(
            pipeline=self._pipeline,
            dataset=dataset,
            agent_id=request.agent_id,
            agent_version=request.agent_version,
            scorers_config=request.scorers or None,
            min_quality_score=request.min_quality_score,
            risk_profile=request.risk_profile,
            callback_url=request.callback_url,
        )
        status_code = 200 if result.get("status") == "completed" else 202
        return JSONResponse(result, status_code=status_code)

    # -- Server lifecycle ---------------------------------------------------

    def _start_grpc(self, grpc_port: int) -> None:
        """Start the gRPC ControlService in a background thread."""
        self._grpc_server = GrpcControlServer(
            pipeline=self._pipeline,
            storage_factory=lambda: self.storage,
            bridge=self._bridge,
            memory_accessor=lambda: self._memory,
        )
        self._grpc_thread = threading.Thread(
            target=self._grpc_server.start,
            kwargs={"port": grpc_port, "block": True},
            daemon=True,
            name="grpc-control",
        )
        self._grpc_thread.start()
        logger.info("gRPC control plane started on port %d (background thread)", grpc_port)

    def stop_grpc(self) -> None:
        """Stop the gRPC server if running."""
        if self._grpc_server is not None:
            self._grpc_server.stop()

    def start(self, port: int | None = None, grpc_port: int | None = None) -> None:
        """Start the control server (blocks). Uses settings.control_port by default.

        Also starts a gRPC ControlService on ``grpc_port`` (default: settings.grpc_control_port).
        """
        port = port or self._settings.control_port
        grpc_port = grpc_port or getattr(self._settings, "grpc_control_port", port + _GRPC_PORT_OFFSET)

        # Start gRPC in background thread
        self._start_grpc(grpc_port)

        # Start HTTP (blocks)
        logger.info("HTTP control plane starting on port %d", port)
        uvicorn.run(self.app, host=self._settings.host, port=port, log_level="info")
