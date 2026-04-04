"""ControlService gRPC servicer -- bridges gRPC calls to shared endpoint logic.

This servicer implements control.v1.ControlServiceServicer using the same
shared endpoint functions that power the HTTP ControlServer.  It runs on
port 8001 alongside (or in place of) the FastAPI control server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import grpc

from corail.control.pb.control.v1 import control_pb2 as pb2
from corail.control.pb.control.v1 import control_pb2_grpc as pb2_grpc
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
from corail.core.stream import (
    ComponentEvent,
    PlanEvent,
    StreamEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from corail.events.types import Event

if TYPE_CHECKING:
    from corail.control.bridge import RecifBridge
    from corail.core.pipeline import Pipeline
    from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import threading

# Dedicated event loop for gRPC handlers — avoids conflicts with FastAPI's loop.
# asyncpg pools are bound to a specific loop, so we reuse this one for all gRPC calls.
_grpc_loop: asyncio.AbstractEventLoop | None = None
_grpc_loop_thread: threading.Thread | None = None


def _ensure_grpc_loop() -> asyncio.AbstractEventLoop:
    """Start a dedicated event loop in a background thread (once)."""
    global _grpc_loop, _grpc_loop_thread
    if _grpc_loop is not None and _grpc_loop.is_running():
        return _grpc_loop
    _grpc_loop = asyncio.new_event_loop()
    _grpc_loop_thread = threading.Thread(target=_grpc_loop.run_forever, daemon=True)
    _grpc_loop_thread.start()
    return _grpc_loop


def _run_async(coro: Any) -> Any:
    """Run an async coroutine on the dedicated gRPC event loop.

    gRPC Python servicer methods are synchronous. We submit the coroutine
    to a persistent event loop running in a background thread. This avoids
    creating new loops per call (which breaks asyncpg connection pools).
    """
    loop = _ensure_grpc_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


# ---------------------------------------------------------------------------
# Servicer
# ---------------------------------------------------------------------------

class ControlServiceServicer(pb2_grpc.ControlServiceServicer):
    """Maps gRPC ControlService RPCs to shared Corail endpoint logic."""

    def __init__(
        self,
        pipeline: Pipeline,
        storage_factory: Any,
        bridge: RecifBridge,
        memory_accessor: Any = None,
    ) -> None:
        self._pipeline = pipeline
        self._storage_factory = storage_factory
        self._bridge = bridge
        self._memory_accessor = memory_accessor
        self._active_generations: dict[str, str] = {}
        self._bg_tasks: set[asyncio.Task[None]] = set()

    # -- Storage / Memory accessors -----------------------------------------

    @property
    def _storage(self) -> StoragePort:
        return self._storage_factory()

    @property
    def _memory(self) -> Any:
        if self._memory_accessor is not None:
            return self._memory_accessor()
        return self._pipeline.memory

    # -- Chat ---------------------------------------------------------------

    def Chat(self, request: pb2.ChatRequest, context: grpc.ServicerContext) -> pb2.ChatResponse:
        options = dict(request.options) if request.options else {}
        result = _run_async(handle_chat(
            self._pipeline, self._storage,
            request.input, request.conversation_id or None, options,
        ))
        return pb2.ChatResponse(
            output=result["output"],
            conversation_id=result["conversation_id"],
        )

    def ChatStream(self, request: pb2.ChatRequest, context: grpc.ServicerContext) -> Any:
        options = dict(request.options) if request.options else {}

        cid, generator = _run_async(handle_chat_stream(
            self._pipeline, self._storage,
            request.input, request.conversation_id or None, options,
            active_generations=self._active_generations,
            bg_tasks=self._bg_tasks,
        ))

        # generator yields SSE strings like 'data: {"token": "..."}\n\n'
        # We need to convert them to ChatStreamEvent proto messages.
        for sse_line in _run_async(_drain_async_generator(generator)):
            event = _sse_to_proto(sse_line, cid)
            if event is not None:
                yield event

    # -- Conversations ------------------------------------------------------

    def ListConversations(
        self, request: pb2.ListConversationsRequest, context: grpc.ServicerContext,
    ) -> pb2.ListConversationsResponse:
        convos = _run_async(list_conversations(self._storage))
        return pb2.ListConversationsResponse(
            conversations=[
                pb2.Conversation(
                    id=c.get("id", c.get("conversation_id", "")),
                    title=c.get("title", ""),
                    created_at=c.get("created_at", ""),
                    message_count=c.get("message_count", 0),
                )
                for c in convos
            ]
        )

    def GetConversation(
        self, request: pb2.GetConversationRequest, context: grpc.ServicerContext,
    ) -> pb2.GetConversationResponse:
        data = _run_async(get_conversation(self._storage, request.conversation_id))
        if data is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Conversation not found")
            return pb2.GetConversationResponse()
        return pb2.GetConversationResponse(
            conversation_id=data["conversation_id"],
            messages=[
                pb2.Message(role=m["role"], content=m["content"])
                for m in data.get("messages", [])
            ],
        )

    def DeleteConversation(
        self, request: pb2.DeleteConversationRequest, context: grpc.ServicerContext,
    ) -> pb2.DeleteConversationResponse:
        deleted = _run_async(delete_conversation(self._storage, request.conversation_id))
        if not deleted:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Conversation not found")
        return pb2.DeleteConversationResponse(deleted=deleted)

    def GetGenerationStatus(
        self, request: pb2.GetGenerationStatusRequest, context: grpc.ServicerContext,
    ) -> pb2.GetGenerationStatusResponse:
        status = _run_async(get_generation_status(self._active_generations, request.conversation_id))
        return pb2.GetGenerationStatusResponse(
            generating=status.get("generating", False),
            partial=status.get("partial", ""),
        )

    # -- Memory -------------------------------------------------------------

    def ListMemories(
        self, request: pb2.ListMemoriesRequest, context: grpc.ServicerContext,
    ) -> pb2.ListMemoriesResponse:
        limit = request.limit if request.limit > 0 else 50
        data = _run_async(list_memories(self._memory, limit=limit))
        return pb2.ListMemoriesResponse(
            memories=[_memory_entry_to_proto(m) for m in data.get("memories", [])],
            count=data.get("count", 0),
        )

    def MemoryStatus(
        self, request: pb2.MemoryStatusRequest, context: grpc.ServicerContext,
    ) -> pb2.MemoryStatusResponse:
        data = _run_async(memory_status(self._memory))
        return pb2.MemoryStatusResponse(
            enabled=data.get("enabled", False),
            backend=data.get("backend", ""),
            backend_label=data.get("backend_label", ""),
            persistent=data.get("persistent", False),
            search_type=data.get("search_type", ""),
            search_label=data.get("search_label", ""),
            scope=data.get("scope", ""),
            scope_label=data.get("scope_label", ""),
            storage_location=data.get("storage_location", ""),
            count=data.get("count", 0),
        )

    def StoreMemory(
        self, request: pb2.StoreMemoryRequest, context: grpc.ServicerContext,
    ) -> pb2.StoreMemoryResponse:
        result = _run_async(store_memory(
            self._memory, request.content, request.category, request.source,
        ))
        if "error" in result:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(result["error"])
            return pb2.StoreMemoryResponse(stored=False)
        return pb2.StoreMemoryResponse(stored=True)

    def SearchMemories(
        self, request: pb2.SearchMemoriesRequest, context: grpc.ServicerContext,
    ) -> pb2.SearchMemoriesResponse:
        top_k = request.top_k if request.top_k > 0 else 10
        entries = _run_async(search_memories(self._memory, request.query, top_k))
        return pb2.SearchMemoriesResponse(
            memories=[_memory_entry_to_proto(m) for m in entries],
        )

    def DeleteMemory(
        self, request: pb2.DeleteMemoryRequest, context: grpc.ServicerContext,
    ) -> pb2.DeleteMemoryResponse:
        result = _run_async(delete_memory_entry(self._memory, request.entry_id))
        if "error" in result:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(result["error"])
            return pb2.DeleteMemoryResponse(deleted=False)
        return pb2.DeleteMemoryResponse(deleted=True)

    # -- Agent control ------------------------------------------------------

    def GetStatus(
        self, request: pb2.GetStatusRequest, context: grpc.ServicerContext,
    ) -> pb2.GetStatusResponse:
        bus = self._bridge._bus
        return pb2.GetStatusResponse(
            agent_id=self._bridge._agent_id,
            phase=self._bridge.phase,
            active_sessions=0,
            tools_count=0,
            kbs_count=0,
            event_subscribers=bus.subscriber_count,
            event_history_size=len(bus.history),
        )

    def UpdateConfig(
        self, request: pb2.UpdateConfigRequest, context: grpc.ServicerContext,
    ) -> pb2.UpdateConfigResponse:
        config = dict(request.config) if request.config else {}
        result = _run_async(self._bridge._commands.execute("update_config", {"config": config}))
        return pb2.UpdateConfigResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    def Reload(
        self, request: pb2.ReloadRequest, context: grpc.ServicerContext,
    ) -> pb2.ReloadResponse:
        result = _run_async(self._bridge._commands.execute("reload", {
            "target": request.target,
            "reason": request.reason,
        }))
        return pb2.ReloadResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    def Pause(
        self, request: pb2.PauseRequest, context: grpc.ServicerContext,
    ) -> pb2.PauseResponse:
        result = _run_async(self._bridge._commands.execute("pause", {}))
        return pb2.PauseResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    def Resume(
        self, request: pb2.ResumeRequest, context: grpc.ServicerContext,
    ) -> pb2.ResumeResponse:
        result = _run_async(self._bridge._commands.execute("resume", {}))
        return pb2.ResumeResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    # -- Evaluation ---------------------------------------------------------

    def Evaluate(
        self, request: pb2.EvaluateRequest, context: grpc.ServicerContext,
    ) -> pb2.EvaluateResponse:
        # Convert proto repeated EvalCase to list[dict]
        dataset = [
            {"input": c.input, "expected_output": c.expected_output}
            for c in request.cases
        ]

        # Convert scorers config from proto map (JSON strings) to dict
        scorers_config: dict[str, dict] | None = None
        if request.scorers_config_json:
            scorers_config = {
                name: json.loads(config_json)
                for name, config_json in request.scorers_config_json.items()
            }

        result = _run_async(handle_evaluate(
            pipeline=self._pipeline,
            dataset=dataset,
            agent_id=request.agent_id,
            agent_version=request.agent_version,
            scorers_config=scorers_config,
            min_quality_score=request.min_quality_score,
            risk_profile=request.risk_profile or "standard",
            callback_url=request.callback_url or None,
        ))

        return pb2.EvaluateResponse(
            run_id=result.get("run_id", ""),
            status=result.get("status", ""),
            scores={k: float(v) for k, v in result.get("scores", {}).items()},
            passed=result.get("passed", False),
            verdict=result.get("verdict", ""),
            mlflow_run_id=result.get("mlflow_run_id", ""),
        )

    # -- Events stream ------------------------------------------------------

    def StreamEvents(
        self, request: pb2.StreamEventsRequest, context: grpc.ServicerContext,
    ) -> Any:
        """Server-streaming RPC: yields AgentEvent messages until the client disconnects."""
        queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._bridge._event_queues.add(queue)

        try:
            while context.is_active():
                try:
                    event = _run_async(_queue_get_timeout(queue, timeout=1.0))
                except TimeoutError:
                    continue
                if event is None:
                    break
                event_dict = event.to_dict()
                yield pb2.AgentEvent(
                    type=event_dict.get("type", ""),
                    data_json=json.dumps(event_dict.get("data", {})),
                    timestamp=event_dict.get("timestamp", ""),
                )
        finally:
            self._bridge._event_queues.discard(queue)

    # -- Health -------------------------------------------------------------

    def Health(
        self, request: pb2.HealthRequest, context: grpc.ServicerContext,
    ) -> pb2.HealthResponse:
        return pb2.HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _memory_entry_to_proto(m: dict[str, Any]) -> pb2.MemoryEntry:
    """Convert a memory dict to a MemoryEntry proto."""
    return pb2.MemoryEntry(
        id=str(m.get("id", "")),
        content=m.get("content", ""),
        category=m.get("category", ""),
        source=m.get("source", ""),
        relevance=float(m.get("relevance", 0.0)),
        timestamp=str(m.get("timestamp", "")),
    )


async def _drain_async_generator(gen: Any) -> list[str]:
    """Collect all items from an async generator into a list."""
    items: list[str] = []
    async for item in gen:
        items.append(item)
    return items


async def _queue_get_timeout(queue: asyncio.Queue[Any], timeout: float) -> Any:
    """Get from an asyncio queue with a timeout."""
    return await asyncio.wait_for(queue.get(), timeout=timeout)


def _sse_to_proto(sse_line: str, conversation_id: str) -> pb2.ChatStreamEvent | None:
    """Parse an SSE data line into a ChatStreamEvent proto message."""
    if not sse_line.startswith("data: "):
        return None
    payload_str = sse_line.removeprefix("data: ").strip()
    if not payload_str:
        return None

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return None

    # Token event
    if "token" in payload:
        return pb2.ChatStreamEvent(token=payload["token"])

    # Done event
    if payload.get("done"):
        return pb2.ChatStreamEvent(
            done=pb2.DoneEvent(
                conversation_id=payload.get("conversation_id", conversation_id),
            ),
        )

    event_type = payload.get("type", "")

    # Tool start
    if event_type == "tool_start":
        return pb2.ChatStreamEvent(
            tool_start=pb2.ToolStartEvent(
                tool=payload.get("tool", ""),
                args_json=json.dumps(payload.get("args", {})),
                call_id=payload.get("call_id", ""),
            ),
        )

    # Tool end
    if event_type == "tool_end":
        return pb2.ChatStreamEvent(
            tool_end=pb2.ToolEndEvent(
                tool=payload.get("tool", ""),
                output=payload.get("output", ""),
                success=payload.get("success", True),
                call_id=payload.get("call_id", ""),
            ),
        )

    # Component
    if event_type == "component":
        return pb2.ChatStreamEvent(
            component=pb2.ComponentEvent(
                component=payload.get("component", ""),
                props_json=json.dumps(payload.get("props", {})),
            ),
        )

    # Plan
    if event_type == "plan":
        plan_data = payload.get("plan", "{}")
        if isinstance(plan_data, str):
            try:
                plan_data = json.loads(plan_data)
            except json.JSONDecodeError:
                plan_data = {}
        return pb2.ChatStreamEvent(
            plan=pb2.PlanEvent(
                plan_goal=plan_data.get("goal", ""),
                step_description=plan_data.get("step", ""),
                step_status=plan_data.get("status", ""),
                step_index=plan_data.get("index", 0),
                total_steps=plan_data.get("total", 0),
            ),
        )

    return None
