"""REST channel — FastAPI server with persistent conversation history + SSE streaming.

Serves both the agent channel (end users) and Récif control plane on the same port.
Routes under /control/* are aliases for the same handlers, used by the Récif proxy.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

try:
    from corail.tracing.mlflow_listener import get_collected_events, reset_events
except ImportError:

    def get_collected_events() -> list:
        return []

    def reset_events() -> None:
        pass


def _set_mlflow_model() -> None:
    """Re-assert active model in trace context (may be lost across async boundaries)."""
    _agent_name = os.environ.get("CORAIL_AGENT_NAME", "default")
    _agent_version = os.environ.get("RECIF_AGENT_VERSION", "unknown")
    safe_version = _agent_version.replace(".", "-")
    mlflow.set_active_model(name=f"{_agent_name}-v{safe_version}")


def _log_chat_trace(
    user_input: str,
    conversation_id: str,
    history_length: int,
    clean_output: str,
    raw_output: str,
    collected_events: list[dict] | None = None,
) -> str | None:
    """Log a streaming chat trace with child spans reconstructed from events.

    Returns the MLflow trace_id for feedback linking.
    """
    if not _HAS_MLFLOW:
        return None
    collected = collected_events or []
    _set_mlflow_model()

    @mlflow.trace(name="chat_stream", span_type="AGENT")
    def _trace_fn(user_input: str, conversation_id: str, history_length: int) -> str:
        _version = os.environ.get("RECIF_AGENT_VERSION", "unknown")
        _agent = os.environ.get("CORAIL_AGENT_NAME", "default")
        mlflow.update_current_trace(
            tags={
                "agent_type": "corail",
                "conversation_id": conversation_id,
                "recif.agent_version": _version,
                "recif.agent_name": _agent,
            },
            metadata={"mlflow.trace.session": conversation_id, "mlflow.trace.user": "default"},
        )

        # Reconstruct child spans from collected events.
        # tool_call events carry args, tool_result events carry output — pair them by name.
        pending_args: dict[str, dict] = {}
        for evt in collected:
            etype = evt.get("type", "")
            if etype == "llm_call_completed":
                with mlflow.start_span(name=f"llm_call_{evt.get('round', 0)}", span_type="CHAT_MODEL"):
                    mlflow.get_current_active_span().set_attributes({"stop_reason": evt.get("stop_reason", "")})
            elif etype == "tool_call":
                pending_args[evt.get("name", "")] = evt.get("args", {})
            elif etype == "tool_result":
                name = evt.get("name", "?")
                span_type = "RETRIEVER" if name.startswith("search_") else "TOOL"
                with mlflow.start_span(name=f"tool:{name}", span_type=span_type):
                    span = mlflow.get_current_active_span()
                    span.set_inputs({"tool": name, "args": pending_args.pop(name, {})})
                    span.set_outputs({"output": evt.get("output", "")[:500], "success": True})
            elif etype == "tool_error":
                name = evt.get("name", "?")
                with mlflow.start_span(name=f"tool:{name}", span_type="TOOL"):
                    span = mlflow.get_current_active_span()
                    span.set_inputs({"tool": name, "args": pending_args.pop(name, {})})
                    span.set_outputs({"error": evt.get("error", "")})
                    span.set_status("ERROR")
            elif etype == "guard_blocked":
                with mlflow.start_span(name="guard_blocked", span_type="CHAIN"):
                    mlflow.get_current_active_span().set_attributes(
                        {"direction": evt.get("direction", ""), "reason": evt.get("reason", "")}
                    )

        return clean_output

    _trace_fn(user_input, conversation_id, history_length)
    try:
        trace_id = mlflow.get_last_active_trace()
        return trace_id.info.trace_id if trace_id else None
    except Exception:
        return None


import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from corail import __version__
from corail.channels.base import Channel
from corail.config import Settings
from corail.core.pipeline import Pipeline
from corail.core.stream import StreamEvent, StreamToken
from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)

# Hard deadline for post-response background work that keeps the SSE
# connection open (suggestions). Beyond this the stream is closed so proxies
# don't drop the connection mid-chunk. Override with CORAIL_SUGGESTIONS_TIMEOUT.
_SUGGESTIONS_TIMEOUT = float(os.environ.get("CORAIL_SUGGESTIONS_TIMEOUT", "5"))

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_llm_title(raw: str) -> str:
    """Extract a short title from raw LLM output, defending against common
    artefacts: <think> blocks, markdown fences, "Title:" prefixes, quotes,
    trailing punctuation and multiline output. Returns an empty string when
    nothing usable is left."""
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    if not text:
        return ""
    # First non-empty line — LLMs sometimes add a preamble or trailing notes.
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line:
            text = line
            break
    else:
        return ""
    # Strip wrapping quotes, then trailing punctuation, then quotes again
    # (handles mixed endings like `"Coral Reef".`). rstrip with `.` is safe
    # for internal dots ("Build 3.0.1" ends with `1`, not `.`).
    text = text.strip('"').strip("'")
    text = text.rstrip(".!?,:;")
    text = text.strip('"').strip("'").strip()
    if text.lower().startswith("title:"):
        text = text[6:].strip().strip('"').strip("'")
    return text


class ChatRequest(BaseModel):
    input: str = Field(..., min_length=1)
    conversation_id: str | None = None
    options: dict = Field(default_factory=dict)


class MemoryStoreRequest(BaseModel):
    content: str = Field(..., min_length=1)
    category: str = "observation"
    source: str = ""


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 10


class RestChannel(Channel):
    """REST channel with SSE streaming. Also serves /control/* for Récif proxy."""

    def __init__(self, pipeline: Pipeline, settings: Settings, storage: StoragePort | None = None) -> None:
        super().__init__(pipeline, settings)
        self._storage = storage
        self._bg_tasks: set[asyncio.Task] = set()
        self._active_generations: dict[str, str] = {}
        self._suggestion_provider = self._build_suggestion_provider()
        self.app = FastAPI(title="Corail Agent", version=__version__)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Channel routes (end users)
        self.app.get("/healthz")(self._healthz)
        self.app.post("/chat")(self._chat)
        self.app.post("/chat/stream")(self._chat_stream)
        self.app.get("/conversations")(self._list_conversations)
        self.app.get("/conversations/{conversation_id}")(self._get_conversation)
        self.app.delete("/conversations/{conversation_id}")(self._delete_conversation)
        self.app.get("/conversations/{conversation_id}/status")(self._get_generation_status)
        self.app.get("/memory")(self._list_memories)
        self.app.get("/memory/status")(self._memory_status)
        self.app.post("/memory")(self._store_memory)
        self.app.post("/memory/search")(self._search_memories)
        self.app.delete("/memory/{entry_id}")(self._delete_memory)
        self.app.get("/suggestions")(self._get_static_suggestions)

        # Control plane aliases (Récif proxy uses /control/* prefix)
        self.app.post("/control/chat")(self._chat)
        self.app.post("/control/chat/stream")(self._chat_stream)
        self.app.get("/control/conversations")(self._list_conversations)
        self.app.get("/control/conversations/{conversation_id}")(self._get_conversation)
        self.app.delete("/control/conversations/{conversation_id}")(self._delete_conversation)
        self.app.get("/control/conversations/{conversation_id}/status")(self._get_generation_status)
        self.app.get("/control/memory")(self._list_memories)
        self.app.get("/control/memory/status")(self._memory_status)
        self.app.post("/control/memory")(self._store_memory)
        self.app.post("/control/memory/search")(self._search_memories)
        self.app.delete("/control/memory/{entry_id}")(self._delete_memory)
        self.app.get("/control/suggestions")(self._get_static_suggestions)

    @property
    def storage(self) -> StoragePort:
        if self._storage is None:
            from corail.storage.factory import StorageFactory

            self._storage = StorageFactory.create(self.settings.storage)
        return self._storage

    @property
    def _memory(self):
        return self.pipeline.memory

    # ------------------------------------------------------------------
    # Suggestion provider
    # ------------------------------------------------------------------

    def _build_suggestion_provider(self):
        """Build the appropriate suggestion provider based on config."""
        from corail.models.factory import background_model
        from corail.suggestions.provider import LLMSuggestionProvider, StaticSuggestionProvider

        static_suggestions = self._parse_static_suggestions()
        provider_type = self.settings.suggestions_provider

        # Suggestions run on the background model (see background_model()) so a
        # heavy chat model doesn't keep the SSE connection open for minutes.
        _PROVIDER_BUILDERS = {
            "static": lambda: StaticSuggestionProvider(static_suggestions) if static_suggestions else None,
            "llm": lambda: LLMSuggestionProvider(background_model(self.pipeline._strategy.model)),
        }

        builder = _PROVIDER_BUILDERS.get(provider_type)
        return builder() if builder else None

    def _parse_static_suggestions(self) -> list[str]:
        """Parse CORAIL_SUGGESTIONS env var as a JSON array."""
        raw = self.settings.suggestions
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(s) for s in parsed if isinstance(s, str) and s.strip()]
        except (json.JSONDecodeError, ValueError):
            logger.warning("Invalid CORAIL_SUGGESTIONS JSON: %s", raw)
        return []

    async def _generate_suggestions(self, last_response: str, history: list[dict[str, str]] | None = None) -> list[str]:
        """Generate follow-up suggestions. Returns empty list on failure (graceful degradation)."""
        if self._suggestion_provider is None:
            return []
        try:
            return await self._suggestion_provider.generate(last_response, history=history)
        except Exception:
            logger.debug("Suggestion generation failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    async def _get_or_create_conversation(self, conversation_id: str | None) -> tuple[str, list[dict[str, str]]]:
        cid = conversation_id or str(uuid.uuid4())
        if not await self.storage.conversation_exists(cid):
            await self.storage.create_conversation(cid)
        history = await self.storage.get_messages(cid)
        return cid, history

    async def _set_title(self, conversation_id: str, user_message: str) -> None:
        """Set a first-pass conversation title from the user's opening message.

        Falls back to a truncated copy of the message so there's always
        *something* visible immediately. If a background model is configured,
        :meth:`_upgrade_title_llm` will then replace it with a model-generated
        title a few seconds later.
        """
        title = user_message.strip()[:60]
        if len(user_message.strip()) > 60:
            title = title.rsplit(" ", 1)[0] + "..."
        if title:
            await self.storage.update_title(conversation_id, title)

    async def _upgrade_title_llm(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Replace the fallback title with a short LLM-generated one.

        Only runs when ``CORAIL_BACKGROUND_MODEL`` is set; otherwise we'd be
        calling the (potentially very slow) chat model just to rename a row.
        Failures are swallowed — the truncated title from :meth:`_set_title`
        remains as a fine fallback.
        """
        if not os.environ.get("CORAIL_BACKGROUND_MODEL"):
            return
        from corail.models.factory import background_model

        model = background_model(self.pipeline._strategy.model)
        prompt = (
            "Summarise this conversation in 3 to 6 words, no punctuation, "
            "no quotes, title case. Return ONLY the title.\n\n"
            f"User: {user_message.strip()[:400]}\n"
            f"Assistant: {assistant_response.strip()[:400]}"
        )
        try:
            raw = await model.generate(messages=[{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.debug("title upgrade skipped: %s: %s", type(exc).__name__, exc)
            return
        title = _clean_llm_title(raw)
        if title:
            await self.storage.update_title(conversation_id, title[:80])

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def _healthz(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def _chat(self, request: ChatRequest) -> JSONResponse:
        cid, history = await self._get_or_create_conversation(request.conversation_id)
        await self.storage.append_message(cid, "user", request.input)

        # Wrap execute() in an MLflow root span so real-time child spans
        # (LLM calls, tool calls) created inside the strategy loop are
        # nested under it with proper timestamps and durations.
        output = await self._traced_execute(
            cid,
            request.input,
            history,
            request.options,
        )

        await self.storage.append_message(cid, "assistant", output)
        if len(history) == 0:
            await self._set_title(cid, request.input)
        return JSONResponse({"output": output, "conversation_id": cid})

    async def _traced_execute(
        self,
        cid: str,
        user_input: str,
        history: list[dict[str, str]],
        options: dict,
    ) -> str:
        """Run pipeline.execute inside an MLflow root span."""
        if not _HAS_MLFLOW:
            return await self.pipeline.execute(user_input, history=history, **options)
        _set_mlflow_model()
        with mlflow.start_span(name="chat", span_type="AGENT") as root:
            mlflow.update_current_trace(
                tags={"agent_type": "corail", "conversation_id": cid},
                metadata={"mlflow.trace.session": cid, "mlflow.trace.user": "default"},
            )
            root.set_inputs({"user_input": user_input, "history_length": len(history)})
            output = await self.pipeline.execute(user_input, history=history, **options)
            root.set_outputs(output)
            return output

    async def _chat_stream(self, request: ChatRequest) -> StreamingResponse:
        cid, history = await self._get_or_create_conversation(request.conversation_id)
        await self.storage.append_message(cid, "user", request.input)
        is_first_exchange = len(history) == 0
        user_input = request.input
        options = request.options

        queue: asyncio.Queue[StreamToken | None] = asyncio.Queue()
        self._active_generations[cid] = ""
        trace_info: dict[str, str | None] = {"trace_id": None}
        stream_meta: dict[str, object] = {"suggestions": []}

        async def _collect() -> None:
            full_response = ""
            reset_events()  # Clear events from previous request
            try:
                async for token in self.pipeline.execute_stream(user_input, history=history, **options):
                    if isinstance(token, str):
                        full_response += token
                        self._active_generations[cid] = full_response
                    await queue.put(token)
            except Exception as exc:
                logger.exception("Stream collection error for conversation %s", cid)
                await queue.put({"_error": f"{type(exc).__name__}: {exc}"})
            finally:
                self._active_generations.pop(cid, None)
                await queue.put(None)

                if full_response:
                    clean = full_response

                    def _xml_tool_result(m: re.Match) -> str:
                        return f"\n<tool_result>{m.group(1).strip()}</tool_result>\n"

                    clean = re.sub(
                        r"\n*---\n\*\*Executing tool\.\.\.\*\*\n+\*\*Result:\*\*\n```\n(.*?)\n```\n+---\n*",
                        _xml_tool_result,
                        clean,
                        flags=re.DOTALL,
                    )

                    def _xml_tool_call(m: re.Match) -> str:
                        return f"<tool_use>{m.group(1).strip()}</tool_use>"

                    clean = re.sub(r"```tool_call\n(.*?)\n```", _xml_tool_call, clean, flags=re.DOTALL)
                    clean = clean.strip()
                    await self.storage.append_message(cid, "assistant", clean or full_response)
                    if is_first_exchange:
                        await self._set_title(cid, user_input)

                # Log trace synchronously (with timeout) so trace_id is available for SSE
                if _HAS_MLFLOW and full_response:
                    _clean = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                    _events = list(get_collected_events())
                    reset_events()
                    try:
                        tid = await asyncio.wait_for(
                            asyncio.to_thread(
                                _log_chat_trace, user_input, cid, len(history), _clean, full_response, _events
                            ),
                            timeout=5.0,
                        )
                        if tid:
                            trace_info["trace_id"] = tid
                            await queue.put({"_trace_id": tid})
                    except TimeoutError:
                        logger.debug("MLflow trace logging timed out")
                    except Exception:
                        pass

                # Generate suggestions with a hard deadline. The SSE connection
                # stays open while we wait, so a slow background model (Ollama
                # swapping between chat + bg model can take 30s+) would cause
                # proxies to drop the connection with ERR_INCOMPLETE_CHUNKED_ENCODING.
                try:
                    suggestions = await asyncio.wait_for(
                        self._generate_suggestions(clean or full_response, history=history),
                        timeout=_SUGGESTIONS_TIMEOUT,
                    )
                    if suggestions:
                        await queue.put({"_suggestions": suggestions})
                except TimeoutError:
                    logger.debug("suggestions skipped: timed out after %ss", _SUGGESTIONS_TIMEOUT)
                except Exception as exc:
                    logger.debug("suggestions skipped: %s: %s", type(exc).__name__, exc)
                # Final signal to close the SSE connection
                await queue.put("_close")

                # Best-effort title upgrade via the background model. Runs
                # fully detached: the SSE response has already been closed
                # so a slow model or a failure here never affects the user.
                if is_first_exchange and full_response:
                    upgrade_task = asyncio.create_task(
                        self._upgrade_title_llm(cid, user_input, clean or full_response),
                    )
                    self._bg_tasks.add(upgrade_task)
                    upgrade_task.add_done_callback(self._bg_tasks.discard)

        task = asyncio.create_task(_collect())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

        async def event_stream() -> AsyncGenerator[str]:
            while True:
                token = await queue.get()
                # End of LLM stream — send done immediately so frontend unlocks
                if token is None:
                    done_payload: dict[str, object] = {"done": True, "conversation_id": cid}
                    if trace_info.get("trace_id"):
                        done_payload["trace_id"] = trace_info["trace_id"]
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    # Keep listening for suggestions and close signal
                    continue
                # Trace ID arrives after done (for feedback linking)
                if isinstance(token, dict) and "_trace_id" in token:
                    yield f"data: {json.dumps({'trace_id': token['_trace_id']})}\n\n"
                    continue
                # Suggestions arrive after done (user already unblocked)
                if isinstance(token, dict) and "_suggestions" in token:
                    yield f"data: {json.dumps({'suggestions': token['_suggestions']})}\n\n"
                    continue
                # Pipeline raised — forward the error message to the client.
                if isinstance(token, dict) and "_error" in token:
                    yield f"data: {json.dumps({'error': token['_error']})}\n\n"
                    continue
                # Final close
                if token == "_close":
                    break
                if isinstance(token, StreamEvent):
                    yield f"data: {json.dumps(token.to_sse_data())}\n\n"
                    continue
                yield f"data: {json.dumps({'token': token})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def _list_conversations(self) -> JSONResponse:
        conversations = await self.storage.list_conversations()
        return JSONResponse({"conversations": conversations})

    async def _get_conversation(self, conversation_id: str) -> JSONResponse:
        if not await self.storage.conversation_exists(conversation_id):
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        messages = await self.storage.get_messages(conversation_id)
        return JSONResponse({"conversation_id": conversation_id, "messages": messages})

    async def _delete_conversation(self, conversation_id: str) -> JSONResponse:
        if not await self.storage.conversation_exists(conversation_id):
            return JSONResponse({"error": "Conversation not found"}, status_code=404)
        await self.storage.delete_conversation(conversation_id)
        return JSONResponse({"deleted": True, "conversation_id": conversation_id})

    async def _get_generation_status(self, conversation_id: str) -> JSONResponse:
        partial = self._active_generations.get(conversation_id)
        if partial is not None:
            return JSONResponse({"generating": True, "partial": partial})
        return JSONResponse({"generating": False})

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    async def _get_static_suggestions(self) -> JSONResponse:
        """Return static suggestions from agent config (for empty chat state)."""
        return JSONResponse({"suggestions": self._parse_static_suggestions()})

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    async def _list_memories(self, limit: int = 50) -> JSONResponse:
        if self._memory is None:
            return JSONResponse({"memories": [], "count": 0})
        entries = await self._memory._storage.list_recent(limit=limit)
        return JSONResponse(
            {
                "memories": [
                    {
                        "id": e.id,
                        "content": e.content,
                        "category": e.category,
                        "source": e.source,
                        "relevance": e.relevance,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                    }
                    for e in entries
                ],
                "count": len(entries),
            }
        )

    async def _memory_status(self) -> JSONResponse:
        from corail.memory.in_memory import InMemoryStorage
        from corail.memory.pgvector import PgVectorMemoryStorage

        if self._memory is None:
            return JSONResponse(
                {
                    "enabled": False,
                    "backend": "none",
                    "persistent": False,
                    "search_type": "none",
                    "scope": "none",
                    "storage_location": "N/A",
                    "count": 0,
                }
            )
        storage = self._memory._storage
        if isinstance(storage, PgVectorMemoryStorage):
            info = {
                "enabled": True,
                "backend": "pgvector",
                "backend_label": "PostgreSQL + pgvector",
                "persistent": True,
                "search_type": "semantic",
                "search_label": "Vector similarity (cosine)",
                "scope": "long-term",
                "scope_label": "Cross-session, survives pod restarts",
                "storage_location": "PostgreSQL (agent_memories table)",
            }
        elif isinstance(storage, InMemoryStorage):
            info = {
                "enabled": True,
                "backend": "in_memory",
                "backend_label": "In-Memory (Python dict)",
                "persistent": False,
                "search_type": "keyword",
                "search_label": "Keyword overlap scoring",
                "scope": "short-term",
                "scope_label": "Session only, lost on pod restart",
                "storage_location": "Pod RAM (ephemeral)",
            }
        else:
            info = {
                "enabled": True,
                "backend": storage.__class__.__name__,
                "backend_label": storage.__class__.__name__,
                "persistent": False,
                "search_type": "unknown",
                "search_label": "Unknown",
                "scope": "unknown",
                "scope_label": "Unknown",
                "storage_location": "Unknown",
            }
        count = len(await storage.list_recent(limit=9999))
        info["count"] = count
        return JSONResponse(info)

    async def _store_memory(self, request: MemoryStoreRequest) -> JSONResponse:
        if self._memory is None:
            return JSONResponse({"error": "Memory not configured"}, status_code=503)
        await self._memory.remember(
            content=request.content, category=request.category, source=request.source or "manual"
        )
        return JSONResponse({"stored": True})

    async def _search_memories(self, request: MemorySearchRequest) -> JSONResponse:
        if self._memory is None:
            return JSONResponse({"memories": []})
        entries = await self._memory.recall(request.query, top_k=request.top_k)
        return JSONResponse(
            {
                "memories": [
                    {
                        "id": e.id,
                        "content": e.content,
                        "category": e.category,
                        "source": e.source,
                        "relevance": e.relevance,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                    }
                    for e in entries
                ],
            }
        )

    async def _delete_memory(self, entry_id: str) -> JSONResponse:
        if self._memory is None:
            return JSONResponse({"error": "Memory not configured"}, status_code=503)
        await self._memory._storage.delete(entry_id)
        return JSONResponse({"deleted": True, "id": entry_id})

    def start(self) -> None:
        uvicorn.run(self.app, host=self.settings.host, port=self.settings.port, log_level="info")

    def stop(self) -> None:
        pass
