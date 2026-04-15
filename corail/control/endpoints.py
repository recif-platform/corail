"""Shared endpoint logic for chat, conversations, memory, and evaluation.

These async functions encapsulate the core business logic used by both
the RestChannel (port 8000) and the ControlServer (port 8001).
Neither FastAPI nor HTTP details leak in here — callers handle serialization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from corail.core.stream import StreamEvent, StreamToken

if TYPE_CHECKING:
    from corail.core.pipeline import Pipeline
    from corail.memory.manager import MemoryManager
    from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)

# Background evaluation tasks — prevents GC and ensures clean shutdown.
_eval_bg_tasks: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


async def get_or_create_conversation(
    storage: StoragePort,
    conversation_id: str | None,
) -> tuple[str, list[dict[str, str]]]:
    """Return (cid, history), creating the conversation if needed."""
    cid = conversation_id or str(uuid.uuid4())
    if not await storage.conversation_exists(cid):
        await storage.create_conversation(cid)
    history = await storage.get_messages(cid)
    return cid, history


async def set_title(storage: StoragePort, conversation_id: str, user_message: str) -> None:
    """Derive a short title from the first user message."""
    title = user_message.strip()[:60]
    if len(user_message.strip()) > 60:
        title = title.rsplit(" ", 1)[0] + "..."
    if title:
        await storage.update_title(conversation_id, title)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def handle_chat(
    pipeline: Pipeline,
    storage: StoragePort,
    input_text: str,
    conversation_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single-turn chat, persist messages, return {output, conversation_id}."""
    options = options or {}
    cid, history = await get_or_create_conversation(storage, conversation_id)
    await storage.append_message(cid, "user", input_text)
    output = await pipeline.execute(input_text, history=history, **options)
    await storage.append_message(cid, "assistant", output)
    if len(history) == 0:
        await set_title(storage, cid, input_text)
    return {"output": output, "conversation_id": cid}


def _clean_response(raw: str) -> str:
    """Strip verbose tool-execution markdown from a raw streamed response."""
    clean = raw

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
    return clean.strip()


async def handle_chat_stream(
    pipeline: Pipeline,
    storage: StoragePort,
    input_text: str,
    conversation_id: str | None = None,
    options: dict[str, Any] | None = None,
    active_generations: dict[str, str] | None = None,
    bg_tasks: set[asyncio.Task] | None = None,
    on_complete: Callable[[str, str, str, list], Awaitable[None]] | None = None,
) -> tuple[str, AsyncGenerator[str]]:
    """Set up streaming chat. Returns (conversation_id, sse_generator).

    The caller is responsible for returning the generator as an SSE response.
    ``active_generations`` and ``bg_tasks`` are optional shared state dicts
    that the caller can pass for generation-status tracking.
    """
    options = options or {}
    active_generations = active_generations if active_generations is not None else {}
    bg_tasks = bg_tasks if bg_tasks is not None else set()

    cid, history = await get_or_create_conversation(storage, conversation_id)
    await storage.append_message(cid, "user", input_text)
    is_first_exchange = len(history) == 0

    queue: asyncio.Queue[StreamToken | None] = asyncio.Queue()
    active_generations[cid] = ""

    async def _collect() -> None:
        from corail.channels.base import get_collected_events, reset_events
        reset_events()
        full_response = ""
        try:
            async for token in pipeline.execute_stream(input_text, history=history, **options):
                if isinstance(token, str):
                    full_response += token
                    active_generations[cid] = full_response
                await queue.put(token)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).exception("Stream collection error: %s", exc)
        finally:
            await queue.put(None)
            active_generations.pop(cid, None)
            if full_response:
                clean = _clean_response(full_response)
                await storage.append_message(cid, "assistant", clean or full_response)
                if is_first_exchange:
                    await set_title(storage, cid, input_text)
                if on_complete:
                    try:
                        await on_complete(input_text, cid, clean or full_response, get_collected_events())
                    except Exception:
                        pass

    task = asyncio.create_task(_collect())
    bg_tasks.add(task)
    task.add_done_callback(bg_tasks.discard)

    async def event_stream() -> AsyncGenerator[str]:
        while True:
            token = await queue.get()
            if token is None:
                yield f"data: {json.dumps({'done': True, 'conversation_id': cid})}\n\n"
                break
            if isinstance(token, StreamEvent):
                yield f"data: {json.dumps(token.to_sse_data())}\n\n"
                continue
            yield f"data: {json.dumps({'token': token})}\n\n"

    return cid, event_stream()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


async def list_conversations(storage: StoragePort) -> list[dict[str, Any]]:
    """Return all conversations."""
    return await storage.list_conversations()


async def get_conversation(storage: StoragePort, cid: str) -> dict[str, Any] | None:
    """Return conversation detail or None if not found."""
    if not await storage.conversation_exists(cid):
        return None
    messages = await storage.get_messages(cid)
    return {"conversation_id": cid, "messages": messages}


async def delete_conversation(storage: StoragePort, cid: str) -> bool:
    """Delete a conversation. Returns False if it did not exist."""
    if not await storage.conversation_exists(cid):
        return False
    await storage.delete_conversation(cid)
    return True


async def get_generation_status(active_generations: dict[str, str], cid: str) -> dict[str, Any]:
    """Return generation status for a conversation."""
    partial = active_generations.get(cid)
    if partial is not None:
        return {"generating": True, "partial": partial}
    return {"generating": False}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def list_memories(memory: MemoryManager | None, limit: int = 50) -> dict[str, Any]:
    """List recent memories."""
    if memory is None:
        return {"memories": [], "count": 0}
    entries = await memory._storage.list_recent(limit=limit)
    return {
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


async def memory_status(memory: MemoryManager | None) -> dict[str, Any]:
    """Return metadata about the memory system."""
    from corail.memory.in_memory import InMemoryStorage
    from corail.memory.pgvector import PgVectorMemoryStorage

    if memory is None:
        return {
            "enabled": False,
            "backend": "none",
            "persistent": False,
            "search_type": "none",
            "scope": "none",
            "storage_location": "N/A",
            "count": 0,
        }

    storage = memory._storage
    backend = storage.__class__.__name__

    if isinstance(storage, PgVectorMemoryStorage):
        info: dict[str, Any] = {
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
            "backend": backend,
            "backend_label": backend,
            "persistent": False,
            "search_type": "unknown",
            "search_label": "Unknown",
            "scope": "unknown",
            "scope_label": "Unknown",
            "storage_location": "Unknown",
        }

    count = len(await storage.list_recent(limit=9999))
    info["count"] = count
    return info


async def store_memory(
    memory: MemoryManager | None, content: str, category: str = "observation", source: str = ""
) -> dict[str, Any]:
    """Store a memory entry. Returns error dict if memory is not configured."""
    if memory is None:
        return {"error": "Memory not configured"}
    await memory.remember(content=content, category=category, source=source or "manual")
    return {"stored": True}


async def search_memories(memory: MemoryManager | None, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search memories by query."""
    if memory is None:
        return []
    entries = await memory.recall(query, top_k=top_k)
    return [
        {
            "id": e.id,
            "content": e.content,
            "category": e.category,
            "source": e.source,
            "relevance": e.relevance,
            "timestamp": e.timestamp.isoformat() if e.timestamp else "",
        }
        for e in entries
    ]


async def delete_memory_entry(memory: MemoryManager | None, entry_id: str) -> dict[str, Any]:
    """Delete a single memory entry."""
    if memory is None:
        return {"error": "Memory not configured"}
    await memory._storage.delete(entry_id)
    return {"deleted": True, "id": entry_id}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def handle_evaluate(
    pipeline: Pipeline,
    dataset: list[dict[str, Any]],
    agent_id: str,
    agent_version: str,
    scorers_config: dict[str, dict[str, Any]] | None = None,
    min_quality_score: float = 0.0,
    risk_profile: str = "standard",
    callback_url: str | None = None,
) -> dict[str, Any]:
    """Run evaluation against the pipeline.

    Two modes:
    - **Synchronous** (no ``callback_url``): blocks until done, returns full results.
    - **Asynchronous** (with ``callback_url``): returns immediately with ``{status: "running"}``,
      POSTs results to ``callback_url`` when complete.

    Returns
    -------
    dict with keys: run_id, status, scores, passed, verdict, mlflow_run_id
    """
    from corail.evaluation.mlflow_evaluator import MLflowEvaluator

    evaluator = MLflowEvaluator()

    # Synchronous mode — block and return results
    if callback_url is None:
        result = await evaluator.evaluate(
            pipeline=pipeline,
            dataset=dataset,
            agent_id=agent_id,
            agent_version=agent_version,
            scorers_config=scorers_config,
            min_quality_score=min_quality_score,
            risk_profile=risk_profile,
        )
        return _eval_result_to_dict(result)

    # Asynchronous mode — fire and forget, POST results to callback
    run_id = f"eval_{uuid.uuid4().hex[:12]}"

    async def _run_and_callback() -> None:
        try:
            result = await evaluator.evaluate(
                pipeline=pipeline,
                dataset=dataset,
                agent_id=agent_id,
                agent_version=agent_version,
                scorers_config=scorers_config,
                min_quality_score=min_quality_score,
                risk_profile=risk_profile,
            )
            payload = _eval_result_to_dict(result)
        except Exception:
            logger.exception("Evaluation failed for agent %s v%s", agent_id, agent_version)
            payload = {
                "run_id": run_id,
                "status": "failed",
                "scores": {},
                "passed": False,
                "verdict": "Evaluation error",
            }

        # POST results to callback
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(callback_url, json=payload)
        except Exception:
            logger.exception("Failed to POST eval results to callback %s", callback_url)

    task = asyncio.create_task(_run_and_callback())
    # Track to prevent GC and ensure clean shutdown — same pattern as handle_chat_stream
    _eval_bg_tasks.add(task)
    task.add_done_callback(_eval_bg_tasks.discard)
    return {"run_id": run_id, "status": "running"}


def _eval_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert EvalRunResult dataclass to a plain dict."""
    return {
        "run_id": result.run_id,
        "status": result.status,
        "scores": result.scores,
        "passed": result.passed,
        "verdict": result.verdict,
        "mlflow_run_id": result.mlflow_run_id,
    }
