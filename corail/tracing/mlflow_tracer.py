"""MLflow tracing for Corail agent pipelines.

Auto-traces every conversation, tool call, RAG retrieval, and guard check.
Integrates with MLflow Prompt Registry for prompt versioning.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Track if MLflow is available
_mlflow = None
_initialized = False


def init_tracing(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    agent_name: str = "default",
) -> bool:
    """Initialize MLflow tracing. Returns True if successful."""
    global _mlflow, _initialized

    if _initialized:
        return _mlflow is not None

    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow-system.svc.cluster.local:5000")
    experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT", f"recif/agents/{agent_name}")

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.autolog(disable=True, log_traces=False)
        # Increase truncation limits for trace metadata (default 250 chars is too short)
        try:
            from mlflow.tracing import constant as _tc
            _tc.MAX_CHARS_IN_TRACE_INFO_METADATA = 4096
            _tc.TRACE_REQUEST_RESPONSE_PREVIEW_MAX_LENGTH_OSS = 10000
        except Exception:
            pass
        _mlflow = mlflow
        _initialized = True
        logger.info("MLflow tracing initialized: uri=%s experiment=%s", tracking_uri, experiment_name)
        return True
    except ImportError:
        logger.warning("mlflow not installed. Tracing disabled. Install with: pip install mlflow")
        _initialized = True
        return False
    except Exception as e:
        logger.warning("Failed to initialize MLflow: %s", e)
        _initialized = True
        return False


class MLflowTracer:
    """Traces agent pipeline execution to MLflow."""

    def __init__(self, agent_name: str = "default"):
        self.agent_name = agent_name
        init_tracing(agent_name=agent_name)

    @asynccontextmanager
    async def trace_request(self, user_input: str, metadata: dict[str, Any] | None = None):
        """Trace a full chat request."""
        if _mlflow is None:
            yield {"trace_id": None}
            return

        span = _mlflow.start_span(name="chat_request")
        span.set_inputs({"user_input": user_input, **(metadata or {})})
        t0 = time.monotonic()

        context = {
            "trace_id": span.span_id if hasattr(span, 'span_id') else None,
            "span": span,
        }

        try:
            yield context
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            span.set_attributes({"latency_ms": latency_ms, "agent": self.agent_name})
            span.end()

    def trace_tool_call(self, tool_name: str, args: dict, result: Any, latency_ms: float):
        """Log a tool call as a child span."""
        if _mlflow is None:
            return
        try:
            span = _mlflow.start_span(name=f"tool:{tool_name}")
            span.set_inputs({"tool": tool_name, "args": args})
            span.set_outputs({"result": str(result)})
            span.set_attributes({"latency_ms": latency_ms})
            span.end()
        except Exception as e:
            logger.debug("Failed to trace tool call: %s", e)

    def trace_rag_retrieval(self, query: str, chunks: list[dict], latency_ms: float):
        """Log a RAG retrieval as a child span."""
        if _mlflow is None:
            return
        try:
            span = _mlflow.start_span(name="rag_retrieval")
            span.set_inputs({"query": query})
            span.set_outputs({"chunk_count": len(chunks), "top_score": chunks[0].get("score", 0) if chunks else 0})
            span.set_attributes({"latency_ms": latency_ms})
            span.end()
        except Exception as e:
            logger.debug("Failed to trace RAG: %s", e)

    def trace_guard_check(self, guard_name: str, passed: bool, details: str = ""):
        """Log a guard check as a child span."""
        if _mlflow is None:
            return
        try:
            span = _mlflow.start_span(name=f"guard:{guard_name}")
            span.set_inputs({"guard": guard_name})
            span.set_outputs({"passed": passed, "details": details})
            span.end()
        except Exception as e:
            logger.debug("Failed to trace guard: %s", e)

    def log_response(self, response: str, tokens: int = 0, model: str = ""):
        """Log the final response."""
        if _mlflow is None:
            return
        try:
            _mlflow.log_metrics({
                "response_tokens": tokens,
            })
        except Exception as e:
            logger.debug("Failed to log response: %s", e)

    def log_feedback(self, trace_id: str, name: str, value: float, source: dict | None = None):
        """Log user or expert feedback on a trace."""
        if _mlflow is None:
            return
        try:
            _mlflow.log_feedback(
                trace_id=trace_id,
                name=name,
                value=value,
                source=source or {"type": "user"},
            )
        except Exception as e:
            logger.debug("Failed to log feedback: %s", e)


def load_prompt(name: str, version: int | None = None, alias: str | None = None) -> str | None:
    """Load a prompt from MLflow Prompt Registry."""
    if _mlflow is None:
        return None
    try:
        if alias:
            prompt = _mlflow.genai.load_prompt(name, alias=alias)
        elif version:
            prompt = _mlflow.genai.load_prompt(name, version=version)
        else:
            prompt = _mlflow.genai.load_prompt(name)
        return prompt.template if hasattr(prompt, 'template') else str(prompt)
    except Exception as e:
        logger.debug("Failed to load prompt %s: %s", name, e)
        return None


def register_prompt(name: str, template: str, commit_message: str = "") -> int | None:
    """Register a new prompt version in MLflow Prompt Registry."""
    if _mlflow is None:
        return None
    try:
        result = _mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message or f"Updated prompt for {name}",
        )
        return result.version if hasattr(result, 'version') else None
    except Exception as e:
        logger.debug("Failed to register prompt %s: %s", name, e)
        return None
