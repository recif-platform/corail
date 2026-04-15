"""Channel abstract base class."""

import os
from abc import ABC, abstractmethod

from corail.config import Settings
from corail.core.pipeline import Pipeline

try:
    import mlflow
    from corail.tracing.mlflow_listener import get_collected_events, reset_events
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

    def get_collected_events() -> list:
        return []

    def reset_events() -> None:
        pass


class Channel(ABC):
    """Base class for I/O channels. Channel is the OUTER layer that starts the server."""

    def __init__(self, pipeline: Pipeline, settings: Settings) -> None:
        self.pipeline = pipeline
        self.settings = settings

    @abstractmethod
    def start(self) -> None:
        """Start the channel (blocks until shutdown)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the channel gracefully."""
        ...

    def log_chat_trace(
        self,
        user_input: str,
        conversation_id: str,
        full_response: str,
        collected_events: list[dict] | None = None,
    ) -> str | None:
        """Log a chat interaction as an MLflow trace. Returns trace_id or None."""
        if not _HAS_MLFLOW:
            return None
        try:
            agent_name = os.environ.get("CORAIL_AGENT_NAME", "default")
            agent_version = os.environ.get("RECIF_AGENT_VERSION", "unknown").replace(".", "-")
            mlflow.set_active_model(name=f"{agent_name}-v{agent_version}")

            collected = collected_events or []

            @mlflow.trace(name="chat_stream", span_type="AGENT")
            def _trace_fn(user_input: str, conversation_id: str) -> str:
                mlflow.update_current_trace(
                    tags={
                        "agent_type": "corail",
                        "channel": self.__class__.__name__.replace("Channel", "").lower(),
                        "conversation_id": conversation_id,
                        "recif.agent_name": agent_name,
                        "recif.agent_version": agent_version,
                    },
                    metadata={"mlflow.trace.session": conversation_id},
                )
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
                return full_response

            _trace_fn(user_input, conversation_id)
            trace = mlflow.get_last_active_trace()
            return trace.info.trace_id if trace else None
        except Exception:
            return None
