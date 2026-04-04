"""ChannelAdapter protocol — port for channel-specific request/response normalization."""

from typing import Any, Protocol, runtime_checkable

from corail.core.agent_config import ExecutionResponse


@runtime_checkable
class ChannelAdapter(Protocol):
    """Protocol for communication channel adapters (REST, WebSocket, Slack, Teams)."""

    def normalize_input(self, raw_input: dict[str, Any]) -> str:
        """Extract user message from channel-specific input format."""
        ...

    def format_response(self, response: ExecutionResponse, request_id: str = "") -> dict[str, Any]:
        """Format execution response for channel-specific output."""
        ...
