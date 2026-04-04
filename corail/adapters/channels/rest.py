"""REST channel adapter — normalizes HTTP JSON to/from ExecutionResponse."""

from typing import Any

from corail.core.agent_config import ExecutionResponse


class RestChannelAdapter:
    """Adapter for REST API channel normalization."""

    def normalize_input(self, raw_input: dict[str, Any]) -> str:
        """Extract user message from REST JSON body."""
        return str(raw_input.get("input", ""))

    def format_response(self, response: ExecutionResponse, request_id: str = "") -> dict[str, Any]:
        """Format response for REST API consumers."""
        return {
            "data": response.model_dump(),
            "meta": {"request_id": request_id},
        }
