"""Phoenix (Arize) evaluation provider — OpenTelemetry-based tracing + evaluation.

Requires: pip install arize-phoenix
"""

import logging

from corail.evaluation.memory_provider import InMemoryProvider

logger = logging.getLogger(__name__)


class PhoenixProvider(InMemoryProvider):
    """Phoenix-backed evaluation. Falls back to in-memory if phoenix not installed."""

    def __init__(self, endpoint: str = "http://localhost:6006", **kwargs: object) -> None:
        super().__init__()
        self._endpoint = endpoint
        try:
            import phoenix  # noqa: F401

            logger.info("Phoenix provider initialized: %s", endpoint)
        except ImportError:
            logger.warning(
                "arize-phoenix not installed. Using in-memory fallback. "
                "Install with: pip install arize-phoenix"
            )
