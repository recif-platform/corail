"""RAGStrategy — deprecated, use agent-react with KB tools instead.

Kept for backward compatibility: ``strategy: rag`` still boots but logs a
deprecation warning and delegates to UnifiedAgentStrategy.
"""

import logging

from corail.strategies.agent import UnifiedAgentStrategy

logger = logging.getLogger(__name__)


class RAGStrategy(UnifiedAgentStrategy):
    """Deprecated: use agent-react strategy with KB tools instead."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "RAGStrategy is deprecated. Use strategy='agent-react' with "
            "knowledgeBases in your Agent CRD — KB search is now tool-based."
        )
        kwargs.pop("retriever", None)
        super().__init__(*args, **kwargs)
