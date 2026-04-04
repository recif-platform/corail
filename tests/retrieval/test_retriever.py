"""Tests for RAG retrieval, deprecation shim, and retriever factory."""

import logging
from unittest.mock import AsyncMock

import pytest

from corail.retrieval.base import RetrievalResult, Retriever
from corail.retrieval.factory import RetrieverFactory
from corail.strategies.rag import RAGStrategy

# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_basic_fields(self):
        result = RetrievalResult(content="hello world", score=0.95, metadata={"filename": "doc.pdf"})
        assert result.content == "hello world"
        assert result.score == 0.95
        assert result.metadata == {"filename": "doc.pdf"}

    def test_default_metadata(self):
        result = RetrievalResult(content="text", score=0.5)
        assert result.metadata == {}

    def test_equality(self):
        a = RetrievalResult(content="a", score=0.9, metadata={})
        b = RetrievalResult(content="a", score=0.9, metadata={})
        assert a == b


# ---------------------------------------------------------------------------
# RAGStrategy deprecation shim
# ---------------------------------------------------------------------------


class TestRAGStrategyDeprecation:
    """RAGStrategy is now a deprecation shim that delegates to UnifiedAgentStrategy."""

    async def test_logs_deprecation_warning(self, caplog):
        model = AsyncMock()
        model.supports_tool_use = False
        model.generate = AsyncMock(return_value="response")
        with caplog.at_level(logging.WARNING):
            strategy = RAGStrategy(model=model)
        assert "deprecated" in caplog.text.lower()

    async def test_drops_retriever_kwarg(self):
        model = AsyncMock()
        model.supports_tool_use = False
        model.generate = AsyncMock(return_value="response")
        retriever = AsyncMock(spec=Retriever)
        # Should not raise — retriever kwarg is silently dropped
        strategy = RAGStrategy(model=model, retriever=retriever)
        # UnifiedAgentStrategy no longer accepts retriever — verify it wasn't stored
        assert not hasattr(strategy, "retriever")

    async def test_execute_works_as_unified_agent(self):
        model = AsyncMock()
        model.supports_tool_use = False
        model.generate = AsyncMock(return_value="The answer is 42.")
        strategy = RAGStrategy(model=model)
        result = await strategy.execute("What is the answer?")
        assert result == "The answer is 42."

    async def test_stream_works_as_unified_agent(self):
        model = AsyncMock()
        model.supports_tool_use = False

        async def _fake_stream(messages, **kwargs):
            yield "Hello"
            yield " world"

        model.generate_stream = _fake_stream
        strategy = RAGStrategy(model=model)
        tokens = [t async for t in strategy.execute_stream("question") if isinstance(t, str)]
        assert "Hello" in tokens
        assert " world" in tokens


# ---------------------------------------------------------------------------
# RetrieverFactory registry
# ---------------------------------------------------------------------------


class TestRetrieverFactory:
    def test_available_includes_pgvector(self):
        assert "pgvector" in RetrieverFactory.available()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.create("nonexistent")
