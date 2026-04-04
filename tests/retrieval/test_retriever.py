"""Tests for RAG retrieval and strategy."""

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
# RAGStrategy._build_rag_prompt
# ---------------------------------------------------------------------------


class TestBuildRagPrompt:
    def _make_strategy(self):
        model = AsyncMock()
        return RAGStrategy(model=model)

    def test_prompt_includes_context_and_question(self):
        strategy = self._make_strategy()
        chunks = [
            RetrievalResult(content="Paris is the capital of France.", score=0.9),
            RetrievalResult(content="France is in Europe.", score=0.8),
        ]
        prompt = strategy._build_rag_prompt("What is the capital of France?", chunks)
        assert "Paris is the capital of France." in prompt
        assert "France is in Europe." in prompt
        assert "What is the capital of France?" in prompt
        assert "---" in prompt  # separator between chunks

    def test_empty_chunks(self):
        strategy = self._make_strategy()
        prompt = strategy._build_rag_prompt("question?", [])
        assert "question?" in prompt
        assert "Context:" in prompt


# ---------------------------------------------------------------------------
# RAGStrategy.execute — mock retriever + mock model
# ---------------------------------------------------------------------------


class TestRAGStrategyExecute:
    @pytest.fixture
    def mock_chunks(self):
        return [
            RetrievalResult(content="Chunk A content.", score=0.95, metadata={"filename": "a.pdf"}),
            RetrievalResult(content="Chunk B content.", score=0.80, metadata={"filename": "b.pdf"}),
        ]

    @pytest.fixture
    def mock_retriever(self, mock_chunks):
        retriever = AsyncMock(spec=Retriever)
        retriever.search.return_value = mock_chunks
        return retriever

    @pytest.fixture
    def mock_model(self):
        model = AsyncMock()
        model.generate.return_value = "The answer is 42."
        return model

    async def test_execute_calls_retriever_and_model(self, mock_retriever, mock_model):
        strategy = RAGStrategy(model=mock_model, retriever=mock_retriever)
        result = await strategy.execute("What is the answer?")

        mock_retriever.search.assert_awaited_once_with("What is the answer?", top_k=5)
        mock_model.generate.assert_awaited_once()
        assert result == "The answer is 42."

    async def test_execute_without_retriever(self, mock_model):
        strategy = RAGStrategy(model=mock_model)
        result = await strategy.execute("Hello")

        mock_model.generate.assert_awaited_once()
        assert result == "The answer is 42."

    async def test_execute_passes_history(self, mock_retriever, mock_model):
        strategy = RAGStrategy(model=mock_model, retriever=mock_retriever)
        history = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "resp"}]
        await strategy.execute("follow up", history=history)

        call_args = mock_model.generate.call_args
        messages = call_args.kwargs["messages"]
        # System + 2 history + 1 user = 4 messages
        assert len(messages) == 4
        assert messages[1]["content"] == "prior"

    async def test_execute_injects_context_into_prompt(self, mock_retriever, mock_model, mock_chunks):
        strategy = RAGStrategy(model=mock_model, retriever=mock_retriever)
        await strategy.execute("Tell me about chunks")

        call_args = mock_model.generate.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]["content"]
        assert "Chunk A content." in user_msg
        assert "Chunk B content." in user_msg


# ---------------------------------------------------------------------------
# RAGStrategy.execute_stream — sources first, then tokens
# ---------------------------------------------------------------------------


async def _fake_stream_multi(**_kwargs):
    for token in ["Hello", " world"]:
        yield token


async def _fake_stream_single(**_kwargs):
    yield "only token"


async def _fake_stream_answer(**_kwargs):
    yield "answer"


class TestRAGStrategyStream:
    @pytest.fixture
    def mock_chunks(self):
        return [
            RetrievalResult(content="Chunk 1", score=0.9, metadata={"filename": "report.pdf"}),
            RetrievalResult(content="Chunk 2", score=0.8, metadata={"filename": "notes.md"}),
        ]

    @pytest.fixture
    def mock_retriever(self, mock_chunks):
        retriever = AsyncMock(spec=Retriever)
        retriever.search.return_value = mock_chunks
        return retriever

    async def test_stream_yields_sources_first(self, mock_retriever):
        model = AsyncMock()
        model.generate_stream = _fake_stream_multi

        strategy = RAGStrategy(model=model, retriever=mock_retriever)
        tokens = [token async for token in strategy.execute_stream("question")]

        # First token should be sources
        assert tokens[0].startswith("*Sources:")
        assert "notes.md" in tokens[0]
        assert "report.pdf" in tokens[0]
        # Remaining tokens are LLM output
        assert tokens[1:] == ["Hello", " world"]

    async def test_stream_no_sources_without_retriever(self):
        model = AsyncMock()
        model.generate_stream = _fake_stream_single

        strategy = RAGStrategy(model=model)
        tokens = [token async for token in strategy.execute_stream("question")]

        assert tokens == ["only token"]
        assert not any(t.startswith("*Sources:") for t in tokens)

    async def test_stream_no_sources_when_empty_results(self):
        model = AsyncMock()
        retriever = AsyncMock(spec=Retriever)
        retriever.search.return_value = []
        model.generate_stream = _fake_stream_answer

        strategy = RAGStrategy(model=model, retriever=retriever)
        tokens = [token async for token in strategy.execute_stream("question")]

        assert tokens == ["answer"]


# ---------------------------------------------------------------------------
# RetrieverFactory registry
# ---------------------------------------------------------------------------


class TestRetrieverFactory:
    def test_available_includes_pgvector(self):
        assert "pgvector" in RetrieverFactory.available()

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.create("nonexistent")
