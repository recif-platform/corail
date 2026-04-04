"""RAG Strategy — Retrieve relevant context, augment the prompt, generate response.

Supports:
- Multiple knowledge bases (via MultiRetriever)
- Per-request KB toggle (via options.active_kbs)
- Streaming with source attribution

Future (pluggable via interfaces):
- QueryTransformer (HyDE, step-back, expansion)
- Reranker (cross-encoder, score threshold)
- Hybrid search (BM25 + vector)
"""

from collections.abc import AsyncIterator

from corail.models.base import Model
from corail.retrieval.base import RetrievalResult, Retriever
from corail.retrieval.multi import MultiRetriever
from corail.strategies.base import AgentStrategy

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on provided context."


class RAGStrategy(AgentStrategy):
    """RAG: Retrieve relevant context -> Augment prompt -> Generate response."""

    def __init__(self, model: Model, system_prompt: str = "", retriever: Retriever | None = None) -> None:
        super().__init__(model, system_prompt)
        self.retriever = retriever

    def _build_rag_prompt(self, user_input: str, context_chunks: list[RetrievalResult]) -> str:
        context = "\n\n---\n\n".join([c.content for c in context_chunks])
        return (
            f"Use the following context to answer the question. "
            f"If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}"
        )

    def _build_messages(
        self, user_input: str, chunks: list[RetrievalResult], history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt or _DEFAULT_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        if chunks:
            messages.append({"role": "user", "content": self._build_rag_prompt(user_input, chunks)})
        else:
            messages.append({"role": "user", "content": user_input})
        return messages

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object) -> str:
        chunks = await self._retrieve(user_input, **kwargs)
        messages = self._build_messages(user_input, chunks, history)
        return await self.model.generate(messages=messages)

    async def execute_stream(
        self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object,
    ) -> AsyncIterator[str]:
        chunks = await self._retrieve(user_input, **kwargs)

        if chunks:
            sources = ", ".join(sorted({c.metadata.get("filename", c.metadata.get("kb_id", "unknown")) for c in chunks}))
            yield f"*Sources: {sources}*\n\n"

        messages = self._build_messages(user_input, chunks, history)
        async for token in self.model.generate_stream(messages=messages):
            yield token

    async def _retrieve(self, query: str, **kwargs: object) -> list[RetrievalResult]:
        """Retrieve chunks, respecting use_rag, active_kbs, and minimum relevance threshold."""
        use_rag = bool(kwargs.get("use_rag", True))
        if not use_rag or self.retriever is None:
            return []

        min_score = float(kwargs.get("min_score", 0.3))

        # If retriever is MultiRetriever, pass active_kbs filter
        active_kbs = kwargs.get("active_kbs")
        if isinstance(self.retriever, MultiRetriever) and active_kbs is not None:
            results = await self.retriever.search(query, top_k=5, active_kbs=list(active_kbs))
        else:
            results = await self.retriever.search(query, top_k=5)

        # Filter out low-relevance chunks
        return [r for r in results if r.score >= min_score]
