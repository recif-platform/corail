"""Multi-KB retriever — searches across multiple knowledge bases, filtered by active KBs."""

from corail.retrieval.base import RetrievalResult, Retriever


class MultiRetriever(Retriever):
    """Wraps multiple retrievers (one per KB). Filters by active_kbs at query time.

    This is NOT a base class — it's a compositor. Each inner retriever
    handles its own embedding + vector search.
    """

    def __init__(self, retrievers: dict[str, Retriever]) -> None:
        """retrievers: mapping of kb_id → Retriever instance."""
        self._retrievers = retrievers

    @property
    def kb_ids(self) -> list[str]:
        """Return sorted list of available KB IDs."""
        return sorted(self._retrievers.keys())

    async def search(
        self,
        query: str,
        top_k: int = 5,
        active_kbs: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search across active KBs, merge and rank results.

        Args:
            query: The user's question.
            top_k: Max total results to return.
            active_kbs: If provided, only search these KBs. If None, search all.
        """
        # Determine which retrievers to use
        if active_kbs is not None:
            retrievers = {k: v for k, v in self._retrievers.items() if k in active_kbs}
        else:
            retrievers = self._retrievers

        if not retrievers:
            return []

        # Search each KB (could be parallelized later with asyncio.gather)
        all_results: list[RetrievalResult] = []
        per_kb_k = max(1, top_k // len(retrievers) + 1)  # Over-fetch per KB, then trim

        for kb_id, retriever in retrievers.items():
            results = await retriever.search(query, top_k=per_kb_k)
            # Tag results with KB source
            for r in results:
                r.metadata["kb_id"] = kb_id
            all_results.extend(results)

        # Rank by score (highest first) and trim to top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    async def close(self) -> None:
        for retriever in self._retrievers.values():
            await retriever.close()
