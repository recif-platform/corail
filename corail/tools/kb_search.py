"""Knowledge base search tool — per-KB tool for agentic RAG.

Each attached knowledge base becomes a separate search tool that the agent
calls through the react loop only when it decides the KB is relevant.
"""

import logging

from corail.retrieval.base import Retriever
from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class KBSearchTool(ToolExecutor):
    """Search a single knowledge base. Created per-KB, registered in ToolRegistry."""

    def __init__(
        self,
        name: str,
        description: str,
        retriever: Retriever,
        kb_id: str,
        top_k: int = 5,
    ) -> None:
        self._name = name
        self._description = description
        self._retriever = retriever
        self._kb_id = kb_id
        self._top_k = top_k

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query to find relevant documents",
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        query = str(kwargs.get("query", ""))
        if not query.strip():
            return ToolResult(success=False, output="", error="Empty query")

        try:
            results = await self._retriever.search(query, top_k=self._top_k)
        except Exception as exc:
            logger.warning("KB search failed for %s: %s", self._kb_id, exc)
            return ToolResult(success=False, output="", error=f"KB search failed: {exc}")

        if not results:
            return ToolResult(
                success=True,
                output=f"No relevant documents found in {self._name}.",
            )

        parts: list[str] = []
        sources: list[dict] = []
        for i, r in enumerate(results):
            filename = r.metadata.get("filename", f"chunk-{i}")
            chunk_idx = r.metadata.get("chunk_index", i)
            parts.append(f"[Source: {filename}]\n{r.content}")
            sources.append({
                "filename": filename,
                "score": round(r.score, 4),
                "chunk_index": chunk_idx,
                "content_preview": r.content[:300].replace("\n", " ").strip(),
            })

        output = "\n\n---\n\n".join(parts)
        output += (
            "\n\n---\n"
            "CITATION INSTRUCTION: You MUST cite the [Source: ...] document names above "
            "in your response so the user knows where the information comes from."
        )
        return ToolResult(success=True, output=output, props={"sources": sources})

    async def close(self) -> None:
        """Close the underlying retriever's connection pool."""
        if hasattr(self._retriever, "close"):
            await self._retriever.close()
