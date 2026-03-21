"""Web search tool — pluggable backend (DuckDuckGo, SearXNG).

Default: DuckDuckGo via `ddgs` library (free, no API key).
Enterprise: SearXNG self-hosted (unlimited, private, no rate limits).

The admin chooses the backend via CORAIL_SEARCH_BACKEND env var.

This tool returns titles, URLs, and snippets only. To actually read
the content of a result page, the agent must call the separate
``fetch_url`` tool with one of the URLs. Splitting the two produces
distinct MLflow spans (``tool:web_search`` and ``tool:fetch_url``) so
failures are attributable to the right step, and it lets evaluators
score the agent's URL-selection decision independently of its query.
"""

import logging
import os
from abc import ABC, abstractmethod

from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Search provider interface                                          #
# ------------------------------------------------------------------ #


class SearchProvider(ABC):
    """Abstract interface for web search backends."""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web. Returns list of {"title", "url", "snippet"}."""
        ...


# ------------------------------------------------------------------ #
#  DuckDuckGo provider (default — free, no API key)                   #
# ------------------------------------------------------------------ #


class DuckDuckGoProvider(SearchProvider):
    """Web search via DuckDuckGo using the `ddgs` library.

    Install: pip install ddgs
    No API key required. Rate limited by DDG (generous for agent use).
    """

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("ddgs not installed. Install with: pip install ddgs")
            return [{"title": "Error", "url": "", "snippet": "ddgs library not installed. Run: pip install ddgs"}]

        try:
            results = DDGS().text(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return [{"title": "Search error", "url": "", "snippet": str(e)}]


# ------------------------------------------------------------------ #
#  SearXNG provider (self-hosted — unlimited, private)                #
# ------------------------------------------------------------------ #


class SearXNGProvider(SearchProvider):
    """Web search via a self-hosted SearXNG instance.

    Configure: CORAIL_SEARXNG_URL=http://localhost:8080
    SearXNG must have JSON format enabled in settings.yml.
    No rate limits, no API key, full privacy.
    """

    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url or os.environ.get("CORAIL_SEARXNG_URL", "http://localhost:8080")

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed")
            return [{"title": "Error", "url": "", "snippet": "httpx not installed"}]

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self._base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": "general",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])[:max_results]
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning("SearXNG search failed: %s", e)
            return [{"title": "Search error", "url": "", "snippet": str(e)}]


# ------------------------------------------------------------------ #
#  Provider registry                                                   #
# ------------------------------------------------------------------ #

_PROVIDERS: dict[str, type[SearchProvider]] = {
    "ddgs": DuckDuckGoProvider,
    "searxng": SearXNGProvider,
}


def register_search_provider(name: str, cls: type[SearchProvider]) -> None:
    """Register a custom search provider."""
    _PROVIDERS[name] = cls


def _get_provider() -> SearchProvider:
    """Resolve the search provider from CORAIL_SEARCH_BACKEND env var."""
    backend = os.environ.get("CORAIL_SEARCH_BACKEND", "ddgs")
    cls = _PROVIDERS.get(backend)
    if cls is None:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        logger.warning("Unknown search backend '%s', falling back to ddgs. Available: %s", backend, available)
        cls = DuckDuckGoProvider
    return cls()


# ------------------------------------------------------------------ #
#  Tool implementation                                                 #
# ------------------------------------------------------------------ #


class WebSearchTool(ToolExecutor):
    """Searches the web using a pluggable backend (DuckDuckGo or SearXNG).

    Backend is selected via CORAIL_SEARCH_BACKEND env var:
    - "ddgs" (default): DuckDuckGo, free, no API key
    - "searxng": Self-hosted SearXNG instance, unlimited
    """

    def __init__(self) -> None:
        self._provider = _get_provider()
        logger.info("Web search backend: %s", self._provider.__class__.__name__)

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "Search the web and return titles, URLs, and snippets. "
                "Does NOT download page content — to actually read a page, "
                "call fetch_url with the URL from the result you want."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query",
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return (default: 5)",
                    required=False,
                    default=5,
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        query = str(kwargs.get("query", ""))
        if not query.strip():
            return ToolResult(success=False, output="", error="Empty search query")

        max_results = int(kwargs.get("max_results", 5))
        results = await self._provider.search(query, max_results=max_results)

        if not results:
            return ToolResult(success=True, output="No results found.")

        formatted = [f"{i}. **{r['title']}**\n   {r['url']}\n   {r['snippet']}" for i, r in enumerate(results, 1)]
        output = f"Web search results for: {query}\n\n" + "\n\n".join(formatted)
        return ToolResult(success=True, output=output)
