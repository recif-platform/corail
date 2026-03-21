"""fetch_url — download one URL and return its readable text content.

Split out of ``web_search`` so the LLM can (and must) explicitly choose
which URL to read after seeing the search titles and snippets. This
gives two distinct MLflow spans — ``tool:web_search`` and
``tool:fetch_url`` — making it possible to tell at a glance whether a
failure came from the search or from a specific page download.

The extraction is stdlib-only (``html.parser``): no trafilatura, no
lxml, no beautifulsoup. Good enough to give the LLM real prose for
grounding without adding a new dependency.
"""

import logging
import re
from html.parser import HTMLParser

import httpx

from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_CHARS = 4000
# Real Chrome UA. The previous "compatible; CorailAgent/1.0" was trivially
# detected by Cloudflare and returned 403 on sites like Investing.com,
# Boursorama's bot tier, and major news outlets. A real browser UA clears
# the bar for the vast majority of sites; aggressive anti-bot
# (Bloomberg, Reuters, some press sites) will still refuse us — nothing
# we can do short-of a headless browser.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
_ACCEPT_LANGUAGE = "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"

# When a fetch gets redirected to a consent / cookie / sign-in wall,
# the content we extract is a language selector or "accept cookies"
# banner — useless for grounding. Return an explicit failure so the
# agent loop's retry rules kick in and the LLM picks another URL from
# its search results instead of narrating a dead-end.
#
# Pattern-based: any host starting with one of these prefixes counts
# as a wall. Covers consent.google.com, consent.yahoo.com,
# consent.youtube.com, login.microsoftonline.com, accounts.google.com,
# idp.nytimes.com, auth.*, etc. without needing to enumerate every
# property ahead of time.
_WALL_HOST_PREFIXES = (
    "consent.",
    "accounts.",
    "auth.",
    "login.",
    "idp.",
    "signin.",
)


def _is_wall_host(host: str) -> bool:
    h = (host or "").lower()
    return any(h.startswith(p) for p in _WALL_HOST_PREFIXES)


# Only real container tags. Void tags (meta, link, br, img, input, ...) have
# no closing tag, so tracking them in skip_depth would leave the parser stuck
# above zero forever and drop every subsequent handle_data call.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "svg"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "br",
        "div",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "aside",
    }
)


def _fail(error: str) -> ToolResult:
    """Shorthand for the many failure branches in ``execute``."""
    return ToolResult(success=False, output="", error=error)


# Shared httpx client, lazily created on first use. Before this, every
# fetch_url call spun up a fresh AsyncClient inside an async-with block,
# paying the TCP/TLS handshake + connection-pool setup cost on each call.
# Agents chain 3–4 fetches per turn on the fallback path, so sharing
# one client with keep-alive roughly halves the latency of the retry chain.
_shared_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=_DEFAULT_TIMEOUT_SECONDS,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": _ACCEPT,
                "Accept-Language": _ACCEPT_LANGUAGE,
            },
        )
    return _shared_client


class _ReadableTextExtractor(HTMLParser):
    """Minimal HTML→text extractor.

    Drops script/style content, adds newlines on block-level tags, and
    collapses whitespace. Stops collecting once accumulated text
    exceeds ``budget`` characters — the caller will truncate to its
    final max_chars anyway, so buffering a 10 MB page is pointless and
    an OOM risk on pathological inputs.
    """

    def __init__(self, budget: int) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._accumulated = 0
        self._budget = budget

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._accumulated >= self._budget:
            return
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)
            self._accumulated += len(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_readable_text(html: str, max_chars: int) -> str:
    """Parse HTML and return collapsed plain text, truncated to ``max_chars``.

    The parser stops collecting once it has ~3× the final max_chars of
    raw text, leaving headroom for whitespace collapse while still
    truncating oversized pages instead of buffering the whole DOM.
    """
    try:
        parser = _ReadableTextExtractor(budget=max_chars * 3)
        parser.feed(html)
        parser.close()
        text = parser.get_text()
    except Exception as exc:
        logger.debug("HTML extraction failed: %s", exc)
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    return text


class FetchURLTool(ToolExecutor):
    """Downloads one URL and returns its extracted text content.

    Typical flow: the agent first calls ``web_search`` to get candidate
    URLs, then calls ``fetch_url`` on the most promising one(s). Each
    call produces its own MLflow span so failures are attributable to a
    specific URL rather than hidden inside a composite search-and-fetch
    step.
    """

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fetch_url",
            description=(
                "Download a single web page and return its readable text content. "
                "Use this after web_search to actually read the page behind a "
                "promising result URL. Returns plain text extracted from the "
                "HTML, with scripts, styles, and boilerplate removed."
            ),
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The absolute URL to download (http:// or https://).",
                ),
                ToolParameter(
                    name="max_chars",
                    type="integer",
                    description=(
                        f"Maximum characters of extracted text to return "
                        f"(default: {_DEFAULT_MAX_CHARS}). Pages longer than "
                        "this are truncated with a trailing ellipsis."
                    ),
                    required=False,
                    default=_DEFAULT_MAX_CHARS,
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return _fail("Empty URL")
        if not (url.startswith("http://") or url.startswith("https://")):
            return _fail(f"URL must start with http:// or https://, got: {url}")

        try:
            max_chars = int(kwargs.get("max_chars", _DEFAULT_MAX_CHARS))
        except (TypeError, ValueError):
            max_chars = _DEFAULT_MAX_CHARS

        try:
            resp = await _get_client().get(url)
        except httpx.TimeoutException:
            return _fail(f"Timeout after {_DEFAULT_TIMEOUT_SECONDS}s while fetching {url}")
        except httpx.HTTPError as exc:
            return _fail(f"HTTP error fetching {url}: {type(exc).__name__}: {exc}")

        if resp.status_code >= 400:
            return _fail(f"HTTP {resp.status_code} from {url}")

        final_host = resp.url.host or ""
        if _is_wall_host(final_host):
            return _fail(
                f"Redirected to a consent/login wall at {final_host} "
                f"(original URL: {url}). Try a different result URL.",
            )

        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return _fail(
                f"Unsupported content-type for text extraction: {ctype or 'unknown'} ({url})",
            )

        text = _extract_readable_text(resp.text, max_chars)
        if not text:
            return _fail(
                f"No readable text extracted from {url} "
                f"(HTTP {resp.status_code}, content-type={ctype}). "
                "The page may be JS-rendered or behind a consent/login wall.",
            )

        header = f"Fetched: {resp.url}\nStatus: {resp.status_code} {ctype}\n\n"
        return ToolResult(success=True, output=header + text)
