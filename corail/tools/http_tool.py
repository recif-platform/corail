"""HTTP Tool Executor — calls external REST APIs."""

import json

import httpx

from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult


def _detect_render_hint(data: object) -> tuple[str, dict]:
    """Detect an appropriate render hint from parsed response data.

    Returns (render_type, props) tuple.
    JSON arrays of dicts → table with auto-detected columns.
    """
    if not isinstance(data, list) or not data:
        return "text", {}
    # All items must be dicts to form a table
    if not all(isinstance(item, dict) for item in data):
        return "text", {}
    # Extract column names from first row's keys
    columns = list(data[0].keys())
    rows = [[row.get(col, "") for col in columns] for row in data]
    return "table", {"columns": columns, "rows": rows}


class HTTPTool(ToolExecutor):
    """Executes HTTP requests to external APIs. Configurable via constructor."""

    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        parameters: list[ToolParameter] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._name = name
        self._description = description
        self._url = url
        self._method = method.upper()
        self._headers = headers or {}
        self._parameters = parameters or []
        self._timeout = timeout

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        try:
            # Build URL with path params
            url = self._url.format(**{k: v for k, v in kwargs.items() if isinstance(v, str)})

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                if self._method in ("GET", "DELETE"):
                    response = await client.request(self._method, url, headers=self._headers, params=kwargs)
                else:
                    response = await client.request(self._method, url, headers=self._headers, json=kwargs)

                response.raise_for_status()

                # Try JSON, fall back to text
                try:
                    data = response.json()
                    output = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception:
                    data = None
                    output = response.text

                # Detect tabular data: JSON arrays of objects → table render
                render, props = _detect_render_hint(data)
                return ToolResult(
                    success=True,
                    output=output[:4000],  # Truncate for LLM context
                    render=render,
                    props=props,
                )

        except httpx.HTTPStatusError as e:
            return ToolResult(success=False, output="", error=f"HTTP {e.response.status_code}: {e.response.text[:500]}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
