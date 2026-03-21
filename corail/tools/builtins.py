"""Built-in tools — available out of the box."""

import json
from datetime import UTC, datetime

from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult


class DateTimeTool(ToolExecutor):
    """Returns the current date and time."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="datetime",
            description="Get the current date and time in UTC",
            parameters=[],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        now = datetime.now(UTC)
        return ToolResult(
            success=True,
            output=json.dumps(
                {
                    "datetime": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "day": now.strftime("%A"),
                }
            ),
            render="json",
        )


class CalculatorTool(ToolExecutor):
    """Evaluates a mathematical expression safely."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, (), sqrt, abs, round.",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="The math expression to evaluate, e.g. '(15 * 37) + sqrt(144)'",
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        import math

        expr = str(kwargs.get("expression", ""))
        # Whitelist safe math operations
        allowed = {"sqrt": math.sqrt, "abs": abs, "round": round, "pi": math.pi, "e": math.e}
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Math error: {e}")
