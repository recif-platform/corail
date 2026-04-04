"""Tool Registry — manages the set of tools available to an agent."""

from corail.tools.base import ToolDefinition, ToolExecutor, ToolResult


class ToolRegistry:
    """Holds all tools available to an agent. Used by ReAct strategy."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolExecutor] = {}

    def register(self, tool: ToolExecutor) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolExecutor | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions (for LLM prompt injection)."""
        return [t.definition() for t in self._tools.values()]

    def names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    async def execute(self, name: str, **kwargs: object) -> ToolResult:
        """Execute a tool by name. Returns error result if tool not found."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}. Available: {', '.join(self.names())}")
        return await tool.execute(**kwargs)

    def build_system_prompt_section(self) -> str:
        """Generate the tools section for the LLM system prompt."""
        if not self._tools:
            return ""
        tools_desc = "\n".join(d.to_prompt_schema() for d in self.list_definitions())
        return (
            "\n\n## Available Tools\n"
            "You can call tools by outputting a tool_call block:\n"
            "```tool_call\n"
            '{"name": "tool_name", "args": {"param1": "value1"}}\n'
            "```\n"
            "After calling a tool, you will receive the result and can continue your response.\n\n"
            f"Tools:\n{tools_desc}"
        )

    def __len__(self) -> int:
        return len(self._tools)
