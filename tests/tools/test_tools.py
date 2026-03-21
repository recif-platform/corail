"""Tests for tool infrastructure."""

from corail.tools.base import ToolDefinition, ToolParameter
from corail.tools.builtins import CalculatorTool, DateTimeTool
from corail.tools.registry import ToolRegistry


class TestToolDefinition:
    def test_to_prompt_schema(self):
        d = ToolDefinition(
            name="test",
            description="A test tool",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        schema = d.to_prompt_schema()
        assert "**test**" in schema
        assert "query: string (required)" in schema

    def test_no_params(self):
        d = ToolDefinition(name="simple", description="No params")
        schema = d.to_prompt_schema()
        assert "(no parameters)" in schema


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = DateTimeTool()
        registry.register(tool)
        assert registry.get("datetime") is tool
        assert registry.get("nonexistent") is None

    def test_names(self):
        registry = ToolRegistry()
        registry.register(DateTimeTool())
        registry.register(CalculatorTool())
        assert registry.names() == ["calculator", "datetime"]

    def test_list_definitions(self):
        registry = ToolRegistry()
        registry.register(DateTimeTool())
        defs = registry.list_definitions()
        assert len(defs) == 1
        assert defs[0].name == "datetime"

    def test_build_system_prompt_section(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        section = registry.build_system_prompt_section()
        assert "## Available Tools" in section
        assert "calculator" in section
        assert "tool_call" in section

    def test_empty_registry_prompt(self):
        registry = ToolRegistry()
        assert registry.build_system_prompt_section() == ""

    async def test_execute_known_tool(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.execute("calculator", expression="2 + 2")
        assert result.success
        assert result.output == "4"

    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent")
        assert not result.success
        assert "Unknown tool" in result.error


class TestDateTimeTool:
    async def test_execute(self):
        tool = DateTimeTool()
        result = await tool.execute()
        assert result.success
        assert "datetime" in result.output
        assert "date" in result.output

    def test_definition(self):
        tool = DateTimeTool()
        d = tool.definition()
        assert d.name == "datetime"
        assert len(d.parameters) == 0


class TestCalculatorTool:
    async def test_basic_math(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="15 * 37")
        assert result.success
        assert result.output == "555"

    async def test_complex_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="sqrt(144) + abs(-5)")
        assert result.success
        assert result.output == "17.0"

    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="import os")
        assert not result.success
        assert "error" in result.error.lower()

    def test_definition(self):
        tool = CalculatorTool()
        d = tool.definition()
        assert d.name == "calculator"
        assert len(d.parameters) == 1
        assert d.parameters[0].name == "expression"
