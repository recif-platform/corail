"""ReAct Strategy — Reasoning + Acting with tool calls.

The LLM thinks, decides to call tools, observes results, and continues.
Supports streaming: tokens stream live, tool calls execute between rounds.
"""

import json
import re
from collections.abc import AsyncIterator

from corail.core.stream import (
    ComponentEvent,
    ConfirmEvent,
    StreamToken,
    ToolEndEvent,
    ToolStartEvent,
)
from corail.strategies.base import AgentStrategy
from corail.tools.base import ToolResult
from corail.tools.registry import ToolRegistry

# Detect both markdown ```tool_call and XML <tool_use> formats
TOOL_CALL_PATTERNS = [
    re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"<tool_use>(.*?)</tool_use>", re.DOTALL),
]
MAX_TOOL_ROUNDS = 5


class ReActStrategy(AgentStrategy):
    """ReAct: Reason + Act. The LLM can call tools and use their results."""

    def __init__(self, model: "Model", system_prompt: str = "", tools: ToolRegistry | None = None) -> None:
        super().__init__(model, system_prompt)
        self.tools = tools or ToolRegistry()

    def _build_system_prompt(self) -> str:
        base = self.system_prompt or "You are a helpful assistant."
        return base + self.tools.build_system_prompt_section()

    def _build_messages(self, user_input: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> wrapper but keep the content visible."""
        return re.sub(r"</?think>", "", text)

    def _extract_tool_call(self, text: str) -> tuple[str, str] | None:
        """Extract first tool_call from text. Supports both markdown and XML formats."""
        clean = self._strip_think_tags(text)
        for pattern in TOOL_CALL_PATTERNS:
            match = pattern.search(clean)
            if match:
                return match.group(1).strip(), match.group(0)
        return None

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        messages = self._build_messages(user_input, history)

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.model.generate(messages=messages)
            tool_call = self._extract_tool_call(response)
            if not tool_call:
                return response

            call_json, _ = tool_call
            result = await self._execute_tool_call(call_json)
            messages.append({"role": "assistant", "content": self._strip_think_tags(response)})
            messages.append({"role": "user", "content": f"[Tool Result]\n{result}"})

        return response

    async def execute_stream(
        self, user_input: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[StreamToken]:
        """Stream with tool execution between rounds.

        Each round:
        1. Stream all LLM tokens to the client (thinking + response)
        2. After stream ends, check for tool_call blocks
        3. If found: execute tool, yield structured events, start next round
        4. If not found: done
        """
        messages = self._build_messages(user_input, history)

        for round_num in range(MAX_TOOL_ROUNDS):
            full_response = ""

            # Stream all tokens from this round
            async for token in self.model.generate_stream(messages=messages):
                full_response += token
                yield token

            # Check for tool calls in the complete response
            tool_call = self._extract_tool_call(full_response)
            if not tool_call:
                return  # No tool call — done

            # Execute the tool with structured events
            call_json, _ = tool_call
            result = await self._execute_tool_call_full(call_json)

            tool_name = self._extract_tool_name(call_json)
            tool_args = self._extract_tool_args(call_json)

            # Check if tool requires confirmation
            tool_defn = self._get_tool_definition(tool_name)
            if tool_defn and tool_defn.risk_level == "confirm":
                yield ConfirmEvent(
                    call_id=f"react_v1_{round_num}",
                    tool=tool_name,
                    args=tool_args,
                    message=f"Execute {tool_name}?",
                )

            yield ToolStartEvent(tool=tool_name, args=tool_args)
            result_text = result.output if result.success else f"Error: {result.error}"
            yield ToolEndEvent(tool=tool_name, output=result_text, success=result.success)

            # Emit component event for rich render hints
            if result.success and result.render != "text":
                component_name = result.component if result.render == "react" else result.render
                props = result.props or self._try_parse_json(result.output)
                yield ComponentEvent(component=component_name, props=props)

            # Prepare next round
            clean_response = self._strip_think_tags(full_response)
            messages.append({"role": "assistant", "content": clean_response})
            messages.append({"role": "user", "content": f"[Tool Result]\n{result_text}"})

    async def _execute_tool_call(self, call_json: str) -> str:
        result = await self._execute_tool_call_full(call_json)
        return result.output if result.success else f"Error: {result.error}"

    async def _execute_tool_call_full(self, call_json: str) -> ToolResult:
        """Execute a tool call and return the full ToolResult."""
        try:
            call = json.loads(call_json.strip())
            name = call.get("name", "")
            args = call.get("args", {})
            return await self.tools.execute(name, **args)
        except json.JSONDecodeError:
            return ToolResult(success=False, output="", error=f"Invalid tool call JSON: {call_json[:200]}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error executing tool: {e}")

    def _extract_tool_name(self, call_json: str) -> str:
        """Extract tool name from call JSON."""
        try:
            return json.loads(call_json.strip()).get("name", "unknown")
        except (json.JSONDecodeError, AttributeError):
            return "unknown"

    def _extract_tool_args(self, call_json: str) -> dict:
        """Extract tool args from call JSON."""
        try:
            return json.loads(call_json.strip()).get("args", {})
        except (json.JSONDecodeError, AttributeError):
            return {}

    def _get_tool_definition(self, name: str):
        """Look up a tool's definition by name."""
        tool = self.tools.get(name)
        return tool.definition() if tool else None

    @staticmethod
    def _try_parse_json(text: str) -> dict:
        """Attempt to parse text as JSON dict; return empty dict on failure."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except (json.JSONDecodeError, TypeError):
            return {}
