"""ReAct V2 Strategy — native tool calling with prompt-based fallback.

Uses the model's native tool_use when available, falls back to prompt-based
ReAct (V1) for models without native support. Integrates GuardPipeline
and EventBus as optional pluggable components.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from corail.core.stream import (
    ComponentEvent,
    ConfirmEvent,
    StreamToken,
    ToolEndEvent,
    ToolStartEvent,
)
from corail.events.bus import EventBus
from corail.events.types import Event, EventType
from corail.guards.base import GuardResult
from corail.guards.pipeline import GuardPipeline
from corail.models.base import Model, ModelResponse, ToolCall
from corail.strategies._shared import (
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_ROUNDS,
    _DEFAULT_MAX_TOKENS,
    _MAX_TOOL_RETRIES,
)
from corail.strategies.base import AgentStrategy
from corail.tools.base import ToolDefinition, ToolParameter, ToolResult
from corail.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class BudgetOptions:
    """Budget constraints for the execution loop."""

    max_rounds: int = _DEFAULT_MAX_ROUNDS
    max_tokens: int = _DEFAULT_MAX_TOKENS


class _EventEmitter:
    """Thin wrapper that no-ops when EventBus is None."""

    __slots__ = ("_bus",)

    def __init__(self, bus: EventBus | None) -> None:
        self._bus = bus

    async def emit(self, event_type: EventType, **data: Any) -> None:
        if self._bus is not None:
            await self._bus.emit(Event(type=event_type, data=data))


class _GuardRunner:
    """Thin wrapper that no-ops when GuardPipeline is None."""

    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: GuardPipeline | None) -> None:
        self._pipeline = pipeline

    async def check_input(self, content: str) -> GuardResult:
        if self._pipeline is None:
            return GuardResult(allowed=True)
        return await self._pipeline.check_input(content)

    async def check_output(self, content: str) -> GuardResult:
        if self._pipeline is None:
            return GuardResult(allowed=True)
        return await self._pipeline.check_output(content)


# --- Tool schema converters ---


def _parameter_to_json_schema(param: ToolParameter) -> dict[str, Any]:
    """Convert a ToolParameter to JSON Schema property."""
    return {"type": param.type, "description": param.description}


def _definition_to_anthropic_tool(defn: ToolDefinition) -> dict[str, Any]:
    """Convert a ToolDefinition to Anthropic tool format."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in defn.parameters:
        properties[param.name] = _parameter_to_json_schema(param)
        if param.required:
            required.append(param.name)

    return {
        "name": defn.name,
        "description": defn.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _try_parse_json(text: str) -> dict[str, Any]:
    """Attempt to parse text as JSON dict; return empty dict on failure."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}


def _build_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Build Anthropic-formatted tool schemas from a ToolRegistry."""
    return [_definition_to_anthropic_tool(defn) for defn in registry.list_definitions()]


# --- Message builders for native tool calling ---


def _tool_result_message(tool_call: ToolCall, result_text: str) -> dict[str, Any]:
    """Build a tool_result message for the Anthropic conversation."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result_text,
            }
        ],
    }


def _assistant_message_from_response(response: ModelResponse) -> dict[str, Any]:
    """Build an assistant message containing text + tool_use blocks."""
    content: list[dict[str, Any]] = []
    if response.content:
        content.append({"type": "text", "text": response.content})
    for tc in response.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.args,
            }
        )
    return {"role": "assistant", "content": content}


class ReActV2Strategy(AgentStrategy):
    """ReAct V2: native tool calling with prompt-based fallback.

    When the model supports native tool_use, calls generate_with_tools directly.
    Otherwise, delegates to the prompt-based ReActStrategy (V1).

    Optional components:
    - GuardPipeline: input/output security checks
    - EventBus: event emission for observability
    - BudgetOptions: max_rounds / max_tokens constraints
    """

    def __init__(
        self,
        model: Model,
        system_prompt: str = "",
        tools: ToolRegistry | None = None,
        guard_pipeline: GuardPipeline | None = None,
        event_bus: EventBus | None = None,
        budget: BudgetOptions | None = None,
    ) -> None:
        super().__init__(model, system_prompt)
        self.tools = tools or ToolRegistry()
        self._guards = _GuardRunner(guard_pipeline)
        self._events = _EventEmitter(event_bus)
        self._budget = budget or BudgetOptions()
        self._fallback: AgentStrategy | None = None

    def _get_fallback(self) -> AgentStrategy:
        """Lazy-load the prompt-based ReActStrategy for non-tool-use models."""
        if self._fallback is None:
            from corail.strategies.react import ReActStrategy

            self._fallback = ReActStrategy(
                model=self.model,
                system_prompt=self.system_prompt,
                tools=self.tools,
            )
        return self._fallback

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        """Execute strategy: native tool calling or prompt-based fallback."""
        if self.model.supports_tool_use:
            return await self._execute_native(user_input, history)
        return await self._get_fallback().execute(user_input, history=history)

    async def execute_stream(
        self, user_input: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[StreamToken]:
        """Stream execution: native tool calling or prompt-based fallback."""
        if self.model.supports_tool_use:
            async for token in self._stream_native(user_input, history):
                yield token
        else:
            async for token in self._get_fallback().execute_stream(user_input, history=history):
                yield token

    # --- Native tool calling path ---

    async def _execute_native(self, user_input: str, history: list[dict[str, str]] | None = None) -> str:
        """Full native tool calling loop with guards, events, and budget."""
        # Guard input
        input_result = await self._guards.check_input(user_input)
        if not input_result.allowed:
            await self._events.emit(EventType.GUARD_BLOCKED, direction="input", reason=input_result.reason)
            return f"[Blocked] {input_result.reason}"

        effective_input = input_result.sanitized or user_input
        await self._events.emit(EventType.MESSAGE_RECEIVED, content=effective_input)

        messages = self._build_messages(effective_input, history)
        tool_schemas = _build_tool_schemas(self.tools)
        final_content = ""
        tokens_used = 0

        for round_num in range(self._budget.max_rounds):
            await self._events.emit(EventType.LLM_CALL_STARTED, round=round_num)
            response = await self.model.generate_with_tools(messages, tool_schemas)
            tokens_used += len(response.content) // _CHARS_PER_TOKEN
            await self._events.emit(EventType.LLM_CALL_COMPLETED, round=round_num, stop_reason=response.stop_reason)

            # End turn — no more tool calls
            if response.stop_reason != "tool_use":
                final_content = response.content
                break

            # Process tool calls
            messages.append(_assistant_message_from_response(response))
            for tc in response.tool_calls:
                result_text = await self._execute_tool_with_retry(tc)
                messages.append(_tool_result_message(tc, result_text))

            # Budget check
            if tokens_used >= self._budget.max_tokens:
                await self._events.emit(EventType.BUDGET_EXCEEDED, tokens_used=tokens_used)
                final_content = response.content or "[Budget exceeded]"
                break
        else:
            await self._events.emit(EventType.BUDGET_EXCEEDED, rounds=self._budget.max_rounds)
            final_content = response.content or "[Max rounds exceeded]"

        # Guard output
        output_result = await self._guards.check_output(final_content)
        if not output_result.allowed:
            await self._events.emit(EventType.GUARD_BLOCKED, direction="output", reason=output_result.reason)
            return f"[Blocked] {output_result.reason}"

        final = output_result.sanitized or final_content
        await self._events.emit(EventType.MESSAGE_RESPONSE, content=final)
        return final

    async def _stream_native(
        self, user_input: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[StreamToken]:
        """Streaming native tool calling: stream text, execute tools inline."""
        # Guard input
        input_result = await self._guards.check_input(user_input)
        if not input_result.allowed:
            yield f"[Blocked] {input_result.reason}"
            return

        effective_input = input_result.sanitized or user_input
        messages = self._build_messages(effective_input, history)
        tool_schemas = _build_tool_schemas(self.tools)

        for round_num in range(self._budget.max_rounds):
            response = await self.model.generate_with_tools(messages, tool_schemas)

            if response.content:
                yield response.content

            if response.stop_reason != "tool_use":
                break

            # Execute tool calls with structured events
            messages.append(_assistant_message_from_response(response))
            for tc in response.tool_calls:
                # Check if tool requires confirmation
                tool_defn = self._get_tool_definition(tc.name)
                if tool_defn and tool_defn.risk_level == "confirm":
                    yield ConfirmEvent(
                        call_id=tc.id,
                        tool=tc.name,
                        args=tc.args,
                        message=f"Execute {tc.name}?",
                    )

                yield ToolStartEvent(tool=tc.name, args=tc.args, call_id=tc.id)
                result = await self._execute_tool_with_retry_full(tc)
                result_text = result.output if result.success else f"Error: {result.error}"
                yield ToolEndEvent(
                    tool=tc.name,
                    output=result_text,
                    success=result.success,
                    call_id=tc.id,
                )

                # Emit component event for rich render hints
                if result.success and result.render != "text":
                    component_name = result.component if result.render == "react" else result.render
                    props = result.props or _try_parse_json(result.output)
                    yield ComponentEvent(component=component_name, props=props)

                messages.append(_tool_result_message(tc, result_text))

    # --- Tool helpers ---

    def _get_tool_definition(self, name: str) -> ToolDefinition | None:
        """Look up a tool's definition by name."""
        tool = self.tools.get(name)
        return tool.definition() if tool else None

    # --- Tool execution with retry ---

    async def _execute_tool_with_retry_full(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call with retry, returning the full ToolResult."""
        await self._events.emit(EventType.TOOL_CALLED, name=tool_call.name, args=tool_call.args)

        last_error = ""
        for attempt in range(_MAX_TOOL_RETRIES + 1):
            result = await self.tools.execute(tool_call.name, **tool_call.args)
            if result.success:
                await self._events.emit(EventType.TOOL_RESULT, name=tool_call.name, output=result.output)
                return result

            last_error = result.error
            await self._events.emit(
                EventType.TOOL_ERROR,
                name=tool_call.name,
                error=last_error,
                attempt=attempt,
            )
            logger.warning(
                "Tool %s failed (attempt %d/%d): %s",
                tool_call.name,
                attempt + 1,
                _MAX_TOOL_RETRIES + 1,
                last_error,
            )

        return ToolResult(
            success=False,
            output="",
            error=f"After {_MAX_TOOL_RETRIES + 1} attempts: {last_error}",
        )

    async def _execute_tool_with_retry(self, tool_call: ToolCall) -> str:
        """Execute a tool call with retry on failure."""
        await self._events.emit(EventType.TOOL_CALLED, name=tool_call.name, args=tool_call.args)

        last_error = ""
        for attempt in range(_MAX_TOOL_RETRIES + 1):
            result = await self.tools.execute(tool_call.name, **tool_call.args)
            if result.success:
                await self._events.emit(EventType.TOOL_RESULT, name=tool_call.name, output=result.output)
                return result.output

            last_error = result.error
            await self._events.emit(
                EventType.TOOL_ERROR,
                name=tool_call.name,
                error=last_error,
                attempt=attempt,
            )
            logger.warning(
                "Tool %s failed (attempt %d/%d): %s",
                tool_call.name,
                attempt + 1,
                _MAX_TOOL_RETRIES + 1,
                last_error,
            )

        return f"Error after {_MAX_TOOL_RETRIES + 1} attempts: {last_error}"

    # --- Message building ---

    def _build_messages(self, user_input: str, history: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
        system_prompt = self.system_prompt or "You are a helpful assistant."
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages
