"""UnifiedAgentStrategy — single adaptive strategy that replaces simple, react, react_v2, and rag.

Behavior is determined by what capabilities are injected, not by strategy name:
- No tools                       -> simple LLM chat
- Tools injected                 -> ReAct tool-calling loop
- KB search tools registered     -> agentic RAG (agent decides when to search)
- Planner injected               -> task decomposition for complex goals
- Memory injected                -> persistent memory across sessions
- Guards injected                -> input/output security
- All of the above               -> full agentic behavior

Works with both native tool_use (Anthropic/Bedrock) and prompt-based (Ollama).
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from corail.core.stream import (
    ComponentEvent,
    ConfirmEvent,
    PlanEvent,
    SourcesEvent,
    StreamToken,
    ToolEndEvent,
    ToolStartEvent,
)
from corail.events.bus import EventBus
from corail.events.types import EventType
from corail.guards.pipeline import GuardPipeline
from corail.memory.manager import MemoryManager
from corail.models.base import Model, ModelResponse, ToolCall
from corail.planning.correction import SelfCorrector
from corail.planning.planner import Plan, Planner, PlanStep
from corail.skills.registry import SkillRegistry
from corail.strategies._shared import (
    StopReason,
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_ROUNDS,
    _DEFAULT_MAX_TOKENS,
    _EventEmitter,
    _GuardRunner,
    _MAX_TOOL_RETRIES,
    build_system_prompt,
)
from corail.strategies.base import AgentStrategy
from corail.tools.base import ToolDefinition, ToolParameter, ToolResult
from corail.tools.registry import ToolRegistry
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    # Create a no-op decorator fallback
    class _NoOpMLflow:
        @staticmethod
        def trace(name: str = ""):
            def decorator(fn):
                return fn
            return decorator
        class tracing:
            @staticmethod
            def enable(): pass
        @staticmethod
        def set_tracking_uri(uri): pass
        @staticmethod
        def set_experiment(name): pass
    mlflow = _NoOpMLflow()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Human-readable suffixes appended to the final response when a loop
# terminates abnormally, so users and MLflow traces can see *why*.
_STOP_MAX_ROUNDS_MSG = "\n\n[Stopped: max rounds reached]"
_STOP_TOKEN_BUDGET_MSG = "\n\n[Stopped: token budget exceeded]"

# Prompt-based tool call patterns (for models without native tool_use)
_TOOL_CALL_PATTERNS = [
    re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL),
    re.compile(r"<tool_use>(.*?)</tool_use>", re.DOTALL),
]

# ---------------------------------------------------------------------------
# Schema converters (native tool_use format)
# ---------------------------------------------------------------------------

def _parameter_to_json_schema(param: ToolParameter) -> dict[str, Any]:
    return {"type": param.type, "description": param.description}


def _definition_to_anthropic_tool(defn: ToolDefinition) -> dict[str, Any]:
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


def _build_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    return [_definition_to_anthropic_tool(d) for d in registry.list_definitions()]


def _tool_result_message(tool_call: ToolCall, result_text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result_text}],
    }


def _assistant_message_from_response(response: ModelResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if response.content:
        content.append({"type": "text", "text": response.content})
    for tc in response.tool_calls:
        content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args})
    return {"role": "assistant", "content": content}


def _try_parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# UnifiedAgentStrategy
# ---------------------------------------------------------------------------

class UnifiedAgentStrategy(AgentStrategy):
    """The unified agent. Adapts behavior based on available capabilities.

    Inject only what you need:
    - Nothing extra         -> simple LLM chat
    - tools                 -> ReAct tool loop (native or prompt-based)
    - KB search tools       -> agentic RAG (agent decides when to search)
    - planner + corrector   -> multi-step planning with self-correction
    - memory                -> persistent cross-session memory
    - guards                -> input/output security pipeline
    """

    def __init__(
        self,
        model: Model,
        system_prompt: str = "",
        tools: ToolRegistry | None = None,
        guard_pipeline: GuardPipeline | None = None,
        event_bus: EventBus | None = None,
        planner: Planner | None = None,
        corrector: SelfCorrector | None = None,
        memory: MemoryManager | None = None,
        skills: SkillRegistry | None = None,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        grounding_strict: bool = True,
    ) -> None:
        super().__init__(model, system_prompt)
        self.tools = tools or ToolRegistry()
        self._guards = _GuardRunner(guard_pipeline)
        self._events = _EventEmitter(event_bus)
        self._planner = planner
        self._corrector = corrector
        self._memory = memory
        self._skills = skills or SkillRegistry()
        if self._memory is not None and self._memory._model is None:
            # Memory extraction is a background task — use the cheap model
            # when CORAIL_BACKGROUND_MODEL is configured so a slow 35B chat
            # model doesn't block summarisation after every turn.
            from corail.models.factory import background_model
            self._memory._model = background_model(model)
        self._max_rounds = max_rounds
        self._max_tokens = max_tokens
        self._grounding_strict = grounding_strict
        # MLflow tracing is initialized by the CLI layer — no double-init here.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object) -> str:
        # 1. Guard input
        input_result = await self._guards.check_input(user_input)
        if not input_result.allowed:
            await self._events.emit(EventType.GUARD_BLOCKED, direction="input", reason=input_result.reason)
            return f"[Blocked] {input_result.reason}"

        effective_input = input_result.sanitized or user_input
        await self._events.emit(EventType.MESSAGE_RECEIVED, content=effective_input)

        # 2. Memory recall
        memory_context = await self._recall_memory(effective_input)

        # 3. Build system prompt with memory
        system = self._build_system_prompt(memory_context)

        # 4. Decide: plan or direct execution
        plan: Plan | None = None
        if self._planner is not None and Planner.needs_planning(effective_input):
            plan = await self._planner.create_plan(effective_input, self.tools.names())
            await self._events.emit(EventType.PLAN_CREATED, goal=plan.goal, steps=len(plan.steps))

        # 5. Execute
        if plan is not None and plan.steps:
            final_content = await self._execute_with_plan(plan, system, effective_input, history)
        else:
            final_content = await self._execute_direct(system, effective_input, history)

        # 7. Memory: extract learnings in background (don't block response)
        if self._memory is not None and final_content:
            import asyncio
            full_history = list(history or [])
            full_history.append({"role": "user", "content": effective_input})
            full_history.append({"role": "assistant", "content": final_content})
            asyncio.create_task(self._memory.extract_from_conversation(full_history))

        # 8. Guard output
        output_result = await self._guards.check_output(final_content)
        if not output_result.allowed:
            await self._events.emit(EventType.GUARD_BLOCKED, direction="output", reason=output_result.reason)
            return f"[Blocked] {output_result.reason}"

        final = output_result.sanitized or final_content
        await self._events.emit(EventType.MESSAGE_RESPONSE, content=final)
        return final

    async def execute_stream(
        self, user_input: str, history: list[dict[str, str]] | None = None, **kwargs: object,
    ) -> AsyncIterator[StreamToken]:
        # 1. Guard input
        input_result = await self._guards.check_input(user_input)
        if not input_result.allowed:
            yield f"[Blocked] {input_result.reason}"
            return

        effective_input = input_result.sanitized or user_input

        # 2. Memory recall
        memory_context = await self._recall_memory(effective_input)

        # 3. System prompt
        system = self._build_system_prompt(memory_context)

        # 4. Plan?
        plan: Plan | None = None
        if self._planner is not None and Planner.needs_planning(effective_input):
            plan = await self._planner.create_plan(effective_input, self.tools.names())
            await self._events.emit(EventType.PLAN_CREATED, goal=plan.goal, steps=len(plan.steps))
            yield PlanEvent(
                plan_goal=plan.goal,
                step_description="Plan created",
                step_status="created",
                step_index=0,
                total_steps=len(plan.steps),
            )

        # 5. Stream execution — accumulate text for memory extraction
        accumulated_text = ""
        token_count = 0
        if plan is not None and plan.steps:
            async for token in self._stream_with_plan(plan, system, effective_input, history):
                if isinstance(token, str):
                    accumulated_text += token
                    token_count += 1
                yield token
        else:
            async for token in self._stream_direct(system, effective_input, history):
                if isinstance(token, str):
                    accumulated_text += token
                    token_count += 1
                yield token

        # 7. Tracing is handled by the channel layer (rest.py)

        # 8. Memory: extract learnings in background (don't block stream completion)
        if self._memory is not None and accumulated_text:
            import asyncio
            full_history = list(history or [])
            full_history.append({"role": "user", "content": effective_input})
            full_history.append({"role": "assistant", "content": accumulated_text})
            asyncio.create_task(self._memory.extract_from_conversation(full_history))

    # ------------------------------------------------------------------
    # Direct execution (no plan)
    # ------------------------------------------------------------------

    async def _execute_direct(
        self, system: str, user_input: str, history: list[dict[str, str]] | None,
    ) -> str:
        messages = self._build_messages(system, user_input, history)

        # Native tool_use path
        if self.model.supports_tool_use and len(self.tools) > 0:
            return await self._native_tool_loop(messages)

        # Prompt-based tool path (Ollama, etc.)
        if len(self.tools) > 0:
            return await self._prompt_tool_loop(messages)

        # Simple LLM call
        return await self.model.generate(messages=messages)

    async def _stream_direct(
        self, system: str, user_input: str, history: list[dict[str, str]] | None,
    ) -> AsyncIterator[StreamToken]:
        messages = self._build_messages(system, user_input, history)

        if self.model.supports_tool_use and len(self.tools) > 0:
            async for token in self._stream_native_tools(messages):
                yield token
        elif len(self.tools) > 0:
            async for token in self._stream_prompt_tools(messages):
                yield token
        else:
            async for token in self.model.generate_stream(messages=messages):
                yield token

    # ------------------------------------------------------------------
    # Planned execution
    # ------------------------------------------------------------------

    async def _execute_with_plan(
        self, plan: Plan, system: str, user_input: str, history: list[dict[str, str]] | None,
    ) -> str:
        results: list[str] = []

        for step_idx, step in enumerate(plan.steps):
            step.mark_started()
            await self._events.emit(
                EventType.PLAN_STEP_STARTED, step=step.description, index=step_idx,
            )

            step_prompt = self._step_prompt(plan, step, results, user_input)
            messages = self._build_messages(system, step_prompt, history)

            try:
                step_result = await self._execute_step(messages)
                step.mark_completed(step_result)
                results.append(step_result)
                await self._events.emit(
                    EventType.PLAN_STEP_COMPLETED, step=step.description, index=step_idx,
                )
            except Exception as exc:
                error = str(exc)
                step.mark_failed(error)
                await self._events.emit(
                    EventType.PLAN_STEP_FAILED, step=step.description, error=error, index=step_idx,
                )
                # Self-correction
                correction = await self._try_correction(step, error)
                if correction:
                    results.append(f"[Corrected] {correction}")
                else:
                    results.append(f"[Failed] {error}")

        await self._events.emit(EventType.PLAN_COMPLETED, goal=plan.goal, progress=plan.progress)

        # Final synthesis
        synthesis_prompt = (
            f"Original goal: {plan.goal}\n\n"
            f"Step results:\n" + "\n".join(f"- {r}" for r in results) + "\n\n"
            f"Provide a final comprehensive answer based on these results."
        )
        synth_messages = self._build_messages(system, synthesis_prompt, None)
        return await self.model.generate(messages=synth_messages)

    async def _stream_with_plan(
        self, plan: Plan, system: str, user_input: str, history: list[dict[str, str]] | None,
    ) -> AsyncIterator[StreamToken]:
        results: list[str] = []

        for step_idx, step in enumerate(plan.steps):
            step.mark_started()
            yield PlanEvent(
                plan_goal=plan.goal,
                step_description=step.description,
                step_status="in_progress",
                step_index=step_idx,
                total_steps=len(plan.steps),
            )

            step_prompt = self._step_prompt(plan, step, results, user_input)
            messages = self._build_messages(system, step_prompt, history)

            try:
                step_result = await self._execute_step(messages)
                step.mark_completed(step_result)
                results.append(step_result)
                yield PlanEvent(
                    plan_goal=plan.goal,
                    step_description=step.description,
                    step_status="completed",
                    step_index=step_idx,
                    total_steps=len(plan.steps),
                )
            except Exception as exc:
                error = str(exc)
                step.mark_failed(error)
                correction = await self._try_correction(step, error)
                if correction:
                    results.append(f"[Corrected] {correction}")
                else:
                    results.append(f"[Failed] {error}")
                yield PlanEvent(
                    plan_goal=plan.goal,
                    step_description=step.description,
                    step_status="failed",
                    step_index=step_idx,
                    total_steps=len(plan.steps),
                )

        # Stream final synthesis
        synthesis_prompt = (
            f"Original goal: {plan.goal}\n\n"
            f"Step results:\n" + "\n".join(f"- {r}" for r in results) + "\n\n"
            f"Provide a final comprehensive answer based on these results."
        )
        synth_messages = self._build_messages(system, synthesis_prompt, None)
        async for token in self.model.generate_stream(messages=synth_messages):
            yield token

    async def _execute_step(self, messages: list[dict[str, Any]]) -> str:
        """Execute a single plan step with the appropriate tool path."""
        if self.model.supports_tool_use and len(self.tools) > 0:
            return await self._native_tool_loop(messages)
        if len(self.tools) > 0:
            return await self._prompt_tool_loop(messages)
        return await self.model.generate(messages=messages)

    # ------------------------------------------------------------------
    # Native tool calling loop
    # ------------------------------------------------------------------

    async def _native_tool_loop(self, messages: list[dict[str, Any]]) -> str:
        tool_schemas = _build_tool_schemas(self.tools)
        tokens_used = 0
        response: ModelResponse | None = None

        for round_num in range(self._max_rounds):
            await self._events.emit(EventType.TURN_STARTED, round=round_num)
            await self._events.emit(EventType.LLM_CALL_STARTED, round=round_num)

            with self._llm_span(round_num):
                response = await self.model.generate_with_tools(messages, tool_schemas)
                self._end_llm_span(response, messages)

            tokens_used += len(response.content) // _CHARS_PER_TOKEN
            await self._events.emit(EventType.LLM_CALL_COMPLETED, round=round_num, stop_reason=response.stop_reason)

            if response.stop_reason != "tool_use":
                await self._emit_turn_ended(round_num, StopReason.END_TURN)
                return response.content

            messages.append(_assistant_message_from_response(response))
            for tc in response.tool_calls:
                with self._tool_span(tc):
                    result = await self._execute_tool_with_retry_full(tc)
                    result_text = result.output if result.success else f"Error: {result.error}"
                    self._end_tool_span(tc, result)
                messages.append(_tool_result_message(tc, result_text))

            if tokens_used >= self._max_tokens:
                await self._events.emit(EventType.BUDGET_EXCEEDED, tokens_used=tokens_used)
                await self._emit_turn_ended(round_num, StopReason.TOKEN_BUDGET)
                logger.warning("Agent loop stopped: token budget exceeded (%d tokens)", tokens_used)
                return (response.content or "") + _STOP_TOKEN_BUDGET_MSG

        await self._emit_turn_ended(self._max_rounds - 1, StopReason.MAX_ROUNDS)
        logger.warning("Agent loop stopped: max rounds (%d) reached", self._max_rounds)
        return ((response.content if response else "") or "") + _STOP_MAX_ROUNDS_MSG

    async def _stream_native_tools(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamToken]:
        tool_schemas = _build_tool_schemas(self.tools)

        for round_num in range(self._max_rounds):
            await self._events.emit(EventType.TURN_STARTED, round=round_num)

            with self._llm_span(round_num):
                response = await self.model.generate_with_tools(messages, tool_schemas)
                self._end_llm_span(response, messages)

            if response.content:
                yield response.content

            if response.stop_reason != "tool_use":
                await self._emit_turn_ended(round_num, StopReason.END_TURN)
                return

            messages.append(_assistant_message_from_response(response))
            for tc in response.tool_calls:
                with self._tool_span(tc):
                    result = await self._execute_tool_with_retry_full(tc)
                    self._end_tool_span(tc, result)
                result_text = result.output if result.success else f"Error: {result.error}"
                # Still yield stream events for the frontend
                yield ToolStartEvent(tool=tc.name, args=tc.args, call_id=tc.id)
                yield ToolEndEvent(tool=tc.name, output=result_text, success=result.success, call_id=tc.id)
                if result.props.get("sources"):
                    yield SourcesEvent(sources=result.props["sources"])
                messages.append(_tool_result_message(tc, result_text))

        await self._emit_turn_ended(self._max_rounds - 1, StopReason.MAX_ROUNDS)
        logger.warning("Stream agent loop stopped: max rounds (%d) reached", self._max_rounds)
        yield _STOP_MAX_ROUNDS_MSG

    # ------------------------------------------------------------------
    # Prompt-based tool calling loop (Ollama, etc.)
    # ------------------------------------------------------------------

    async def _prompt_tool_loop(self, messages: list[dict[str, Any]]) -> str:
        messages = self._inject_tool_prompt(messages)
        response = ""

        for round_num in range(self._max_rounds):
            await self._events.emit(EventType.TURN_STARTED, round=round_num)

            with self._llm_span(round_num):
                response = await self.model.generate(messages=messages)
                # Prompt-based: model returns text, not ModelResponse
                self._end_llm_span(ModelResponse(content=response, stop_reason="end_turn"), messages)

            tool_call = self._extract_prompt_tool_call(response)

            if tool_call is None:
                await self._emit_turn_ended(round_num, StopReason.END_TURN)
                return self._strip_think_tags(response)

            call_json, _ = tool_call
            result = await self._execute_prompt_tool_full(call_json)
            result_text = result.output if result.success else f"Error: {result.error}"
            messages.append({"role": "assistant", "content": self._strip_think_tags(response)})
            messages.append({"role": "user", "content": f"[Tool Result]\n{result_text}"})

        await self._emit_turn_ended(self._max_rounds - 1, StopReason.MAX_ROUNDS)
        logger.warning("Prompt-tool loop stopped: max rounds (%d) reached", self._max_rounds)
        return self._strip_think_tags(response) + _STOP_MAX_ROUNDS_MSG

    # Patterns that signal the start of a tool-call block in the stream.
    _TOOL_CALL_MARKERS = ("```tool_call", "<tool_use>")

    async def _stream_prompt_tools(self, messages: list[dict[str, Any]]) -> AsyncIterator[StreamToken]:
        messages = self._inject_tool_prompt(messages)

        for round_num in range(self._max_rounds):
            await self._events.emit(EventType.TURN_STARTED, round=round_num)

            # Stream tokens, suppressing tool_call JSON blocks from the user
            full_response = ""
            suppressing = False
            with self._llm_span(round_num):
                async for token in self.model.generate_stream(messages=messages):
                    full_response += token
                    if not suppressing:
                        if any(marker in full_response for marker in self._TOOL_CALL_MARKERS):
                            suppressing = True
                        else:
                            yield token
                self._end_llm_span(ModelResponse(content=full_response, stop_reason="end_turn"), messages)

            tool_call = self._extract_prompt_tool_call(full_response)

            if tool_call is None:
                if suppressing:
                    for marker in self._TOOL_CALL_MARKERS:
                        idx = full_response.find(marker)
                        if idx >= 0:
                            yield full_response[idx:]
                            break
                await self._emit_turn_ended(round_num, StopReason.END_TURN)
                return

            call_json, _ = tool_call
            tool_name = self._extract_tool_name(call_json)
            tool_args = self._extract_tool_args(call_json)

            yield ToolStartEvent(tool=tool_name, args=tool_args)
            result = await self._execute_prompt_tool_full(call_json)
            result_text = result.output if result.success else f"Error: {result.error}"
            yield ToolEndEvent(tool=tool_name, output=result_text, success=result.success)

            if result.props.get("sources"):
                yield SourcesEvent(sources=result.props["sources"])

            if result.success and result.render != "text":
                component_name = result.component if result.render == "react" else result.render
                props = result.props or _try_parse_json(result.output)
                yield ComponentEvent(component=component_name, props=props)

            messages.append({"role": "assistant", "content": self._strip_think_tags(full_response)})
            messages.append({"role": "user", "content": f"[Tool Result]\n{result_text}"})

        await self._emit_turn_ended(self._max_rounds - 1, StopReason.MAX_ROUNDS)
        logger.warning("Stream prompt-tool loop stopped: max rounds (%d) reached", self._max_rounds)
        yield _STOP_MAX_ROUNDS_MSG

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    async def _stream_tool_execution(self, tc: ToolCall) -> AsyncIterator[StreamToken]:
        """Execute a native tool call with structured stream events."""
        tool_defn = self._get_tool_definition(tc.name)
        if tool_defn and tool_defn.risk_level == "confirm":
            yield ConfirmEvent(call_id=tc.id, tool=tc.name, args=tc.args, message=f"Execute {tc.name}?")

        yield ToolStartEvent(tool=tc.name, args=tc.args, call_id=tc.id)
        result = await self._execute_tool_with_retry_full(tc)
        result_text = result.output if result.success else f"Error: {result.error}"
        yield ToolEndEvent(tool=tc.name, output=result_text, success=result.success, call_id=tc.id)

        if result.props.get("sources"):
            yield SourcesEvent(sources=result.props["sources"])

        if result.success and result.render != "text":
            component_name = result.component if result.render == "react" else result.render
            props = result.props or _try_parse_json(result.output)
            yield ComponentEvent(component=component_name, props=props)

    # ------------------------------------------------------------------
    # MLflow real-time span helpers (nesting via OTel context propagation)
    # ------------------------------------------------------------------

    def _llm_span(self, round_num: int):
        """Context manager — no-op. Real-time spans disabled to avoid orphan traces.
        MLflow tracing happens post-hoc via _log_chat_trace in the channel layer.
        """
        from contextlib import nullcontext
        return nullcontext()

    def _end_llm_span(self, response: ModelResponse, messages: list[dict] | None = None) -> None:
        if not _HAS_MLFLOW:
            return
        span = mlflow.get_current_active_span()
        if span is None:
            return
        model_id = getattr(self.model, "model_id", "unknown")
        # Inputs: model name + last few messages (truncated for readability)
        inputs: dict[str, Any] = {"model": model_id}
        if messages:
            inputs["messages"] = [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:200]}
                for m in messages[-4:]
            ]
        span.set_inputs(inputs)
        # Outputs: full response (content, tool_calls, stop_reason)
        outputs: dict[str, Any] = {"stop_reason": response.stop_reason}
        if response.content:
            outputs["content"] = response.content
        if response.tool_calls:
            outputs["tool_calls"] = [
                {"name": tc.name, "args": tc.args} for tc in response.tool_calls
            ]
        span.set_outputs(outputs)
        span.set_attribute("model", model_id)

    def _tool_span(self, tc: ToolCall):
        """Context manager — no-op. Real-time spans disabled to avoid orphan traces."""
        from contextlib import nullcontext
        return nullcontext()

    def _end_tool_span(self, tc: ToolCall, result: ToolResult) -> None:
        """Set inputs/outputs on the current tool span from the full ToolResult."""
        if not _HAS_MLFLOW:
            return
        span = mlflow.get_current_active_span()
        if span is None:
            return
        span.set_inputs({"tool": tc.name, "args": tc.args})

        if not result.success:
            span.set_outputs({"error": result.error})
            span.set_status("ERROR")
            return

        # RETRIEVER spans: structured documents for MLflow UI
        sources = result.props.get("sources")
        if tc.name.startswith("search_") and sources:
            span.set_outputs([
                {
                    "page_content": s.get("content_preview", ""),
                    "metadata": {
                        "source": s.get("filename", ""),
                        "score": s.get("score", 0),
                        "chunk_index": s.get("chunk_index", 0),
                    },
                }
                for s in sources
            ])
        else:
            span.set_outputs({"output": result.output})

    async def _emit_turn_ended(self, round_num: int, stop_reason: StopReason) -> None:
        """Emit TURN_ENDED exactly once per turn, carrying the real stop reason.

        Called only at terminal exits so listeners never see a misleading
        mid-loop stop_reason on the final TURN_ENDED.
        """
        await self._events.emit(
            EventType.TURN_ENDED, round=round_num, stop_reason=stop_reason.value,
        )

    async def _execute_tool_with_retry_full(self, tool_call: ToolCall) -> ToolResult:
        await self._events.emit(EventType.TOOL_CALLED, name=tool_call.name, args=tool_call.args)

        last_error = ""
        for attempt in range(_MAX_TOOL_RETRIES + 1):
            # pi-agent pattern: unexpected exceptions become tool_result errors
            # so the LLM can retry or explain, instead of bubbling to the user.
            try:
                result = await self.tools.execute(tool_call.name, **tool_call.args)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await self._events.emit(
                    EventType.TOOL_ERROR, name=tool_call.name, error=last_error, attempt=attempt,
                )
                logger.warning(
                    "Tool %s raised (attempt %d/%d): %s",
                    tool_call.name, attempt + 1, _MAX_TOOL_RETRIES + 1, last_error,
                )
                continue

            if result.success:
                await self._events.emit(
                    EventType.TOOL_RESULT, name=tool_call.name,
                    output=result.output, sources=result.props.get("sources"),
                )
                return result

            last_error = result.error
            await self._events.emit(
                EventType.TOOL_ERROR, name=tool_call.name, error=last_error, attempt=attempt,
            )
            logger.warning("Tool %s failed (attempt %d/%d): %s", tool_call.name, attempt + 1, _MAX_TOOL_RETRIES + 1, last_error)

        return ToolResult(success=False, output="", error=f"After {_MAX_TOOL_RETRIES + 1} attempts: {last_error}")

    async def _execute_tool_with_retry(self, tool_call: ToolCall) -> str:
        result = await self._execute_tool_with_retry_full(tool_call)
        return result.output if result.success else f"Error: {result.error}"

    async def _execute_prompt_tool(self, call_json: str) -> str:
        result = await self._execute_prompt_tool_full(call_json)
        return result.output if result.success else f"Error: {result.error}"

    async def _execute_prompt_tool_full(self, call_json: str) -> ToolResult:
        try:
            call = json.loads(call_json.strip())
            name = call.get("name", "")
            args = call.get("args", {})
            await self._events.emit(EventType.TOOL_CALLED, name=name, args=args)
            result = await self.tools.execute(name, **args)
            if result.success:
                await self._events.emit(EventType.TOOL_RESULT, name=name, output=result.output[:500])
            else:
                await self._events.emit(EventType.TOOL_ERROR, name=name, error=result.error)
            return result
        except json.JSONDecodeError:
            return ToolResult(success=False, output="", error=f"Invalid tool call JSON: {call_json[:200]}")
        except Exception as e:
            await self._events.emit(EventType.TOOL_ERROR, name="unknown", error=str(e))
            return ToolResult(success=False, output="", error=f"Tool execution error: {e}")

    def _get_tool_definition(self, name: str) -> ToolDefinition | None:
        tool = self.tools.get(name)
        return tool.definition() if tool else None

    # ------------------------------------------------------------------
    # Prompt-based tool call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prompt_tool_call(text: str) -> tuple[str, str] | None:
        clean = re.sub(r"</?think>", "", text)
        for pattern in _TOOL_CALL_PATTERNS:
            match = pattern.search(clean)
            if match:
                return match.group(1).strip(), match.group(0)
        return None

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        return re.sub(r"</?think>", "", text)

    @staticmethod
    def _extract_tool_name(call_json: str) -> str:
        try:
            return json.loads(call_json.strip()).get("name", "unknown")
        except (json.JSONDecodeError, AttributeError):
            return "unknown"

    @staticmethod
    def _extract_tool_args(call_json: str) -> dict[str, Any]:
        try:
            return json.loads(call_json.strip()).get("args", {})
        except (json.JSONDecodeError, AttributeError):
            return {}

    def _inject_tool_prompt(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add tool descriptions to the system prompt for prompt-based tool calling."""
        tool_section = self.tools.build_system_prompt_section()
        if not tool_section:
            return messages

        updated = list(messages)
        if updated and updated[0].get("role") == "system":
            updated[0] = {**updated[0], "content": updated[0]["content"] + tool_section}
        else:
            updated.insert(0, {"role": "system", "content": tool_section})
        return updated

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _recall_memory(self, query: str) -> str:
        if self._memory is None:
            return ""
        context = await self._memory.build_context(query)
        if context:
            await self._events.emit(EventType.MEMORY_RECALLED, query=query)
        return context

    # ------------------------------------------------------------------
    # Self-correction
    # ------------------------------------------------------------------

    async def _try_correction(self, step: PlanStep, error: str) -> str | None:
        if self._corrector is None:
            return None
        try:
            alternative = await self._corrector.suggest_alternative(
                step.description, error, self.tools.names(),
            )
            await self._events.emit(EventType.CORRECTION_ATTEMPTED, step=step.description, alternative=alternative)
            return alternative
        except Exception:
            logger.warning("Self-correction failed for step: %s", step.description, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """Assemble the system prompt using the shared builder.

        The shared builder applies grounding rules by default (disable via
        grounding_strict=False) and merges memory context and skill directives.
        """
        return build_system_prompt(
            self.system_prompt,
            memory=self._memory,
            memory_context=memory_context,
            skills=self._skills,
            channel="rest",
            grounding_strict=self._grounding_strict,
        )

    def _build_messages(
        self,
        system: str,
        user_input: str,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _step_prompt(plan: Plan, step: PlanStep, previous_results: list[str], original_input: str) -> str:
        context_parts = [f"Original goal: {original_input}"]
        if previous_results:
            context_parts.append("Previous results:\n" + "\n".join(f"- {r}" for r in previous_results))
        context_parts.append(f"Current step: {step.description}")
        context_parts.append("Complete this step and provide the result.")
        return "\n\n".join(context_parts)
