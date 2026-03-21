"""Tests for ReActV2Strategy — native tool calling with fallback, guards, events, budget."""

from unittest.mock import AsyncMock

import pytest

from corail.events.bus import EventBus
from corail.events.types import EventType
from corail.guards.base import Guard, GuardDirection, GuardResult
from corail.guards.pipeline import GuardPipeline
from corail.models.base import Model, ModelResponse, ToolCall
from corail.strategies.react_v2 import BudgetOptions, ReActV2Strategy
from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult
from corail.tools.registry import ToolRegistry

# --- Test fixtures ---


class _NativeModel(Model):
    """Mock model that supports native tool calling."""

    def __init__(self) -> None:
        self._generate_with_tools = AsyncMock()

    @property
    def supports_tool_use(self) -> bool:
        return True

    async def generate(self, messages, **kwargs):
        return "fallback response"

    async def generate_with_tools(self, messages, tools, **kwargs):
        return await self._generate_with_tools(messages, tools, **kwargs)


class _PromptModel(Model):
    """Mock model that does NOT support native tool calling."""

    def __init__(self) -> None:
        self._generate = AsyncMock(return_value="prompt-based response")

    @property
    def supports_tool_use(self) -> bool:
        return False

    async def generate(self, messages, **kwargs):
        return await self._generate(messages, **kwargs)


class _EchoTool(ToolExecutor):
    """Simple tool that echoes its input."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echoes input",
            parameters=[ToolParameter(name="text", type="string", description="Text to echo")],
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Echo: {kwargs.get('text', '')}")


class _FailingTool(ToolExecutor):
    """Tool that fails a configurable number of times then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._call_count = 0

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="flaky", description="Flaky tool")

    async def execute(self, **kwargs) -> ToolResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return ToolResult(success=False, output="", error="Temporary failure")
        return ToolResult(success=True, output="Success after retries")


class _BlockingGuard(Guard):
    """Guard that blocks all input."""

    @property
    def name(self) -> str:
        return "blocker"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.INPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        return GuardResult(allowed=False, reason="Blocked by test guard")


class _SanitizingGuard(Guard):
    """Guard that sanitizes output."""

    @property
    def name(self) -> str:
        return "sanitizer"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.OUTPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        sanitized = content.replace("secret", "[REDACTED]")
        return GuardResult(allowed=True, sanitized=sanitized if sanitized != content else "")


@pytest.fixture
def tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def native_model() -> _NativeModel:
    return _NativeModel()


@pytest.fixture
def prompt_model() -> _PromptModel:
    return _PromptModel()


# --- Test: Native vs Fallback path selection ---


class TestPathSelection:
    async def test_native_path_used_when_supported(self, native_model, tools):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="native response", tool_calls=[], stop_reason="end_turn"
        )
        strategy = ReActV2Strategy(model=native_model, tools=tools)
        result = await strategy.execute("hello")
        assert result == "native response"
        native_model._generate_with_tools.assert_awaited_once()

    async def test_fallback_used_when_not_supported(self, prompt_model, tools):
        strategy = ReActV2Strategy(model=prompt_model, tools=tools)
        result = await strategy.execute("hello")
        assert result == "prompt-based response"
        prompt_model._generate.assert_awaited_once()

    async def test_stream_native_path(self, native_model, tools):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="streamed native", tool_calls=[], stop_reason="end_turn"
        )
        strategy = ReActV2Strategy(model=native_model, tools=tools)
        tokens = []
        async for token in strategy.execute_stream("hello"):
            tokens.append(token)
        assert "streamed native" in "".join(tokens)

    async def test_stream_fallback_path(self, prompt_model, tools):
        strategy = ReActV2Strategy(model=prompt_model, tools=tools)
        tokens = []
        async for token in strategy.execute_stream("hello"):
            tokens.append(token)
        assert "prompt-based response" in "".join(tokens)


# --- Test: Native tool calling loop ---


class TestNativeToolCalling:
    async def test_single_tool_call_round(self, native_model, tools):
        """Model calls a tool, gets result, then produces final response."""
        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="Let me echo that.",
                tool_calls=[ToolCall(id="tc_1", name="echo", args={"text": "hello"})],
                stop_reason="tool_use",
            ),
            ModelResponse(
                content="The echo said: Echo: hello",
                tool_calls=[],
                stop_reason="end_turn",
            ),
        ]
        strategy = ReActV2Strategy(model=native_model, tools=tools)
        result = await strategy.execute("echo hello")
        assert "Echo: hello" in result

    async def test_multiple_tool_calls_in_response(self, native_model, tools):
        """Model returns multiple tool_use blocks in one response."""
        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc_1", name="echo", args={"text": "first"}),
                    ToolCall(id="tc_2", name="echo", args={"text": "second"}),
                ],
                stop_reason="tool_use",
            ),
            ModelResponse(content="Done with both.", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = ReActV2Strategy(model=native_model, tools=tools)
        result = await strategy.execute("do both")
        assert result == "Done with both."
        assert native_model._generate_with_tools.await_count == 2


# --- Test: Guard integration ---


class TestGuardIntegration:
    async def test_input_guard_blocks(self, native_model, tools):
        pipeline = GuardPipeline(guards=[_BlockingGuard()])
        strategy = ReActV2Strategy(model=native_model, tools=tools, guard_pipeline=pipeline)
        result = await strategy.execute("anything")
        assert "[Blocked]" in result
        native_model._generate_with_tools.assert_not_awaited()

    async def test_output_guard_sanitizes(self, native_model, tools):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="The secret is 42", tool_calls=[], stop_reason="end_turn"
        )
        pipeline = GuardPipeline(guards=[_SanitizingGuard()])
        strategy = ReActV2Strategy(model=native_model, tools=tools, guard_pipeline=pipeline)
        result = await strategy.execute("tell me")
        assert "[REDACTED]" in result
        assert "secret" not in result

    async def test_no_guards_passes_through(self, native_model, tools):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="raw output", tool_calls=[], stop_reason="end_turn"
        )
        strategy = ReActV2Strategy(model=native_model, tools=tools, guard_pipeline=None)
        result = await strategy.execute("hello")
        assert result == "raw output"

    async def test_stream_input_guard_blocks(self, native_model, tools):
        pipeline = GuardPipeline(guards=[_BlockingGuard()])
        strategy = ReActV2Strategy(model=native_model, tools=tools, guard_pipeline=pipeline)
        tokens = []
        async for token in strategy.execute_stream("anything"):
            tokens.append(token)
        assert "[Blocked]" in "".join(tokens)


# --- Test: Budget tracking ---


class TestBudget:
    async def test_max_rounds_stops_loop(self, native_model, tools):
        """When max_rounds is reached, execution stops."""
        # Model always wants to call tools — should be stopped by budget
        native_model._generate_with_tools.return_value = ModelResponse(
            content="still going",
            tool_calls=[ToolCall(id="tc_x", name="echo", args={"text": "loop"})],
            stop_reason="tool_use",
        )
        budget = BudgetOptions(max_rounds=2)
        strategy = ReActV2Strategy(model=native_model, tools=tools, budget=budget)
        result = await strategy.execute("loop forever")
        # Should have been called exactly max_rounds times
        assert native_model._generate_with_tools.await_count == 2
        # Result should contain content or budget message
        assert result in ("still going", "[Max rounds exceeded]")

    async def test_max_tokens_stops_loop(self, native_model, tools):
        """When token budget is exceeded, execution stops."""
        long_content = "x" * 200
        native_model._generate_with_tools.return_value = ModelResponse(
            content=long_content,
            tool_calls=[ToolCall(id="tc_x", name="echo", args={"text": "loop"})],
            stop_reason="tool_use",
        )
        budget = BudgetOptions(max_rounds=100, max_tokens=50)
        strategy = ReActV2Strategy(model=native_model, tools=tools, budget=budget)
        result = await strategy.execute("go")
        # Should stop after first round due to token budget
        assert native_model._generate_with_tools.await_count == 1


# --- Test: Tool error retry ---


class TestToolRetry:
    async def test_retry_on_tool_error(self, native_model):
        """Tool that fails once then succeeds should be retried."""
        registry = ToolRegistry()
        flaky = _FailingTool(fail_count=1)
        registry.register(flaky)

        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="tc_1", name="flaky", args={})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="Got it.", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = ReActV2Strategy(model=native_model, tools=registry)
        result = await strategy.execute("try it")
        assert result == "Got it."
        # Flaky tool should have been called twice (1 fail + 1 success)
        assert flaky._call_count == 2

    async def test_max_retries_exhausted(self, native_model):
        """Tool that always fails should exhaust retries and return error."""
        registry = ToolRegistry()
        always_fail = _FailingTool(fail_count=999)
        registry.register(always_fail)

        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="tc_1", name="flaky", args={})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="Error handled.", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = ReActV2Strategy(model=native_model, tools=registry)
        result = await strategy.execute("try it")
        # Should have tried 3 times total (1 + 2 retries)
        assert always_fail._call_count == 3


# --- Test: Event emission ---


class TestEventEmission:
    async def test_events_emitted_on_execute(self, native_model, tools, event_bus):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="done", tool_calls=[], stop_reason="end_turn"
        )
        strategy = ReActV2Strategy(model=native_model, tools=tools, event_bus=event_bus)
        await strategy.execute("hello")

        event_types = [e.type for e in event_bus.history]
        assert EventType.MESSAGE_RECEIVED in event_types
        assert EventType.LLM_CALL_STARTED in event_types
        assert EventType.LLM_CALL_COMPLETED in event_types
        assert EventType.MESSAGE_RESPONSE in event_types

    async def test_tool_events_emitted(self, native_model, tools, event_bus):
        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="",
                tool_calls=[ToolCall(id="tc_1", name="echo", args={"text": "hi"})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="done", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = ReActV2Strategy(model=native_model, tools=tools, event_bus=event_bus)
        await strategy.execute("echo hi")

        event_types = [e.type for e in event_bus.history]
        assert EventType.TOOL_CALLED in event_types
        assert EventType.TOOL_RESULT in event_types

    async def test_guard_blocked_event_emitted(self, native_model, tools, event_bus):
        pipeline = GuardPipeline(guards=[_BlockingGuard()])
        strategy = ReActV2Strategy(
            model=native_model,
            tools=tools,
            guard_pipeline=pipeline,
            event_bus=event_bus,
        )
        await strategy.execute("anything")

        event_types = [e.type for e in event_bus.history]
        assert EventType.GUARD_BLOCKED in event_types

    async def test_budget_exceeded_event(self, native_model, tools, event_bus):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="looping",
            tool_calls=[ToolCall(id="tc_x", name="echo", args={"text": "x"})],
            stop_reason="tool_use",
        )
        budget = BudgetOptions(max_rounds=1)
        strategy = ReActV2Strategy(
            model=native_model,
            tools=tools,
            event_bus=event_bus,
            budget=budget,
        )
        await strategy.execute("loop")

        event_types = [e.type for e in event_bus.history]
        assert EventType.BUDGET_EXCEEDED in event_types

    async def test_no_event_bus_does_not_crash(self, native_model, tools):
        native_model._generate_with_tools.return_value = ModelResponse(
            content="ok", tool_calls=[], stop_reason="end_turn"
        )
        strategy = ReActV2Strategy(model=native_model, tools=tools, event_bus=None)
        result = await strategy.execute("hello")
        assert result == "ok"


# --- Test: Tool schema conversion ---


class TestToolSchemaConversion:
    def test_definitions_to_anthropic_format(self):
        from corail.strategies.react_v2 import _build_tool_schemas

        registry = ToolRegistry()
        registry.register(_EchoTool())
        schemas = _build_tool_schemas(registry)

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "echo"
        assert schema["description"] == "Echoes input"
        assert "input_schema" in schema
        assert "text" in schema["input_schema"]["properties"]
        assert "text" in schema["input_schema"]["required"]


# --- Test: Strategy factory registration ---


class TestFactoryRegistration:
    def test_agent_react_in_registry(self):
        from corail.strategies.factory import StrategyFactory

        assert "agent-react" in StrategyFactory.available()

    def test_agent_react_create(self, native_model):
        from corail.strategies.factory import StrategyFactory

        strategy = StrategyFactory.create("agent-react", model=native_model)
        from corail.strategies.agent import UnifiedAgentStrategy

        assert isinstance(strategy, UnifiedAgentStrategy)
