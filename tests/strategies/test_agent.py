"""Tests for UnifiedAgentStrategy — the single adaptive agent strategy."""

import json
from unittest.mock import AsyncMock

import pytest

from corail.core.stream import PlanEvent, ToolEndEvent, ToolStartEvent
from corail.events.bus import EventBus
from corail.events.types import EventType
from corail.guards.base import Guard, GuardDirection, GuardResult
from corail.guards.pipeline import GuardPipeline
from corail.memory.in_memory import InMemoryStorage
from corail.memory.manager import MemoryManager
from corail.models.base import Model, ModelResponse, ToolCall
from corail.planning.correction import SelfCorrector
from corail.planning.planner import Planner
from corail.strategies.agent import UnifiedAgentStrategy
from corail.tools.base import ToolDefinition, ToolExecutor, ToolParameter, ToolResult
from corail.tools.registry import ToolRegistry


# --- Mock models ---

class _NativeModel(Model):
    """Model with native tool_use support."""

    def __init__(self) -> None:
        self._generate = AsyncMock(return_value="native response")
        self._generate_with_tools = AsyncMock()
        self._generate_stream = AsyncMock()

    @property
    def supports_tool_use(self) -> bool:
        return True

    async def generate(self, messages, **kwargs):
        return await self._generate(messages, **kwargs)

    async def generate_with_tools(self, messages, tools, **kwargs):
        return await self._generate_with_tools(messages, tools, **kwargs)

    async def generate_stream(self, messages, **kwargs):
        result = await self._generate(messages, **kwargs)
        yield result


class _PromptModel(Model):
    """Model without native tool_use (Ollama-style)."""

    def __init__(self) -> None:
        self._generate = AsyncMock(return_value="prompt response")

    @property
    def supports_tool_use(self) -> bool:
        return False

    async def generate(self, messages, **kwargs):
        return await self._generate(messages, **kwargs)

    async def generate_stream(self, messages, **kwargs):
        result = await self._generate(messages, **kwargs)
        yield result


# --- Mock tools ---

class _EchoTool(ToolExecutor):
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echoes input",
            parameters=[ToolParameter(name="text", type="string", description="Text to echo")],
        )

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Echo: {kwargs.get('text', '')}")


# --- Mock KB search tool ---

class _MockKBSearchTool(ToolExecutor):
    """Mock KB search tool that returns pre-configured results."""

    def __init__(self, name: str = "search_test_kb", results: list[dict] | None = None) -> None:
        self._tool_name = name
        self._results = results or []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._tool_name,
            description="Search the test knowledge base",
            parameters=[ToolParameter(name="query", type="string", description="Search query")],
        )

    async def execute(self, **kwargs) -> ToolResult:
        if not self._results:
            return ToolResult(success=True, output=f"No relevant documents found in {self._tool_name}.")
        parts = [f"[Source: {r['filename']}]\n{r['content']}" for r in self._results]
        sources = [{"filename": r["filename"], "score": r.get("score", 0.9), "chunk_index": i, "content_preview": r["content"][:100]} for i, r in enumerate(self._results)]
        return ToolResult(success=True, output="\n\n---\n\n".join(parts), props={"sources": sources})


# --- Mock guard ---

class _BlockingGuard(Guard):
    @property
    def name(self) -> str:
        return "blocker"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.INPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        return GuardResult(allowed=False, reason="Blocked by test guard")


class _OutputSanitizer(Guard):
    @property
    def name(self) -> str:
        return "sanitizer"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.OUTPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        sanitized = content.replace("secret", "[REDACTED]")
        return GuardResult(allowed=True, sanitized=sanitized if sanitized != content else "")


# --- Fixtures ---

@pytest.fixture
def native_model() -> _NativeModel:
    return _NativeModel()


@pytest.fixture
def prompt_model() -> _PromptModel:
    return _PromptModel()


@pytest.fixture
def tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


# --- Test: Simple mode (no tools, no retriever) ---

class TestSimpleMode:
    """With no tools or retriever, the agent behaves like SimpleStrategy."""

    async def test_simple_generate(self, native_model):
        native_model._generate.return_value = "Hello from simple mode"
        strategy = UnifiedAgentStrategy(model=native_model)
        result = await strategy.execute("hello")
        assert result == "Hello from simple mode"

    async def test_simple_with_prompt_model(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model)
        result = await strategy.execute("hello")
        assert result == "prompt response"

    async def test_simple_stream(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model)
        tokens = []
        async for token in strategy.execute_stream("hello"):
            tokens.append(token)
        assert "prompt response" in "".join(str(t) for t in tokens)


# --- Test: Tool calling mode ---

class TestToolCallingMode:
    """With tools injected, the agent uses native or prompt-based tool calling."""

    async def test_native_tool_calling(self, native_model, tools):
        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="Let me echo.",
                tool_calls=[ToolCall(id="tc1", name="echo", args={"text": "hi"})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="Echo said: hi", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = UnifiedAgentStrategy(model=native_model, tools=tools)
        result = await strategy.execute("echo hi")
        assert "Echo said: hi" in result

    async def test_prompt_tool_calling(self, prompt_model, tools):
        # First call returns a tool_call block, second call returns final response
        prompt_model._generate.side_effect = [
            '```tool_call\n{"name": "echo", "args": {"text": "test"}}\n```',
            "The echo returned: Echo: test",
        ]
        strategy = UnifiedAgentStrategy(model=prompt_model, tools=tools)
        result = await strategy.execute("echo test")
        assert "echo" in result.lower() or "Echo" in result

    async def test_native_tool_stream_yields_events(self, native_model, tools):
        native_model._generate_with_tools.side_effect = [
            ModelResponse(
                content="Calling tool.",
                tool_calls=[ToolCall(id="tc1", name="echo", args={"text": "stream"})],
                stop_reason="tool_use",
            ),
            ModelResponse(content="Done.", tool_calls=[], stop_reason="end_turn"),
        ]
        strategy = UnifiedAgentStrategy(model=native_model, tools=tools)
        events = []
        async for token in strategy.execute_stream("echo stream"):
            events.append(token)

        event_types = [type(e) for e in events if not isinstance(e, str)]
        assert ToolStartEvent in event_types
        assert ToolEndEvent in event_types


# --- Test: KB search tool mode ---

class TestKBSearchToolMode:
    """KB search tools are registered in the ToolRegistry — agent decides when to search."""

    async def test_kb_tool_registered(self, native_model):
        kb_tool = _MockKBSearchTool(
            name="search_test_kb",
            results=[{"filename": "python_guide.pdf", "content": "Python was created by Guido"}],
        )
        registry = ToolRegistry()
        registry.register(kb_tool)

        strategy = UnifiedAgentStrategy(model=native_model, tools=registry)
        tool_names = strategy.tools.names()
        assert "search_test_kb" in tool_names

    async def test_no_kb_tools_without_kbs(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model)
        tool_names = strategy.tools.names()
        assert not any(n.startswith("search_") for n in tool_names)

    async def test_kb_tool_returns_sources_in_props(self, native_model):
        kb_tool = _MockKBSearchTool(
            name="search_docs",
            results=[{"filename": "guide.pdf", "content": "Important info"}],
        )
        result = await kb_tool.execute(query="test")
        assert result.success
        assert result.props.get("sources")
        assert result.props["sources"][0]["filename"] == "guide.pdf"


# --- Test: Planner mode ---

class TestPlannerMode:
    """With planner injected, complex tasks get decomposed into steps."""

    async def test_complex_task_creates_plan(self, prompt_model):
        # Planner model response (JSON array of steps)
        prompt_model._generate.side_effect = [
            json.dumps(["Analyze the codebase", "Write new tests"]),  # Plan creation
            "Codebase analyzed: 5 modules found",  # Step 1 execution
            "Tests written: 3 test files",  # Step 2 execution
            "Final answer: Analysis complete, 3 test files added.",  # Synthesis
        ]
        planner = Planner(model=prompt_model)
        strategy = UnifiedAgentStrategy(
            model=prompt_model,
            planner=planner,
        )
        result = await strategy.execute(
            "Analyze the codebase structure and then write new tests and also update the CI configuration"
        )
        # Should have called generate multiple times (plan + steps + synthesis)
        assert prompt_model._generate.await_count >= 3

    async def test_simple_question_skips_planning(self, prompt_model):
        planner = Planner(model=prompt_model)
        prompt_model._generate.return_value = "42"
        strategy = UnifiedAgentStrategy(model=prompt_model, planner=planner)
        result = await strategy.execute("What is 6 * 7?")

        # Simple question: generate called once (no planning)
        assert result == "42"
        assert prompt_model._generate.await_count == 1

    async def test_plan_stream_yields_plan_events(self, prompt_model):
        # generate_stream also calls _generate internally, so each step's
        # _execute_step calls generate (1 call each), and the synthesis
        # calls generate_stream which also calls _generate (1 call).
        prompt_model._generate.side_effect = [
            json.dumps(["Step A", "Step B"]),  # Plan creation
            "Result A",  # Step 1 execution
            "Result B",  # Step 2 execution
            "Summary of results",  # Synthesis via generate_stream
        ]
        planner = Planner(model=prompt_model)
        strategy = UnifiedAgentStrategy(model=prompt_model, planner=planner)

        events = []
        async for token in strategy.execute_stream(
            "Create the new authentication module and then write comprehensive integration tests and also deploy the updated service to staging"
        ):
            events.append(token)

        plan_events = [e for e in events if isinstance(e, PlanEvent)]
        # Should have: 1 initial "created" + 2 steps "in_progress" + 2 steps "completed" = 5
        assert len(plan_events) >= 3


# --- Test: Memory mode ---

class TestMemoryMode:
    """With memory injected, the agent recalls and stores memories."""

    async def test_memory_context_injected(self, prompt_model):
        storage = InMemoryStorage()
        memory = MemoryManager(storage=storage)
        await memory.remember("User prefers dark mode theme", category="preference")

        strategy = UnifiedAgentStrategy(model=prompt_model, memory=memory)
        # Query shares the word "dark" with the memory
        await strategy.execute("What dark theme should I use?")

        # Use first call (main generate), not last (memory extraction)
        call_args = prompt_model._generate.call_args_list[0][0][0]
        system_prompt = call_args[0]["content"]
        assert "dark mode" in system_prompt

    async def test_no_memory_injection_when_no_match(self, prompt_model):
        storage = InMemoryStorage()
        memory = MemoryManager(storage=storage)
        await memory.remember("cats are nice", category="fact")

        strategy = UnifiedAgentStrategy(
            model=prompt_model,
            memory=memory,
            system_prompt="You are helpful.",
        )
        await strategy.execute("quantum physics")

        # Use first call (main generate), not last (memory extraction)
        call_args = prompt_model._generate.call_args_list[0][0][0]
        system_prompt = call_args[0]["content"]
        # No match, so system prompt should be clean
        assert "cats" not in system_prompt

    async def test_no_memory_when_none(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model, memory=None)
        await strategy.execute("hello")
        # Should not crash
        assert prompt_model._generate.await_count == 1


# --- Test: Guards ---

class TestGuardIntegration:
    async def test_input_guard_blocks(self, native_model):
        pipeline = GuardPipeline(guards=[_BlockingGuard()])
        strategy = UnifiedAgentStrategy(model=native_model, guard_pipeline=pipeline)
        result = await strategy.execute("anything")
        assert "[Blocked]" in result

    async def test_output_guard_sanitizes(self, prompt_model):
        prompt_model._generate.return_value = "The secret code is 42"
        pipeline = GuardPipeline(guards=[_OutputSanitizer()])
        strategy = UnifiedAgentStrategy(model=prompt_model, guard_pipeline=pipeline)
        result = await strategy.execute("tell me the code")
        assert "[REDACTED]" in result
        assert "secret" not in result

    async def test_stream_input_guard_blocks(self, native_model):
        pipeline = GuardPipeline(guards=[_BlockingGuard()])
        strategy = UnifiedAgentStrategy(model=native_model, guard_pipeline=pipeline)
        tokens = []
        async for token in strategy.execute_stream("anything"):
            tokens.append(token)
        assert "[Blocked]" in "".join(str(t) for t in tokens)

    async def test_no_guards_passes_through(self, prompt_model):
        prompt_model._generate.return_value = "raw output"
        strategy = UnifiedAgentStrategy(model=prompt_model, guard_pipeline=None)
        result = await strategy.execute("hello")
        assert result == "raw output"


# --- Test: Event emission ---

class TestEventEmission:
    async def test_events_emitted(self, prompt_model, event_bus):
        strategy = UnifiedAgentStrategy(model=prompt_model, event_bus=event_bus)
        await strategy.execute("hello")

        event_types = [e.type for e in event_bus.history]
        assert EventType.MESSAGE_RECEIVED in event_types
        assert EventType.MESSAGE_RESPONSE in event_types

    async def test_no_event_bus_does_not_crash(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model, event_bus=None)
        result = await strategy.execute("hello")
        assert result == "prompt response"


# --- Test: Factory registration ---

class TestFactoryRegistration:
    def test_agent_in_registry(self):
        from corail.strategies.factory import StrategyFactory
        assert "agent-react" in StrategyFactory.available()

    def test_agent_create(self, prompt_model):
        from corail.strategies.factory import StrategyFactory
        strategy = StrategyFactory.create("agent-react", model=prompt_model)
        assert isinstance(strategy, UnifiedAgentStrategy)


# --- Test: Combined capabilities ---

class TestCombinedCapabilities:
    """Test multiple capabilities together."""

    async def test_kb_tools_plus_regular_tools(self, native_model, tools):
        kb_tool = _MockKBSearchTool(
            name="search_docs",
            results=[{"filename": "guide.pdf", "content": "Relevant context from KB"}],
        )
        tools.register(kb_tool)
        native_model._generate_with_tools.return_value = ModelResponse(
            content="Answer using KB and tools", tool_calls=[], stop_reason="end_turn",
        )
        strategy = UnifiedAgentStrategy(model=native_model, tools=tools)
        result = await strategy.execute("question with context")
        assert result == "Answer using KB and tools"
        assert "echo" in strategy.tools.names()
        assert "search_docs" in strategy.tools.names()

    async def test_memory_plus_kb_tools(self, prompt_model):
        storage = InMemoryStorage()
        memory = MemoryManager(storage=storage)
        await memory.remember("User prefers Python version 3", category="preference")

        kb_tool = _MockKBSearchTool(name="search_python_docs")
        registry = ToolRegistry()
        registry.register(kb_tool)

        strategy = UnifiedAgentStrategy(
            model=prompt_model, tools=registry, memory=memory,
        )
        # Query shares "Python" with memory entry
        await strategy.execute("What Python version should I use?")

        # Use first call (main generate), not last (memory extraction)
        call_args = prompt_model._generate.call_args_list[0][0][0]
        system_prompt = call_args[0]["content"]
        # Memory in system prompt
        assert "Python" in system_prompt
        # KB tool is registered but agent decides whether to call it
        assert "search_python_docs" in strategy.tools.names()
