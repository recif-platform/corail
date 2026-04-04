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
from corail.retrieval.base import RetrievalResult, Retriever
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


# --- Mock retriever ---

class _MockRetriever(Retriever):
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results or []

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self._results[:top_k]


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


# --- Test: RAG mode ---

class TestRAGMode:
    """With retriever injected, the agent augments prompts with retrieved context."""

    async def test_rag_augments_prompt(self, prompt_model):
        retriever = _MockRetriever(results=[
            RetrievalResult(content="Python was created by Guido", score=0.9, metadata={}),
        ])
        strategy = UnifiedAgentStrategy(model=prompt_model, retriever=retriever)
        await strategy.execute("Who created Python?")

        # The model should have been called with a prompt containing the context
        call_args = prompt_model._generate.call_args[0][0]
        messages_text = json.dumps(call_args)
        assert "Guido" in messages_text

    async def test_no_rag_without_retriever(self, prompt_model):
        strategy = UnifiedAgentStrategy(model=prompt_model, retriever=None)
        await strategy.execute("Who created Python?")

        call_args = prompt_model._generate.call_args[0][0]
        messages_text = json.dumps(call_args)
        assert "Guido" not in messages_text

    async def test_rag_disabled_via_use_rag_false(self, prompt_model):
        retriever = _MockRetriever(results=[
            RetrievalResult(content="Context data", score=0.9),
        ])
        strategy = UnifiedAgentStrategy(model=prompt_model, retriever=retriever)
        await strategy.execute("query", use_rag=False)

        call_args = prompt_model._generate.call_args[0][0]
        messages_text = json.dumps(call_args)
        assert "Context data" not in messages_text

    async def test_rag_filters_low_score(self, prompt_model):
        retriever = _MockRetriever(results=[
            RetrievalResult(content="Low quality", score=0.1),
        ])
        strategy = UnifiedAgentStrategy(model=prompt_model, retriever=retriever)
        await strategy.execute("query", min_score=0.5)

        call_args = prompt_model._generate.call_args[0][0]
        messages_text = json.dumps(call_args)
        assert "Low quality" not in messages_text


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

    async def test_rag_plus_tools(self, native_model, tools):
        retriever = _MockRetriever(results=[
            RetrievalResult(content="Relevant context from KB", score=0.95),
        ])
        native_model._generate_with_tools.return_value = ModelResponse(
            content="Answer using context and tools", tool_calls=[], stop_reason="end_turn",
        )
        strategy = UnifiedAgentStrategy(
            model=native_model, tools=tools, retriever=retriever,
        )
        result = await strategy.execute("question with context")
        assert result == "Answer using context and tools"

    async def test_memory_plus_rag(self, prompt_model):
        storage = InMemoryStorage()
        memory = MemoryManager(storage=storage)
        await memory.remember("User prefers Python version 3", category="preference")

        retriever = _MockRetriever(results=[
            RetrievalResult(content="Python 3.13 docs", score=0.9),
        ])

        strategy = UnifiedAgentStrategy(
            model=prompt_model, retriever=retriever, memory=memory,
        )
        # Query shares "Python" with memory entry
        await strategy.execute("What Python version should I use?")

        # Use first call (main generate), not last (memory extraction)
        call_args = prompt_model._generate.call_args_list[0][0][0]
        system_prompt = call_args[0]["content"]
        user_message = call_args[-1]["content"]
        # Memory in system prompt, RAG in user message
        assert "Python" in system_prompt
        assert "Python 3.13" in user_message
