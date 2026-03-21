"""Tests for the pluggable evaluation system."""

import json
from pathlib import Path

import pytest

from corail.evaluation.base import EvalCase, EvalResult, EvalRun, EvalScore
from corail.evaluation.dataset import load_jsonl
from corail.evaluation.factory import available_providers, create_provider
from corail.evaluation.memory_provider import InMemoryProvider
from corail.evaluation.runner import ContainsScorer, ExactMatchScorer, LatencyScorer

# ── Dataclass creation ──────────────────────────────────────────────


class TestDataclasses:
    def test_eval_case(self):
        case = EvalCase(input="hello", expected_output="world")
        assert case.input == "hello"
        assert case.expected_output == "world"
        assert case.context == ""
        assert case.metadata == {}

    def test_eval_score(self):
        score = EvalScore(name="accuracy", value=0.95, passed=True)
        assert score.name == "accuracy"
        assert score.value == 0.95
        assert score.threshold == 0.0
        assert score.passed is True

    def test_eval_result(self):
        case = EvalCase(input="test")
        result = EvalResult(case=case, output="response", latency_ms=42.5)
        assert result.case is case
        assert result.output == "response"
        assert result.latency_ms == 42.5
        assert result.scores == []

    def test_eval_run(self):
        run = EvalRun(id="r1", agent_id="a1", agent_version="v1", dataset_name="ds1")
        assert run.id == "r1"
        assert run.status == "running"
        assert run.results == []
        assert run.aggregate_scores == {}
        assert run.completed_at is None


# ── InMemoryProvider ────────────────────────────────────────────────


class TestInMemoryProvider:
    @pytest.fixture
    def provider(self):
        return InMemoryProvider()

    @pytest.mark.asyncio
    async def test_start_run(self, provider):
        run_id = await provider.start_run("agent-1", "v1.0", "test-dataset")
        assert isinstance(run_id, str)
        assert len(run_id) == 8

    @pytest.mark.asyncio
    async def test_log_result(self, provider):
        run_id = await provider.start_run("agent-1", "v1.0", "ds")
        case = EvalCase(input="hi")
        result = EvalResult(case=case, output="hello", latency_ms=10.0)
        await provider.log_result(run_id, result)
        run = await provider.get_run(run_id)
        assert run is not None
        assert len(run.results) == 1
        assert run.results[0].output == "hello"

    @pytest.mark.asyncio
    async def test_complete_run(self, provider):
        run_id = await provider.start_run("agent-1", "v1.0", "ds")
        await provider.complete_run(run_id, {"accuracy": 0.9, "latency": 0.8})
        run = await provider.get_run(run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.aggregate_scores["accuracy"] == 0.9
        assert run.completed_at is not None

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, provider):
        result = await provider.get_run("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_runs(self, provider):
        await provider.start_run("agent-1", "v1.0", "ds1")
        await provider.start_run("agent-1", "v1.1", "ds2")
        await provider.start_run("agent-2", "v1.0", "ds1")

        runs = await provider.list_runs("agent-1")
        assert len(runs) == 2
        assert all(r.agent_id == "agent-1" for r in runs)

    @pytest.mark.asyncio
    async def test_list_runs_limit(self, provider):
        for i in range(5):
            await provider.start_run("agent-1", f"v{i}", "ds")
        runs = await provider.list_runs("agent-1", limit=3)
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_compare_runs(self, provider):
        run_a = await provider.start_run("agent-1", "v1.0", "ds")
        run_b = await provider.start_run("agent-1", "v1.1", "ds")
        await provider.complete_run(run_a, {"accuracy": 0.8, "latency": 0.7})
        await provider.complete_run(run_b, {"accuracy": 0.9, "latency": 0.6})

        comparison = await provider.compare_runs(run_a, run_b)
        assert comparison["run_a"] == run_a
        assert comparison["run_b"] == run_b
        assert "metrics" in comparison
        assert comparison["metrics"]["accuracy"]["winner"] == "b"
        assert comparison["metrics"]["latency"]["winner"] == "a"

    @pytest.mark.asyncio
    async def test_compare_runs_not_found(self, provider):
        result = await provider.compare_runs("x", "y")
        assert "error" in result


# ── Factory ─────────────────────────────────────────────────────────


class TestFactory:
    def test_create_memory_provider(self):
        provider = create_provider("memory")
        assert isinstance(provider, InMemoryProvider)

    def test_available_providers(self):
        providers = available_providers()
        assert "memory" in providers
        assert "mlflow" in providers
        assert "phoenix" in providers

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown evaluation provider"):
            create_provider("nonexistent")


# ── Scorers ─────────────────────────────────────────────────────────


class TestScorers:
    def test_exact_match_pass(self):
        scorer = ExactMatchScorer()
        case = EvalCase(input="q", expected_output="hello")
        score = scorer.score(case, "hello", 100.0)
        assert score.name == "exact_match"
        assert score.value == 1.0
        assert score.passed is True

    def test_exact_match_fail(self):
        scorer = ExactMatchScorer()
        case = EvalCase(input="q", expected_output="hello")
        score = scorer.score(case, "world", 100.0)
        assert score.value == 0.0
        assert score.passed is False

    def test_exact_match_no_expected(self):
        scorer = ExactMatchScorer()
        case = EvalCase(input="q", expected_output="")
        score = scorer.score(case, "anything", 100.0)
        assert score.value == 1.0
        assert score.passed is True

    def test_exact_match_strips_whitespace(self):
        scorer = ExactMatchScorer()
        case = EvalCase(input="q", expected_output="hello")
        score = scorer.score(case, "  hello  ", 100.0)
        assert score.value == 1.0

    def test_contains_pass(self):
        scorer = ContainsScorer()
        case = EvalCase(input="q", expected_output="Paris")
        score = scorer.score(case, "The capital of France is Paris.", 100.0)
        assert score.name == "contains"
        assert score.value == 1.0
        assert score.passed is True

    def test_contains_case_insensitive(self):
        scorer = ContainsScorer()
        case = EvalCase(input="q", expected_output="PARIS")
        score = scorer.score(case, "paris is beautiful", 100.0)
        assert score.value == 1.0

    def test_contains_fail(self):
        scorer = ContainsScorer()
        case = EvalCase(input="q", expected_output="London")
        score = scorer.score(case, "Paris is a city", 100.0)
        assert score.value == 0.0
        assert score.passed is False

    def test_contains_no_expected(self):
        scorer = ContainsScorer()
        case = EvalCase(input="q", expected_output="")
        score = scorer.score(case, "anything", 100.0)
        assert score.value == 1.0

    def test_latency_fast(self):
        scorer = LatencyScorer(max_ms=1000)
        case = EvalCase(input="q")
        score = scorer.score(case, "out", 200.0)
        assert score.name == "latency"
        assert score.value == 0.8
        assert score.passed is True

    def test_latency_slow(self):
        scorer = LatencyScorer(max_ms=1000)
        case = EvalCase(input="q")
        score = scorer.score(case, "out", 1500.0)
        assert score.passed is False
        assert score.value == 0.0  # clamped to 0

    def test_latency_at_threshold(self):
        scorer = LatencyScorer(max_ms=1000)
        case = EvalCase(input="q")
        score = scorer.score(case, "out", 1000.0)
        assert score.passed is True
        assert score.value == 0.0


# ── Dataset loader ──────────────────────────────────────────────────


class TestDatasetLoader:
    def test_load_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"input": "What is 2+2?", "expected_output": "4"}),
            json.dumps({"input": "Capital of France?", "expected_output": "Paris", "context": "geography"}),
            json.dumps({"input": "Open question", "metadata": {"difficulty": "hard"}}),
        ]
        jsonl_file.write_text("\n".join(lines) + "\n")

        cases = load_jsonl(str(jsonl_file))
        assert len(cases) == 3
        assert cases[0].input == "What is 2+2?"
        assert cases[0].expected_output == "4"
        assert cases[1].context == "geography"
        assert cases[2].expected_output == ""
        assert cases[2].metadata == {"difficulty": "hard"}

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        content = '{"input": "hello"}\n\n{"input": "world"}\n'
        jsonl_file.write_text(content)

        cases = load_jsonl(str(jsonl_file))
        assert len(cases) == 2

    def test_load_golden_example(self):
        """Load the actual golden example file."""
        path = Path(__file__).parent.parent.parent / "eval" / "golden-example.jsonl"
        if path.exists():
            cases = load_jsonl(str(path))
            assert len(cases) == 5
            assert cases[0].input == "What is 2 + 2?"
            assert cases[0].expected_output == "4"
