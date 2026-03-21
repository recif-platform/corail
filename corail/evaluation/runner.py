"""Evaluation runner — executes datasets against agent pipelines."""

import time
from abc import ABC, abstractmethod

from corail.core.pipeline import Pipeline
from corail.evaluation.base import EvalCase, EvalResult, EvalScore, EvaluationProvider


class Scorer(ABC):
    """Base class for evaluation scorers."""

    @abstractmethod
    def score(self, case: EvalCase, output: str, latency_ms: float) -> EvalScore:
        """Score a single evaluation case."""
        ...


class ExactMatchScorer(Scorer):
    """Scores 1.0 if output exactly matches expected output."""

    def score(self, case: EvalCase, output: str, latency_ms: float) -> EvalScore:
        if not case.expected_output:
            return EvalScore(name="exact_match", value=1.0, passed=True, details="No expected output")
        match = output.strip() == case.expected_output.strip()
        return EvalScore(name="exact_match", value=1.0 if match else 0.0, passed=match)


class ContainsScorer(Scorer):
    """Scores 1.0 if expected output is contained in output (case-insensitive)."""

    def score(self, case: EvalCase, output: str, latency_ms: float) -> EvalScore:
        if not case.expected_output:
            return EvalScore(name="contains", value=1.0, passed=True)
        contains = case.expected_output.strip().lower() in output.strip().lower()
        return EvalScore(name="contains", value=1.0 if contains else 0.0, passed=contains)


class LatencyScorer(Scorer):
    """Scores based on response latency relative to a maximum threshold."""

    def __init__(self, max_ms: float = 5000) -> None:
        self.max_ms = max_ms

    def score(self, case: EvalCase, output: str, latency_ms: float) -> EvalScore:
        passed = latency_ms <= self.max_ms
        value = max(0, 1.0 - (latency_ms / self.max_ms))
        return EvalScore(
            name="latency",
            value=round(value, 3),
            threshold=self.max_ms,
            passed=passed,
            details=f"{latency_ms:.0f}ms",
        )


class EvalRunner:
    """Runs evaluation datasets against agent pipelines."""

    def __init__(self, provider: EvaluationProvider) -> None:
        self._provider = provider

    async def run(
        self,
        pipeline: Pipeline,
        dataset: list[EvalCase],
        agent_id: str,
        agent_version: str,
        dataset_name: str = "default",
        scorers: list[Scorer] | None = None,
    ) -> str:
        """Run evaluation. Returns run ID."""
        run_id = await self._provider.start_run(agent_id, agent_version, dataset_name)

        if scorers is None:
            scorers = [ExactMatchScorer(), ContainsScorer(), LatencyScorer()]

        aggregate: dict[str, list[float]] = {}

        for case in dataset:
            t0 = time.monotonic()
            try:
                output = await pipeline.execute(case.input)
            except Exception as e:
                output = f"[Error] {e}"
            latency = (time.monotonic() - t0) * 1000

            scores = []
            for scorer in scorers:
                score = scorer.score(case, output, latency)
                scores.append(score)
                aggregate.setdefault(score.name, []).append(score.value)

            result = EvalResult(case=case, output=output, scores=scores, latency_ms=latency)
            await self._provider.log_result(run_id, result)

        final_scores = {k: sum(v) / len(v) for k, v in aggregate.items()}
        await self._provider.complete_run(run_id, final_scores)

        return run_id
