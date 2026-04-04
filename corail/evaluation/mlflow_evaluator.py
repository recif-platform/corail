"""MLflow GenAI evaluation engine — real LLM-judge scoring for agent pipelines.

Uses mlflow.genai.evaluate() with a registry-based scorer resolution pattern.
Scorers are resolved by name from a registry dict, never via if/elif chains.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corail.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scorer registry — maps scorer names to (module_path, class_name)
# ---------------------------------------------------------------------------

_SCORER_REGISTRY: dict[str, tuple[str, str]] = {
    # Response quality
    "safety": ("mlflow.genai.scorers", "Safety"),
    "relevance_to_query": ("mlflow.genai.scorers", "RelevanceToQuery"),
    "correctness": ("mlflow.genai.scorers", "Correctness"),
    "completeness": ("mlflow.genai.scorers", "Completeness"),
    "fluency": ("mlflow.genai.scorers", "Fluency"),
    "equivalence": ("mlflow.genai.scorers", "Equivalence"),
    "summarization": ("mlflow.genai.scorers", "Summarization"),
    "guidelines": ("mlflow.genai.scorers", "Guidelines"),
    "expectations_guidelines": ("mlflow.genai.scorers", "ExpectationsGuidelines"),
    # RAG
    "retrieval_relevance": ("mlflow.genai.scorers", "RetrievalRelevance"),
    "retrieval_groundedness": ("mlflow.genai.scorers", "RetrievalGroundedness"),
    "retrieval_sufficiency": ("mlflow.genai.scorers", "RetrievalSufficiency"),
    # Tool calls
    "tool_call_correctness": ("mlflow.genai.scorers", "ToolCallCorrectness"),
    "tool_call_efficiency": ("mlflow.genai.scorers", "ToolCallEfficiency"),
}


def register_scorer(name: str, module_path: str, class_name: str) -> None:
    """Register a custom scorer for use in evaluations."""
    _SCORER_REGISTRY[name] = (module_path, class_name)


def available_scorers() -> list[str]:
    """Return sorted list of registered scorer names."""
    return sorted(_SCORER_REGISTRY.keys())


def resolve_scorer(name: str, config: dict[str, Any] | None = None) -> Any:
    """Resolve a scorer by name from the registry. Returns an instantiated scorer."""
    entry = _SCORER_REGISTRY.get(name)
    if entry is None:
        available = ", ".join(available_scorers())
        raise ValueError(f"Unknown scorer: {name}. Available: {available}")
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**(config or {}))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalRunResult:
    """Result of an evaluation run."""

    run_id: str
    status: str  # completed, failed
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    verdict: str = ""
    mlflow_run_id: str = ""
    per_case_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default scorer presets per risk profile
# ---------------------------------------------------------------------------

_RISK_PROFILE_SCORERS: dict[str, list[str]] = {
    "low": ["safety", "relevance_to_query"],
    "standard": ["safety", "relevance_to_query", "correctness"],
    "high": ["safety", "relevance_to_query", "correctness", "guidelines", "retrieval_groundedness", "tool_call_correctness"],
}


def scorers_for_risk_profile(profile: str) -> list[str]:
    """Return the default scorer list for a given risk profile."""
    return _RISK_PROFILE_SCORERS.get(profile, _RISK_PROFILE_SCORERS["standard"])


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MLflowEvaluator:
    """Runs mlflow.genai.evaluate() against a Corail pipeline.

    Resolves scorers from the registry, wraps the async pipeline into a sync
    predict_fn, and returns structured results.
    """

    def __init__(self, judge_model: str = "openai:/gpt-4o-mini") -> None:
        self._judge_model = judge_model

    async def evaluate(
        self,
        pipeline: Pipeline,
        dataset: list[dict[str, Any]],
        agent_id: str,
        agent_version: str,
        scorers_config: dict[str, dict[str, Any]] | None = None,
        min_quality_score: float = 0.0,
        risk_profile: str = "standard",
    ) -> EvalRunResult:
        """Run evaluation. Returns structured results with pass/fail verdict.

        Parameters
        ----------
        pipeline : Pipeline
            The agent pipeline to evaluate.
        dataset : list[dict]
            List of test cases: ``[{inputs: {question: ...}, expectations: {expected_response: ...}}]``
        agent_id : str
            Agent identifier for MLflow experiment.
        agent_version : str
            Agent version tag for the evaluation run.
        scorers_config : dict | None
            Scorer name → config dict. If None, uses risk_profile defaults.
        min_quality_score : float
            Minimum average score (0.0–1.0) to pass. 0.0 means always pass.
        risk_profile : str
            One of "low", "standard", "high". Used to select default scorers.
        """
        import mlflow

        # Resolve scorers — merge judge_model as default into each scorer's config
        scorer_names = list(scorers_config.keys()) if scorers_config else scorers_for_risk_profile(risk_profile)
        default_config = {"model": self._judge_model}
        scorers = [
            resolve_scorer(name, {**default_config, **(scorers_config or {}).get(name, {})})
            for name in scorer_names
        ]

        # Build predict_fn — sync wrapper around async pipeline
        predict_fn = _make_predict_fn(pipeline)

        # Normalize dataset format for MLflow
        eval_data = _normalize_dataset(dataset)

        # Set experiment
        mlflow.set_experiment(f"recif/agents/{agent_id}")

        # Run evaluation (blocking — must be called from a thread)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: mlflow.genai.evaluate(
                data=eval_data,
                predict_fn=predict_fn,
                scorers=scorers,
            ),
        )

        # Extract scores from results
        scores = _extract_scores(results)
        avg_score = sum(scores.values()) / max(len(scores), 1)
        passed = avg_score >= min_quality_score

        verdict = (
            f"PASSED (avg={avg_score:.3f} >= {min_quality_score:.3f})"
            if passed
            else f"REJECTED (avg={avg_score:.3f} < {min_quality_score:.3f})"
        )

        return EvalRunResult(
            run_id=getattr(results, "run_id", ""),
            status="completed",
            scores=scores,
            passed=passed,
            verdict=verdict,
            mlflow_run_id=getattr(results, "run_id", ""),
            per_case_results=_extract_per_case(results),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_predict_fn(pipeline: Pipeline):
    """Create a sync predict_fn for mlflow.genai.evaluate from an async pipeline.

    MLflow's evaluate() calls predict_fn synchronously from its own thread pool.
    We use a dedicated persistent event loop (like the gRPC servicer pattern)
    to avoid creating/destroying loops per call and to keep asyncpg pools alive.
    """
    import threading

    _eval_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    _eval_thread = threading.Thread(target=_eval_loop.run_forever, daemon=True)
    _eval_thread.start()

    def predict_fn(inputs: dict[str, Any]) -> str:
        # _normalize_dataset guarantees inputs.question exists
        question = str(inputs.get("question", ""))
        future = asyncio.run_coroutine_threadsafe(pipeline.execute(question), _eval_loop)
        return future.result(timeout=120)

    return predict_fn


def _normalize_dataset(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize dataset entries to MLflow's expected format.

    MLflow expects: ``{inputs: {question: ...}, expectations: {expected_response: ...}}``
    We accept both this format and the simpler ``{input: ..., expected_output: ...}``.
    """
    normalized = []
    for entry in dataset:
        # Already in MLflow format
        if "inputs" in entry:
            normalized.append(entry)
            continue
        # Convert from simple format
        normalized.append({
            "inputs": {"question": entry.get("input", "")},
            "expectations": {"expected_response": entry.get("expected_output", "")},
        })
    return normalized


def _extract_scores(results: Any) -> dict[str, float]:
    """Extract aggregate scores from MLflow EvaluationResult."""
    scores: dict[str, float] = {}
    metrics = getattr(results, "metrics", {})
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            scores[key] = round(float(value), 4)
    return scores


def _extract_per_case(results: Any) -> list[dict[str, Any]]:
    """Extract per-case results from MLflow EvaluationResult."""
    table = getattr(results, "tables", {})
    eval_results = table.get("eval_results", None)
    if eval_results is None:
        return []
    try:
        return eval_results.to_dict("records")
    except Exception:
        return []
