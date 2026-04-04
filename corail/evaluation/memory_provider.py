"""In-memory evaluation provider — for development and testing."""

import uuid
from datetime import UTC, datetime

from corail.evaluation.base import EvalResult, EvalRun, EvaluationProvider


class InMemoryProvider(EvaluationProvider):
    """Simple in-memory evaluation store. No persistence across restarts."""

    def __init__(self, **kwargs: object) -> None:
        self._runs: dict[str, EvalRun] = {}

    async def start_run(
        self,
        agent_id: str,
        agent_version: str,
        dataset_name: str,
        metadata: dict | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        self._runs[run_id] = EvalRun(
            id=run_id,
            agent_id=agent_id,
            agent_version=agent_version,
            dataset_name=dataset_name,
            metadata=metadata or {},
        )
        return run_id

    async def log_result(self, run_id: str, result: EvalResult) -> None:
        if run_id in self._runs:
            self._runs[run_id].results.append(result)

    async def complete_run(self, run_id: str, aggregate_scores: dict[str, float]) -> None:
        if run_id in self._runs:
            run = self._runs[run_id]
            run.aggregate_scores = aggregate_scores
            run.status = "completed"
            run.completed_at = datetime.now(UTC)

    async def get_run(self, run_id: str) -> EvalRun | None:
        return self._runs.get(run_id)

    async def list_runs(self, agent_id: str, limit: int = 20) -> list[EvalRun]:
        runs = [r for r in self._runs.values() if r.agent_id == agent_id]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    async def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        a = self._runs.get(run_id_a)
        b = self._runs.get(run_id_b)
        if not a or not b:
            return {"error": "Run not found"}
        comparison = {}
        all_metrics = set(a.aggregate_scores.keys()) | set(b.aggregate_scores.keys())
        for metric in all_metrics:
            va = a.aggregate_scores.get(metric, 0)
            vb = b.aggregate_scores.get(metric, 0)
            comparison[metric] = {
                "a": va,
                "b": vb,
                "diff": round(vb - va, 4),
                "winner": "b" if vb > va else "a" if va > vb else "tie",
            }
        return {"run_a": run_id_a, "run_b": run_id_b, "metrics": comparison}
