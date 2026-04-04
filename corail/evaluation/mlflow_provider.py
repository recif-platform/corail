"""MLflow evaluation provider — integrates with MLflow GenAI evaluation.

Requires: pip install mlflow
Uses MLflow tracking for experiment runs and MLflow.evaluate() for scoring.
"""

import logging

from corail.evaluation.base import EvalResult, EvalRun, EvaluationProvider

logger = logging.getLogger(__name__)


class MLflowProvider(EvaluationProvider):
    """MLflow-backed evaluation with experiment tracking and GenAI metrics."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "recif-agents",
        **kwargs: object,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._runs: dict[str, dict] = {}  # local cache
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            logger.info("MLflow provider initialized: %s", tracking_uri)
        except ImportError:
            logger.error("mlflow not installed. Install with: pip install mlflow")
            self._mlflow = None

    async def start_run(
        self,
        agent_id: str,
        agent_version: str,
        dataset_name: str,
        metadata: dict | None = None,
    ) -> str:
        if self._mlflow is None:
            raise RuntimeError("mlflow not installed")
        run = self._mlflow.start_run(
            run_name=f"{agent_id}-{agent_version}-{dataset_name}",
            tags={
                "agent_id": agent_id,
                "agent_version": agent_version,
                "dataset": dataset_name,
                "platform": "recif",
            },
        )
        run_id = run.info.run_id
        self._runs[run_id] = {
            "agent_id": agent_id,
            "agent_version": agent_version,
            "results": [],
        }
        self._mlflow.log_params(
            {
                "agent_id": agent_id,
                "agent_version": agent_version,
                "dataset": dataset_name,
                **(metadata or {}),
            }
        )
        return run_id

    async def log_result(self, run_id: str, result: EvalResult) -> None:
        if run_id in self._runs:
            self._runs[run_id]["results"].append(result)
            step = len(self._runs[run_id]["results"])
            for score in result.scores:
                self._mlflow.log_metric(f"eval_{score.name}", score.value, step=step)
            self._mlflow.log_metric("latency_ms", result.latency_ms, step=step)

    async def complete_run(self, run_id: str, aggregate_scores: dict[str, float]) -> None:
        if self._mlflow is None:
            return
        for name, value in aggregate_scores.items():
            self._mlflow.log_metric(f"avg_{name}", value)
        self._mlflow.log_metric(
            "total_cases",
            len(self._runs.get(run_id, {}).get("results", [])),
        )
        self._mlflow.end_run()

    async def get_run(self, run_id: str) -> EvalRun | None:
        if self._mlflow is None:
            return None
        try:
            run = self._mlflow.get_run(run_id)
            cached = self._runs.get(run_id, {})
            return EvalRun(
                id=run_id,
                agent_id=run.data.params.get("agent_id", ""),
                agent_version=run.data.params.get("agent_version", ""),
                dataset_name=run.data.params.get("dataset", ""),
                aggregate_scores={
                    k.replace("avg_", ""): v
                    for k, v in run.data.metrics.items()
                    if k.startswith("avg_")
                },
                status="completed" if run.info.status == "FINISHED" else "running",
                results=cached.get("results", []),
            )
        except Exception:
            return None

    async def list_runs(self, agent_id: str, limit: int = 20) -> list[EvalRun]:
        if self._mlflow is None:
            return []
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(self._tracking_uri)
            experiment = client.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                return []
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.agent_id = '{agent_id}'",
                max_results=limit,
                order_by=["start_time DESC"],
            )
            return [
                EvalRun(
                    id=r.info.run_id,
                    agent_id=r.data.params.get("agent_id", ""),
                    agent_version=r.data.params.get("agent_version", ""),
                    dataset_name=r.data.params.get("dataset", ""),
                    aggregate_scores={
                        k.replace("avg_", ""): v
                        for k, v in r.data.metrics.items()
                        if k.startswith("avg_")
                    },
                    status="completed" if r.info.status == "FINISHED" else "running",
                )
                for r in runs
            ]
        except Exception as e:
            logger.warning("Failed to list MLflow runs: %s", e)
            return []

    async def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        a = await self.get_run(run_id_a)
        b = await self.get_run(run_id_b)
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
