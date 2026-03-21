"""Base types and provider interface for the evaluation system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EvalCase:
    """A single evaluation test case."""

    input: str
    expected_output: str = ""
    context: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalScore:
    """Score for a single metric."""

    name: str
    value: float  # 0.0 - 1.0
    threshold: float = 0.0
    passed: bool = True
    details: str = ""


@dataclass
class EvalResult:
    """Result of evaluating one case."""

    case: EvalCase
    output: str
    scores: list[EvalScore] = field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0


@dataclass
class EvalRun:
    """A complete evaluation run."""

    id: str
    agent_id: str
    agent_version: str
    dataset_name: str
    results: list[EvalResult] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: str = "running"  # running, completed, failed
    metadata: dict = field(default_factory=dict)


class EvaluationProvider(ABC):
    """Pluggable evaluation backend."""

    @abstractmethod
    async def start_run(
        self,
        agent_id: str,
        agent_version: str,
        dataset_name: str,
        metadata: dict | None = None,
    ) -> str:
        """Start an evaluation run. Returns run ID."""
        ...

    @abstractmethod
    async def log_result(self, run_id: str, result: EvalResult) -> None:
        """Log a single evaluation result."""
        ...

    @abstractmethod
    async def complete_run(self, run_id: str, aggregate_scores: dict[str, float]) -> None:
        """Mark a run as complete with aggregate scores."""
        ...

    @abstractmethod
    async def get_run(self, run_id: str) -> EvalRun | None:
        """Get evaluation run details."""
        ...

    @abstractmethod
    async def list_runs(self, agent_id: str, limit: int = 20) -> list[EvalRun]:
        """List recent evaluation runs for an agent."""
        ...

    @abstractmethod
    async def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """Compare two evaluation runs. Returns comparison metrics."""
        ...
