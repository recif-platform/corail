"""Regression detection between agent versions."""

from dataclasses import dataclass


@dataclass
class RegressionResult:
    """Result of comparing two evaluation runs."""

    has_regression: bool
    score_delta: float
    previous_score: float
    current_score: float
    degraded_scenarios: list[str]


def detect_regression(
    current_score: float,
    previous_score: float,
    current_details: list[dict[str, float]] | None = None,
    previous_details: list[dict[str, float]] | None = None,
) -> RegressionResult:
    """Compare scores and detect regression."""
    delta = current_score - previous_score
    degraded: list[str] = []

    if current_details and previous_details:
        for curr, prev in zip(current_details, previous_details, strict=False):
            curr_score = curr.get("score", 0)
            prev_score = prev.get("score", 0)
            if curr_score < prev_score:
                degraded.append(str(curr.get("scenario_id", "unknown")))

    return RegressionResult(
        has_regression=delta < 0,
        score_delta=round(delta, 2),
        previous_score=previous_score,
        current_score=current_score,
        degraded_scenarios=degraded,
    )
