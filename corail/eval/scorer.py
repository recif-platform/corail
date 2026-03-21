"""Scoring logic for evaluation results."""


def score_response(actual: str, expected: str) -> float:
    """Score a response against expected output. Returns 0-100."""
    if not expected:
        return 100.0  # No expectation = always pass

    actual_lower = actual.strip().lower()
    expected_lower = expected.strip().lower()

    if actual_lower == expected_lower:
        return 100.0

    # Partial match: check if expected is contained in actual
    if expected_lower in actual_lower:
        return 80.0

    # Basic word overlap scoring
    actual_words = set(actual_lower.split())
    expected_words = set(expected_lower.split())

    if not expected_words:
        return 100.0

    overlap = actual_words & expected_words
    return round((len(overlap) / len(expected_words)) * 100, 2)


def aggregate_scores(scores: list[float]) -> float:
    """Compute overall score from per-scenario scores."""
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)
