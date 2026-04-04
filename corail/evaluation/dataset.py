"""Golden dataset loaders — JSONL and JSON formats."""

import json

from corail.evaluation.base import EvalCase


def load_jsonl(path: str) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    input=data["input"],
                    expected_output=data.get("expected_output", ""),
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            )
    return cases


def load_json(path: str) -> list[EvalCase]:
    """Load evaluation cases from a JSON array file."""
    with open(path) as f:
        data = json.load(f)
    return [
        EvalCase(
            input=d["input"],
            expected_output=d.get("expected_output", ""),
            context=d.get("context", ""),
            metadata=d.get("metadata", {}),
        )
        for d in data
    ]
