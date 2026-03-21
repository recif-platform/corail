"""Evaluation system — pluggable providers for agent quality assessment."""

from corail.evaluation.base import EvalCase, EvalResult, EvalRun, EvalScore, EvaluationProvider
from corail.evaluation.dataset import load_json, load_jsonl
from corail.evaluation.factory import available_providers, create_provider
from corail.evaluation.runner import EvalRunner

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "EvalRunner",
    "EvalScore",
    "EvaluationProvider",
    "available_providers",
    "create_provider",
    "load_json",
    "load_jsonl",
]
