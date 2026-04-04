"""Evaluation system — pluggable providers for agent quality assessment."""

from corail.evaluation.base import EvalCase, EvalResult, EvalRun, EvalScore, EvaluationProvider
from corail.evaluation.factory import available_providers, create_provider
from corail.evaluation.runner import EvalRunner
from corail.evaluation.dataset import load_json, load_jsonl

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "EvalScore",
    "EvaluationProvider",
    "EvalRunner",
    "create_provider",
    "available_providers",
    "load_jsonl",
    "load_json",
]
