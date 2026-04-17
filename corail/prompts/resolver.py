"""Resolve prompt references from MLflow Prompt Registry.

Supports three reference formats:
    "my-prompt"           → load latest version
    "my-prompt/3"         → load version 3
    "my-prompt@champion"  → load alias @champion

Falls back to inline text if MLflow is unavailable or the prompt is not found.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pattern: "name/version" or "name@alias" or just "name"
_REF_PATTERN = re.compile(r"^(?P<name>[^/@]+)(?:/(?P<version>\d+)|@(?P<alias>[a-zA-Z_][a-zA-Z0-9_-]*))?$")


def parse_prompt_ref(ref: str) -> tuple[str, int | None, str | None]:
    """Parse a prompt reference into (name, version, alias).

    Returns:
        (name, version, alias) — exactly one of version/alias is set, or both None (latest).
    """
    ref = ref.strip()
    m = _REF_PATTERN.match(ref)
    if not m:
        raise ValueError(f"Invalid prompt reference format: {ref!r}")
    name = m.group("name")
    version = int(m.group("version")) if m.group("version") else None
    alias = m.group("alias")
    return name, version, alias


def resolve_prompt(prompt_ref: str, fallback: str) -> str:
    """Load a prompt from MLflow Prompt Registry, or return fallback text.

    Args:
        prompt_ref: MLflow prompt reference (e.g., "my-prompt@champion").
                    Empty string skips MLflow and returns fallback.
        fallback: Inline system prompt text to use if MLflow is unavailable.

    Returns:
        The resolved prompt template text.
    """
    if not prompt_ref:
        return fallback

    try:
        import mlflow.genai

        name, version, alias = parse_prompt_ref(prompt_ref)

        if alias:
            prompt = mlflow.genai.load_prompt(name, alias=alias)
        elif version:
            prompt = mlflow.genai.load_prompt(name, version=version)
        else:
            prompt = mlflow.genai.load_prompt(name)

        template = prompt.template if hasattr(prompt, "template") else str(prompt)
        logger.info("Prompt loaded from MLflow: %s (ref=%s)", name, prompt_ref)
        return template

    except ImportError:
        logger.warning("mlflow not installed — using fallback prompt")
        return fallback
    except Exception as e:
        logger.warning("Failed to load prompt %r from MLflow: %s — using fallback", prompt_ref, e)
        return fallback
