"""SelfCorrector — LLM-driven recovery from failed plan steps."""

import logging

from corail.models.base import Model

logger = logging.getLogger(__name__)

_CORRECTION_PROMPT = """\
A step in our plan failed. Suggest an alternative approach.

Failed step: {failed_step}
Error: {error}
Available tools: {tools}

Respond with a single sentence describing an alternative action to achieve the same goal.
Alternative:"""


class SelfCorrector:
    """Suggests alternative approaches when a plan step fails."""

    def __init__(self, model: Model) -> None:
        self._model = model

    async def suggest_alternative(
        self, failed_step: str, error: str, available_tools: list[str]
    ) -> str:
        """Ask the LLM for an alternative approach to a failed step."""
        prompt = _CORRECTION_PROMPT.format(
            failed_step=failed_step,
            error=error,
            tools=", ".join(available_tools) if available_tools else "(none)",
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self._model.generate(messages=messages)
        return response.strip()
