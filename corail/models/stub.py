"""StubModel — echoes input for testing without API keys."""

from corail.models.base import Model


class StubModel(Model):
    """Deterministic echo model. No API key needed."""

    def __init__(self, model_id: str = "stub-echo") -> None:
        self.model_id = model_id

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Return echo of the last user message."""
        last = messages[-1]["content"] if messages else ""
        return f"Echo: {last}"
