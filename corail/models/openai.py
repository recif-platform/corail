"""OpenAI adapter — GPT models via OpenAI API."""

import os

import httpx

from corail.models.base import Model


class OpenAIModel(Model):
    """Connects to OpenAI API for LLM generation."""

    def __init__(self, model_id: str = "gpt-4", api_key: str = "") -> None:
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to OpenAI and return the response."""
        if not self.api_key:
            msg = "OPENAI_API_KEY not set"
            raise ValueError(msg)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_id,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
