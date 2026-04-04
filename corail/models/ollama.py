"""Ollama adapter — local LLM via Ollama REST API with streaming + thinking support."""

import json
import os
from collections.abc import AsyncIterator

import httpx

from corail.models.base import Model


class OllamaModel(Model):
    """Connects to Ollama for LLM generation. Supports streaming and thinking mode."""

    def __init__(self, model_id: str = "qwen3.5:4b", base_url: str = "") -> None:
        self.model_id = model_id
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Ollama and return the full response."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model_id, "messages": messages, "stream": False, "think": True},
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream tokens from Ollama. Thinking tokens are wrapped in <think>...</think>."""
        async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={"model": self.model_id, "messages": messages, "stream": True, "think": True},
        ) as response:
            response.raise_for_status()
            in_thinking = False
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message", {})
                thinking = msg.get("thinking", "")
                content = msg.get("content", "")

                # Thinking phase
                if thinking:
                    if not in_thinking:
                        yield "<think>"
                        in_thinking = True
                    yield thinking

                # Content phase
                if content:
                    if in_thinking:
                        yield "</think>"
                        in_thinking = False
                    yield content

            # Close thinking tag if stream ended during thinking
            if in_thinking:
                yield "</think>"
