"""Ollama adapter — local LLM via Ollama REST API with streaming + thinking support."""

import json
import os
from collections.abc import AsyncIterator

import httpx

from corail.models.base import Model

# Non-streaming requests (memory extraction, suggestions, title generation…)
# feed the whole conversation to a potentially large local model on CPU/Metal,
# which can take several minutes on the first call after a cold start while
# Ollama loads the model into memory. Keep the default generous, but let ops
# shrink it via CORAIL_OLLAMA_TIMEOUT when running cloud models.
_DEFAULT_TIMEOUT = float(os.environ.get("CORAIL_OLLAMA_TIMEOUT", "300"))

# How long Ollama should keep each model loaded after a request. When a chat
# agent uses a heavy model (35B+) for chat and a light one (4B) for background
# tasks, Ollama's default 5m window causes one of them to be unloaded and
# reloaded on every interleaved call, which blocks the SSE stream for tens of
# seconds. A long keep_alive lets Ollama hold both models simultaneously as
# long as the host has enough RAM. Tune via CORAIL_OLLAMA_KEEP_ALIVE (Ollama
# duration syntax: "30m", "1h", "-1" for forever).
_KEEP_ALIVE = os.environ.get("CORAIL_OLLAMA_KEEP_ALIVE", "30m")


class OllamaModel(Model):
    """Connects to Ollama for LLM generation. Supports streaming and thinking mode."""

    def __init__(self, model_id: str = "qwen3.5:4b", base_url: str = "") -> None:
        self.model_id = model_id
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

    def _request_body(self, messages: list[dict[str, str]], *, stream: bool) -> dict:
        return {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "think": True,
            "keep_alive": _KEEP_ALIVE,
        }

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Ollama and return the full response."""
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=self._request_body(messages, stream=False),
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream tokens from Ollama. Thinking tokens are wrapped in <think>...</think>."""
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client, client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=self._request_body(messages, stream=True),
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
