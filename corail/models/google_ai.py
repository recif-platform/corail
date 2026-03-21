"""Google AI Studio adapter — Gemini models via Google AI (Makersuite) API."""

import json
import os
from collections.abc import AsyncIterator

import httpx

from corail.models.base import Model

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GoogleAIModel(Model):
    """Connects to Google AI Studio (Makersuite) for Gemini LLM generation."""

    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: str = "") -> None:
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("GOOGLE_AI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    def _url(self, method: str) -> str:
        return f"{_API_BASE}/models/{self.model_id}:{method}?key={self.api_key}"

    def _validate_api_key(self) -> None:
        if not self.api_key:
            msg = "GOOGLE_AI_API_KEY (or GOOGLE_API_KEY) not set"
            raise ValueError(msg)

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Gemini format."""
        system = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return system, contents

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Google AI and return the response."""
        self._validate_api_key()
        system, contents = self._convert_messages(messages)

        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._url("generateContent"),
                headers={"Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response from Google AI."""
        self._validate_api_key()
        system, contents = self._convert_messages(messages)

        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                self._url("streamGenerateContent") + "&alt=sse",
                headers={"Content-Type": "application/json"},
                json=body,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        yield part["text"]
