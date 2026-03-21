"""Anthropic adapter — Claude models via Anthropic API with native tool_use support."""

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from corail.models.base import Model, ModelResponse, ToolCall

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


class AnthropicModel(Model):
    """Connects to Anthropic API for Claude LLM generation. Supports native tool calling."""

    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str = "") -> None:
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def supports_tool_use(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def _validate_api_key(self) -> None:
        if not self.api_key:
            msg = "ANTHROPIC_API_KEY not set"
            raise ValueError(msg)

    def _extract_system_and_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
        """Separate system message from chat messages, converting to Anthropic format."""
        system = ""
        chat_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        return system, chat_messages

    def _build_body(
        self,
        chat_messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        return body

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Anthropic and return the response."""
        self._validate_api_key()
        system, chat_messages = self._extract_system_and_messages(messages)
        body = self._build_body(chat_messages, system)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(_API_URL, headers=self._headers(), json=body)
            response.raise_for_status()
            data = response.json()
            return self._extract_text_from_content(data.get("content", []))

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response token by token from Anthropic."""
        self._validate_api_key()
        system, chat_messages = self._extract_system_and_messages(messages)
        body = self._build_body(chat_messages, system)
        body["stream"] = True

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                _API_URL,
                headers=self._headers(),
                json=body,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                data = json.loads(payload)
                event_type = data.get("type", "")
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    async def generate_with_tools(self, messages: list[dict], tools: list[dict], **kwargs: object) -> ModelResponse:
        """Generate with native tool calling via Anthropic API."""
        self._validate_api_key()
        system, chat_messages = self._extract_system_and_messages(messages)
        max_tokens = int(kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS))
        body = self._build_body(chat_messages, system, tools=tools, max_tokens=max_tokens)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_API_URL, headers=self._headers(), json=body)
            response.raise_for_status()
            data = response.json()
            return self._parse_tool_response(data)

    def _parse_tool_response(self, data: dict[str, Any]) -> ModelResponse:
        """Parse Anthropic API response into a ModelResponse with tool calls."""
        content_blocks = data.get("content", [])
        stop_reason = data.get("stop_reason", "end_turn")

        text = self._extract_text_from_content(content_blocks)
        tool_calls = [
            ToolCall(
                id=block["id"],
                name=block["name"],
                args=block.get("input", {}),
            )
            for block in content_blocks
            if block.get("type") == "tool_use"
        ]

        return ModelResponse(content=text, tool_calls=tool_calls, stop_reason=stop_reason)

    @staticmethod
    def _extract_text_from_content(content_blocks: list[dict[str, Any]]) -> str:
        """Extract concatenated text from Anthropic content blocks."""
        return "".join(block.get("text", "") for block in content_blocks if block.get("type") == "text")
