"""Vertex AI adapter — Gemini models via Google Cloud Vertex AI REST API."""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from corail.models.base import Model, ModelResponse, ToolCall

logger = logging.getLogger(__name__)

# Safety margin: refresh token 5 minutes before actual expiry.
_TOKEN_REFRESH_MARGIN = 300


def _build_tool_use_name_map(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Walk messages to build ``{tool_use_id: tool_name}``.

    Corail's ``_tool_result_message`` carries only ``tool_use_id``, but
    Gemini's ``functionResponse`` needs the tool's *name*. We resolve the
    mapping by looking at the preceding assistant ``tool_use`` blocks.
    """
    mapping: dict[str, str] = {}
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id and name:
                    mapping[tool_id] = name
    return mapping


def _anthropic_to_gemini_tool(tool: dict) -> dict:
    """Convert an Anthropic-style tool schema to a Gemini functionDeclaration.

    Anthropic: ``{"name":..., "description":..., "input_schema": {"type":"object", ...}}``
    Gemini:    ``{"name":..., "description":..., "parameters": {"type":"object", ...}}``
    """
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
    }


def _parse_gemini_response(data: dict) -> ModelResponse:
    """Parse a Gemini generateContent response into a ``ModelResponse``.

    Gemini returns a single candidate whose ``content.parts`` is a mix of
    ``text`` and ``functionCall`` entries. We concatenate text into
    ``content`` and turn each functionCall into a Corail ``ToolCall``.
    The stop_reason is derived from the presence of tool calls so the
    agent loop's ``if response.stop_reason != "tool_use"`` check keeps
    working unchanged.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        logger.warning("Vertex generate_with_tools: no candidates: %s", data)
        return ModelResponse(content="", tool_calls=[], stop_reason="end_turn")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for idx, part in enumerate(parts):
        if "text" in part:
            text_chunks.append(part["text"])
            continue
        fc = part.get("functionCall")
        if fc:
            tool_calls.append(ToolCall(
                id=f"call_{idx}",
                name=fc.get("name", ""),
                args=fc.get("args") or {},
            ))

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return ModelResponse(
        content="".join(text_chunks),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
    )


def _extract_vertex_text(data: dict) -> str:
    """Extract text from a Vertex AI response, tolerating missing fields.

    Gemini may return a candidate without `parts` when `finishReason` is
    `MAX_TOKENS`, `SAFETY`, `RECITATION`, `OTHER`, etc. Return an empty
    string so the caller can decide how to handle an empty completion
    instead of crashing with a KeyError.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        logger.warning("Vertex response has no candidates: %s", data)
        return ""
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text_parts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    if not text_parts:
        finish = candidate.get("finishReason", "unknown")
        logger.warning("Vertex returned no text parts (finishReason=%s)", finish)
        return ""
    return "".join(text_parts)


class VertexAIModel(Model):
    """Connects to Vertex AI for Gemini LLM generation.

    Authentication priority:
    1. Service account key or ADC file (``GOOGLE_APPLICATION_CREDENTIALS``)
    2. GCP metadata server (GCE / GKE / Cloud Run)
    3. Explicit ``GOOGLE_ACCESS_TOKEN`` env var
    """

    def __init__(self, model_id: str = "gemini-2.5-flash", project: str = "", location: str = "") -> None:
        self.model_id = model_id
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._cached_token: str = ""
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()
        self._credentials: dict[str, Any] | None = None

        # Auto-detect project_id from credentials file at init time,
        # so GOOGLE_CLOUD_PROJECT env var is not required when a SA key is used.
        if not self.project:
            self.project = self._detect_project_from_credentials()

    @staticmethod
    def _detect_project_from_credentials() -> str:
        """Read project_id from the GOOGLE_APPLICATION_CREDENTIALS file, if available."""
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not adc_path or not os.path.exists(adc_path):
            return ""
        try:
            with open(adc_path) as f:
                creds = json.load(f)
            return creds.get("project_id", "")
        except (json.JSONDecodeError, OSError):
            return ""

    def _base_url(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1"
            f"/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{self.model_id}"
        )

    async def _get_access_token(self) -> str:
        """Return a valid access token, using the cache when possible."""
        if self._cached_token and time.time() < self._token_expiry:
            return self._cached_token

        async with self._token_lock:
            # Double-check after acquiring lock (another coroutine may have refreshed)
            if self._cached_token and time.time() < self._token_expiry:
                return self._cached_token

            token = await self._fetch_access_token()
            self._cached_token = token
            self._token_expiry = time.time() + 3600 - _TOKEN_REFRESH_MARGIN
            return token

    async def _fetch_access_token(self) -> str:
        """Fetch a fresh access token via ADC file, metadata server, or env var."""
        # 1. Application Default Credentials file (service account or user credentials)
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if adc_path and os.path.exists(adc_path):
            with open(adc_path) as f:
                creds = json.load(f)

            token = await self._token_from_credentials(creds)
            if token:
                if not self.project and creds.get("project_id"):
                    self.project = creds["project_id"]
                return token

            msg = f"Failed to obtain token from credentials file: {adc_path}"
            raise ValueError(msg)

        # 2. Metadata server (GCE/GKE/Cloud Run — no file needed)
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                if resp.status_code == 200:
                    return resp.json()["access_token"]
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

        # 3. Explicit token env var (short-lived, for quick tests)
        token = os.environ.get("GOOGLE_ACCESS_TOKEN", "")
        if token:
            return token

        msg = "No Vertex AI credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_ACCESS_TOKEN."
        raise ValueError(msg)

    async def _token_from_credentials(self, creds: dict[str, Any]) -> str | None:
        """Exchange credentials for an access token.

        Supports both credential types:
        - ``authorized_user`` — refresh-token flow (``gcloud auth application-default login``)
        - ``service_account`` — signed JWT assertion (service-account key file)
        """
        cred_type = creds.get("type", "")

        if cred_type == "service_account":
            return await self._token_from_service_account(creds)

        # Default: authorized_user (refresh token flow)
        refresh_token = creds.get("refresh_token")
        client_id = creds.get("client_id")
        client_secret = creds.get("client_secret")
        if not (refresh_token and client_id and client_secret):
            return None

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
            )
            if resp.status_code == 200:
                return resp.json()["access_token"]

            logger.error("token exchange failed (authorized_user): %s %s", resp.status_code, resp.text)
        return None

    async def _token_from_service_account(self, creds: dict[str, Any]) -> str | None:
        """Create a signed JWT and exchange it for an access token."""
        try:
            import jwt
        except ImportError:
            msg = "PyJWT[crypto] required for Vertex AI service account auth: pip install 'PyJWT[crypto]'"
            raise ImportError(msg) from None

        client_email = creds.get("client_email")
        private_key = creds.get("private_key")
        if not client_email or not private_key:
            msg = "Service account key file is missing 'client_email' or 'private_key' fields."
            raise ValueError(msg)

        now = int(time.time())
        payload = {
            "iss": client_email,
            "sub": client_email,
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
        }

        signed_jwt = jwt.encode(payload, private_key, algorithm="RS256")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": signed_jwt,
                },
            )
            if resp.status_code == 200:
                return resp.json()["access_token"]

            msg = f"GCP token exchange failed ({resp.status_code}): {resp.text}"
            logger.error(msg)
            raise ValueError(msg)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict]]:
        """Convert Corail-style messages to Vertex AI format.

        Handles three message shapes:
        - Plain ``content: str`` (user/system/assistant text)
        - Anthropic-style assistant with ``tool_use`` blocks (from
          ``_assistant_message_from_response`` in the strategy loop)
        - Anthropic-style user with a ``tool_result`` block (from
          ``_tool_result_message``)
        """
        tool_name_by_id = _build_tool_use_name_map(messages)
        system = ""
        contents: list[dict] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system = content
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(content, list):
                parts: list[dict] = []
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append({"text": block.get("text", "")})
                    elif btype == "tool_use":
                        parts.append({
                            "functionCall": {
                                "name": block.get("name", ""),
                                "args": block.get("input", {}) or {},
                            }
                        })
                    elif btype == "tool_result":
                        name = tool_name_by_id.get(block.get("tool_use_id", ""), "")
                        parts.append({
                            "functionResponse": {
                                "name": name,
                                "response": {"content": block.get("content", "")},
                            }
                        })
                if parts:
                    contents.append({"role": gemini_role, "parts": parts})
                continue

            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        return system, contents

    def _build_body(self, messages: list[dict[str, Any]]) -> dict:
        """Build the Vertex AI request body from Corail-style messages."""
        system, contents = self._convert_messages(messages)
        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @property
    def supports_tool_use(self) -> bool:
        return True

    async def generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict], **kwargs: object,
    ) -> ModelResponse:
        """Call Gemini with native function declarations and parse the response
        back into Corail's ``ModelResponse`` shape (text + tool_calls)."""
        token = await self._get_access_token()
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT not set. Set the env var or use a service account key file."
            raise ValueError(msg)

        body = self._build_body(messages)
        body["tools"] = [{"functionDeclarations": [_anthropic_to_gemini_tool(t) for t in tools]}]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url()}:generateContent",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()

        return _parse_gemini_response(data)

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Vertex AI and return the response."""
        token = await self._get_access_token()
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT not set. Set the env var or use a service account key file."
            raise ValueError(msg)
        body = self._build_body(messages)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url()}:generateContent",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return _extract_vertex_text(data)

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response from Vertex AI."""
        token = await self._get_access_token()
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT not set. Set the env var or use a service account key file."
            raise ValueError(msg)
        body = self._build_body(messages)

        async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
            "POST",
            f"{self._base_url()}:streamGenerateContent?alt=sse",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        yield part["text"]
