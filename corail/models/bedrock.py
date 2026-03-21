"""AWS Bedrock adapter — Claude/Titan/Llama models via Bedrock Converse API."""

import hashlib
import hmac
import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx

from corail.models.base import Model


class BedrockModel(Model):
    """Connects to AWS Bedrock for LLM generation via the Converse API."""

    def __init__(self, model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0", region: str = "") -> None:
        self.model_id = model_id
        self.region = region or os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        self._access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        self._secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        self._session_token = os.environ.get("AWS_SESSION_TOKEN", "")

    def _endpoint(self) -> str:
        return f"https://bedrock-runtime.{self.region}.amazonaws.com"

    def _sign_request(self, method: str, url: str, headers: dict, body: bytes) -> dict:
        """AWS Signature V4 signing."""
        if not self._access_key or not self._secret_key:
            msg = "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
            raise ValueError(msg)

        now = datetime.now(tz=UTC)
        datestamp = now.strftime("%Y%m%d")
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        service = "bedrock"

        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname
        canonical_uri = parsed.path

        signed_headers = "content-type;host;x-amz-date"
        header_vals = {"content-type": "application/json", "host": host, "x-amz-date": amz_date}
        if self._session_token:
            signed_headers += ";x-amz-security-token"
            header_vals["x-amz-security-token"] = self._session_token

        canonical_headers = "".join(f"{k}:{v}\n" for k, v in sorted(header_vals.items()))
        payload_hash = hashlib.sha256(body).hexdigest()
        canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

        scope = f"{datestamp}/{self.region}/{service}/aws4_request"
        string_to_sign = (
            f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"
        )

        def _sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode(), hashlib.sha256).digest()

        k_date = _sign(f"AWS4{self._secret_key}".encode(), datestamp)
        k_region = _sign(k_date, self.region)
        k_service = _sign(k_region, service)
        k_signing = _sign(k_service, "aws4_request")
        signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

        auth = f"AWS4-HMAC-SHA256 Credential={self._access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"

        result = {
            "Content-Type": "application/json",
            "x-amz-date": amz_date,
            "Authorization": auth,
        }
        if self._session_token:
            result["x-amz-security-token"] = self._session_token
        return result

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
        """Convert OpenAI-style messages to Bedrock Converse format."""
        system = []
        converse_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system.append({"text": msg["content"]})
            else:
                role = "assistant" if msg["role"] == "assistant" else "user"
                converse_messages.append({"role": role, "content": [{"text": msg["content"]}]})
        return system, converse_messages

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send messages to Bedrock Converse and return the response."""
        system, converse_messages = self._convert_messages(messages)
        url = f"{self._endpoint()}/model/{self.model_id}/converse"

        body = {"messages": converse_messages}
        if system:
            body["system"] = system

        body_bytes = json.dumps(body).encode()
        headers = self._sign_request("POST", url, {}, body_bytes)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, content=body_bytes)
            response.raise_for_status()
            data = response.json()
            return data["output"]["message"]["content"][0]["text"]

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response from Bedrock Converse."""
        system, converse_messages = self._convert_messages(messages)
        url = f"{self._endpoint()}/model/{self.model_id}/converse-stream"

        body = {"messages": converse_messages}
        if system:
            body["system"] = system

        body_bytes = json.dumps(body).encode()
        headers = self._sign_request("POST", url, {}, body_bytes)

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                url,
                headers=headers,
                content=body_bytes,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    delta = data.get("contentBlockDelta", {}).get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
                except json.JSONDecodeError:
                    continue
