"""Vertex AI adapter — Gemini models via Google Cloud Vertex AI REST API."""

import json
import os
from collections.abc import AsyncIterator

import httpx

from corail.models.base import Model


class VertexAIModel(Model):
    """Connects to Vertex AI for Gemini LLM generation. Uses Application Default Credentials or API key."""

    def __init__(self, model_id: str = "gemini-2.5-flash", project: str = "", location: str = "") -> None:
        self.model_id = model_id
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")

    def _base_url(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1"
            f"/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{self.model_id}"
        )

    async def _get_access_token(self) -> str:
        """Get access token via ADC file, metadata server, or env var."""
        # Try Application Default Credentials file (local dev with gcloud)
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if adc_path and os.path.exists(adc_path):
            token = await self._token_from_adc(adc_path)
            if token:
                return token

        # Try metadata server (works in GCE/GKE/Cloud Run)
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                    headers={"Metadata-Flavor": "Google"},
                )
                if resp.status_code == 200:
                    return resp.json()["access_token"]
            except httpx.ConnectError:
                pass

        # Fall back to GOOGLE_ACCESS_TOKEN env var
        token = os.environ.get("GOOGLE_ACCESS_TOKEN", "")
        if token:
            return token

        msg = "No Vertex AI credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_ACCESS_TOKEN."
        raise ValueError(msg)

    async def _token_from_adc(self, adc_path: str) -> str | None:
        """Exchange ADC refresh token for an access token."""
        with open(adc_path) as f:
            creds = json.load(f)

        refresh_token = creds.get("refresh_token")
        client_id = creds.get("client_id")
        client_secret = creds.get("client_secret")
        if not all([refresh_token, client_id, client_secret]):
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
        return None

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Vertex AI format."""
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
        """Send messages to Vertex AI and return the response."""
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT not set"
            raise ValueError(msg)

        token = await self._get_access_token()
        system, contents = self._convert_messages(messages)

        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url()}:generateContent",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_stream(self, messages: list[dict[str, str]], **kwargs: object) -> AsyncIterator[str]:
        """Stream response from Vertex AI."""
        if not self.project:
            msg = "GOOGLE_CLOUD_PROJECT not set"
            raise ValueError(msg)

        token = await self._get_access_token()
        system, contents = self._convert_messages(messages)

        body: dict = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

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
