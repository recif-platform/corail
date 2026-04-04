"""Tests for VertexAIModel — token caching, SA auth, error handling."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from corail.models.vertex import _TOKEN_REFRESH_MARGIN, VertexAIModel


class TestVertexAIModelInit:
    def test_defaults(self) -> None:
        m = VertexAIModel()
        assert m.model_id == "gemini-2.5-flash"
        assert m.location == "us-central1"
        assert m._cached_token == ""
        assert m._token_expiry == 0.0

    def test_custom_params(self) -> None:
        m = VertexAIModel("gemini-2.0-pro", project="my-proj", location="europe-west1")
        assert m.model_id == "gemini-2.0-pro"
        assert m.project == "my-proj"
        assert m.location == "europe-west1"

    def test_base_url(self) -> None:
        m = VertexAIModel("gemini-2.5-flash", project="p", location="eu-west1")
        assert m._base_url() == (
            "https://eu-west1-aiplatform.googleapis.com/v1"
            "/projects/p/locations/eu-west1"
            "/publishers/google/models/gemini-2.5-flash"
        )

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
        m = VertexAIModel()
        assert m.project == "env-proj"
        assert m.location == "asia-east1"


class TestConvertMessages:
    def test_system_extracted(self) -> None:
        m = VertexAIModel()
        system, contents = m._convert_messages([
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ])
        assert system == "Be helpful"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_mapped_to_model(self) -> None:
        m = VertexAIModel()
        _, contents = m._convert_messages([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        assert contents[1]["role"] == "model"


class TestBuildBody:
    def test_with_system(self) -> None:
        m = VertexAIModel()
        body = m._build_body([
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi"},
        ])
        assert "systemInstruction" in body
        assert body["systemInstruction"]["parts"][0]["text"] == "You are a bot"

    def test_without_system(self) -> None:
        m = VertexAIModel()
        body = m._build_body([{"role": "user", "content": "Hi"}])
        assert "systemInstruction" not in body


class TestTokenCaching:
    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        m = VertexAIModel(project="p")
        m._cached_token = "cached-tok"  # noqa: S105
        m._token_expiry = time.time() + 600

        token = await m._get_access_token()
        assert token == "cached-tok"  # noqa: S105

    @pytest.mark.asyncio
    async def test_cache_expired_triggers_fetch(self) -> None:
        m = VertexAIModel(project="p")
        m._cached_token = "old-tok"  # noqa: S105
        m._token_expiry = time.time() - 1  # expired

        with patch.object(m, "_fetch_access_token", new_callable=AsyncMock, return_value="new-tok"):
            token = await m._get_access_token()

        assert token == "new-tok"  # noqa: S105
        assert m._cached_token == "new-tok"  # noqa: S105
        assert m._token_expiry > time.time()

    @pytest.mark.asyncio
    async def test_cache_sets_expiry_with_margin(self) -> None:
        m = VertexAIModel(project="p")
        before = time.time()

        with patch.object(m, "_fetch_access_token", new_callable=AsyncMock, return_value="tok"):
            await m._get_access_token()

        expected_min = before + 3600 - _TOKEN_REFRESH_MARGIN
        assert m._token_expiry >= expected_min - 1


class TestTokenFromCredentials:
    @pytest.mark.asyncio
    async def test_authorized_user_flow(self) -> None:
        creds = {
            "type": "authorized_user",
            "refresh_token": "rt",
            "client_id": "cid",
            "client_secret": "csec",
        }
        m = VertexAIModel()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "user-tok"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            token = await m._token_from_credentials(creds)

        assert token == "user-tok"  # noqa: S105

    @pytest.mark.asyncio
    async def test_authorized_user_missing_fields(self) -> None:
        creds = {"type": "authorized_user", "refresh_token": "rt"}
        m = VertexAIModel()
        token = await m._token_from_credentials(creds)
        assert token is None

    @pytest.mark.asyncio
    async def test_service_account_missing_fields_raises(self) -> None:
        creds = {"type": "service_account", "client_email": "a@b.com"}
        m = VertexAIModel()
        with pytest.raises(ValueError, match="missing 'client_email' or 'private_key'"):
            await m._token_from_service_account(creds)


class TestFetchAccessToken:
    @pytest.mark.asyncio
    async def test_adc_file_raises_on_failure(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ADC file exists but token exchange fails, raise immediately."""
        cred_file = tmp_path / "bad-creds.json"
        cred_file.write_text(json.dumps({"type": "authorized_user"}))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred_file))

        m = VertexAIModel()
        with pytest.raises(ValueError, match="Failed to obtain token from credentials file"):
            await m._fetch_access_token()

    @pytest.mark.asyncio
    async def test_env_token_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "manual-tok")

        m = VertexAIModel()

        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("no metadata")):
            token = await m._fetch_access_token()

        assert token == "manual-tok"  # noqa: S105

    @pytest.mark.asyncio
    async def test_no_credentials_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)

        m = VertexAIModel()

        with (
            patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("no metadata")),
            pytest.raises(ValueError, match="No Vertex AI credentials found"),
        ):
            await m._fetch_access_token()

    @pytest.mark.asyncio
    async def test_project_auto_detected_from_sa(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cred_file = tmp_path / "sa.json"
        cred_file.write_text(json.dumps({
            "type": "service_account",
            "project_id": "auto-proj",
            "client_email": "sa@auto-proj.iam.gserviceaccount.com",
            "private_key": "fake-key",
        }))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred_file))

        m = VertexAIModel()
        # project_id is now auto-detected from credentials.json at init time
        assert m.project == "auto-proj"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "sa-tok"}

        with (
            patch("jwt.encode", return_value="signed"),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            token = await m._fetch_access_token()

        assert token == "sa-tok"  # noqa: S105


class TestGenerate:
    @pytest.mark.asyncio
    async def test_missing_project_raises(self) -> None:
        m = VertexAIModel(project="")
        # Token fetch must run first (it auto-detects project), then check
        with patch.object(m, "_get_access_token", new_callable=AsyncMock, return_value="tok"):
            with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT not set"):
                await m.generate([{"role": "user", "content": "Hi"}])
