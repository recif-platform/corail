"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Corail runtime settings. All configurable via CORAIL_* env vars."""

    model_config = SettingsConfigDict(env_prefix="CORAIL_")

    # Runtime
    channel: str = "rest"
    strategy: str = "agent-react"
    model_type: str = "stub"
    model_id: str = "stub-echo"
    system_prompt: str = "You are a helpful assistant."

    # Server (for HTTP-based channels)
    port: int = 8000
    control_port: int = 8001
    grpc_control_port: int = 9001  # gRPC ControlService (control_port + 1000)
    host: str = "0.0.0.0"  # noqa: S104

    # Environment
    env: str = "dev"
    log_level: str = "INFO"
    log_format: str = "json"

    # Storage (conversation persistence)
    storage: str = "memory"  # memory | postgresql | redis | s3
    database_url: str = ""

    # Memory (agent working memory)
    memory_backend: str = "in_memory"  # in_memory | pgvector

    # Web search
    search_backend: str = "ddgs"  # ddgs | searxng
    searxng_url: str = "http://localhost:8080"

    # Skills (JSON array of skill names)
    skills: str = ""  # JSON: ["agui-render", "code-review"]

    # Tools (JSON array from ConfigMap or env var)
    tools: str = ""  # JSON: [{"name":"...", "type":"http", "endpoint":"..."}]

    # Knowledge bases (JSON for RAG strategy)
    knowledge_bases: str = ""  # JSON: [{"type":"pgvector", "connection_url":"...", "kb_id":"..."}]

    # Suggestions (follow-up chips after agent responses)
    suggestions: str = ""  # JSON: ["What can you do?", "Show me examples"]
    suggestions_provider: str = "llm"  # static | llm

    # Récif control plane
    recif_grpc_addr: str = "localhost:50051"

    # Auth (optional — trusted headers from Istio)
    jwt_public_key: str = ""
