"""ModelFactory — registry-based model resolution."""

import logging
import os

from corail.models.base import Model

logger = logging.getLogger(__name__)

# Registry: model_type → (module_path, class_name, default_model_id)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "stub": ("corail.models.stub", "StubModel", "stub-echo"),
    "ollama": ("corail.models.ollama", "OllamaModel", "qwen3.5:35b"),
    "openai": ("corail.models.openai", "OpenAIModel", "gpt-4"),
    "anthropic": ("corail.models.anthropic", "AnthropicModel", "claude-sonnet-4-20250514"),
    "vertex-ai": ("corail.models.vertex", "VertexAIModel", "gemini-2.5-flash"),
    "bedrock": ("corail.models.bedrock", "BedrockModel", "anthropic.claude-sonnet-4-20250514-v1:0"),
    "google-ai": ("corail.models.google_ai", "GoogleAIModel", "gemini-2.5-flash"),
}


def register_model(model_type: str, module_path: str, class_name: str, default_id: str = "") -> None:
    """Register a new model type. Allows external plugins to add models."""
    _REGISTRY[model_type] = (module_path, class_name, default_id)


class ModelFactory:
    """Creates model instances via registry lookup. Lazy imports for minimal startup cost."""

    @staticmethod
    def create(model_type: str, model_id: str = "") -> Model:
        """Create a model by type. Uses registry for O(1) lookup."""
        entry = _REGISTRY.get(model_type)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown model type: {model_type}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name, default_id = entry
        resolved_id = model_id or default_id

        # Lazy import — only load the module when needed
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        return cls(model_id=resolved_id)

    @staticmethod
    def from_uri(uri: str) -> Model:
        """Create a model from a ``provider:model_id`` or ``provider:/model_id``
        string. Both forms are accepted so users can reuse the MLflow-style
        ``openai:/gpt-4o-mini`` notation already used by ``RECIF_JUDGE_MODEL``.

        Examples::

            ollama:qwen3.5:4b         → Ollama / qwen3.5:4b
            ollama:/qwen3.5:4b        → Ollama / qwen3.5:4b
            openai:/gpt-4o-mini       → OpenAI / gpt-4o-mini
            vertex-ai:gemini-2.5-flash → Vertex AI / gemini-2.5-flash
        """
        provider, sep, model_id = uri.partition(":")
        if not sep or not provider:
            msg = f"Invalid model URI {uri!r}, expected 'provider:model_id'"
            raise ValueError(msg)
        model_id = model_id.lstrip("/")
        if not model_id:
            msg = f"Invalid model URI {uri!r}, missing model id"
            raise ValueError(msg)
        return ModelFactory.create(provider, model_id)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered model types."""
        return sorted(_REGISTRY.keys())


def background_model(fallback: Model) -> Model:
    """Return the model to use for background tasks (memory extraction,
    suggestions, titles, etc.).

    Reads ``CORAIL_BACKGROUND_MODEL`` as a ``provider:model_id`` URI. When the
    variable is unset or parses to an error, returns ``fallback`` so existing
    deployments keep working with a single model. The point of the split is
    that the main chat model can be a slow/expensive 35B+ while background
    tasks run on a cheap small model — users on 4B laptops can leave this
    unset, users on 35B/70B should point it at a 4B or a cloud mini model.
    """
    uri = os.environ.get("CORAIL_BACKGROUND_MODEL", "").strip()
    if not uri:
        return fallback
    try:
        return ModelFactory.from_uri(uri)
    except Exception as exc:
        logger.warning(
            "CORAIL_BACKGROUND_MODEL=%r could not be resolved (%s), falling back to main chat model",
            uri,
            exc,
        )
        return fallback
