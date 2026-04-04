"""ModelFactory — registry-based model resolution."""


from corail.models.base import Model

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
    def available() -> list[str]:
        """Return list of registered model types."""
        return sorted(_REGISTRY.keys())
