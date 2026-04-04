"""ChannelFactory — registry-based channel resolution."""

import importlib

from corail.channels.base import Channel
from corail.config import Settings
from corail.core.pipeline import Pipeline

# Registry: channel_name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "rest": ("corail.channels.rest", "RestChannel"),
    "discord": ("corail.channels.discord", "DiscordChannel"),
}


def register_channel(name: str, module_path: str, class_name: str) -> None:
    """Register a new channel. Allows external plugins to add channels."""
    _REGISTRY[name] = (module_path, class_name)


class ChannelFactory:
    """Creates channel instances via registry lookup."""

    @staticmethod
    def create(name: str, pipeline: Pipeline, settings: Settings) -> Channel:
        """Create a channel by name."""
        entry = _REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            msg = f"Unknown channel: {name}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(pipeline=pipeline, settings=settings)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered channel types."""
        return sorted(_REGISTRY.keys())
