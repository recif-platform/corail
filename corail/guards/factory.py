"""GuardFactory — registry-based guard resolution."""

import importlib

from corail.guards.base import Guard

_REGISTRY: dict[str, tuple[str, str]] = {
    "prompt_injection": ("corail.guards.builtins", "PromptInjectionGuard"),
    "pii": ("corail.guards.builtins", "PIIGuard"),
    "secrets": ("corail.guards.builtins", "SecretGuard"),
}


def register_guard(name: str, module_path: str, class_name: str) -> None:
    _REGISTRY[name] = (module_path, class_name)


class GuardFactory:
    @staticmethod
    def create(name: str, **kwargs: object) -> Guard:
        entry = _REGISTRY.get(name)
        if entry is None:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(f"Unknown guard: {name}. Available: {available}")
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def available() -> list[str]:
        return sorted(_REGISTRY.keys())
