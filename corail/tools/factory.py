"""ToolFactory — registry-based tool resolution."""

import importlib

from corail.tools.base import ToolExecutor

# Registry: tool_type → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "http": ("corail.tools.http_tool", "HTTPTool"),
    "cli": ("corail.tools.cli_tool", "CLIToolExecutor"),
}

# Built-in tools: name → (module_path, class_name)
_BUILTINS: dict[str, tuple[str, str]] = {
    "datetime": ("corail.tools.builtins", "DateTimeTool"),
    "calculator": ("corail.tools.builtins", "CalculatorTool"),
    "web_search": ("corail.tools.web_search", "WebSearchTool"),
    "fetch_url": ("corail.tools.fetch_url", "FetchURLTool"),
}


def register_tool(tool_type: str, module_path: str, class_name: str) -> None:
    """Register a new tool type. Allows external plugins to add tools."""
    _REGISTRY[tool_type] = (module_path, class_name)


def register_builtin(name: str, module_path: str, class_name: str) -> None:
    """Register a new built-in tool by name."""
    _BUILTINS[name] = (module_path, class_name)


class ToolFactory:
    """Creates tool instances via registry lookup. Lazy imports for minimal startup cost."""

    @staticmethod
    def create(tool_type: str, **kwargs: object) -> ToolExecutor:
        """Create a tool by type. For 'builtin' type, looks up by 'name' kwarg."""
        # Built-in tools: resolve by name
        if tool_type == "builtin":
            name = str(kwargs.get("name", "")).replace("-", "_")
            entry = _BUILTINS.get(name)
            if entry is None:
                available = ", ".join(sorted(_BUILTINS.keys()))
                msg = f"Unknown builtin tool: {name}. Available: {available}"
                raise ValueError(msg)
            module_path, class_name = entry
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls()

        # Other tool types: resolve by type
        entry = _REGISTRY.get(tool_type)
        if entry is None:
            available = ", ".join(sorted(list(_REGISTRY.keys()) + ["builtin"]))
            msg = f"Unknown tool type: {tool_type}. Available: {available}"
            raise ValueError(msg)

        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def available() -> list[str]:
        """Return list of registered tool types."""
        return sorted(list(_REGISTRY.keys()) + ["builtin"])

    @staticmethod
    def available_builtins() -> list[str]:
        """Return list of registered built-in tool names."""
        return sorted(_BUILTINS.keys())
