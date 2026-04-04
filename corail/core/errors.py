"""Core error types with structured error information."""

from typing import Any


class CoreError(Exception):
    """Base error for Corail core operations."""

    def __init__(self, message: str, code: str = "CORE_ERROR", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class AdapterNotFoundError(CoreError):
    """Raised when a requested adapter (framework or LLM) is not registered."""

    def __init__(self, adapter_type: str, name: str) -> None:
        super().__init__(
            message=f"{adapter_type} adapter '{name}' not found",
            code="ADAPTER_NOT_FOUND",
            details={"adapter_type": adapter_type, "name": name},
        )


class ConfigValidationError(CoreError):
    """Raised when agent configuration fails validation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="CONFIG_VALIDATION_ERROR", details=details)


class LLMError(CoreError):
    """Raised when an LLM provider returns an error."""

    def __init__(self, message: str, code: str = "LLM_ERROR", details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code=code, details=details)


class ExecutionError(CoreError):
    """Raised when agent execution fails."""

    def __init__(self, message: str, code: str = "EXECUTION_ERROR", details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code=code, details=details)
