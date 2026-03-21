"""Guard — pluggable input/output security interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class GuardDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


@dataclass
class GuardResult:
    """Result of a guard check."""

    allowed: bool
    reason: str = ""
    sanitized: str = ""  # If not empty, use this instead of original content
    guard_name: str = ""
    details: dict = field(default_factory=dict)


class Guard(ABC):
    """Abstract guard. Checks content for security/policy violations.

    Guards are pluggable via registry pattern.
    Each guard declares which direction it operates on (input, output, or both).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.BOTH

    @abstractmethod
    async def check(self, content: str, direction: GuardDirection) -> GuardResult: ...
