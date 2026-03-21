"""Built-in guards — prompt injection, PII, secrets detection."""

import re

from corail.guards.base import Guard, GuardDirection, GuardResult


class PromptInjectionGuard(Guard):
    """Detects common prompt injection patterns in user input."""

    PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"override\s+(all\s+)?rules",
        r"act\s+as\s+(if\s+)?(you\s+are\s+)?a\s+",
        r"pretend\s+(you\s+are|to\s+be)\s+",
        r"jailbreak",
        r"DAN\s+mode",
    ]

    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    @property
    def name(self) -> str:
        return "prompt_injection"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.INPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        for pattern in self._compiled:
            match = pattern.search(content)
            if match:
                return GuardResult(
                    allowed=False,
                    reason=f"Prompt injection detected: '{match.group()}'",
                    details={"pattern": match.group(), "position": match.start()},
                )
        return GuardResult(allowed=True)


class PIIGuard(Guard):
    """Detects and optionally masks PII in output."""

    PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_fr": r"(?:(?:\+33|0)\s?[1-9])(?:[\s.-]?\d{2}){4}",
        "phone_intl": r"\+\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}[\s.-]?\d{0,4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "iban": r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{0,4}\b",
    }

    def __init__(self, block: bool = False, mask: bool = True) -> None:
        self._block = block
        self._mask = mask
        self._compiled = {k: re.compile(v) for k, v in self.PATTERNS.items()}

    @property
    def name(self) -> str:
        return "pii"

    @property
    def direction(self) -> GuardDirection:
        return GuardDirection.OUTPUT

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        found: dict[str, list[str]] = {}
        sanitized = content

        for pii_type, pattern in self._compiled.items():
            matches = pattern.findall(content)
            if matches:
                found[pii_type] = matches
                if self._mask:
                    sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)

        if found:
            if self._block:
                return GuardResult(
                    allowed=False,
                    reason=f"PII detected: {', '.join(found.keys())}",
                    details={"pii_types": list(found.keys())},
                )
            return GuardResult(
                allowed=True,
                sanitized=sanitized if self._mask else "",
                details={"pii_types": list(found.keys()), "masked": self._mask},
            )

        return GuardResult(allowed=True)


class SecretGuard(Guard):
    """Detects API keys, tokens, passwords in output."""

    PATTERNS = {
        "api_key": r"(?:api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "bearer_token": r"Bearer\s+[a-zA-Z0-9_\-.]{20,}",
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "github_token": r"gh[ps]_[a-zA-Z0-9]{36,}",
        "generic_secret": r"(?:password|secret|token|key)\s*[=:]\s*['\"]([^'\"]{8,})['\"]",
        "private_key": r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    }

    def __init__(self) -> None:
        self._compiled = {k: re.compile(v, re.IGNORECASE) for k, v in self.PATTERNS.items()}

    @property
    def name(self) -> str:
        return "secrets"

    async def check(self, content: str, direction: GuardDirection) -> GuardResult:
        for secret_type, pattern in self._compiled.items():
            if pattern.search(content):
                return GuardResult(
                    allowed=False,
                    reason=f"Secret/credential detected: {secret_type}",
                    details={"secret_type": secret_type},
                )
        return GuardResult(allowed=True)
