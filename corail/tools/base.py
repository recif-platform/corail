"""Tool — base interface for all tool executors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """A single parameter in a tool's input schema."""

    name: str
    type: str  # string, integer, number, boolean, array, object
    description: str
    required: bool = True
    default: object = None


@dataclass
class ToolDefinition:
    """Describes a tool's interface — what the LLM sees to decide when to call it."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    risk_level: str = "safe"  # safe, confirm, blocked

    def to_prompt_schema(self) -> str:
        """Generate a description for injection into the LLM system prompt."""
        params = []
        for p in self.parameters:
            req = " (required)" if p.required else " (optional)"
            params.append(f"    - {p.name}: {p.type}{req} — {p.description}")
        params_str = "\n".join(params) if params else "    (no parameters)"
        return f"- **{self.name}**: {self.description}\n  Parameters:\n{params_str}"


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    error: str = ""
    render: str = "text"  # text, table, chart, json, code, react
    component: str = ""   # React component name (if render="react")
    props: dict = field(default_factory=dict)  # Props for the component


class ToolExecutor(ABC):
    """Base interface for tool execution. Implementations are pluggable via registry."""

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's schema definition."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the tool with the given arguments."""
        ...

    @property
    def name(self) -> str:
        return self.definition().name
