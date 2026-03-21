"""Skill — a mini-package of metadata, instructions, and optional resources.

Aligned with the Anthropic skills format:
  SKILL.md (frontmatter + body) + scripts/ + references/ + assets/

Progressive disclosure:
  1. Metadata (~100 words) — always in context, triggers the skill
  2. SKILL.md body — full instructions, loaded when activated
  3. Resources — scripts, references, assets, loaded on demand
"""

from dataclasses import dataclass, field


@dataclass
class SkillDefinition:
    """A skill is a mini-package: metadata + instructions + optional resources.

    Attributes:
        name: Unique skill identifier (e.g. "agui-render", "code-review").
        description: Human-readable summary (~100 words). Always in context, used for triggering.
        instructions: Full SKILL.md body (markdown). Loaded when activated.
        category: Grouping for UI/filtering (general, rendering, analysis, writing).
        version: Semantic version string.
        author: Skill author identifier.
        compatibility: Required tools/dependencies for this skill to function.
        channel_filter: When non-empty, skill only activates for these channels.
                        Empty list means the skill applies to all channels.
        tools: Built-in tool names this skill auto-registers (e.g. ["calculator"]).
        scripts: Executable code bundled with the skill (name -> content).
        references: Documentation loaded on demand (name -> content).
        assets: Templates, images, fonts bundled with the skill (name -> content or path).
        source: Origin of the skill: "builtin", "custom", "github:org/repo/path".
    """

    name: str
    description: str
    instructions: str
    category: str = "general"
    version: str = "1.0.0"
    author: str = ""
    compatibility: list[str] = field(default_factory=list)
    channel_filter: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    scripts: dict[str, str] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)
    assets: dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def prompt_fragment(self) -> str:
        """Backward-compatible alias for instructions."""
        return self.instructions

    @prompt_fragment.setter
    def prompt_fragment(self, value: str) -> None:
        """Backward-compatible setter: writing to prompt_fragment sets instructions."""
        self.instructions = value
