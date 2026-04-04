"""SkillRegistry — manages the set of skills available to an agent."""

from corail.skills.base import SkillDefinition


class SkillRegistry:
    """Holds all skills and builds the combined prompt for injection.

    Progressive disclosure:
      - Level 1: description (always in context, triggers the skill)
      - Level 2: full instructions (loaded when skill is activated)
      - Level 3: scripts/references/assets (mentioned, loaded on demand)
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill by its name."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def build_prompt(self, channel: str = "") -> str:
        """Build combined system prompt from all active skills.

        Args:
            channel: Current channel (e.g. "rest", "cli"). Empty string matches all.

        Returns:
            Combined prompt string. Empty if no skills match.
        """
        parts: list[str] = []
        for skill in self._skills.values():
            if skill.channel_filter and channel and channel not in skill.channel_filter:
                continue
            if skill.channel_filter and not channel:
                # No channel specified but skill has a filter -- skip it
                continue

            section = f"\n## Skill: {skill.name}\n{skill.instructions}"
            if skill.scripts:
                section += f"\n\nAvailable scripts: {', '.join(skill.scripts.keys())}"
            if skill.references:
                section += f"\n\nReference docs: {', '.join(skill.references.keys())}"
            parts.append(section)
        return "\n".join(parts) if parts else ""

    def tool_names(self) -> list[str]:
        """Return all tool names required by registered skills."""
        names: list[str] = []
        for skill in self._skills.values():
            names.extend(skill.tools)
        return sorted(set(names))

    def names(self) -> list[str]:
        """Return sorted list of registered skill names."""
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)
