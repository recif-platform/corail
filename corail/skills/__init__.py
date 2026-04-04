"""Skills -- declarative agent capabilities via prompt injection.

Aligned with the Anthropic skills format: SKILL.md + scripts/ + references/ + assets/.
"""

from corail.skills.base import SkillDefinition
from corail.skills.factory import available_skills, create_skill, load_skill
from corail.skills.loader import load_from_directory, load_from_github, load_from_string
from corail.skills.registry import SkillRegistry

__all__ = [
    "SkillDefinition",
    "SkillRegistry",
    "available_skills",
    "create_skill",
    "load_skill",
    "load_from_directory",
    "load_from_github",
    "load_from_string",
]
