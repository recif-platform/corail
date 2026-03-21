"""Tests for the Skills system -- definitions, registry, factory, and loader."""

import pytest

from corail.skills.base import SkillDefinition
from corail.skills.factory import available_skills, create_skill, register_skill
from corail.skills.loader import load_from_string
from corail.skills.registry import SkillRegistry

# --- Test: SkillDefinition ---


class TestSkillDefinition:
    def test_basic_creation(self):
        skill = SkillDefinition(
            name="test-skill",
            description="A test skill",
            instructions="You are a test expert.",
        )
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.instructions == "You are a test expert."
        assert skill.category == "general"
        assert skill.version == "1.0.0"
        assert skill.author == ""
        assert skill.source == ""
        assert skill.compatibility == []
        assert skill.channel_filter == []
        assert skill.tools == []
        assert skill.scripts == {}
        assert skill.references == {}
        assert skill.assets == {}

    def test_creation_with_all_fields(self):
        skill = SkillDefinition(
            name="full-skill",
            description="Full skill",
            instructions="Full instructions.",
            category="rendering",
            version="2.0.0",
            author="recif",
            source="builtin",
            compatibility=["python3"],
            channel_filter=["rest"],
            tools=["calculator", "datetime"],
            scripts={"run.py": "print('hello')"},
            references={"docs.md": "# Docs"},
            assets={"logo.png": "/path/to/logo.png"},
        )
        assert skill.category == "rendering"
        assert skill.version == "2.0.0"
        assert skill.author == "recif"
        assert skill.source == "builtin"
        assert skill.compatibility == ["python3"]
        assert skill.channel_filter == ["rest"]
        assert skill.tools == ["calculator", "datetime"]
        assert skill.scripts == {"run.py": "print('hello')"}
        assert skill.references == {"docs.md": "# Docs"}
        assert skill.assets == {"logo.png": "/path/to/logo.png"}

    def test_prompt_fragment_backward_compat(self):
        skill = SkillDefinition(
            name="compat",
            description="Compat test",
            instructions="The actual instructions.",
        )
        # prompt_fragment reads instructions
        assert skill.prompt_fragment == "The actual instructions."

    def test_prompt_fragment_setter_backward_compat(self):
        skill = SkillDefinition(
            name="compat",
            description="Compat test",
            instructions="Original.",
        )
        skill.prompt_fragment = "Updated via setter."
        assert skill.instructions == "Updated via setter."
        assert skill.prompt_fragment == "Updated via setter."

    def test_default_mutable_fields_are_independent(self):
        skill_a = SkillDefinition(name="a", description="A", instructions="A")
        skill_b = SkillDefinition(name="b", description="B", instructions="B")
        skill_a.channel_filter.append("rest")
        assert skill_b.channel_filter == []
        skill_a.scripts["test.py"] = "print(1)"
        assert skill_b.scripts == {}


# --- Test: SkillRegistry ---


class TestSkillRegistry:
    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="test", description="Test", instructions="Prompt.")
        registry.register(skill)
        assert registry.get("test") is skill

    def test_get_unknown_returns_none(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_names(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="beta", description="B", instructions="B"))
        registry.register(SkillDefinition(name="alpha", description="A", instructions="A"))
        assert registry.names() == ["alpha", "beta"]

    def test_len(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        registry.register(SkillDefinition(name="x", description="X", instructions="X"))
        assert len(registry) == 1

    def test_build_prompt_no_skills(self):
        registry = SkillRegistry()
        assert registry.build_prompt() == ""

    def test_build_prompt_all_channels(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="general",
                description="General",
                instructions="General prompt.",
            )
        )
        result = registry.build_prompt(channel="rest")
        assert "General prompt." in result
        assert "## Skill: general" in result

    def test_build_prompt_with_channel_filter_match(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="rest-only",
                description="REST only",
                instructions="REST prompt.",
                channel_filter=["rest"],
            )
        )
        result = registry.build_prompt(channel="rest")
        assert "REST prompt." in result

    def test_build_prompt_with_channel_filter_no_match(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="rest-only",
                description="REST only",
                instructions="REST prompt.",
                channel_filter=["rest"],
            )
        )
        result = registry.build_prompt(channel="cli")
        assert result == ""

    def test_build_prompt_channel_filter_skipped_when_no_channel(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="rest-only",
                description="REST only",
                instructions="REST prompt.",
                channel_filter=["rest"],
            )
        )
        # No channel specified -- filtered skills are skipped
        result = registry.build_prompt()
        assert result == ""

    def test_build_prompt_combines_multiple(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="skill-a",
                description="A",
                instructions="Fragment A.",
            )
        )
        registry.register(
            SkillDefinition(
                name="skill-b",
                description="B",
                instructions="Fragment B.",
            )
        )
        result = registry.build_prompt(channel="rest")
        assert "Fragment A." in result
        assert "Fragment B." in result

    def test_build_prompt_mixed_filters(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="universal",
                description="Universal",
                instructions="Universal.",
            )
        )
        registry.register(
            SkillDefinition(
                name="rest-only",
                description="REST",
                instructions="REST only.",
                channel_filter=["rest"],
            )
        )
        # CLI channel: only universal should appear
        result = registry.build_prompt(channel="cli")
        assert "Universal." in result
        assert "REST only." not in result

        # REST channel: both should appear
        result = registry.build_prompt(channel="rest")
        assert "Universal." in result
        assert "REST only." in result

    def test_build_prompt_includes_scripts_references(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="resourceful",
                description="Has resources",
                instructions="Do things.",
                scripts={"run.py": "print(1)", "build.sh": "echo ok"},
                references={"api.md": "# API"},
            )
        )
        result = registry.build_prompt(channel="rest")
        assert "Available scripts:" in result
        assert "run.py" in result
        assert "build.sh" in result
        assert "Reference docs: api.md" in result

    def test_tool_names(self):
        registry = SkillRegistry()
        registry.register(
            SkillDefinition(
                name="a",
                description="A",
                instructions="A",
                tools=["calculator", "datetime"],
            )
        )
        registry.register(
            SkillDefinition(
                name="b",
                description="B",
                instructions="B",
                tools=["calculator", "web_search"],
            )
        )
        assert registry.tool_names() == ["calculator", "datetime", "web_search"]

    def test_tool_names_empty(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="x", description="X", instructions="X"))
        assert registry.tool_names() == []


# --- Test: Factory ---


class TestFactory:
    def test_create_skill_agui_render(self):
        skill = create_skill("agui-render")
        assert skill.name == "agui-render"
        assert skill.category == "rendering"
        assert skill.channel_filter == ["rest"]
        assert skill.version == "1.0.0"
        assert skill.author == "recif"
        assert skill.source == "builtin"
        assert "chart" in skill.instructions.lower()
        # Backward compat: prompt_fragment still works
        assert "chart" in skill.prompt_fragment.lower()

    def test_create_skill_code_review(self):
        skill = create_skill("code-review")
        assert skill.name == "code-review"
        assert skill.category == "analysis"
        assert "security" in skill.instructions.lower()

    def test_create_skill_doc_writer(self):
        skill = create_skill("doc-writer")
        assert skill.name == "doc-writer"
        assert skill.category == "writing"

    def test_create_skill_data_analyst(self):
        skill = create_skill("data-analyst")
        assert skill.name == "data-analyst"
        assert skill.channel_filter == ["rest"]
        assert "calculator" in skill.tools

    def test_create_skill_infra_deployer(self):
        skill = create_skill("infra-deployer")
        assert skill.name == "infra-deployer"
        assert "récif" in skill.instructions.lower() or "recif" in skill.instructions.lower()

    def test_unknown_skill_raises(self):
        with pytest.raises(ValueError, match="Unknown skill"):
            create_skill("nonexistent-skill")

    def test_available_skills(self):
        skills = available_skills()
        assert "agui-render" in skills
        assert "code-review" in skills
        assert "doc-writer" in skills
        assert "data-analyst" in skills
        assert "infra-deployer" in skills
        assert skills == sorted(skills)

    def test_register_custom_skill(self):
        register_skill("custom", "corail.skills.builtins", "CODE_REVIEW")
        skill = create_skill("custom")
        assert skill.name == "code-review"  # It's the CODE_REVIEW constant


# --- Test: Loader (load_from_string) ---


class TestLoader:
    def test_load_from_string_with_frontmatter(self):
        content = """---
name: my-skill
description: A test skill loaded from a string
category: analysis
version: 2.0.0
author: test-author
tools: [calculator, datetime]
channel_filter: [rest]
---

## My Skill Instructions

Do something useful.
"""
        skill = load_from_string(content)
        assert skill.name == "my-skill"
        assert skill.description == "A test skill loaded from a string"
        assert skill.category == "analysis"
        assert skill.version == "2.0.0"
        assert skill.author == "test-author"
        assert skill.tools == ["calculator", "datetime"]
        assert skill.channel_filter == ["rest"]
        assert "## My Skill Instructions" in skill.instructions
        assert "Do something useful." in skill.instructions
        assert skill.source == "inline"

    def test_load_from_string_without_frontmatter(self):
        content = "Just some instructions without frontmatter."
        skill = load_from_string(content, name="plain-skill")
        assert skill.name == "plain-skill"
        assert skill.instructions == "Just some instructions without frontmatter."
        assert skill.description == ""

    def test_load_from_string_with_name_override(self):
        content = """---
name: original-name
description: Original
---
Instructions here.
"""
        # Frontmatter name takes precedence
        skill = load_from_string(content, name="fallback")
        assert skill.name == "original-name"

    def test_load_from_string_fallback_name(self):
        content = """---
description: No name in frontmatter
---
Instructions.
"""
        skill = load_from_string(content, name="fallback-name")
        assert skill.name == "fallback-name"


# --- Test: Integration (agui-render channel filter) ---


class TestAguiRenderChannelFilter:
    """agui-render should only appear for channel='rest'."""

    def test_agui_render_appears_for_rest(self):
        registry = SkillRegistry()
        registry.register(create_skill("agui-render"))
        result = registry.build_prompt(channel="rest")
        assert "Rich Content Rendering" in result

    def test_agui_render_hidden_for_cli(self):
        registry = SkillRegistry()
        registry.register(create_skill("agui-render"))
        result = registry.build_prompt(channel="cli")
        assert result == ""

    def test_agui_render_hidden_for_empty_channel(self):
        registry = SkillRegistry()
        registry.register(create_skill("agui-render"))
        result = registry.build_prompt(channel="")
        assert result == ""
