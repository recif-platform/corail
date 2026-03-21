"""Skill loader — parse skills from filesystem, GitHub, or raw strings.

Supports the Anthropic skills format:
  my-skill/
    SKILL.md          <- YAML frontmatter + markdown instructions
    scripts/          <- executable code (python, bash)
    references/       <- docs loaded on demand
    assets/           <- templates, images, fonts
"""

import re
from pathlib import Path

from corail.skills.base import SkillDefinition

# ---------------------------------------------------------------------------
# Frontmatter parser (lightweight, no PyYAML dependency required)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Simple key: value pairs (strings, lists as comma-separated)
_YAML_LIST_RE = re.compile(r"^\[(.+)]$")


def _parse_frontmatter(content: str) -> tuple[dict[str, str | list[str]], str]:
    """Extract YAML frontmatter and markdown body from SKILL.md content.

    Returns (metadata_dict, body_markdown).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end() :]

    metadata: dict[str, str | list[str]] = {}
    for line in raw_yaml.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        # Handle YAML-style lists: [a, b, c]
        list_match = _YAML_LIST_RE.match(value)
        if list_match:
            items = [item.strip().strip("'\"") for item in list_match.group(1).split(",")]
            metadata[key] = items
        else:
            # Strip surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            metadata[key] = value

    return metadata, body


def _metadata_to_skill(metadata: dict, body: str, name: str = "") -> SkillDefinition:
    """Convert parsed metadata + body into a SkillDefinition."""
    skill_name = metadata.get("name", name) or "unnamed-skill"
    description = metadata.get("description", "")
    category = metadata.get("category", "general")
    version = metadata.get("version", "1.0.0")
    author = metadata.get("author", "")
    source = metadata.get("source", "")

    # List fields
    compatibility = metadata.get("compatibility", [])
    if isinstance(compatibility, str):
        compatibility = [c.strip() for c in compatibility.split(",") if c.strip()]

    channel_filter = metadata.get("channel_filter", [])
    if isinstance(channel_filter, str):
        channel_filter = [c.strip() for c in channel_filter.split(",") if c.strip()]

    tools = metadata.get("tools", [])
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]

    return SkillDefinition(
        name=str(skill_name),
        description=str(description),
        instructions=body.strip(),
        category=str(category),
        version=str(version),
        author=str(author),
        compatibility=list(compatibility),
        channel_filter=list(channel_filter),
        tools=list(tools),
        source=str(source),
    )


# ---------------------------------------------------------------------------
# Load from directory
# ---------------------------------------------------------------------------


def _load_dir_files(directory: Path) -> dict[str, str]:
    """Load all files from a directory into a name->content dict."""
    result: dict[str, str] = {}
    if not directory.is_dir():
        return result
    for filepath in sorted(directory.iterdir()):
        if filepath.is_file():
            try:
                result[filepath.name] = filepath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                result[filepath.name] = f"<binary: {filepath.name}>"
    return result


async def load_from_directory(path: str) -> SkillDefinition:
    """Load a skill from a local directory containing SKILL.md + optional resources.

    Expected structure:
        path/
            SKILL.md
            scripts/
            references/
            assets/
    """
    skill_dir = Path(path)
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.is_file():
        msg = f"SKILL.md not found in {path}"
        raise FileNotFoundError(msg)

    content = skill_md.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(content)

    # Default name from directory name
    name = metadata.get("name", skill_dir.name)
    skill = _metadata_to_skill(metadata, body, name=str(name))

    # Load resource subdirectories
    skill.scripts = _load_dir_files(skill_dir / "scripts")
    skill.references = _load_dir_files(skill_dir / "references")
    skill.assets = _load_dir_files(skill_dir / "assets")

    if not skill.source:
        skill.source = f"local:{path}"

    return skill


# ---------------------------------------------------------------------------
# Load from GitHub
# ---------------------------------------------------------------------------


async def load_from_github(repo: str, skill_path: str, token: str = "") -> SkillDefinition:
    """Load a skill from a GitHub repo.

    Args:
        repo: GitHub repo, e.g. 'anthropics/skills'.
        skill_path: Path within the repo, e.g. 'skills/pdf'.
        token: Optional GitHub personal access token for private repos.

    Returns:
        SkillDefinition parsed from the remote SKILL.md.
    """
    try:
        import httpx
    except ImportError as exc:
        msg = "httpx is required for GitHub skill loading: pip install httpx"
        raise ImportError(msg) from exc

    # Normalize path separators
    skill_path = skill_path.strip("/")
    url = f"https://raw.githubusercontent.com/{repo}/main/{skill_path}/SKILL.md"

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, follow_redirects=True)
        if resp.status_code != 200:
            msg = f"Failed to fetch SKILL.md from {url}: HTTP {resp.status_code}"
            raise RuntimeError(msg)
        content = resp.text

    metadata, body = _parse_frontmatter(content)

    # Default name from the last path segment
    default_name = skill_path.split("/")[-1] if "/" in skill_path else skill_path
    skill = _metadata_to_skill(metadata, body, name=default_name)

    if not skill.source:
        skill.source = f"github:{repo}/{skill_path}"

    return skill


# ---------------------------------------------------------------------------
# Load from string
# ---------------------------------------------------------------------------


def load_from_string(content: str, name: str = "") -> SkillDefinition:
    """Parse a SKILL.md string into a SkillDefinition.

    Args:
        content: Raw SKILL.md content (YAML frontmatter + markdown body).
        name: Fallback name if not specified in frontmatter.

    Returns:
        SkillDefinition parsed from the string.
    """
    metadata, body = _parse_frontmatter(content)
    skill = _metadata_to_skill(metadata, body, name=name)

    if not skill.source:
        skill.source = "inline"

    return skill
