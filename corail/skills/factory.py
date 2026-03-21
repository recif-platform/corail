"""SkillFactory -- registry-based skill resolution with multi-source loading."""

import asyncio
import importlib

from corail.skills.base import SkillDefinition

# Registry: skill_name -> (module_path, variable_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "agui-render": ("corail.skills.builtins", "AGUI_RENDER"),
    "code-review": ("corail.skills.builtins", "CODE_REVIEW"),
    "doc-writer": ("corail.skills.builtins", "DOC_WRITER"),
    "data-analyst": ("corail.skills.builtins", "DATA_ANALYST"),
    "infra-deployer": ("corail.skills.builtins", "INFRASTRUCTURE"),
}


def register_skill(name: str, module_path: str, var_name: str) -> None:
    """Register a new skill. Allows external plugins to add skills."""
    _REGISTRY[name] = (module_path, var_name)


def create_skill(name: str) -> SkillDefinition:
    """Create (import) a builtin skill by name. Returns the SkillDefinition instance."""
    entry = _REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        msg = f"Unknown skill: {name}. Available: {available}"
        raise ValueError(msg)

    module_path, var_name = entry
    module = importlib.import_module(module_path)
    return getattr(module, var_name)


def load_skill(source: str, **kwargs: object) -> SkillDefinition:
    """Load a skill from any source.

    Args:
        source: Can be one of:
            - A builtin name: "agui-render"
            - A local path: "/path/to/skill/"
            - A GitHub ref: "github:anthropics/skills/skills/pdf"
        **kwargs: Additional arguments (e.g. token for GitHub).

    Returns:
        SkillDefinition loaded from the specified source.
    """
    # Builtin name
    if source in _REGISTRY:
        return create_skill(source)

    # GitHub reference
    if source.startswith("github:"):
        from corail.skills.loader import load_from_github

        ref = source[len("github:") :]
        # Parse: org/repo/path/to/skill -> repo=org/repo, skill_path=path/to/skill
        parts = ref.split("/", 2)
        if len(parts) < 3:
            msg = f"Invalid GitHub skill reference: {source}. Expected: github:org/repo/path/to/skill"
            raise ValueError(msg)
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        token = str(kwargs.get("token", ""))

        # Run async loader synchronously if no event loop, or use existing loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context -- create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    load_from_github(repo, skill_path, token=token),
                )
                return future.result()
        else:
            return asyncio.run(load_from_github(repo, skill_path, token=token))

    # Local filesystem path
    if source.startswith("/") or source.startswith("./") or source.startswith("~"):
        import os

        from corail.skills.loader import load_from_directory

        expanded = os.path.expanduser(source)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, load_from_directory(expanded))
                return future.result()
        else:
            return asyncio.run(load_from_directory(expanded))

    msg = f"Unknown skill source: {source}. Expected: builtin name, /local/path, or github:org/repo/path"
    raise ValueError(msg)


def available_skills() -> list[str]:
    """Return sorted list of registered skill names."""
    return sorted(_REGISTRY.keys())
