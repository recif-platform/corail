"""Strategy initializers — build extra kwargs for each strategy type.

Each strategy declares what dependencies it needs. The CLI calls the initializer
to build the kwargs, without knowing anything about the strategy internals.
No if/elif chains. Just a registry.
"""

import json
import logging

from corail.config import Settings

logger = logging.getLogger(__name__)

# Registry: strategy_name → initializer function
_INITIALIZERS: dict[str, callable] = {}


def register_initializer(strategy_name: str):
    """Decorator to register a strategy initializer."""
    def decorator(fn):
        _INITIALIZERS[strategy_name] = fn
        return fn
    return decorator


def build_strategy_kwargs(settings: Settings) -> dict[str, object]:
    """Build extra kwargs for the current strategy. Returns {} if no initializer registered."""
    initializer = _INITIALIZERS.get(settings.strategy)
    if initializer is None:
        return {}
    return initializer(settings)


def _build_guard_pipeline(settings: Settings) -> "GuardPipeline | None":
    """Build a GuardPipeline from settings. Returns None if no guards configured."""
    from corail.events.bus import EventBus
    from corail.guards.factory import GuardFactory
    from corail.guards.pipeline import GuardPipeline

    guards_config = getattr(settings, "guards", "")
    if not guards_config:
        return None

    try:
        guard_names = json.loads(guards_config) if isinstance(guards_config, str) else guards_config
        guards = []
        for name in guard_names:
            try:
                guards.append(GuardFactory.create(name))
            except ValueError as e:
                logger.warning("Skipping guard '%s': %s", name, e)
        return GuardPipeline(guards=guards, event_bus=EventBus()) if guards else None
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Invalid guards config: %s", e)
        return None


@register_initializer("agent-react")
def _init_agent(settings: Settings) -> dict[str, object]:
    """Build all components for the unified agent strategy."""
    from corail.events.bus import EventBus
    from corail.memory.factory import create_memory_storage
    from corail.memory.manager import MemoryManager
    from corail.planning.correction import SelfCorrector
    from corail.planning.planner import Planner
    from corail.skills.factory import available_skills, create_skill
    from corail.skills.registry import SkillRegistry
    from corail.tools.base import ToolParameter
    from corail.tools.factory import ToolFactory
    from corail.tools.registry import ToolRegistry

    # Tools
    tools = _build_tools_registry(settings)
    _register_kb_tools(settings, tools)
    tool_names = tools.names()
    logger.info("Agent tools: %s", ", ".join(tool_names) if tool_names else "(none)")

    # Skills
    skills_registry = _build_skills_registry(settings)
    skill_names = skills_registry.names()
    logger.info("Agent skills: %s", ", ".join(skill_names) if skill_names else "(none)")

    # Guards
    guard_pipeline = _build_guard_pipeline(settings)

    # Event bus + MLflow tracing listener
    event_bus = EventBus()
    try:
        from corail.tracing.mlflow_listener import MLflowTracingListener
        mlflow_listener = MLflowTracingListener()
        mlflow_listener.register(event_bus)
    except Exception:
        pass  # MLflow not available

    # Memory — pgvector backend needs an embedding provider
    memory_backend = getattr(settings, "memory_backend", "in_memory")
    memory_kwargs: dict[str, object] = {}
    if memory_backend == "pgvector":
        from corail.embeddings.factory import EmbeddingProviderFactory
        memory_kwargs["embedding_provider"] = EmbeddingProviderFactory.create("ollama")
        memory_kwargs["connection_url"] = settings.database_url
    storage = create_memory_storage(memory_backend, **memory_kwargs)
    memory = MemoryManager(storage=storage)

    # Planner and corrector need a model — they will be set to None here;
    # the factory creates them with the same model passed to the strategy.
    # We return callables that accept a model to construct them lazily.
    # However, the strategy constructor expects instances, so we pass None.
    # The CLI / server should construct Planner(model) and SelfCorrector(model)
    # when they have the model instance. For initializer, we return None.
    # The StrategyFactory.create() passes model= separately.

    return {
        "tools": tools,
        "skills": skills_registry,
        "guard_pipeline": guard_pipeline,
        "event_bus": event_bus,
        "memory": memory,
        "max_rounds": int(getattr(settings, "max_rounds", 10)),
        "max_tokens": int(getattr(settings, "max_tokens", 100_000)),
    }


def _build_skills_registry(settings: Settings) -> "SkillRegistry":
    """Build a SkillRegistry from CORAIL_SKILLS JSON config.

    CORAIL_SKILLS can contain:
      - Builtin names: ["agui-render", "code-review"]
      - GitHub references: ["github:anthropics/skills/skills/pdf"]
      - Local paths: ["/path/to/my-skill"]
      - Mixed: ["agui-render", "github:anthropics/skills/skills/pdf"]
    """
    from corail.skills.factory import load_skill
    from corail.skills.registry import SkillRegistry

    registry = SkillRegistry()
    if not settings.skills:
        return registry

    try:
        skill_sources = json.loads(settings.skills)
        for source in skill_sources:
            try:
                registry.register(load_skill(source))
            except (ValueError, RuntimeError, FileNotFoundError, ImportError) as e:
                logger.warning("Skipping skill '%s': %s", source, e)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Invalid CORAIL_SKILLS JSON: %s", e)

    return registry


def _build_tools_registry(settings: Settings) -> "ToolRegistry":
    """Build a ToolRegistry from CORAIL_TOOLS JSON config."""
    from corail.tools.base import ToolParameter
    from corail.tools.factory import ToolFactory
    from corail.tools.registry import ToolRegistry

    tools = ToolRegistry()
    if not settings.tools:
        return tools

    try:
        for tc in json.loads(settings.tools):
            tool_type = tc.get("type", "")
            params = [
                ToolParameter(
                    name=p["name"],
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", False),
                )
                for p in tc.get("parameters", [])
            ]
            try:
                tools.register(ToolFactory.create(tool_type, **_build_tool_kwargs(tc, params)))
            except ValueError as e:
                logger.warning("Skipping tool '%s': %s", tc.get("name"), e)
    except json.JSONDecodeError as e:
        logger.warning("Invalid CORAIL_TOOLS JSON: %s", e)

    return tools


def _register_kb_tools(settings: Settings, tools_registry: "ToolRegistry") -> None:
    """Register per-KB search tools from CORAIL_KNOWLEDGE_BASES config."""
    if not settings.knowledge_bases:
        return

    import re as _re

    from corail.embeddings.factory import EmbeddingProviderFactory
    from corail.retrieval.pgvector import PgVectorRetriever
    from corail.tools.kb_search import KBSearchTool

    try:
        kb_configs = json.loads(settings.knowledge_bases)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Invalid CORAIL_KNOWLEDGE_BASES JSON: %s", e)
        return

    if not kb_configs:
        return

    registered_slugs: set[str] = set()
    for kb in kb_configs:
        try:
            kb_id = kb.get("kb_id", kb.get("name", "default"))
            kb_name = kb.get("name", kb_id)
            slug = _re.sub(r"[^a-z0-9_]", "", kb_name.lower().replace(" ", "_").replace("-", "_"))
            tool_name = f"search_{slug}"

            if tool_name in registered_slugs:
                logger.warning("Duplicate KB tool name '%s' from kb_id=%s — skipping", tool_name, kb_id)
                continue
            registered_slugs.add(tool_name)

            embedding_type = kb.get("embedding_provider", "ollama")
            embedding_kwargs = {}
            if kb.get("embedding_model"):
                embedding_kwargs["model"] = kb["embedding_model"]
            embedding_provider = EmbeddingProviderFactory.create(embedding_type, **embedding_kwargs)
            retriever = PgVectorRetriever(
                connection_url=kb.get("connection_url", ""),
                embedding_provider=embedding_provider,
                kb_id=kb_id,
            )

            description = kb.get("description") or (
                f"Search the knowledge base for relevant documents about {kb_name}. "
                f"Use when the user asks about topics that may be covered in this document collection."
            )
            tool = KBSearchTool(
                name=tool_name,
                description=description,
                retriever=retriever,
                kb_id=kb_id,
            )
            tools_registry.register(tool)
            logger.info("KB search tool registered: %s (%s)", tool_name, kb_id)
        except Exception as e:
            logger.warning("Failed to register KB tool for '%s': %s", kb.get("kb_id", "?"), e)


def _build_tool_kwargs(tc: dict, params: list) -> dict[str, object]:
    """Build constructor kwargs for a tool from its CRD config."""
    # Common fields
    kwargs: dict[str, object] = {
        "name": tc["name"],
        "description": tc.get("description", ""),
        "parameters": params,
        "timeout": float(tc.get("timeout", 30)),
    }
    # Type-specific field mapping (CRD field name → constructor param name)
    _FIELD_MAP: dict[str, dict[str, str]] = {
        "http": {"endpoint": "url", "method": "method", "headers": "headers"},
        "cli": {"binary": "binary", "allowedCommands": "allowed_commands"},
    }
    for crd_field, kwarg_name in _FIELD_MAP.get(tc.get("type", ""), {}).items():
        if tc.get(crd_field):
            kwargs[kwarg_name] = tc[crd_field]
    return kwargs
