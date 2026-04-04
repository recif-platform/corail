"""Tests for agent configuration cache."""

from corail.cache.memory import MemoryCache
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig


def _make_config(agent_id: str = "ag_01ARZ3NDEKTSV4RRFFQ69G5FAV") -> AgentConfig:
    return AgentConfig(
        id=agent_id,
        name="Test Agent",
        framework="adk",
        system_prompt="You are helpful.",
        model="gpt-4",
        llm_provider="stub",
    )


class TestAgentConfigCache:
    async def test_set_and_get_agent(self) -> None:
        cache = AgentConfigCache(MemoryCache())
        config = _make_config()
        await cache.set_agent(config)

        result = await cache.get_agent(config.id)
        assert result is not None
        assert result.id == config.id
        assert result.framework == "adk"

    async def test_get_unknown_returns_none(self) -> None:
        cache = AgentConfigCache(MemoryCache())
        assert await cache.get_agent("ag_NONEXISTENT000000000000") is None

    async def test_bulk_load(self) -> None:
        cache = AgentConfigCache(MemoryCache())
        agents = {
            "ag_AGENT1AAAAAAAAAAAAAAAAAAA": _make_config("ag_AGENT1AAAAAAAAAAAAAAAAAAA"),
            "ag_AGENT2BBBBBBBBBBBBBBBBBBBB": _make_config("ag_AGENT2BBBBBBBBBBBBBBBBBBBB"),
        }
        await cache.load_from_stub_registry(agents)

        assert await cache.get_agent("ag_AGENT1AAAAAAAAAAAAAAAAAAA") is not None
        assert await cache.get_agent("ag_AGENT2BBBBBBBBBBBBBBBBBBBB") is not None
