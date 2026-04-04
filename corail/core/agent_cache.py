"""Agent configuration cache — enables resilience when Récif is down."""

from corail.cache.port import CachePort
from corail.core.agent_config import AgentConfig


class AgentConfigCache:
    """Caches agent configurations for resilient operation."""

    KEY_PREFIX = "agent:"

    def __init__(self, cache: CachePort) -> None:
        self._cache = cache

    async def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get an agent config from cache."""
        data = await self._cache.get(f"{self.KEY_PREFIX}{agent_id}")
        if data is None:
            return None
        if isinstance(data, AgentConfig):
            return data
        return AgentConfig.model_validate(data)

    async def set_agent(self, config: AgentConfig) -> None:
        """Store an agent config in cache."""
        await self._cache.set(f"{self.KEY_PREFIX}{config.id}", config)

    async def load_from_stub_registry(self, agents: dict[str, AgentConfig]) -> None:
        """Bulk load agents into cache (temporary until gRPC sync)."""
        for agent_id, config in agents.items():
            await self._cache.set(f"{self.KEY_PREFIX}{agent_id}", config)
