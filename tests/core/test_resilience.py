"""Integration test — resilience when Récif is unavailable."""

from corail.adapters.factory import create_default_registry
from corail.cache.memory import MemoryCache
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig, ExecutionRequest
from corail.core.pipeline import Pipeline
from corail.core.recif_health import RecifHealthChecker


async def test_chat_succeeds_when_recif_is_down() -> None:
    """Agent execution works from cache even when Récif is unreachable."""
    # Set up cache with an agent
    cache = MemoryCache()
    agent_cache = AgentConfigCache(cache)
    config = AgentConfig(
        id="ag_RESILIENCE0000000000000",
        name="Resilient Agent",
        framework="adk",
        system_prompt="You are resilient.",
        model="stub-model",
        llm_provider="stub",
    )
    await agent_cache.set_agent(config)

    # Pipeline works
    registry = create_default_registry()
    pipeline = Pipeline(registry)

    request = ExecutionRequest(agent_config=config, input="Hello from cache")
    response = await pipeline.execute(request)
    assert response.output == "Echo: Hello from cache"

    # Récif is down (unreachable address)
    health = RecifHealthChecker("localhost:99999", check_interval=1)
    is_up = await health.check()
    assert is_up is False

    # Agent still works from cache
    cached_config = await agent_cache.get_agent(config.id)
    assert cached_config is not None

    request2 = ExecutionRequest(agent_config=cached_config, input="Still working")
    response2 = await pipeline.execute(request2)
    assert response2.output == "Echo: Still working"
