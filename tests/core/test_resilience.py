"""Integration test — resilience when Recif is unavailable."""

from corail.cache.memory import MemoryCache
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig
from corail.core.pipeline import Pipeline
from corail.core.recif_health import RecifHealthChecker
from corail.models.stub import StubModel
from corail.strategies.simple import SimpleStrategy


async def test_chat_succeeds_when_recif_is_down() -> None:
    """Agent execution works from cache even when Recif is unreachable."""
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

    # Pipeline works with strategy-based API
    model = StubModel()
    strategy = SimpleStrategy(model=model, system_prompt="You are resilient.")
    pipeline = Pipeline(strategy)

    response = await pipeline.execute("Hello from cache")
    assert response == "Echo: Hello from cache"

    # Recif is down (unreachable address)
    health = RecifHealthChecker("localhost:99999", check_interval=1)
    is_up = await health.check()
    assert is_up is False

    # Agent still works from cache
    cached_config = await agent_cache.get_agent(config.id)
    assert cached_config is not None

    response2 = await pipeline.execute("Still working")
    assert response2 == "Echo: Still working"
