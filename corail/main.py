"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from corail import __version__
from corail.api.errors import core_error_handler, register_error_status
from corail.api.middleware.request_id import RequestIDMiddleware
from corail.api.rest import AgentNotFoundError
from corail.api.rest import router as rest_router
from corail.api.websocket import ws_router
from corail.cache.memory import MemoryCache
from corail.core.agent_cache import AgentConfigCache
from corail.core.agent_config import AgentConfig
from corail.core.errors import CoreError
from corail.core.pipeline import Pipeline
from corail.core.recif_health import RecifHealthChecker
from corail.models.stub import StubModel
from corail.observability.logger import configure_logging
from corail.strategies.simple import SimpleStrategy


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle."""
    configure_logging()
    register_error_status(AgentNotFoundError, 404, "agent-not-found")

    # Initialize cache and pipeline
    cache = MemoryCache()
    agent_cache = AgentConfigCache(cache)
    model = StubModel()
    strategy = SimpleStrategy(model=model, system_prompt="You are a helpful test assistant.")

    app.state.pipeline = Pipeline(strategy)
    app.state.agent_cache = agent_cache
    app.state.cache = cache

    # Preload stub agent for development
    test_agent = AgentConfig(
        id="ag_TESTAGENTSTUB00000000000",
        name="Test Agent",
        framework="adk",
        system_prompt="You are a helpful test assistant.",
        model="stub-model",
        llm_provider="stub",
    )
    await agent_cache.set_agent(test_agent)

    # Start Récif health checker
    from corail.config import Settings

    settings = Settings()
    health_checker = RecifHealthChecker(settings.recif_grpc_addr)
    app.state.recif_health = health_checker
    await health_checker.start()

    # Mount Récif control plane bridge
    from corail.control.bridge import RecifBridge
    from corail.events.bus import EventBus

    event_bus = EventBus()
    bridge = RecifBridge(event_bus)
    bridge.mount(app)
    app.state.event_bus = event_bus
    app.state.recif_bridge = bridge

    yield

    await health_checker.stop()


app = FastAPI(title="Corail", version=__version__, lifespan=lifespan)

# Middleware
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

# Error handlers
app.add_exception_handler(CoreError, core_error_handler)  # type: ignore[arg-type]

# Routes
app.include_router(rest_router, prefix="/api/v1", tags=["agents"])
app.include_router(ws_router, tags=["websocket"])


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Health check — always 200. Corail is healthy regardless of Récif."""
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """Readiness check — includes Récif availability status."""
    recif_health: RecifHealthChecker | None = getattr(app.state, "recif_health", None)
    recif_status = "available" if (recif_health and recif_health.is_available) else "unavailable"

    if hasattr(app.state, "pipeline"):
        return JSONResponse(
            {
                "status": "ready" if recif_status == "available" else "degraded",
                "recif": recif_status,
                "cache": "active",
            }
        )
    return JSONResponse({"status": "not-ready"}, status_code=503)
