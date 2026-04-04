"""Corail CLI — autonomous agent runtime entry point."""

import os
import click

from corail import __version__
from corail.config import Settings


@click.command()
@click.option("--channel", default=None, help="Channel: rest, websocket, slack, cli")
@click.option("--strategy", default=None, help="Strategy: agent-react")
@click.option("--model-type", default=None, help="Model provider: stub, ollama, anthropic, bedrock, vertex")
@click.option("--model-id", default=None, help="Model identifier")
@click.option("--system-prompt", default=None, help="System prompt text")
@click.option("--storage", default=None, help="Storage backend: memory, postgresql")
@click.option("--port", default=None, type=int, help="Port (default: 8000)")
@click.version_option(version=__version__, prog_name="corail")
def main(
    channel: str | None,
    strategy: str | None,
    model_type: str | None,
    model_id: str | None,
    system_prompt: str | None,
    storage: str | None,
    port: int | None,
) -> None:
    """Corail — autonomous AI agent runtime."""
    settings = Settings()

    overrides = {
        "channel": channel, "strategy": strategy, "model_type": model_type,
        "model_id": model_id, "system_prompt": system_prompt, "storage": storage, "port": port,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(settings, key, value)

    click.echo(f"Corail {__version__} — starting agent")
    click.echo(f"  Channel:  {settings.channel}")
    click.echo(f"  Strategy: {settings.strategy}")
    click.echo(f"  Model:    {settings.model_type}/{settings.model_id}")
    click.echo(f"  Storage:  {settings.storage}")
    click.echo(f"  Port:     {settings.port}")
    click.echo("")

    from corail.channels.factory import ChannelFactory
    from corail.core.pipeline import Pipeline
    from corail.models.factory import ModelFactory
    from corail.strategies.factory import StrategyFactory
    from corail.strategies.initializers import build_strategy_kwargs

    # Initialize MLflow tracing early (before strategy construction)
    # Runs in a background thread with timeout to avoid blocking agent startup if MLflow is not yet ready
    agent_name = os.environ.get("CORAIL_AGENT_NAME", "default")
    artifact_version = os.environ.get("RECIF_AGENT_VERSION", "unknown")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        try:
            import mlflow
            import os as _os
            import threading
            # Prevent MLflow from trying to create local artifact directories
            _os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

            mlflow_ready = threading.Event()

            def _init_mlflow():
                try:
                    mlflow.set_tracking_uri(mlflow_uri)
                    mlflow.set_experiment(f"recif/agents/{agent_name}")
                    mlflow.tracing.enable()
                    mlflow.set_active_model(name=f"{agent_name}-v{artifact_version}")
                    mlflow_ready.set()
                except Exception:
                    pass  # MLflow not ready yet — traces will be skipped

            init_thread = threading.Thread(target=_init_mlflow, daemon=True)
            init_thread.start()
            init_thread.join(timeout=10)  # Wait max 10s, then continue without MLflow

            if mlflow_ready.is_set():
                pass  # MLflow initialized successfully
            else:
                click.echo(f"  MLflow:   connecting in background (server may still be starting)...")

            # Enable auto-tracing for LLM providers (captures real token usage + cost)
            try:
                mlflow.openai.autolog()
            except Exception:
                pass
            try:
                mlflow.anthropic.autolog()
            except Exception:
                pass
            # Register production auto-scorers (sampled, non-blocking)
            eval_sample_rate = float(os.environ.get("RECIF_EVAL_SAMPLE_RATE", "0"))
            if eval_sample_rate > 0:
                try:
                    from mlflow.genai.scorers import Safety
                    safety = Safety(model=os.environ.get("RECIF_JUDGE_MODEL", "openai:/gpt-4o-mini"))
                    experiment = mlflow.get_experiment_by_name(f"recif/agents/{agent_name}")
                    if experiment:
                        registered = safety.register(experiment_id=experiment.experiment_id)
                        registered.start(sample_rate=eval_sample_rate)
                        click.echo(f"  Auto-eval: Safety scorer at {eval_sample_rate:.0%} sample rate")
                except Exception:
                    pass  # Auto-scorers are optional — don't block agent startup

            click.echo(f"  MLflow:   {mlflow_uri} (experiment: recif/agents/{agent_name}, version: v{artifact_version})")
        except Exception as e:
            click.echo(f"  MLflow:   disabled ({e})")

    model = ModelFactory.create(settings.model_type, settings.model_id)
    extra_kwargs = build_strategy_kwargs(settings)
    extra_kwargs["agent_name"] = getattr(settings, "agent_name", os.environ.get("CORAIL_AGENT_NAME", "default"))
    strategy_impl = StrategyFactory.create(settings.strategy, model, settings.system_prompt, **extra_kwargs)
    pipeline = Pipeline(strategy_impl)

    # Start ControlServer on port 8001 (always available for Récif control plane)
    import threading
    from corail.control.server import ControlServer
    control_server = ControlServer(pipeline, settings)
    control_thread = threading.Thread(
        target=control_server.start,
        kwargs={"port": 8001},
        daemon=True,
        name="control-server",
    )
    control_thread.start()
    click.echo(f"  Control:  port 8001 (evaluation, status, config)")

    # Channel server on port 8000 (user-facing: chat, conversations, memory)
    channel_impl = ChannelFactory.create(settings.channel, pipeline, settings)
    channel_impl.start()


if __name__ == "__main__":
    main()
