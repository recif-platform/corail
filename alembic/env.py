"""Alembic async migration environment for Corail schema."""

import asyncio
from logging.config import fileConfig

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from environment variable
import os  # noqa: E402

db_url = os.environ.get("CORAIL_DATABASE_URL", "")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

from corail.db.models import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema="corail",
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    """Execute migrations with connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema="corail",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    from sqlalchemy.ext.asyncio import async_engine_from_config

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
