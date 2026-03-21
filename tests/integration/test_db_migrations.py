"""Integration tests for database migrations.

These tests require a running PostgreSQL instance.
Run with: pytest tests/integration/ -m integration
"""

import pytest


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_migration_creates_corail_schema() -> None:
    """Verify that running migrations creates the corail schema."""


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_migration_idempotent() -> None:
    """Verify that running migrations twice produces no errors."""


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_agent_executions_table_exists() -> None:
    """Verify agent_executions table exists with correct columns after migration."""
