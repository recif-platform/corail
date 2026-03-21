"""Integration tests for AgentExecutionRepository.

These tests require a running PostgreSQL instance.
Run with: pytest tests/integration/ -m integration
"""

import pytest


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_create_and_get_execution() -> None:
    """Verify creating and retrieving an agent execution."""


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_list_executions_by_agent() -> None:
    """Verify listing executions filtered by agent_id."""


@pytest.mark.skip(reason="Requires running PostgreSQL — use testcontainers or set CORAIL_DATABASE_URL")
async def test_update_execution_status() -> None:
    """Verify updating execution status."""
