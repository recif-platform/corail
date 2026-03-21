"""Create agent_executions table.

Revision ID: 001
Revises:
Create Date: 2026-03-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create corail schema and agent_executions table."""
    op.execute("CREATE SCHEMA IF NOT EXISTS corail")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "agent_executions",
        sa.Column("id", sa.String(30), primary_key=True),
        sa.Column("agent_id", sa.String(30), nullable=False),
        sa.Column("team_id", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("input", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("output", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema="corail",
    )

    op.create_index(
        "idx_agent_executions_agent_id",
        "agent_executions",
        ["agent_id"],
        schema="corail",
    )
    op.create_index(
        "idx_agent_executions_team_id",
        "agent_executions",
        ["team_id"],
        schema="corail",
    )


def downgrade() -> None:
    """Drop agent_executions table and corail schema."""
    op.drop_index("idx_agent_executions_team_id", table_name="agent_executions", schema="corail")
    op.drop_index("idx_agent_executions_agent_id", table_name="agent_executions", schema="corail")
    op.drop_table("agent_executions", schema="corail")
    op.execute("DROP SCHEMA IF EXISTS corail CASCADE")
