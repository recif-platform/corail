"""SQLAlchemy models for Corail database entities."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all Corail ORM models."""


class AgentExecution(Base):
    """Tracks agent execution lifecycle."""

    __tablename__ = "agent_executions"
    __table_args__ = (
        Index("idx_agent_executions_agent_id", "agent_id"),
        Index("idx_agent_executions_team_id", "team_id"),
        {"schema": "corail"},
    )

    id: Mapped[str] = mapped_column(String(30), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(30), nullable=False)
    team_id: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
