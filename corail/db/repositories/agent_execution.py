"""Repository for agent execution CRUD operations."""

from sqlalchemy import select

from corail.db.models import AgentExecution
from corail.db.repositories.base import BaseRepository


class AgentExecutionRepository(BaseRepository):
    """CRUD operations for agent executions."""

    async def get(self, execution_id: str) -> AgentExecution | None:
        """Get an execution by ID."""
        return await self._session.get(AgentExecution, execution_id)

    async def list_by_agent(self, agent_id: str) -> list[AgentExecution]:
        """List executions for a given agent."""
        stmt = (
            select(AgentExecution).where(AgentExecution.agent_id == agent_id).order_by(AgentExecution.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, execution: AgentExecution) -> AgentExecution:
        """Create a new execution record."""
        self._session.add(execution)
        await self._session.flush()
        await self._session.refresh(execution)
        return execution

    async def update_status(self, execution_id: str, status: str) -> AgentExecution | None:
        """Update execution status."""
        execution = await self.get(execution_id)
        if execution is None:
            return None
        execution.status = status
        await self._session.flush()
        await self._session.refresh(execution)
        return execution
