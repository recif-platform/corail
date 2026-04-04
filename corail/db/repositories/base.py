"""Base repository with common async patterns."""

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base class for async SQLAlchemy repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
