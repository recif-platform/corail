"""Async database session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory from a database URL."""
    engine = create_async_engine(database_url, echo=False)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession]:
    """Yield a database session with automatic cleanup."""
    async with factory() as session:
        yield session
