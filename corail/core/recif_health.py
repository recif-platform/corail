"""Récif health checker — monitors control plane availability."""

import asyncio
import contextlib

import structlog

logger = structlog.get_logger()


class RecifHealthChecker:
    """Periodically checks if Récif is reachable. Non-blocking."""

    def __init__(self, grpc_addr: str, check_interval: int = 30) -> None:
        self._grpc_addr = grpc_addr
        self._check_interval = check_interval
        self._available = False
        self._task: asyncio.Task[None] | None = None

    @property
    def is_available(self) -> bool:
        """Whether Récif was reachable on the last check."""
        return self._available

    async def check(self) -> bool:
        """Attempt to verify Récif connectivity."""
        try:
            host, port_str = self._grpc_addr.rsplit(":", 1)
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port_str)),
                timeout=5.0,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (OSError, TimeoutError, ValueError):
            return False

    async def start(self) -> None:
        """Start periodic health checking as a background task."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background health check task."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        """Background loop that periodically checks Récif health."""
        while True:
            was_available = self._available
            self._available = await self.check()

            if was_available and not self._available:
                await logger.awarning("recif_unavailable", grpc_addr=self._grpc_addr)
            elif not was_available and self._available:
                await logger.ainfo("recif_available", grpc_addr=self._grpc_addr)

            await asyncio.sleep(self._check_interval)
