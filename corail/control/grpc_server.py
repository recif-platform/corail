"""gRPC control plane server for Corail.

Runs alongside (or replaces) the FastAPI ControlServer on port 8001.
Implements the control.v1.ControlService defined in the shared proto.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import TYPE_CHECKING, Any

import grpc

from corail.control.grpc_servicer import ControlServiceServicer
from corail.control.pb.control.v1 import control_pb2_grpc

if TYPE_CHECKING:
    from corail.control.bridge import RecifBridge
    from corail.core.pipeline import Pipeline
    from corail.storage.port import StoragePort

logger = logging.getLogger(__name__)

# Default: 10 worker threads for gRPC handlers.
_MAX_WORKERS = 10


class GrpcControlServer:
    """Standalone gRPC server exposing ControlService on a given port.

    Usage::

        server = GrpcControlServer(pipeline, storage_factory, bridge)
        server.start(port=8001)   # blocks
    """

    def __init__(
        self,
        pipeline: Pipeline,
        storage_factory: Any,
        bridge: RecifBridge,
        memory_accessor: Any = None,
        max_workers: int = _MAX_WORKERS,
    ) -> None:
        self._pipeline = pipeline
        self._storage_factory = storage_factory
        self._bridge = bridge
        self._memory_accessor = memory_accessor
        self._max_workers = max_workers
        self._server: grpc.Server | None = None

    def start(self, port: int = 8001, *, block: bool = True) -> None:
        """Start the gRPC server.

        Args:
            port: TCP port to listen on.
            block: If True, blocks until the server is stopped.
        """
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),   # 50 MB
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )

        servicer = ControlServiceServicer(
            pipeline=self._pipeline,
            storage_factory=self._storage_factory,
            bridge=self._bridge,
            memory_accessor=self._memory_accessor,
        )
        control_pb2_grpc.add_ControlServiceServicer_to_server(servicer, self._server)

        listen_addr = f"[::]:{port}"
        self._server.add_insecure_port(listen_addr)
        self._server.start()
        logger.info("gRPC ControlService listening on %s", listen_addr)

        if block:
            self._server.wait_for_termination()

    def stop(self, grace: float = 5.0) -> None:
        """Gracefully stop the gRPC server."""
        if self._server is not None:
            self._server.stop(grace)
            logger.info("gRPC ControlService stopped")
