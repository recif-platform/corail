"""Control plane -- Recif <-> Corail bridge for commands and event streaming.

Modules:
    bridge          -- RecifBridge: event bus integration, pause/resume, config, SSE events.
    endpoints       -- Shared async functions for chat, conversations, memory.
    server          -- ControlServer: standalone FastAPI + gRPC app on port 8001.
    grpc_server     -- GrpcControlServer: standalone gRPC server for ControlService.
    grpc_servicer   -- ControlServiceServicer: gRPC servicer mapping RPCs to endpoints.
    pb/             -- Generated protobuf and gRPC stubs (control.v1).
"""
