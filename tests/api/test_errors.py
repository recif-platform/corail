"""Tests for RFC 7807 error formatting."""

from corail.api.errors import ProblemDetail, core_error_to_problem
from corail.core.errors import AdapterNotFoundError, ConfigValidationError, ExecutionError, LLMError


class TestRFC7807:
    def test_adapter_not_found_maps_to_400(self) -> None:
        error = AdapterNotFoundError("framework", "unknown")
        problem = core_error_to_problem(error, "/api/v1/agents/x/chat", "req_123")
        assert problem.status == 400
        assert "adapter-not-found" in problem.type

    def test_config_validation_maps_to_422(self) -> None:
        error = ConfigValidationError("Bad config")
        problem = core_error_to_problem(error)
        assert problem.status == 422

    def test_llm_error_maps_to_502(self) -> None:
        error = LLMError("API failed")
        problem = core_error_to_problem(error)
        assert problem.status == 502

    def test_execution_error_maps_to_500(self) -> None:
        error = ExecutionError("Boom")
        problem = core_error_to_problem(error)
        assert problem.status == 500

    def test_problem_detail_serialization(self) -> None:
        detail = ProblemDetail(
            type="https://corail.dev/errors/test",
            title="Test Error",
            status=400,
            detail="Something went wrong",
            instance="/test",
            request_id="req_123",
        )
        data = detail.model_dump()
        assert data["type"] == "https://corail.dev/errors/test"
        assert data["status"] == 400
