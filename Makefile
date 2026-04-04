.PHONY: dev test test-coverage lint format build proto-gen

dev:
	uv run uvicorn corail.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

test-coverage:
	uv run pytest --cov=corail --cov-report=html --cov-report=term

lint:
	uv run ruff check .
	uv run mypy corail/

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	docker build -t corail .

PROTO_SRC ?= ../recif/proto
PROTO_OUT ?= corail/control/pb

proto-gen:
	uv run python -m grpc_tools.protoc \
		--proto_path=$(PROTO_SRC) \
		--python_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		control/v1/control.proto
	@# Fix import path: generated code uses 'from control.v1 import ...' but we
	@# need 'from corail.control.pb.control.v1 import ...' inside the package.
	sed -i '' 's/from control\.v1 import/from corail.control.pb.control.v1 import/' \
		$(PROTO_OUT)/control/v1/control_pb2_grpc.py
	@echo "Python gRPC stubs generated in $(PROTO_OUT)/control/v1/"
