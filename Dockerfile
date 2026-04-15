FROM python:3.13-slim AS builder

# Optional: custom CA certificates (corporate VPN/proxy)
COPY certs/ /tmp/certs/
RUN for f in /tmp/certs/*.pem /tmp/certs/*.crt; do \
      [ -f "$f" ] && cp "$f" /usr/local/share/ca-certificates/"$(basename "${f%.*}").crt" || true; \
    done && update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY corail/ corail/
RUN uv sync --no-dev --frozen --extra vertex

FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
    && addgroup --system corail && adduser --system --ingroup corail corail

# Carry over custom CA certs to runtime image
COPY certs/ /tmp/certs/
RUN for f in /tmp/certs/*.pem /tmp/certs/*.crt; do \
      [ -f "$f" ] && cp "$f" /usr/local/share/ca-certificates/"$(basename "${f%.*}").crt" || true; \
    done && update-ca-certificates && rm -rf /tmp/certs

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/corail /app/corail

RUN mkdir -p /tmp/mlflow-artifacts && chown -R corail:corail /app /tmp/mlflow-artifacts

USER corail

ENV PATH="/app/.venv/bin:$PATH"
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV MLFLOW_ARTIFACT_ROOT=/tmp/mlflow-artifacts
ENV GIT_PYTHON_REFRESH=quiet

EXPOSE 8000 8001

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8001/healthz || exit 1

ENTRYPOINT ["corail"]
CMD []
