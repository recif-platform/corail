<p align="center">
  <img src="https://recif-platform.github.io/logo.png?v=2" alt="Corail" width="80" />
</p>

<h1 align="center">Corail</h1>

<p align="center">
  <strong>The autonomous agent runtime for the Recif platform.</strong>
</p>

<p align="center">
  <a href="https://github.com/recif-platform/corail/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue?style=flat-square" alt="License" /></a>
  <a href="https://github.com/recif-platform/corail/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/recif-platform/corail/ci.yml?style=flat-square&label=CI" alt="CI" /></a>
  <img src="https://img.shields.io/badge/version-v0.2.0-green?style=flat-square" alt="Version" />
  <a href="https://discord.gg/P279TT4ZCp"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord" /></a>
</p>

---

Corail is a Python runtime that powers every AI agent in the [Recif](https://github.com/recif-platform) platform. Each agent runs as its own container with its own model, tools, memory, and skills -- like corals growing independently on a reef. Corail handles the full lifecycle: receiving input through a channel, running it through a reasoning strategy backed by an LLM, calling tools, persisting memory, and streaming the response back via SSE.

---

## Quick Start

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


### Install

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


```bash
pip install corail
```

### Docker

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


```bash
docker run -p 8000:8000 -p 8001:8001 \
  -e CORAIL_MODEL_TYPE=openai \
  -e CORAIL_MODEL_ID=gpt-4o \
  -e CORAIL_STRATEGY=agent-react \
  -e OPENAI_API_KEY=sk-... \
  ghcr.io/recif-platform/corail:latest
```

### Chat

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


```bash
curl -N http://localhost:8000/api/v1/agents/my-agent/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, what can you do?"}'
```

### Local Development

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


```bash
uv sync && make dev
```

---

## Key Features

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


- **7 LLM providers** -- OpenAI, Anthropic, Google AI, Vertex AI (native), Ollama, AWS Bedrock, Stub. Registry pattern with lazy loading.
- **Multi-channel I/O** -- REST API with SSE streaming, WebSocket, Slack, Google Chat, CLI. Pluggable channel adapters.
- **Agentic RAG** -- Retrieval-augmented generation with pgvector, multi-source knowledge bases, and semantic chunking.
- **Tool registry** -- HTTP, CLI, MCP, and built-in tools (web search). Declarative config via JSON or CRDs.
- **Memory and storage** -- In-memory or PostgreSQL-backed conversation persistence. Semantic memory with pgvector.
- **Evaluation scorers** -- 14 MLflow GenAI scorers (correctness, safety, relevance, tool-call accuracy). Auto-eval on production traffic.
- **SSE streaming** -- Real-time token streaming with thinking block support and AG-UI structured content (charts, code, tables).
- **Built-in guards** -- Prompt injection detection, PII masking, secret/credential blocking on every request.

---

## Architecture

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


```
                                +---------------------+
                                |     Recif API       |
                                |     (Go, :8080)     |
                                +----------+----------+
                                           |
                              creates via  |  CRDs
                              Operator     |
                                           v
+--------------------------------------------------------------------------+
|                        Corail Agent Container                            |
|                                                                          |
|   +----------+    +-----------+    +------------+    +-----------+       |
|   | Channel  |--->| Pipeline  |--->|  Strategy  |--->|   LLM     |       |
|   | REST/WS  |    | (Guards)  |    | ReAct/RAG  |    | Provider  |       |
|   +----------+    +-----------+    +-----+------+    +-----------+       |
|       :8000                              |                               |
|                          +---------------+---------------+               |
|                          |               |               |               |
|                          v               v               v               |
|                     +---------+    +---------+    +------------+         |
|                     |  Tools  |    | Memory  |    | Knowledge  |         |
|                     | HTTP/MCP|    | pgvector|    | Base (RAG) |         |
|                     +---------+    +---------+    +------------+         |
|                                                                          |
|   +----------------+                                                     |
|   | Control Server |  <--- Recif Operator (config, reload, pause)        |
|   |     :8001      |                                                     |
|   +----------------+                                                     |
+--------------------------------------------------------------------------+
```

---

## LLM Providers

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Provider | `CORAIL_MODEL_TYPE` | Example Model |
|----------|---------------------|---------------|
| OpenAI | `openai` | `gpt-4o` |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` |
| Google AI | `google-ai` | `gemini-2.5-flash` |
| Vertex AI | `vertex-ai` | `gemini-2.5-flash` |
| Ollama | `ollama` | `qwen3.5:35b` |
| AWS Bedrock | `bedrock` | `anthropic.claude-sonnet-4-20250514-v1:0` |
| Stub | `stub` | `stub-echo` |

---

## Configuration

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Variable | Default | Description |
|----------|---------|-------------|
| `CORAIL_MODEL_TYPE` | `stub` | LLM provider |
| `CORAIL_MODEL_ID` | `stub-echo` | Model identifier |
| `CORAIL_STRATEGY` | `agent-react` | Reasoning strategy: `agent-react`, `simple`, `rag` |
| `CORAIL_CHANNEL` | `rest` | I/O channel: `rest`, `websocket`, `slack`, `cli` |
| `CORAIL_SYSTEM_PROMPT` | `You are a helpful assistant.` | Agent system prompt |
| `CORAIL_STORAGE` | `memory` | Persistence: `memory`, `postgresql` |
| `CORAIL_PORT` | `8000` | User-facing server port |
| `CORAIL_CONTROL_PORT` | `8001` | Control plane port |

See the full [environment variable reference](https://recif-platform.github.io/docs) in the documentation.

---

## Related Repositories

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Repository | Description |
|------------|-------------|
| [recif](https://github.com/recif-platform/recif) | Go API + Next.js dashboard -- the control tower |
| [recif-operator](https://github.com/recif-platform/recif-operator) | Kubernetes operator -- turns Agent CRDs into running containers |
| [helm-charts](https://github.com/recif-platform/helm-charts) | Helm chart for one-command platform installation |

---

## Roadmap

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


### Runtime Core

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Feature | Status | Description |
|---------|:------:|-------------|
| LLM providers (7) | 🟡 | Vertex AI + Ollama tested in prod. OpenAI, Anthropic, Google AI, Bedrock, Stub implemented but not battle-tested |
| Multi-channel I/O | 🟢 | REST + SSE, WebSocket, Slack, Google Chat, CLI |
| Tool registry | 🟢 | HTTP, CLI, MCP, builtins. Declarative JSON/CRD config |
| Agentic RAG | 🟢 | pgvector retrieval, semantic chunking, KB priority rules |
| Memory + storage | 🟢 | In-memory or PostgreSQL, conversation persistence |
| Guards | 🟢 | Prompt injection detection, PII masking, secret blocking |
| Eval scorers | 🟢 | 14 MLflow GenAI scorers, auto-eval on production traces |
| AG-UI rendering | 🟢 | Structured content: charts, code, tables, HTML preview |

### Strategies

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Strategy | Status | Description |
|----------|:------:|-------------|
| `simple` | 🟢 | Single-turn, no tool use |
| `agent-react` | 🟡 | ReAct loop with native tool calling. Works but needs hardening (retry logic, edge cases). |
| `react-v2` | 🟢 | Prompt-based tool calling (for models without native support) |
| `rag` | 🟢 | Retrieval-augmented generation with KB search |
| `assistant` | 🟠 | Autonomous assistant — task planning, multi-step execution, self-correction, parallel tool use. Self-improving agent that learns from interactions. Inspired by OpenClaw/Claude Code. The core is intentionally kept simple for now — the real intelligence layer is the next big milestone. |

### Planned

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


| Feature | Status | Description |
|---------|:------:|-------------|
| Streaming tool calls | 🟡 | Stream tool results as they complete (not wait for all) |
| Agent-to-agent delegation | 🔴 | One agent delegates subtasks to specialized agents |
| Long-running tasks | 🔴 | Background task execution with progress callbacks |
| Plugin system | 🔴 | Third-party strategy/channel/tool plugins via entry points |
| Voice channel | 🔴 | Real-time voice I/O with speech-to-text and text-to-speech |

> 🟢 Done  🟠 In progress  🟡 Designed  🔴 Planned

---

## Contributing

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Write tests for new functionality
4. Ensure `make lint` and `make test` pass
5. Submit a pull request

Follow the existing code patterns: registry-based factories, abstract base classes, lazy imports, and structured logging with `structlog`.

---

## Links

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


- [Documentation](https://recif-platform.github.io/docs)
- [Discord](https://discord.gg/P279TT4ZCp)
- [GitHub Organization](https://github.com/recif-platform)

---

## License

<p align="center">
  <a href="https://youtu.be/9n4S8NRI1zA"><img src="https://img.youtube.com/vi/9n4S8NRI1zA/maxresdefault.jpg" alt="Demo" width="560" /></a>
  <br/><em>Watch the demo (2 min)</em>
</p>


[Apache License 2.0](LICENSE) -- Copyright 2026 Sciences44.
