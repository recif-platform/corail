# ADK Example Agent

A simple agent using the ADK (Agent Development Kit) framework.

## Setup

```bash
# From the corail repo root
cd examples/adk-agent

# Register with Récif
recif register -f agent.yaml

# Test via curl
curl -X POST http://localhost:8000/api/v1/agents/<agent-id>/chat \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the ADK framework?"}'
```

## Evaluate

```bash
recif eval <agent-id> --dataset examples/adk-agent/dataset.json
```
