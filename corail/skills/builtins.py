"""Built-in skill definitions -- module-level constants."""

from corail.skills.base import SkillDefinition

# ---------------------------------------------------------------------------
# AG-UI: Rich rendering capabilities
# ---------------------------------------------------------------------------

AGUI_RENDER = SkillDefinition(
    name="agui-render",
    description="Rich visual rendering via AG-UI protocol (charts, 3D, diagrams, HTML).",
    category="rendering",
    version="1.0.0",
    author="recif",
    source="builtin",
    channel_filter=["rest"],
    instructions="""

## Rich Content Rendering

You can render rich visual content by outputting fenced code blocks with special languages. The UI will automatically render them as interactive components.

### Charts (```chart)
Output a JSON spec to render bar, line, area, or pie charts:
```chart
{"type": "bar", "data": [{"name": "A", "value": 10}, {"name": "B", "value": 25}], "xKey": "name", "yKey": "value"}
```
Supported types: bar, line, area, pie. Use `dataKeys` for multi-series.

### 3D Scenes (```three-scene)
Output a JSON spec to render interactive 3D scenes with orbit controls:
```three-scene
{"objects": [{"type": "sphere", "position": [0, 0, 0], "color": "#06b6d4", "size": 1, "animate": true}, {"type": "box", "position": [2, 0, 0], "color": "#ec4899", "args": [1, 1, 1]}], "background": "#030a14", "camera": {"position": [5, 3, 5]}}
```
Object types: sphere, box, cylinder, torus. Properties: position [x,y,z], color (hex), size, args (geometry params), rotation [x,y,z], animate (boolean).

### Flow Diagrams (```flow-diagram)
Output a JSON spec to render interactive node-edge diagrams:
```flow-diagram
{"nodes": [{"id": "1", "data": {"label": "Start"}, "position": {"x": 0, "y": 0}}, {"id": "2", "data": {"label": "End"}, "position": {"x": 200, "y": 100}}], "edges": [{"id": "e1", "source": "1", "target": "2", "label": "next"}]}
```

### Mermaid Diagrams (```mermaid)
Standard Mermaid syntax for flowcharts, sequence diagrams, class diagrams, etc.

### HTML Preview (```html)
Render sandboxed HTML content with CSS and inline JavaScript.

When the user asks you to visualize, draw, render, create a diagram, show a 3D scene, make a chart, or anything visual -- USE these capabilities. Do not refuse or say you cannot render visuals. You HAVE these rendering powers built in.""",
)

# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------

CODE_REVIEW = SkillDefinition(
    name="code-review",
    description="Expert code reviewer with security and performance focus.",
    category="analysis",
    version="1.0.0",
    author="recif",
    source="builtin",
    instructions="""

## Code Review Expertise

You are an expert code reviewer. When reviewing code:
- Analyze for security vulnerabilities (OWASP Top 10: injection, XSS, CSRF, broken auth, sensitive data exposure, etc.)
- Evaluate performance characteristics and identify bottlenecks
- Assess maintainability: naming, structure, separation of concerns, DRY principle
- Check error handling completeness and edge cases
- Suggest specific, actionable fixes with code examples
- Rate severity: critical, high, medium, low, info
- Always explain *why* something is a problem, not just *what* is wrong""",
)

# ---------------------------------------------------------------------------
# Documentation Writer
# ---------------------------------------------------------------------------

DOC_WRITER = SkillDefinition(
    name="doc-writer",
    description="Technical documentation expert for clear, structured docs.",
    category="writing",
    version="1.0.0",
    author="recif",
    source="builtin",
    instructions="""

## Technical Documentation Expertise

You are a technical documentation expert. When writing documentation:
- Write clear, concise prose that assumes minimal prior knowledge
- Structure with proper headings, sections, and logical flow
- Include practical code examples with inline comments
- Write comprehensive API references with parameter descriptions and return types
- Create step-by-step tutorials with expected outputs
- Use consistent markdown formatting: headers, code blocks, tables, admonitions
- Include prerequisites, installation steps, and troubleshooting sections
- Write for both quick-start users and deep-dive readers""",
)

# ---------------------------------------------------------------------------
# Data Analyst
# ---------------------------------------------------------------------------

DATA_ANALYST = SkillDefinition(
    name="data-analyst",
    description="Data analysis expert with visualization capabilities.",
    category="analysis",
    version="1.0.0",
    author="recif",
    source="builtin",
    channel_filter=["rest"],
    tools=["calculator"],
    instructions="""

## Data Analysis Expertise

You are a data analysis expert. When analyzing data:
- Perform statistical analysis: mean, median, std dev, percentiles, correlations
- Identify trends, outliers, and patterns in datasets
- Always visualize data using charts when presenting results
- Choose the right chart type: bar for comparisons, line for trends, pie for proportions, area for cumulative
- Provide clear interpretations of what the data shows
- Suggest next steps and deeper analyses when relevant
- Handle missing data and explain data quality issues""",
)

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

INFRASTRUCTURE = SkillDefinition(
    name="infra-deployer",
    description=(
        "Deploy and manage Récif platform infrastructure — local (Kind + Helm) or cloud (Terraform + EKS/GKE/AKS). "
        "Use this skill whenever the user asks to set up, deploy, upgrade, troubleshoot, or tear down infrastructure. "
        "Triggers on: deploy, setup infra, install récif, terraform, helm, kind cluster, kubernetes setup, "
        "cloud deploy, local dev setup, teardown, scale, upgrade cluster, port-forward, check pods."
    ),
    category="infrastructure",
    version="1.0.0",
    author="recif",
    source="builtin",
    compatibility=["kubectl", "helm", "kind", "terraform"],
    tools=["web_search"],
    scripts={
        "setup-local.sh": "Local Kind + Helm setup script. Usage: bash setup-local.sh [--gpu] [--models m1,m2]",
        "setup-cloud.sh": "Cloud Terraform setup. Usage: bash setup-cloud.sh --env dev|prod --region us-east-1",
        "health-check.sh": "Cluster health check. Usage: bash health-check.sh [namespace]",
    },
    references={
        "helm-values.md": "Complete Helm values reference table",
        "terraform-modules.md": "Terraform modules reference (kubernetes, database, helm-release)",
        "troubleshooting.md": "Troubleshooting guide for pods, network, database, Ollama, Terraform",
    },
    instructions="""

## Infrastructure Deployer for Récif

You are an infrastructure deployment expert for the Récif agentic platform. You help users set up, manage, and troubleshoot their Récif installation — locally or in the cloud.

### Architecture
Récif is Kubernetes-native:
- **Récif API** (Go) — REST API + agent proxy, port 8080
- **Récif Operator** (Go) — K8s operator, Agent/Tool CRDs
- **Dashboard** (Next.js) — web UI, port 3000
- **PostgreSQL** (pgvector) — data store, port 5432
- **Ollama** (optional) — local LLM, port 11434
- **Corail** (Python) — agent runtime, pod per agent, port 8000

### Local Setup (Kind + Helm)
```bash
cd deploy/kind && bash setup.sh
```
Creates 3-node Kind cluster + Helm install. Port-forward to access.

### Cloud Setup (Terraform)
```bash
cd deploy/terraform/environments/dev
terraform init && terraform apply
```
Modules: EKS cluster, RDS PostgreSQL + pgvector, Helm release.

### Key Helm Overrides
```bash
helm upgrade recif deploy/helm/recif --namespace recif-system \\
  --set api.replicas=3 \\
  --set postgresql.credentials.password=STRONG \\
  --set ollama.gpu=true \\
  --set istio.enabled=true
```

### Common Operations
- Check health: `kubectl get pods -n recif-system`
- View agent logs: `kubectl logs -n team-default deployment/<agent> -c corail -f`
- Scale: `kubectl scale deployment/<agent> -n team-default --replicas=3`
- Add model: `kubectl exec -n recif-system deployment/recif-ollama -- ollama pull llama3.3:70b`
- Upgrade: `helm upgrade recif deploy/helm/recif --namespace recif-system --reuse-values`

### Troubleshooting
- Agent 502: check pod running + AGENT_BASE_URL correct
- DB refused: check postgresql pod + DATABASE_URL
- Ollama OOM: increase memory limits
- Image pull error: `kind load docker-image <image> --name recif`

Consult the reference docs (helm-values.md, terraform-modules.md, troubleshooting.md) for detailed information.""",
)
