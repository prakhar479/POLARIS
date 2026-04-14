# Polaris - Modular Self-Adaptive Systems Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Polaris** is a clean, modular framework for building self-adaptive systems (Implementation of ![POLARIS](./POLARIS_Framework.pdf)). It provides a simple default experience while allowing full customization of every component.

## Quick Start

### Installation

```bash
cd polaris
pip install -c requirements/constraints.txt -e .
```

### Dependency and Version Standardization

Polaris now treats `pyproject.toml` as the single source of truth for package
metadata and uses shared constraints for local dev, CI, and Docker builds.

```bash
# Core runtime install
pip install -c requirements/constraints.txt -e .

# Development install (includes optional LLM/dashboard/connector deps used in CI)
pip install -c requirements/constraints.txt -e .[dev]

# Full-feature runtime install (all optional connectors/providers)
pip install -c requirements/constraints.txt -e .[all]

# Verify dependency policy guard
make dependency-policy-check
```

### Simple Usage (2 lines!)

```python
from polaris import Polaris

# Use all defaults
polaris = Polaris()
await polaris.run()
```

### With Your System

```python
from polaris import Polaris
from your_system import YourConnector

polaris = Polaris(
    connectors=[YourConnector()]
)
await polaris.run()
```

### Custom Strategy

```python
from polaris import Polaris, AdaptationStrategy, ThresholdReactiveStrategy

# Use threshold strategy
polaris = Polaris(
    strategy=ThresholdReactiveStrategy(
        thresholds={'cpu_usage': {'high': 80.0}},
        cooldown_seconds=60
    )
)
```

## Supported Connectors

Polaris includes built-in connectors for common exemplar systems:

- **SWIM**: Connects to SWIM (Simulated Web Infrastructure Manager) via TCP.
- **Wildfire**: Connects to the WildFire multi-UAV fire suppression simulation via REST API.
- **Kubernetes**: Connects to Kubernetes clusters (natively or via kubeconfig) to monitor pods and scale deployments. Requires `pip install kubernetes`.

See [CONFIGURATION.md](./CONFIGURATION.md#connectors) for detailed configuration and metrics/actions for each connector.

## Wildfire simulation (quick run)

1) Start the Wildfire adapter (Flask REST API) so `http://localhost:5000/health` is reachable.

2) Run Polaris with the Wildfire config:

```bash
polaris --config config/wildfire.yaml
```

### Using OpenRouter for LLM strategies

Polaris supports OpenRouter via an OpenAI-compatible client.

- Set `OPENROUTER_API_KEY`
- In YAML, set `provider: openrouter` under `strategy.params` and/or `meta_learner.llm`

See [CONFIGURATION.md](./CONFIGURATION.md#using-openrouter-openai-compatible-gateway) for details.

## Factory-based Registration

Polaris uses factory registries to map configuration type strings to concrete connector and strategy implementations.

- Connector type strings come from `systems[].connector_type` and must match a registered connector factory.
- Strategy type strings come from `strategy.type` and must match a registered strategy factory.

Built-in factories are registered at import time in `polaris.core.factories`. To add your own connector/strategy type without changing Polaris core, register a factory in your code and use the new type string in YAML.

See [CONFIGURATION.md](./CONFIGURATION.md#factory-based-registration-connectors--strategies) for the full pattern and examples.

## CLI Usage

### Basic Commands

```bash
# Run with default config
polaris --config config/default.yaml

# Launch interactive dashboard
polaris --config config/default.yaml --dashboard

# Dashboard + interactive commands in split-screen mode
polaris --config config/default.yaml --both

# Dry-run mode (evaluate decisions without executing actions)
polaris --config config/default.yaml --dry-run

# Run diagnostics (config/env/optional dependencies)
polaris doctor --config config/default.yaml

# Show version
polaris --version
```

### Logging Options

Polaris supports two logging formats with flexible configuration:

```bash
# Human-readable logging (great for development)
polaris --config config/default.yaml --log-format human

# Structured JSON logging (great for production)
polaris --config config/default.yaml --log-format structured

# Export logs to file
polaris --config config/default.yaml --export-logs ./session.log

# Set log level
polaris --config config/default.yaml --log-level DEBUG

# Combine options
polaris --config config/default.yaml --log-format human --export-logs debug.log --log-level DEBUG
```

### Logging Formats

**Human-Readable Format** (colorized, easy to read):
```
10:30:45 INFO    [polaris]    System connected | system_id=swim-001
10:30:46 WARNING [strategy]   High CPU usage detected | cpu_percent=85.2, threshold=80.0
```

**Structured Format** (JSON, machine-parseable):
```json
{"timestamp":"2024-01-15T10:30:45Z","level":"INFO","component":"polaris","message":"System connected","context":{"system_id":"swim-001"}}
```

### Configuration

Configure logging in your `config.yaml`:

```yaml
observability:
  logging:
    type: "human"          # "structured" or "human"
    level: "INFO"          # DEBUG, INFO, WARNING, ERROR
    console: true          # Show in terminal
    file: true             # Save to file
    file_path: "./logs/polaris.log"
    use_colors: true       # Colors for human format (console only)
```

CLI options override config file settings.

### LLM Strategies and Provider Selection

Polaris supports multiple LLM-powered strategies:

- `llm_reasoning`: Single-shot LLM decision-making with reasoning
- `agentic_llm`: Iterative tool-using loop where the LLM calls registered tools (built-ins by default; extensible via tool factories)
- `thread_agentic`: Recursive THREAD-style join-synchronized reasoning with dynamic child-thread spawning
- `multi_agent`: Committee of three specialized agents (Diagnostician → Planner → SafetyValidator); each agent can use its own LLM provider, temperature, and prompt
- `hybrid`: Combine any strategies (including LLM-based ones) with configurable selection mode

You can choose the LLM provider per strategy. Supported canonical providers:
`google`, `openai`, `openrouter`, `groq`, and `ollama`.

LLM strategies run in strict mode: model output must be schema-valid JSON and use
`actions` lists only. A strict-contract violation fails that system iteration while
other systems continue running.

Example (multi-agent strategy with per-agent LLM config):

```yaml
strategy:
  type: multi_agent
  params:
    provider: google           # Shared default provider
    temperature: 0.1
    steps_limit: 3             # Max reasoning steps per agent (new)
    system_description: "SWIM web application server pool"
    diagnostician:
      temperature: 0.0         # Deterministic detection
      steps_limit: 5           # More steps for deep diagnosis
    planner:
      provider: openai         # Stronger model for planning
      temperature: 0.2
      tools:                   # Task-specific tool restriction
        - predict_outcome
        - get_action_history
    validator:
      temperature: 0.0         # Conservative safety check
    resilience:
      rps: 1
      burst: 2
      max_retries: 4
```

Example (agentic LLM strategy):

```yaml
strategy:
  type: agentic_llm
  params:
    provider: google   # or "openai"/"openrouter"/"groq"/"ollama"
    steps_limit: 3
    temperature: 0.1
    max_tool_result_chars: 1200
    native_tools_unsupported_policy: skip_cycle   # skip_cycle | json_fallback | strict_fail
    tools:
      enabled:
        - get_recent_states
        - summarize_metric_trends
        - get_world_model_insights
        - predict_outcome
        - get_action_history
        - list_supported_actions
    resilience:
      rps: 2
      burst: 4
      concurrency: 4
      max_retries: 4
      base_backoff_ms: 200
      max_backoff_ms: 4000
```

Example (THREAD recursive strategy):

```yaml
strategy:
  type: thread_agentic
  params:
    provider: google   # or "openai"/"openrouter"/"groq"/"ollama"
    steps_limit: 4
    max_thread_depth: 3
    max_total_threads: 16
    child_timeout_seconds: 20.0
    max_repeated_spawns: 2
    tools:
      enabled:
        - get_recent_states
        - summarize_metric_trends
        - predict_outcome
        - list_supported_actions
```

For `llm_reasoning`, `agentic_llm`, `thread_agentic`, and `multi_agent`, specify provider under `strategy.params.provider`.
For `hybrid`, each sub-strategy config lives under `strategy.params.strategies[].params` and can set its own provider.

The `agentic_llm` tools use the built-in Knowledge Store and World Model, plus a connector-aware tool `list_supported_actions` that prefers the active Connector's `get_supported_actions()` if available, and falls back to historical inference.
Tool names configured under `strategy.params.tools` are validated against the registered tool factory names.
When `strategy.params.native_tools` references a Polaris tool name, that tool must also be enabled in `strategy.params.tools`.

## Docker

Polaris ships with a root multi-target Dockerfile and a compose file for common
runtime profiles.

### 1) Prerequisites

- Docker Engine / Docker Desktop
- Docker Compose v2 (`docker compose` command)
- A Polaris config file (for example `config/default.yaml`)

Prepare host folders used as bind mounts:

```bash
mkdir -p logs metrics data
```

### 2) Image Targets

- `core`: lean runtime image for threshold/basic deployments
- `full-feature`: runtime image with all optional Polaris extras (`.[all]`)
- `ci`: development/test image with `.[dev]`

Build examples:

```bash
docker build --target core -t polaris:core .
docker build --target full-feature -t polaris:full-feature .
docker build --target ci -t polaris:ci .
```

### 3) Quick Start (Core Image)

Run Polaris in dry-run mode with mounted config and outputs:

```bash
docker run --rm \
  -v "$(pwd)/config:/app/config:ro" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/metrics:/app/metrics" \
  -v "$(pwd)/data:/app/data" \
  polaris:core --config /app/config/default.yaml --dry-run
```

Useful smoke checks:

```bash
docker run --rm polaris:core --version
docker run --rm polaris:core doctor --config /app/config/default.yaml
```

### 4) Compose Profiles

The repository includes `docker-compose.yml` with these profiles:

- Default (`polaris`): core image runtime
- `full` (`polaris-full`): full-feature runtime
- `redis` (`redis`): Redis sidecar for distributed event bus setups
- `ollama` (`ollama`): local Ollama service for local-model workflows

Examples:

```bash
# Core runtime
docker compose up polaris

# Core runtime + Redis
docker compose --profile redis up polaris redis

# Full-feature runtime
docker compose --profile full up polaris-full

# Full-feature runtime + Ollama service
docker compose --profile full --profile ollama up polaris-full ollama
```

### 5) Environment Variables

For LLM-backed strategies/meta-learning, pass provider credentials at runtime.

Common examples:

- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`

Example:

```bash
docker run --rm \
  -e OPENAI_API_KEY="<your-key>" \
  -v "$(pwd)/config:/app/config:ro" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/metrics:/app/metrics" \
  -v "$(pwd)/data:/app/data" \
  polaris:full-feature --config /app/config/default.yaml
```

### 6) Health and Readiness

Container images use Polaris Doctor as the image healthcheck command:

```bash
polaris doctor --config ${POLARIS_CONFIG:-/app/config/default.yaml}
```

This healthcheck fails on true config/runtime failures and allows optional-feature
warnings (for example, `rich` absent in `core`) so healthy minimal deployments are
not marked unhealthy.

### 7) Running Dashboard and Interactive Modes

Dashboard or combined interactive modes require a TTY:

```bash
docker run -it --rm \
  -v "$(pwd)/config:/app/config:ro" \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/metrics:/app/metrics" \
  -v "$(pwd)/data:/app/data" \
  polaris:full-feature --config /app/config/default.yaml --both
```

For non-interactive production deployments, prefer standard mode without `-it`.

### 8) Troubleshooting

- `Config file not found`: confirm mount path and `--config` path inside container (`/app/config/...`).
- `Missing credentials`: export required env vars for your selected LLM provider.
- `Permission denied` writing logs/metrics/data: ensure host directories exist and are writable.
- `Redis bus unreachable`: when using compose, use `redis://redis:6379` (service name), not localhost.
- `Ollama unreachable`: when using compose, use `http://ollama:11434` from Polaris config.

## Architecture

Polaris follows a modular, interface-driven design:

- **`abstractions/`** - Core interfaces (extend these to customize)
- **`core/`** - Framework orchestration (use, don't modify)
- **`strategies/`** - Adaptation logic (default: threshold-based)
- **`world_model/`** - System behavior modeling
- **`knowledge/`** - Historical data storage
- **`meta_learner/`** - Intelligent parameter optimization
- **`infrastructure/`** - Logging, metrics, configuration

## Extension Points

### 1. Custom Connector

Integrate your system:

```python
from polaris import Connector, SystemState

class MyConnector(Connector):
    async def collect_telemetry(self) -> SystemState:
        # Collect your system's metrics
        return SystemState(...)

    async def execute_action(self, action):
        # Execute adaptation on your system
        pass
```

### 2. Custom Strategy

Implement custom adaptation logic:

```python
from polaris import AdaptationStrategy

class MyStrategy(AdaptationStrategy):
    async def assess(self, state, context):
        # Your decision logic
        if should_adapt():
            return AdaptationAction(...)
        return None

    def get_tunable_parameters(self):
        # Declare what can be optimized
        return {...}
```

### 3. Custom Tool

Register a custom tool so it can be enabled in strategy config:

```python
from polaris.tools import register_tool_factory
from polaris.tools.base import Tool


class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "Example custom tool"

    async def execute(self, args, state, context, deps):
        return {"ok": True}


register_tool_factory("my_custom_tool", MyTool)
```

### 4. Swap Any Component

```python
polaris = Polaris(
    strategy=MyStrategy(),
    world_model=MyWorldModel(),
    knowledge_store=PostgreSQLStore(),
    logger=MyLogger()
)
```

## Testing and CI/CD

### Local Development

The project includes a comprehensive CI/CD pipeline with automated testing, formatting, and quality checks.

```bash
# Setup development environment
./scripts/setup-ci.sh

# Install pre-commit hooks
make install-hooks

# Run full CI pipeline locally
make ci

# Quick pre-commit check
make pre-commit

# Available commands
make help
```

### Running Tests

```bash
# Run test suite
make test

# Run with coverage
make test-all

# Verbose output
make test-verbose
```

### Code Quality

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Type checking
make type-check
```

#### Workflow Files

- `.github/workflows/ci-cd.yml` - Main CI/CD pipeline
- `.github/workflows/quality-gate.yml` - PR quality validation
- `.github/workflows/auto-commit.yml` - Auto-format and commit
- `.github/workflows/merge-check.yml` - Merge validation

#### Pre-commit Hooks

Local development includes pre-commit hooks for:

- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- pytest validation
- Security scanning

Setup with:
```bash
make install-hooks
```

## Testing

Run tests with:

```bash
python -m pytest tests/
```

or use the provided script:

```bash
python run_tests.py
```

Validate config/schema/env wiring before runtime:

```bash
python -m polaris.cli doctor --config config/default.yaml
```

For comprehensive testing and quality checks:

```bash
make ci
```
