# Polaris - Modular Self-Adaptive Systems Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Polaris** is a clean, modular framework for building self-adaptive systems (Implementation of ![POLARIS](./POLARIS_Framework.pdf)). It provides a simple default experience while allowing full customization of every component.

## Quick Start

### Installation

```bash
cd polaris
pip install -e .
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
- **Kubernetes**: Connects to Kubernetes clusters (natively or via kubeconfig) to monitor pods and scale deployments.

See [CONFIGURATION.md](./CONFIGURATION.md#connectors) for detailed configuration and metrics/actions for each connector.

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

- `llm_reasoning`: single-shot LLM decision-making
- `agentic_llm`: iterative tool-using loop with the LLM calling safe tools
- `multi_agent`: advanced multi-agent committee (Diagnostician -> Planner -> Validator)
- `hybrid`: combine strategies, including LLM-based ones

You can choose the LLM provider per strategy. Supported providers: `google` (Gemini, default) and `openai`.

Example (agentic LLM strategy):

```yaml
strategy:
  type: agentic_llm
  agentic_llm:
    provider: google   # or "openai"
    steps_limit: 3
    temperature: 0.1
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

For `llm_reasoning`, specify the provider under `strategy.llm_reasoning.provider`. For `hybrid`, each `llm_reasoning` sub-strategy can specify its own provider under its block.

The `agentic_llm` tools use the built-in Knowledge Store and World Model, plus a connector-aware tool `list_supported_actions` that prefers the active Connector's `get_supported_actions()` if available, and falls back to historical inference.

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

### 3. Swap Any Component

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
pytest tests/
```

or use the provided script:

```bash
python run_tests.py
```

For comprehensive testing and quality checks:

```bash
make ci
```
