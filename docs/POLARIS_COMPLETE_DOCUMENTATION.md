# POLARIS Complete Documentation

This document is a consolidated, code-verified reference for Polaris users and developers. It combines and normalizes information from the existing repository documentation and the implementation itself.

Companion guides in this docs folder:

- Strategy deep dive: [STRATEGIES_DETAILED.md](STRATEGIES_DETAILED.md)
- SWIM end-to-end walkthrough: [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)
- Default baseline config: [config/default.yaml](../config/default.yaml)

## Index

- [Getting Started](#1-what-polaris-is)
- [CLI and Runtime Architecture](#5-cli-reference)
- [Configuration and Connectors](#7-configuration-model-code-verified)
- [Strategy and LLM Contracts](#9-strategy-system)
- [Observability and Operations](#15-observability)
- [Reproducibility and Examples](#20-reproducibility-checklist-tool-paper-friendly)
- [Troubleshooting](#22-troubleshooting)
- [Extensibility Guide](#23-developer-extension-guide-and-snippets)
- [Docker Guide](#25-docker-operations)
- [Architecture Map](#26-package-architecture-map)
- [Testing and CI/CD](#27-testing-and-cicd)
- [Citation and Validation](#28-suggested-citation-block-template)
- Companion docs: [STRATEGIES_DETAILED.md](STRATEGIES_DETAILED.md), [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)

## 1. What Polaris Is

Polaris is a modular self-adaptive systems framework for MAPE-K style control loops:

1. Monitor: collect telemetry from one or more managed systems.
2. Analyze: derive state trends and insights using history and optional world models.
3. Plan: decide adaptation actions using pluggable strategies.
4. Execute: validate and execute actions through connectors.
5. Knowledge: store states/actions and optionally optimize strategy parameters over time.

Polaris is designed for:

- Runtime adaptation experiments and research prototypes.
- Comparative strategy studies (rule-based vs LLM-based vs hybrid committee-based).
- Extensible integration of new system connectors and adaptation strategies.

## 2. Core Features

- Multi-system monitoring with per-system collection cadence controls.
- Pluggable connectors: SWIM, Wildfire, Kubernetes, SUAVE.
- Pluggable strategies:
  - `threshold`
  - `llm_reasoning`
  - `agentic_llm`
  - `thread_agentic`
  - `multi_agent`
  - `hybrid`
- Contract-first strict action validation for LLM-backed strategies.
- Hot-reload of strategy parameters and selected meta-learner settings.
- Optional metrics collection and periodic exports (JSON/CSV).
- Optional dashboard and interactive CLI (with `rich`).
- Built-in diagnostics command (`doctor`) for config/env/dependency checks.

## 3. Installation and Setup

## 3.1 Requirements

- Python >= 3.11
- Linux/macOS/Windows (Python-compatible runtime)

## 3.2 Install

Recommended install from repo root (constraints-aligned with CI and Docker):

```bash
pip install -c requirements/constraints.txt -e .
```

Common install profiles:

```bash
# Core runtime
pip install -c requirements/constraints.txt -e .

# Development profile (tests, lint, optional integrations)
pip install -c requirements/constraints.txt -e .[dev]

# Full runtime with all optional providers/connectors
pip install -c requirements/constraints.txt -e .[all]

# Verify dependency policy guard
make dependency-policy-check
```

CLI entry points are available as:

- Console script: `polaris`
- Module form: `python -m polaris.cli`

## 3.3 Optional Dependencies

- Dashboard and interactive UI: install extra `.[dashboard]`
- LLM providers: install extra `.[llm]`
- Kubernetes/Redis connector helpers: install extra `.[connectors]`
- SUAVE connector support (`roslibpy`): install extra `.[suave]`
- Everything optional: install extra `.[all]`

## 4. Quick Start

Primary baseline configuration file:

- [config/default.yaml](../config/default.yaml)

Run with default config:

```bash
polaris --config config/default.yaml
```

Check environment/config wiring:

```bash
polaris doctor --config config/default.yaml
```

Run dashboard:

```bash
polaris --dashboard --config config/default.yaml
```

Run interactive CLI:

```bash
polaris --interactive --config config/default.yaml
```

Run split-screen dashboard + interactive pane:

```bash
polaris --both --config config/default.yaml
```

Source-of-truth references for CLI/config details:

- `README.md`
- `CLI_USAGE.md`
- `CONFIGURATION.md`

If you are starting with SWIM, run the full tutorial in:

- [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)

## 5. CLI Reference

The main CLI supports:

```text
--config/-c
--dashboard/-d
--interactive/-i
--both/-b
--no-clear
--version/-v
--export-logs/-e
--log-format {structured,human}
--log-level {DEBUG,INFO,WARNING,ERROR}
--metrics-export/-m
--metrics-format {json,csv,both}
--metrics-experiment
--disable-metrics
--auto-export-metrics
--monitoring-interval
--dry-run
```

Subcommands:

- `doctor`: diagnostics for runtime/config/env/dependencies.
- `init`: interactive config starter generator.

Examples:

```bash
# version
python -m polaris.cli --version

# diagnostics (strict turns warnings into failures)
python -m polaris.cli doctor --config config/default.yaml --strict

# generate starter config
python -m polaris.cli init --output config/custom.yaml

# dry-run mode: evaluate actions but do not execute
python -m polaris.cli --config config/default.yaml --dry-run
```

## 6. Runtime Architecture

At runtime, `Polaris` orchestrates:

- Connector registry and connector contracts.
- Monitoring loop.
- Adaptation pipeline.
- Optional config hot-reloader.
- Optional meta-learning loop.
- Optional metrics export loop.

High-level flow per cycle:

1. Optional config change detection and hot-apply.
2. For each due connector:
   - collect telemetry
   - store state
   - update world model
   - publish telemetry event
   - run adaptation pipeline (assess -> validate -> execute)
3. Emit monitoring metrics and sleep to maintain cadence.

## 7. Configuration Model (Code-Verified)

Polaris uses strict Pydantic validation (`extra='forbid'` in key models), so unsupported keys can fail startup.

Top-level keys:

- `systems` (list)
- `strategy` (object)
- `world_model` (object, optional)
- `knowledge_store` (object, optional)
- `meta_learner` (object, optional)
- `observability` (object, optional)
- `monitoring` (object, optional)
- `plugin_imports` (list of import strings, optional)
- `max_concurrent_connectors` (int > 0, optional)

Environment variable substitution supports `${VAR}` in YAML; missing vars fail config load.

Important migration rule:

- `knowledge` key is removed. Use `knowledge_store`.

## 8. Systems and Connectors

Each system entry requires:

- `id` (non-empty)
- `connector_type` (registered type)
- optional `enabled` (default true)
- optional `connection` dict
- optional `monitoring.collection_interval > 0`
- optional `action_policy` rules

Built-in connector types:

- `swim`
- `wildfire`
- `kubernetes`
- `suave`

## 8.1 SWIM Connector

- Transport: TCP commands.
- Typical telemetry includes server count, dimmer, response/throughput metrics.
- Supported actions (canonical):
  - `scale_up`
  - `scale_down`
  - `set_dimmer`
- Aliases are normalized internally (for example `add_server` -> `scale_up`).

## 8.2 Wildfire Connector

- Transport: HTTP REST via adapter.
- Typical telemetry includes timestep, MR metrics, burning cells and ratio.
- Supported actions:
  - `wildfire_reset`
  - `wildfire_pause`
  - `wildfire_resume`
  - `wildfire_step`
  - `wildfire_move`
  - `wildfire_batch_actions`

Quick run reminder:

1. Start the Wildfire adapter and ensure `http://localhost:5000/health` is reachable.
2. Run Polaris with Wildfire config:

```bash
polaris --config config/wildfire.yaml
```

## 8.3 Kubernetes Connector

- Requires `kubernetes` Python client.
- Supports in-cluster and kubeconfig usage.
- Telemetry includes pod/deployment counts and health indicators.
- Supported actions:
  - `scale_deployment`
  - `restart_deployment`

## 8.4 SUAVE Connector

- Transport: ROS2/rosbridge integration.
- Mission lifecycle via `/task/request` and `/task/cancel`.
- Adaptation via mode changes on function nodes.
- Supported action families:
  - `start_mission`
  - `stop_mission`
  - `change_mode`

## 9. Strategy System

For complete strategy details and practical templates, see:

- [STRATEGIES_DETAILED.md](STRATEGIES_DETAILED.md)

Polaris uses canonical strategy schema:

```yaml
strategy:
  type: <strategy_name>
  params: ...
```

Legacy type-keyed blocks are diagnosed as invalid by `doctor`.

## 9.1 `threshold`

Rule-based adaptation from metric bounds.

Key params:

- `thresholds.<metric>.high/low`
- `cooldown_seconds`
- optional `action_templates`

Validation includes `high > low` when both exist.

## 9.2 `llm_reasoning`

Single-shot LLM decision with strict JSON response requirements.

Contract requirements:

- Connector-supported action contract must exist.
- LLM output must be valid JSON object.
- Must contain boolean `needs_adaptation`.
- Must contain non-empty `reasoning` string.
- If adaptation is needed, must contain non-empty `actions` list.

## 9.3 `agentic_llm`

Tool-using multi-step LLM strategy.

Key params:

- `provider`
- `temperature`
- `steps_limit`
- `decision_cooldown_seconds`
- `tools.enabled`
- optional `native_tools` (OpenAI-style function schema)
- optional `per_system_prompts`
- optional `resilience`

If native tool-calling is unsupported by provider, behavior follows
`native_tools_unsupported_policy` (`skip_cycle`, `json_fallback`, `strict_fail`).
If native tool response lacks tool calls, strategy falls back to JSON-text parsing path.

## 9.4 `thread_agentic`

Recursive THREAD-inspired strategy with child thread spawning and join synchronization.

Key params include recursion and payload control:

- `max_thread_depth`
- `max_total_threads`
- `child_timeout_seconds`
- `max_repeated_spawns`
- `assessment_cooldown_seconds`
- `phi_mode`, `phi_max_lines`
- `listen_token`, `return_token`

## 9.5 `multi_agent`

Committee pipeline:

1. Diagnostician
2. Planner
3. SafetyValidator

Each role supports per-agent override for:

- provider/client
- temperature
- system prompt
- max tokens
- steps limit
- allowed tools
- resilience behavior (through per-agent client settings)

## 9.6 `hybrid`

Combines sub-strategies.

Key params:

- `selection_mode` in `{first, priority, confidence}`
- `min_confidence`
- `strategies` list, each with `type`, optional `priority`, and `params`

Sub-strategies are validated recursively via `StrategyConfig`.

Recommended production pattern:

- Use threshold as first-stage deterministic guard.
- Use hybrid as the default strategy wrapper for all production recommendations.
- Keep a reactive deterministic threshold path as the first stage in hybrid.
- Use one advanced secondary branch based on system needs: `agentic_llm`, `multi_agent`, or `thread_agentic`.
- Apply cooldown to prevent cross-strategy oscillation.

## 10. Strict Action Contracts

LLM-backed strategies set `requires_system_contract = True`.

At startup, Polaris builds connector contracts from `get_capabilities()` / `get_supported_actions()`.
If no supported action set exists for a strict strategy, that system iteration fails with strict-contract violation instead of executing uncertain actions.

Action names are normalized through connector capabilities + alias mapping to reduce hallucinated or mismatched action strings.

## 11. Built-in Agent Tools

Polaris includes built-in tools used by agentic strategies:

- `get_recent_states`
- `summarize_metric_trends`
- `list_metric_fields`
- `compute_metric_math`
- `get_world_model_insights`
- `predict_outcome`
- `get_action_history`
- `list_supported_actions`
- `sleep`
- `get_system_status`

Tool sets can be constrained by strategy config (`tools.enabled`).

## 12. Knowledge Store and World Model

Knowledge store types:

- `memory`
- `sqlite`

World model:

- `statistical`
  - `window_size` (required positive int)
  - `use_kalman` (bool)

## 13. Meta-Learning

Meta-learner types:

- `statistical`
- `llm`

LLM meta-learner supports:

- periodic performance analysis and proposal generation
- optional `auto_apply`
- global and per-system prompt overrides

Hot-reload support includes selected LLM meta-learner fields (for example `auto_apply`, `temperature`, prompts) when type remains unchanged.

## 14. Hot Reload Behavior

Config file changes are monitored when Polaris runs with `config_path`.

Hot-reload applies:

- strategy parameter updates (`apply_config_update` path)
- selected meta-learner updates for LLM meta-learner instances

Hot-reload does not fully swap strategy class/type at runtime; strategy-type changes require restart.

## 15. Observability

Logging:

- `structured` or `human`
- configurable level, console, file output
- dashboard captures concise recent logs separately

Metrics:

- optional collector (`simple`, `prometheus`, `datadog`)
- component-level collection toggles supported by configuration checks in builder
- export utility supports JSON and CSV files

Auto-export runs when enabled and interval > 0.

## 16. Monitoring Cadence and Concurrency

Global cadence:

- `monitoring.interval_seconds` (default 30)

Per-system cadence:

- `systems[].monitoring.collection_interval`
- effective interval is `max(global_interval, collection_interval)`

Connector processing concurrency:

- bounded by `max_concurrent_connectors` (default 10)

## 17. Action Policies (Per System)

Optional per-system action injection rules:

- `append_each_cycle`
- `inject_when_no_actions`

Each policy can inject a predefined action template under controlled conditions.

## 18. Interactive CLI and Dashboard UX

Interactive CLI commands include:

- `status`
- `systems`
- `metrics [filter]`
- `knowledge <system_id> [hours]`
- `worldmodel [system_id]`
- `predict <system_id> <action> [k=v ...]`
- `export <file> [json|csv]`
- `history [N]`
- `clear`
- `help`
- `quit` / `exit`

Usability features:

- command aliases (`h`, `q`, `wm`, `ks`, `st`)
- command history with `!!`
- tab completion for key commands

## 19. LLM Provider Configuration

Canonical providers:

- `google`
- `openai`
- `openrouter`
- `groq`
- `ollama`

Common environment variables:

- Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Groq: `GROQ_API_KEY`
- Ollama: usually local, key often unnecessary

Multi-key rotation variables are supported by resilience wrapper (for example `OPENAI_API_KEYS`, `GEMINI_API_KEYS`, etc.).

OpenRouter quick note:

- Set `OPENROUTER_API_KEY`.
- Use `provider: openrouter` under `strategy.params` and/or `meta_learner.llm`.

## 20. Reproducibility Checklist (Tool-Paper Friendly)

For experimental reproducibility, report:

1. Polaris version (`2.0.0`) and commit hash.
2. Python version and OS.
3. Full YAML config used.
4. Connector endpoints and simulator versions.
5. Strategy type and all `strategy.params`.
6. LLM provider/model/temperature and resilience settings.
7. Whether native tools were enabled.
8. Monitoring interval and run duration.
9. Metrics export files and aggregation script.
10. Success/failure criteria and statistical method.

Recommended run protocol:

1. Validate config with `doctor`.
2. Warm up managed systems before timed runs.
3. Run N repeated trials per configuration.
4. Export metrics in both JSON and CSV.
5. Store raw logs and config snapshots per trial.

## 21. Minimal Examples

Expanded, runnable examples are available in:

- [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)
- [STRATEGIES_DETAILED.md](STRATEGIES_DETAILED.md)

## 21.1 Threshold (SWIM)

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
    connection:
      host: "localhost"
      port: 4242

monitoring:
  interval_seconds: 30

strategy:
  type: "threshold"
  params:
    thresholds:
      average_response_time:
        high: 800.0
      average_utilization:
        high: 0.85
        low: 0.30
    action_templates:
      default:
        high: {"type": "scale_up",   "parameters": {}}
        low:  {"type": "scale_down", "parameters": {}}
      average_utilization:
        high: {"type": "set_dimmer", "parameters": {"value": 0.5}}
    cooldown_seconds: 60
```

> **`action_templates` is required.** The threshold strategy raises a `ValueError` at the first threshold crossing if this block is absent. Add either a per-metric entry or a `default` fallback.

For a full SWIM operating workflow (doctor -> run modes -> troubleshooting), see:

- [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)

## 21.2 LLM Reasoning (OpenRouter)

```yaml
strategy:
  type: "llm_reasoning"
  params:
    provider: "openrouter"
    model: "openai/gpt-4o-mini"
    temperature: 0.1
    system_description: "SWIM server pool"
    adaptation_goals: "Maintain latency under target while minimizing cost"
```

## 21.3 Multi-Agent

```yaml
strategy:
  type: "multi_agent"
  params:
    provider: "google"
    temperature: 0.1
    steps_limit: 3
    diagnostician:
      temperature: 0.0
      steps_limit: 4
    planner:
      provider: "openai"
      temperature: 0.2
    validator:
      temperature: 0.0
```

## 22. Troubleshooting

## 22.1 Config fails to load

- Run:

```bash
polaris doctor --config <your_config.yaml>
```

- Check for:
  - unsupported keys (strict schema)
  - unsupported strategy or connector type
  - missing `${ENV_VAR}` substitutions

## 22.2 LLM strategy fails immediately

- Ensure API key/env vars are set.
- Ensure provider dependency package is installed.
- Verify connector exposes supported actions (strict contract requirement).

## 22.3 Dashboard or interactive mode fails

- Install `rich`.

## 22.4 Wildfire/Kubernetes connector issues

- Wildfire: verify adapter health endpoint and session handling.
- Kubernetes: verify kubeconfig/in-cluster config and RBAC permissions.

## 23. Developer Extension Guide and Snippets

## 23.1 Add a custom connector

1. Implement `Connector` methods.
2. Provide `get_supported_actions()` and optionally `get_capabilities()`.
3. Register a connector factory and optional config validator via:

```python
from polaris.core.factories import register_connector_factory, register_connector_config_validator
```

4. Use new connector type in YAML `systems[].connector_type`.

## 23.2 Add a custom strategy

1. Implement `AdaptationStrategy` methods.
2. Register strategy factory with:

```python
from polaris.core.factories import register_strategy_factory
```

3. Use strategy type in YAML `strategy.type`.

If strict action correctness is needed, enforce contract checking similarly to built-in LLM strategies.

## 23.3 Plugin discovery

Connectors can be discovered through:

- explicit `plugin_imports`
- entry point group `polaris.connectors`

## 23.4 Connector Skeleton

```python
from polaris import Connector, SystemState


class MyConnector(Connector):
    async def collect_telemetry(self) -> SystemState:
        return SystemState(...)

    async def execute_action(self, action):
        pass
```

## 23.5 Strategy Skeleton

```python
from polaris import AdaptationAction, AdaptationStrategy


class MyStrategy(AdaptationStrategy):
    async def assess(self, state, context):
        if should_adapt():
            return AdaptationAction(...)
        return None

    def get_tunable_parameters(self):
        return {}
```

## 23.6 Tool Skeleton

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

## 23.7 Swap Multiple Components

```python
from polaris import Polaris


polaris = Polaris(
    strategy=MyStrategy(),
    world_model=MyWorldModel(),
    knowledge_store=PostgreSQLStore(),
    logger=MyLogger(),
)
```

## 24. Current Defaults and Notable Constants

- Default monitoring interval: 30s
- Default SWIM timeout: 30s
- Default connector timeout: 10s
- Default max concurrent connectors: 10
- Default Google model constant: `gemini-3-flash-preview`

## 25. Docker Operations

Polaris ships with a root multi-target Dockerfile and a compose file for common runtime profiles.

### 25.1 Prerequisites

- Docker Engine / Docker Desktop
- Docker Compose v2 (`docker compose` command)
- A Polaris config file (for example `config/default.yaml`)

Prepare host folders used as bind mounts:

```bash
mkdir -p logs metrics data
```

### 25.2 Image Targets

- `core`: lean runtime image for threshold/basic deployments
- `full-feature`: runtime image with all optional Polaris extras (`.[all]`)
- `ci`: development/test image with `.[dev]`

Build examples:

```bash
docker build --target core -t polaris:core .
docker build --target full-feature -t polaris:full-feature .
docker build --target ci -t polaris:ci .
```

### 25.3 Quick Start (Core Image)

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

### 25.4 Compose Profiles

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

### 25.5 Environment Variables

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

### 25.6 Health and Readiness

Container images use Polaris Doctor as the image healthcheck command:

```bash
polaris doctor --config ${POLARIS_CONFIG:-/app/config/default.yaml}
```

This healthcheck fails on true config/runtime failures and allows optional-feature warnings (for example, `rich` absent in `core`) so healthy minimal deployments are not marked unhealthy.

### 25.7 Running Dashboard and Interactive Modes

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

### 25.8 Docker Troubleshooting

- `Config file not found`: confirm mount path and `--config` path inside container (`/app/config/...`).
- `Missing credentials`: export required env vars for your selected LLM provider.
- `Permission denied` writing logs/metrics/data: ensure host directories exist and are writable.
- `Redis bus unreachable`: when using compose, use `redis://redis:6379` (service name), not localhost.
- `Ollama unreachable`: when using compose, use `http://ollama:11434` from Polaris config.

## 26. Package Architecture Map

Polaris follows a modular, interface-driven design:

- `abstractions/` - Core interfaces (extend these to customize)
- `core/` - Framework orchestration
- `strategies/` - Adaptation logic
- `world_model/` - System behavior modeling
- `knowledge/` - Historical data storage
- `meta_learner/` - Parameter optimization
- `infrastructure/` - Logging, metrics, configuration, LLM clients

## 27. Testing and CI/CD

### 27.1 Local Development Workflow

```bash
# Setup development environment
./scripts/setup-ci.sh

# Install pre-commit hooks
make install-hooks

# Run full CI pipeline locally
make ci
```

### 27.2 Running Tests

```bash
# Run test suite
make test

# Run with coverage
make test-all

# Verbose output
make test-verbose
```

Alternative test entry points:

```bash
python -m pytest tests/
python run_tests.py
```

### 27.3 Code Quality Commands

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint

# Type checking
make type-check

# Quick pre-commit check
make pre-commit

# List available make commands
make help
```

### 27.4 Workflow Files

- `.github/workflows/ci-cd.yml` - Main CI/CD pipeline
- `.github/workflows/quality-gate.yml` - PR quality validation
- `.github/workflows/auto-commit.yml` - Auto-format and commit
- `.github/workflows/merge-check.yml` - Merge validation

### 27.5 Pre-commit Hooks Included

- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking
- pytest validation
- Security scanning

## 28. Suggested Citation Block (Template)

```text
Polaris Framework (v2.0.0): A modular self-adaptive systems framework with
contract-first adaptation strategies, multi-agent LLM planning, and extensible
connector/strategy factories. Repository: <insert URL>, commit <insert hash>.
```

## 29. Validation Note

This document was cross-checked against:

- CLI implementation (`main`, `doctor`, `init`)
- Configuration schema and validators
- Factory registrations
- Connector implementations
- Strategy implementations
- Runtime orchestration loops

It should be treated as the current canonical operational guide for this repository version.

Detailed companion docs are maintained in the same folder for strategy-level and SWIM-level operational depth.
