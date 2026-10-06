# POLARIS: Agent Engineering & Repository Context Guide

Welcome to the **POLARIS** codebase. This guide provides AI agents and human contributors with essential context, architectural invariants, development workflows, testing procedures, and operational rules for developing, maintaining, and running experiments within this framework.

---

## 1. Project Overview & Philosophy

**POLARIS** (*Proactive Optimization & Learning Architecture for Resilient Intelligent Systems*) is an extensible, modular framework implementing autonomous **MAPE-K** (Monitor, Analyze, Plan, Execute, Knowledge) control loops for self-adaptive systems.

### Core Architecture
- **Monitor (`M`)**: Telemetry collection from distributed managed systems via pluggable asynchronous [Connectors](file:///home/prakhar/dev/prakhar479/polaris/polaris/abstractions/connector.py).
- **Analyze (`A`)**: Trend detection, regime shifts, and predictive modeling using a statistical/Kalman [World Model](file:///home/prakhar/dev/prakhar479/polaris/polaris/world_model/statistical.py).
- **Plan (`P`)**: Adaptation decision-making via swappable [Adaptation Strategies](file:///home/prakhar/dev/prakhar479/polaris/polaris/strategies) (deterministic threshold rules, single-turn LLM reasoning, ReAct agentic tool use, hierarchical THREAD agentic, multi-agent committees, or hybrid cascades).
- **Execute (`E`)**: Strict contract-first validation and execution of adaptation actions against managed systems via connectors.
- **Knowledge (`K`)**: Telemetry and adaptation history stored in [Knowledge Stores](file:///home/prakhar/dev/prakhar479/polaris/polaris/knowledge) (In-Memory or SQLite).
- **Meta-Learning**: Autonomous offline/background parameter optimization (Bayesian optimization, Statistical regression, or LLM-based reflection) to tune strategy thresholds and loop cadences.

---

## 2. Repository Layout & File Taxonomy

```
polaris/
├── polaris/                       # Core Python package
│   ├── abstractions/              # Abstract Base Classes (Protocols & Interfaces)
│   │   ├── connector.py           # Connector ABC & ExecutionResult
│   │   ├── strategy.py            # AdaptationStrategy ABC & AdaptationContext
│   │   ├── world_model.py         # WorldModel ABC
│   │   ├── knowledge_store.py     # KnowledgeStore ABC
│   │   ├── meta_learner.py        # MetaLearner ABC
│   │   ├── system_contract.py     # SystemContract & ActionSchema contracts
│   │   └── observability.py       # Logger & MetricsCollector ABCs
│   ├── cli/                       # CLI commands, dashboard, doctor, init
│   │   ├── main.py                # Primary CLI argument parser & execution loop
│   │   ├── doctor.py              # Pre-flight diagnostic engine (doctor command)
│   │   ├── dashboard.py           # Rich-based terminal UI dashboard
│   │   └── interactive.py         # Interactive terminal REPL
│   ├── connectors/                # Concrete system connectors
│   │   ├── swim.py                # SWIM (web infrastructure manager via TCP)
│   │   ├── wildfire.py            # Wildfire-UAVSim (via HTTP REST API)
│   │   ├── kubernetes_connector.py# Kubernetes pod scaler
│   │   └── suave.py               # SUAVE underwater robotics (ROS/roslibpy)
│   ├── core/                      # Core runtime orchestration
│   │   ├── polaris.py             # Polaris top-level coordinator & lifecycle
│   │   ├── component_builder.py   # Factory builder wiring config to instances
│   │   ├── adaptation_pipeline.py # Assess → Validate → Execute pipeline
│   │   ├── monitoring_loop.py     # Periodic telemetry collection loop
│   │   ├── meta_learning_loop.py  # Background parameter optimization loop
│   │   ├── metrics_export_loop.py # Background JSON/CSV metrics export
│   │   ├── config_reloader.py     # Hot-reloading watcher for configuration
│   │   ├── factories.py           # Connector & strategy registries
│   │   ├── registry.py            # Runtime connector registry
│   │   └── models.py              # Pydantic domain models (SystemState, AdaptationAction)
│   ├── infrastructure/            # Supporting infrastructure & I/O
│   │   ├── config.py              # Pydantic schema validation for YAML configurations
│   │   ├── contract_builder.py    # Auto-generates SystemContract from connectors
│   │   ├── events/                # In-memory and Redis event bus
│   │   ├── llm/                   # Multi-provider LLM clients, resilience, retry
│   │   └── observability/         # Structured logger, metrics counters/gauges/timers
│   ├── knowledge/                 # Knowledge store implementations (Memory, SQLite)
│   │   ├── memory.py              # In-memory ring buffer knowledge store
│   │   └── sqlite_store.py        # Persistent SQLite knowledge store
│   ├── meta_learner/              # Meta-learning algorithms
│   │   ├── bayesian_optimizer.py  # Bayesian Optimization for strategy thresholds
│   │   ├── statistical.py         # Statistical gradient & windowed tuning
│   │   └── llm_based.py           # LLM reflective feedback meta-learner
│   ├── strategies/                # Adaptation decision engines
│   │   ├── threshold.py           # Reactive threshold strategy with cooldowns
│   │   ├── llm_reasoning.py       # Zero/Few-shot structured LLM reasoning
│   │   ├── agentic_llm.py         # Tool-calling ReAct agentic strategy
│   │   ├── thread_agentic.py      # Hierarchical supervisor-worker thread agentic
│   │   ├── multi_agent.py         # Multi-agent voting/committee consensus
│   │   └── hybrid.py              # Cascade/committee combiner of multiple strategies
│   ├── tools/                     # Tooling system for agentic LLM strategies
│   │   ├── base.py                # BaseTool ABC & ToolResult
│   │   ├── registry.py            # ToolRegistry mapping tool names to functions
│   │   ├── factories.py           # Factory registration for custom tools
│   │   └── builtin.py             # Standard tools (metric math, trends, action history)
│   └── world_model/               # World modeling implementations
│       └── statistical.py         # Kalman filter, trends, and regime tracking
├── config/                        # YAML configuration files
│   ├── default.yaml               # Baseline configuration (threshold + LLM hybrid)
│   ├── swim.yaml                  # SWIM web server pool configuration
│   ├── wildfire.yaml              # Wildfire-UAVSim REST configuration
│   ├── suave.yaml                 # SUAVE underwater vehicle configuration
│   └── swim_thread_threshold_hybrid_experiment.yaml # THREAD experimental configuration
├── docs/                          # Detailed architecture and guide documents
├── examples/                      # Runnable examples and scripts
├── requirements/                  # Dependency constraints
│   └── constraints.txt            # Pinned package constraints for deterministic builds
├── scripts/                       # Maintenance and CI utility scripts
├── tests/                         # Pytest test suite (470+ unit and integration tests)
├── wildfire/                      # Mesa simulation code & REST adapter for Wildfire
├── pyproject.toml                 # Package definition and single source of truth for dependencies
├── Makefile                       # Developer command shortcuts
└── README.md                      # Human-facing introduction and quick start
```

---

## 3. Core Domain Models & Abstractions

### Key Types (`polaris/core/models.py`)
- **`SystemState`**: Snapshot of managed system telemetry at a specific UTC timestamp, containing a dictionary of `MetricValue` objects (`name`, `value`, `unit`, `tags`), and `HealthStatus` (`HEALTHY`, `WARNING`, `CRITICAL`, `UNKNOWN`).
- **`AdaptationAction`**: Action proposed by a strategy, containing `action_type`, `parameters` (dict), `priority` (float), `reasoning` (str), and `system_id` (str).
- **`ExecutionResult`**: Result returned by a connector after executing an action: `action_id`, `status` (`SUCCESS`, `FAILED`, `PARTIAL`, `TIMEOUT`), `execution_time_ms`, and optional `error_message`.
- **`SystemContract`**: Machine-readable specification exposed by connectors defining valid actions, parameter bounds, enums, and required telemetry metrics. Used by LLM prompt builders and validators.

### Primary Extension Interfaces (`polaris/abstractions/`)
1. **`Connector`** ([connector.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/abstractions/connector.py)):
   - `connect() -> bool`
   - `disconnect() -> bool`
   - `get_system_id() -> str`
   - `collect_telemetry() -> SystemState`
   - `execute_action(action: AdaptationAction) -> ExecutionResult`
   - `get_supported_actions() -> List[str]`
2. **`AdaptationStrategy`** ([strategy.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/abstractions/strategy.py)):
   - `assess(context: AdaptationContext) -> List[AdaptationAction]`
   - `get_tunable_parameters() -> Dict[str, Any]`
   - `update_parameter(name: str, value: Any) -> bool`
   - `on_action_executed(action: AdaptationAction, result: ExecutionResult) -> None`
3. **`BaseTool`** ([tools/base.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/tools/base.py)):
   - `name: str`, `description: str`, `parameters_schema: Dict[str, Any]`
   - `execute(context: ToolContext, **kwargs) -> ToolResult`
4. **Resilience & Fault Tolerance Guards**:
   - **Circuit Breaker & Strategy Fallback** ([core/adaptation_pipeline.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/core/adaptation_pipeline.py)): Tracks consecutive strategy failures; trips to `OPEN` after `circuit_breaker_threshold` failures and safely delegates adaptation decisions to `fallback_strategy` without hanging the loop. Probes recovery via `HALF_OPEN` state.
   - **Reasoning Cycle Execution Budget** (`max_cycle_time_seconds`): Enforces cumulative wall-clock limits on ReAct and recursive THREAD agentic reasoning loops, automatically returning fallback or skipping to prevent monitoring loop stalls.
   - **Token & Cost Accounting** ([infrastructure/llm/](file:///home/prakhar/dev/prakhar479/polaris/polaris/infrastructure/llm)): Standardizes `prompt_tokens`, `completion_tokens`, and `tokens_used` across OpenAI, Gemini, Groq, OpenRouter, and Ollama.

---

## 4. Development & Testing Commands

POLARIS uses standard Python virtual environments and enforces strict code quality gates.

### Virtual Environment Setup
```bash
# Activate existing local virtual environment
source .venv/bin/activate

# Or create and install development dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -c requirements/constraints.txt -e .[dev]
```

### Running Tests
```bash
# Run complete test suite (498+ tests)
pytest tests/ -v --tb=short

# Run with coverage (fails if under 60%)
pytest tests/ -v --cov=polaris --cov-report=term-missing --cov-fail-under=60

# Run specific test file
pytest tests/test_adaptation_pipeline.py -v
```

### Code Formatting & Quality Checks
```bash
# Format code (black line length 100, isort profile black)
make format

# Check formatting without modifying files
make format-check

# Run flake8 linter
make lint

# Run mypy static type checking (must pass with 0 errors)
make type-check

# Check dependency consistency and policy
make dependency-policy-check

# Run full CI suite locally
make ci
```

---

## 5. CLI & Diagnostic Tools

The POLARIS CLI entry point is `polaris` (mapped to [polaris.cli.main:main](file:///home/prakhar/dev/prakhar479/polaris/polaris/cli/main.py)).

### Pre-flight Diagnostics (`doctor`)
Always run `polaris doctor` before starting a run or experiment to verify environment variables, YAML schema validity, optional dependencies, and tool bindings:
```bash
# Diagnose configuration
polaris doctor --config config/swim.yaml
polaris doctor --config config/default.yaml
```

### Running Polaris
```bash
# Standard execution
polaris --config config/default.yaml

# Dry-run mode (computes adaptations without executing connector actions)
polaris --config config/default.yaml --dry-run

# Run with Rich interactive dashboard
polaris --config config/default.yaml --dashboard

# Run in split-screen mode (dashboard + interactive REPL)
polaris --config config/default.yaml --both

# Export logs and metrics
polaris --config config/swim.yaml \
  --log-format structured \
  --export-logs ./logs/swim_run.log \
  --metrics-export ./metrics/swim \
  --metrics-experiment exp_run_1

# Automated End-to-End Experiment Execution (Simulation Boot + Polaris + Metric Extraction)
./scripts/run_experiment.sh --exemplar swim --config config/swim.yaml --experiment-name swim_exp1

# Compare Multiple Benchmark Experiment Runs (Markdown table + 4-panel visual charts)
python3 scripts/compare_experiments.py \
  --experiments swim_baseline swim_exp1 \
  --metrics-dir metrics/swim \
  --output results/benchmark_comparison.md \
  --chart results/benchmark_comparison.png
```

---

## 6. Supported Exemplars & Experiment Workflows

POLARIS supports multiple real-world and benchmark exemplars for self-adaptive systems:

### A. SWIM (Simulated Web Infrastructure Manager)
- **Protocol**: TCP socket connection to `localhost:4242`.
- **Metrics**: `average_response_time`, `average_utilization`, `server_count`, `active_servers`, `dimmer`.
- **Actions**: `scale_up` (add server), `scale_down` (remove server), `set_dimmer` (float 0.0-1.0 optional content ratio).
- **Automated Workflow**:
  ```bash
  # Single command boots container, waits for socket, runs Polaris, and summarizes SLA
  ./scripts/run_experiment.sh --exemplar swim --config config/swim.yaml --experiment-name swim_baseline
  ```

### B. Wildfire-UAVSim
- **Protocol**: HTTP REST API on `localhost:5000` via [wildfire/adapter.py](file:///home/prakhar/dev/prakhar479/polaris/wildfire/adapter.py).
- **Metrics**: `fire_cells_burning`, `fire_cells_total`, `num_agents`, `mr1_values`, `mr2_value`.
- **Actions**: `wildfire_move` (directions: north, south, east, west, hold), `wildfire_reset`.
- **Running Simulation**:
  ```bash
  # 1. Start Wildfire adapter (in wildfire/ directory)
  python3 wildfire/adapter.py

  # 2. In parallel, run Polaris with Wildfire config
  polaris --config config/wildfire.yaml --metrics-export ./metrics/wildfire --metrics-experiment wf_exp
  ```

### C. SUAVE (Search and Underwater Autonomous Vehicle)
- **Protocol**: ROS via `roslibpy` bridge on `localhost:9090`.
- **Config**: [config/suave.yaml](file:///home/prakhar/dev/prakhar479/polaris/config/suave.yaml).

### D. Kubernetes
- **Protocol**: Kubernetes client / kubeconfig.
- **Metrics**: Pod CPU/memory, replica counts.
- **Actions**: `scale_deployment`.

---

## 7. LLM Providers & Resilience Configuration

POLARIS strategies support multiple LLM providers via a unified abstraction in [polaris/infrastructure/llm/](file:///home/prakhar/dev/prakhar479/polaris/polaris/infrastructure/llm):

| Provider | Config `provider:` | Required Environment Variables | Supported Features |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `openai` | `OPENAI_API_KEY` (or `OPENAI_API_KEYS`) | Structured output, native tool calling |
| **Google Gemini** | `google` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Fast reasoning, native tool calling |
| **Groq** | `groq` | `GROQ_API_KEY` (or `GROQ_API_KEYS`) | Ultra-low latency inference |
| **Ollama** | `ollama` | `OLLAMA_BASE_URL` (default: `http://localhost:11434`) | Local offline inference |
| **OpenRouter** | `openrouter` | `OPENROUTER_API_KEY` | Multi-model routing (OpenAI-compatible) |

### Key Resilience Features
- **Key Rotation**: Provide comma-separated keys in `*_API_KEYS` to automatically rotate across rate limits.
- **Retries & Backoff**: Exponential backoff on HTTP 429/500/503.
- **Resilience Toggle**: Set `LLM_RESILIENCE_ENABLED=1` in `.env` to activate automatic fallback and rotation.

---

## 8. Critical Invariants & Rules for Coding Agents

When editing or extending POLARIS, you **MUST** strictly adhere to the following rules:

1. **Dependency Standardization Policy**:
   - `pyproject.toml` is the **single source of truth** for all project dependencies and versions.
   - Do **NOT** duplicate dependencies in `setup.py` (which is strictly a thin compatibility shim).
   - Every new runtime dependency **MUST** have an explicit upper bound (e.g., `pkg>=1.0.0,<2.0.0`).
   - Run `make dependency-policy-check` before concluding any dependency changes.

2. **Strict Action Contract Enforcement**:
   - Never allow an LLM strategy to emit arbitrary or free-form action names.
   - All proposed actions must pass through `SystemContract` validation in [polaris/core/adaptation_pipeline.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/core/adaptation_pipeline.py) and [polaris/strategies/action_resolution.py](file:///home/prakhar/dev/prakhar479/polaris/polaris/strategies/action_resolution.py).
   - If an action type does not match `connector.get_supported_actions()` or explicit aliases, it is rejected and logged.

3. **No Large Binary Data in Git**:
   - Do **NOT** commit raw simulation trace outputs (e.g. `.vec`, `.sca`, `.out` files) or large datasets to git.
   - Ensure experiment outputs write to `results/` or `metrics/` and are properly ignored by `.gitignore`.

4. **Async Loop & Cancellation Hygiene**:
   - All background loops (`MonitoringLoop`, `MetaLearningLoop`, `MetricsExportLoop`) inherit or follow `asyncio` task patterns.
   - Never use blocking synchronous calls (like `time.sleep`) inside async methods; use `await asyncio.sleep()`.
   - Ensure proper cleanup and task cancellation inside `stop()` and `__aexit__`.

5. **Code Style & Type Annotations**:
   - All code must conform to `black` (100-character line length) and `isort`.
   - All public methods, parameters, and return types must be fully type-annotated. `make type-check` must pass with zero errors.
   - Preserve existing docstrings and comments.
