# Polaris Configuration Guide

Complete reference and guide for configuring the Polaris self-adaptive framework. This document contains all configuration information in one place.

**Quick Navigation:**

- [Quick Start](#quick-start) - Get started in 5 minutes
- [Complete File Structure](#complete-configuration-file-structure) - Full YAML structure
- [Systems Configuration](#systems) - Managed systems setup
- [Monitoring](#monitoring) - Loop frequency settings
- [Adaptation Strategies](#adaptation-strategies) - Decision-making (6 types)
- [World Model & Knowledge](#world-model--knowledge-store) - System insights
- [Meta-Learner](#meta-learner) - Parameter optimization
- [Observability](#observability) - Logging, metrics, export
- [Configuration Examples](#configuration-examples) - Real-world patterns
- [Validation Checklist](#configuration-validation-checklist) - Pre-deployment verification
- [API Reference](#complete-parameter-reference) - All parameters documented
- [Code Mapping](#configuration-variable-mapping) - Where values are used
- [Troubleshooting](#troubleshooting-guide) - Common issues & solutions

Strict runtime semantics:

- LLM-based strategies accept only schema-valid JSON outputs.
- Adaptation outputs must use `actions` (list); singular `action` is not supported.
- Contract violations fail the affected system iteration while the monitoring loop continues for other systems.

## Quick Start

### For New Users (5 minutes)

1. **Create a configuration file** by copying `config/default.yaml`
2. **Update system connection details** (host, port)
3. **Choose your strategy** (recommend: `threshold` for starting)
4. **Validate** using the settings below
5. **Deploy** and monitor

### Minimal Configuration (Development)

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
    connection:
      host: "localhost"
      port: 4242

monitoring:
  interval_seconds: 30
  connector_timeout_seconds: 30

strategy:
  type: "threshold"
  params:
    thresholds:
      cpu_usage: { high: 80.0, low: 20.0 }
      memory_usage: { high: 85.0, low: 25.0 }
    cooldown_seconds: 60

observability:
  logging:
    type: "human"
    level: "INFO"
```

### Production Configuration (LLM-based)

```yaml
systems:
  - id: "swim"
    connector_type: "swim"

strategy:
  type: "llm_reasoning"
  params:
    provider: "openai" # or "openrouter"
    system_description: "SWIM web application server"
    adaptation_goals: "Maintain performance with minimal latency"
    temperature: 0.1

observability:
  logging:
    type: "structured"
    level: "INFO"
  metrics:
    enabled: true
    export:
      enabled: true
      output_dir: "./metrics"
      auto_export_interval_minutes: 60
```

### Using OpenRouter (OpenAI-compatible gateway)

Polaris supports OpenRouter via an OpenAI-compatible client.

Required environment variable:

- `OPENROUTER_API_KEY`

Optional environment variables:

- `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `OPENROUTER_SITE_URL` (sent as `HTTP-Referer`, recommended by OpenRouter)
- `OPENROUTER_APP_NAME` (sent as `X-Title`, recommended by OpenRouter)

Example config:

```yaml
strategy:
  type: "agentic_llm"
  params:
    provider: "openrouter"
    # Optional: override the OpenRouter model routing string
    # model: "anthropic/claude-3.5-sonnet"
    temperature: 0.3

meta_learner:
  enabled: true
  type: "llm"
  llm:
    provider: "openrouter"
    temperature: 0.1
```

---

## Complete Configuration File Structure

```yaml
# Core monitoring settings
monitoring:
  interval_seconds: 30 # Global loop cadence floor (seconds)
  connector_timeout_seconds: 30 # Global timeout for telemetry + adaptation per system (seconds)

# Managed systems
systems:
  - id: "system-name"
    connector_type: "swim" # Must match a registered connector factory (built-ins: swim, wildfire, suave, kubernetes)
    enabled: true
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: 5 # Effective cadence = max(interval_seconds, collection_interval)
      connector_timeout_seconds: 10 # Optional per-system override

# Adaptation strategy
strategy:
  type: "threshold" # Must match a registered strategy factory (built-ins: threshold, llm_reasoning, hybrid, agentic_llm, thread_agentic, multi_agent)
  params:
    thresholds:
      cpu_usage:
        high: 80.0
        low: 20.0
      memory_usage:
        high: 85.0
        low: 25.0
    cooldown_seconds: 60
    enabled: true

  # For llm_reasoning, agentic_llm, multi_agent, thread_agentic, and hybrid,
  # place strategy-specific fields under strategy.params.
  # Type-keyed blocks (for example strategy.threshold / strategy.agentic_llm)
  # are not supported in canonical config.

# World model configuration
world_model:
  type: "statistical"
  statistical:
    # Number of recent samples to keep per metric in memory
    window_size: 100
    # Enable lightweight Kalman-style filtering for smoother predictions
    # and a variance-based confidence score. When false, the model uses
    # simple historical means.
    use_kalman: true

    # Simple regime tracking (HMM-style) is always enabled for the
    # statistical world model. Regimes ("low", "normal", "high") are
    # inferred from metrics such as cpu_usage/response_time and are
    # exposed via world_model.get_insights() and the PredictionResult
    # reasoning string.

# Knowledge store configuration
knowledge_store:
  type: "memory"
  memory:
    max_states_per_system: 1000

# Meta-learner configuration
meta_learner:
  enabled: true
  type: "statistical" # Options: statistical, llm
  analysis_interval_hours: 1.0
  conservative_mode: true
  transparency:
    enabled: true
    output_path: "./logs/meta_learning_updates.jsonl"

  # When a statistical world model is provided to Polaris, the
  # StatisticalMetaLearner automatically consumes its insights via
  # world_model.get_insights(). The meta-learner aggregates basic
  # uncertainty information (e.g., average metric std and regime
  # estimates) into analysis.insights["world_model_uncertainty"], and
  # uses this to adjust Bayesian optimization confidence (more cautious
  # when variability is high). The StatisticalMetaLearner uses
  # Gaussian Process-based Bayesian optimization for intelligent
  # parameter tuning, with automatic fallback to rule-based heuristics
  # when insufficient historical data is available.

# Example: LLM-based meta-learner configuration
# meta_learner:
#   enabled: true
#   type: "llm"
#   analysis_interval_hours: 1.0   # Controls how often the meta-learning loop runs
#   transparency:
#     enabled: true                # Enabled by default when meta-learning is enabled
#     output_path: "./logs/meta_learning_updates.jsonl"
#   llm:
#     provider: google             # one of: "google", "openai", "openrouter", "groq", "ollama"
#     temperature: 0.1             # LLM temperature for analysis and proposals
#     auto_apply: false            # Whether to auto-apply approved proposals
#
#     # Optional: global prompts used for *all* systems
#     # analysis_system_prompt: |
#     #   You are an expert system analyst for self-adaptive systems...
#     # optimization_system_prompt: |
#     #   You are an expert parameter optimizer for self-adaptive systems...
#
#     # Optional: per-system overrides keyed by systems[].id (e.g. "swim", "wildfire").
#     # These take precedence over the global prompts for that system.
#     # per_system_prompts:
#     #   swim:
#     #     analysis_system_prompt: |
#     #       Analyze SWIM-specific performance metrics (response time, throughput, utilization)...
#     #     optimization_system_prompt: |
#     #       Propose threshold and cooldown changes for SWIM's adaptation strategy...
#     #   wildfire:
#     #     analysis_system_prompt: |
#     #       Analyze Wildfire-specific metrics (fire_cells_burning_ratio, mr1_avg, safety)...
#     #     optimization_system_prompt: |
#     #       Propose parameter changes that balance fire containment and UAV safety...
#
#     # Optional: LLM resilience configuration (same semantics as strategy.params.resilience)
#     # resilience:
#     #   rps: 2
#     #   burst: 4
#     #   concurrency: 4
#     #   max_retries: 4
#     #   base_backoff_ms: 200
#     #   max_backoff_ms: 4000

# Observability configuration
observability:
  logging:
    type: "human" # Options: structured, human
    level: "INFO" # DEBUG, INFO, WARNING, ERROR
    console: true
    file: true
    file_path: "./logs/polaris.log"
    use_colors: true

  metrics:
    enabled: true
    collector_type: "simple" # Options: simple, prometheus, datadog

    export:
      enabled: false
      formats: ["json", "csv"]
      output_dir: "./metrics"
      auto_export_interval_minutes: 60
      experiment_name: null
      include_timestamp: true

    simple:
      histogram_max_values: 1000

    components:
      core_framework: true
      monitoring_loop: true
      event_bus: true
      registry: true
      strategy: true
      connectors: true
      world_model: true
      knowledge_store: true
      meta_learner: true
```

## Factory-based Registration (Connectors, Strategies & Tools)

Polaris uses registries to map config identifiers to concrete implementations:

- `systems[].connector_type` -> connector factory
- `strategy.type` -> strategy factory
- `strategy.params.tools` / per-agent `tools` -> tool factories

Built-ins are registered lazily from:

- `polaris.core.factories` (connectors + strategies)
- `polaris.tools.factories` (tools)

Configuration validation uses the registered types and tool names, so custom registrations should happen before loading config.

### Registering a Custom Connector Factory

```python
from polaris.core.factories import register_connector_factory

def my_connector_factory(system_cfg, logger, metrics):
    # system_cfg is a SystemConfig-like object with .id and .connection
    return MyConnector(
        base_url=system_cfg.connection.get("base_url"),
        logger=logger,
        metrics=metrics,
    )

register_connector_factory("my_connector", my_connector_factory)
```

Then use it in YAML:

```yaml
systems:
  - id: "my-system"
    connector_type: "my_connector"
    connection:
      base_url: "http://localhost:1234"
```

### Registering a Custom Strategy Factory

```python
from polaris.core.factories import register_strategy_factory

def my_strategy_factory(strategy_cfg, logger, metrics, knowledge_store, world_model, registry):
    return MyStrategy(logger=logger, metrics=metrics)

register_strategy_factory("my_strategy", my_strategy_factory)
```

Then use it in YAML:

```yaml
strategy:
  type: "my_strategy"
```

### Registering a Custom Tool Factory

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

Then enable it in strategy config:

```yaml
strategy:
  type: "agentic_llm"
  params:
    tools:
      enabled:
        - my_custom_tool
```

## Connectors

### SWIM Connector

The SWIM connector connects to a SWIM (Simulated Web Infrastructure Manager) server via TCP.

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
    enabled: true
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: 5
```

**Connection parameters:**

- `host` (string): SWIM server host. Default: `"localhost"`
- `port` (int): SWIM server port. Default: `4242`

**Metrics provided:**

- `server_count`, `active_servers`, `max_servers`
- `dimmer`
- `basic_response_time`, `basic_throughput`
- `optional_response_time`, `optional_throughput`
- `average_response_time`, `average_utilization`

**Supported actions:**

- `scale_up` / `add_server`
- `scale_down` / `remove_server`
- `set_dimmer` / `adjust_qos`

### Wildfire Connector

The Wildfire connector connects to a WildFire simulation REST API (Flask adapter).

```yaml
systems:
  - id: "wildfire"
    connector_type: "wildfire"
    enabled: true
    connection:
      host: "localhost"
      port: 5000
      # OR use full base_url
      # base_url: "http://localhost:5000"
      timeout: 10.0
      session_id: null # optional: reuse an existing session
    monitoring:
      collection_interval: 5
```

**Connection parameters:**

- `host` (string): Adapter host. Default: `"localhost"`
- `port` (int): Adapter port. Default: `5000`
- `base_url` (string): Full base URL. Overrides `host`/`port` if provided.
- `timeout` (float): HTTP request timeout in seconds. Default: `10.0`
- `session_id` (string, optional): Reuse an existing session ID. If omitted and `auto_create_session` is true, a new session is created automatically.

**Metrics provided:**

- `timestep` (int): Current simulation timestep
- `num_agents` (int): Number of UAV agents
- `mr1_avg` (float): Average MR1 (detection reward) across agents
- `mr2_value` (float): MR2 (collision avoidance) metric
- `fire_cells_burning` (int): Number of currently burning cells
- `fire_cells_burning_ratio` (float): Percentage of fire cells burning

**Supported actions:**

- `wildfire_reset`: Reset simulation to timestep 0
- `wildfire_pause`: Pause simulation
- `wildfire_resume`: Resume simulation
- `wildfire_step`: Execute a single simulation step (no actions)
- `wildfire_move`: Execute UAV moves. Parameters:
  ```yaml
  parameters:
    actions:
      - uav: 0
        move: "north" # one of: north, south, east, west, hold
      - uav: 1
        move: "hold"
  ```
- `wildfire_batch_actions`: Execute multiple timesteps with per-step action lists. Parameters:
  ```yaml
  parameters:
    actions:
      - [{ uav: 0, move: "north" }, { uav: 1, move: "east" }]
      - [{ uav: 0, move: "south" }, { uav: 1, move: "hold" }]
  ```

## Environment Variables

### Required Environment Variables

For LLM-powered features, you need to set API keys:

```bash
# For Google Gemini (both names supported for compatibility)
export GOOGLE_API_KEY="your-google-api-key"
# OR
export GEMINI_API_KEY="your-google-api-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

### Environment Variable Substitution

You can use environment variables in configuration files:

```yaml
systems:
  - id: "production"
    connector_type: "swim"
    connection:
      host: "${SWIM_HOST}"
      port: ${SWIM_PORT}
```

**Important**: Missing environment variables will cause configuration loading to fail with a clear error message.

## Configuration Validation

The framework validates configuration at startup:

### System Configuration Validation

- `id` cannot be empty
- `connector_type` must be one of the types registered in the connector factory registry
  (built-ins: `"swim"`, `"wildfire"`, `"suave"`, `"kubernetes"`)
- `port` must be a valid integer between 1-65535 (for connectors that use it)

### Strategy Configuration Validation

- `type` must be one of the types registered in the strategy factory registry
  (built-ins: `"threshold"`, `"llm_reasoning"`, `"hybrid"`, `"agentic_llm"`, `"thread_agentic"`, `"multi_agent"`)
- For threshold strategy:
  - High thresholds must be greater than low thresholds
  - `cooldown_seconds` must be non-negative

### Example Validation Errors

```python
# This will fail - invalid connector type
systems:
  - id: "test"
    connector_type: "invalid"  # ❌ Not supported

# This will fail - invalid thresholds
strategy:
  type: "threshold"
  params:
    thresholds:
      cpu_usage:
        high: 20.0  # ❌ High < Low
        low: 80.0
```

## CLI Configuration Overrides

Many configuration options can be overridden via CLI arguments:

```bash
# Override logging settings
polaris --config config.yaml --log-level DEBUG --log-format structured

# Override metrics settings
polaris --config config.yaml --metrics-export ./output --disable-metrics

# Override monitoring interval (target seconds between monitoring iterations)
polaris --config config.yaml --monitoring-interval 60
```

### Available CLI Overrides

| CLI Argument            | Config Section                            | Description              |
| ----------------------- | ----------------------------------------- | ------------------------ |
| `--log-level`           | `observability.logging.level`             | Log level                |
| `--log-format`          | `observability.logging.type`              | Log format               |
| `--export-logs`         | `observability.logging.file_path`         | Log file path            |
| `--metrics-export`      | `observability.metrics.export.output_dir` | Metrics export directory |
| `--disable-metrics`     | `observability.metrics.enabled`           | Disable metrics          |
| `--monitoring-interval` | `monitoring.interval_seconds`             | Monitoring loop interval |

## Strategy Configuration

### Threshold Strategy

```yaml
strategy:
  type: "threshold"
  params:
    thresholds:
      cpu_usage:
        high: 80.0 # Trigger scale-up at 80% CPU
        low: 20.0 # Trigger scale-down at 20% CPU
      memory_usage:
        high: 85.0
        low: 25.0
      response_time:
        high: 500.0 # milliseconds
    cooldown_seconds: 60 # Wait 60s between adaptations
    enabled: true
```

### LLM Reasoning Strategy

```yaml
strategy:
  type: "llm_reasoning"
  params:
    system_description: "SWIM web application server pool"
    adaptation_goals: "Maintain performance with minimal resource usage"
    temperature: 0.1 # Lower = more deterministic
    # Optional: global system prompt template. The following placeholders are supported:
    #   {system_id}, {system_description}, {adaptation_goals}
    # system_prompt: |
    #   You are an intelligent adaptation controller for a self-adaptive system.
    #   System Description: {system_description}
    #   Adaptation Goals: {adaptation_goals}
    #   Managed System ID: {system_id}
    #
    # Optional: per-system prompt overrides keyed by systems[].id (e.g. "swim", "wildfire").
    # When present, these take precedence over system_prompt for that system_id.
    # per_system_prompts:
    #   swim: |
    #     You are an adaptation controller for the SWIM web application server pool.
    #     Focus on response time, throughput, and server utilization when
    #     deciding whether to scale up/down or adjust QoS dimmer settings.
    #   wildfire: |
    #     You are an adaptation controller for the Wildfire UAV-fire simulation.
    #     Focus on fire_cells_burning_ratio, mr1_avg, and safety metrics when
    #     deciding which wildfire_* actions to trigger.
```

**Requirements**:

- Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` environment variable
- Install LLM client libraries: `pip install google-generativeai` or `pip install openai`

### Multi-Agent Strategy

A committee of three specialized LLM agents — **Diagnostician**, **Planner**, and **SafetyValidator** — collaborate sequentially. Each agent can use its own LLM provider, temperature, and system prompt.

```yaml
strategy:
  type: "multi_agent"
  params:
    provider: google # Shared default LLM provider
    temperature: 0.1 # Shared default temperature
    steps_limit: 3 # Default max reasoning steps per agent (new)
    system_description: "SWIM web application server pool"

    # --- Per-agent overrides (all optional) ---
    # Omit any block to inherit shared values.

    diagnostician:
      provider: google # Fast model for anomaly detection
      temperature: 0.0 # Deterministic
      max_tokens: 1024
      # system_prompt: |
      #   You are a precise diagnostician for {system_description}.
      #   Return strict JSON only.
      resilience:
        rps: 2
        burst: 4
        max_retries: 4
        base_backoff_ms: 200
        max_backoff_ms: 4000

    planner:
      provider: openai # Stronger creative model for planning
      temperature: 0.2
      max_tokens: 1500
      # system_prompt: |
      #   You are a creative Planner for {system_description}.
      #   Propose concrete, reversible actions.

    validator:
      provider: google
      temperature: 0.0 # Conservative: reject borderline plans
      max_tokens: 1500
      tools: # Restricted tools for validator
        - get_world_model_insights
        - predict_outcome
      # system_prompt: |
      #   You are a conservative Safety Validator for {system_description}.
      #   Only approve safe, reversible plans.

    resilience: # Shared resilience defaults
      rps: 1
      burst: 2
      concurrency: 2
      max_retries: 4
      base_backoff_ms: 200
      max_backoff_ms: 4000
```

**Agent pipeline:**

1. **Diagnostician** — Receives system state → returns `is_anomaly_detected`, `issues`, `root_causes`, `severity`.
2. **Planner** — Receives diagnosis → returns `plans` (list of typed actions) and `rationale`.
3. **SafetyValidator** — Receives plan → returns `approved`, `reasoning`, and `safe_actions` (approved subset).

**Hot-reloadable parameters** (without restart):

The following parameters can be updated via the configuration file and will take effect immediately:

- `temperature`, `steps_limit`, `system_description`
- `diagnostician.temperature`, `planner.temperature`, `validator.temperature`
- `diagnostician.steps_limit`, `planner.steps_limit`, `validator.steps_limit`
- `diagnostician.system_prompt`, `planner.system_prompt`, `validator.system_prompt`
- `diagnostician.max_tokens`, `planner.max_tokens`, `validator.max_tokens`
- `resilience.*` (shared + per-agent, when using resilient LLM client)

### LLM Resilience (Retries, Rate Limiting, Key Rotation)

You can enable a resilience layer for LLM calls that adds retries with exponential backoff, async rate limiting, optional concurrency caps, and API key rotation.

Enable via strategy config (recommended):

```yaml
strategy:
  type: "llm_reasoning"
  params:
    temperature: 0.1
    resilience:
      rps: 2 # tokens refill rate per second for token-bucket
      burst: 4 # bucket capacity for short bursts
      concurrency: 4 # max concurrent LLM requests
      max_retries: 4
      base_backoff_ms: 200
      max_backoff_ms: 4000
      # keys_env_var: OPENAI_API_KEYS  # optional custom env var for key rotation
```

Or enable globally via environment variables:

```bash
export LLM_RESILIENCE_ENABLED=1
# For key rotation (comma-separated):
export OPENAI_API_KEYS="key1,key2,key3"
export GEMINI_API_KEYS="keyA,keyB"
```

Notes:

- If a `resilience` block is present in config or multi-key env vars are set, the client uses the resilience wrapper automatically.
- When rate limited (429/quota), the client rotates to the next key (if configured) and retries with backoff.
- Structured logs for LLM latency, error types, and response previews are written to `./logs/llm_debug.log` for debugging.

## Connector Configuration

### SWIM Connector

```yaml
systems:
  - id: "swim-pool"
    connector_type: "swim"
    enabled: true
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: 5 # seconds
```

### Wildfire Connector

See the full [Wildfire Connector](#wildfire-connector) section earlier in this document.

### Kubernetes Connector

Connects to a Kubernetes cluster to monitor pods and scale deployments. Works both in-cluster (running inside a pod) and externally via a kubeconfig file.

```yaml
systems:
  - id: "k8s-prod"
    connector_type: "kubernetes"
    enabled: true
    connection:
      kubeconfig_path: null # Path to kubeconfig file (null = ~/.kube/config)
      in_cluster: false # Set to true when running as a pod inside the cluster
      namespace: "default" # Kubernetes namespace to monitor and manage
    monitoring:
      collection_interval: 10
```

**Connection parameters:**

- `kubeconfig_path` (string, optional): Absolute path to a kubeconfig file. When `null`, uses the default kubeconfig (`~/.kube/config` or `KUBECONFIG` env var).
- `in_cluster` (boolean): Set to `true` when Polaris itself runs as a Kubernetes pod. Default: `false`.
- `namespace` (string): Namespace to watch. Default: `"default"`.

**Metrics provided:**

- `pod_count` (int): Total pods in namespace
- `running_pods` (int): Pods in Running state
- `pending_pods` (int): Pods in Pending state
- `failed_pods` (int): Pods in Failed state
- `deployment_available_replicas` (int): Available replicas across deployments
- `deployment_desired_replicas` (int): Desired replicas across deployments

**Supported actions:**

- `scale_deployment`: Scale a deployment to a target replica count.
  ```yaml
  parameters:
    deployment: "my-deployment"
    replicas: 3
  ```
- `restart_deployment`: Trigger a rolling restart of a deployment.
  ```yaml
  parameters:
    deployment: "my-deployment"
  ```

**Installation requirements:**

```bash
pip install kubernetes
```

**Permissions:** The service account or kubeconfig must allow `get`, `list`, `watch`, `update`, `patch` on `pods`, `deployments`, and `replicasets`.

## Observability Configuration

### Logging

```yaml
observability:
  logging:
    type: "human" # "structured" for JSON, "human" for readable
    level: "INFO" # DEBUG, INFO, WARNING, ERROR
    console: true # Log to console
    file: true # Log to file
    file_path: "./logs/polaris.log"
    use_colors: true # Colored output (console only)
```

### Metrics

```yaml
observability:
  metrics:
    enabled: true
    collector_type: "simple"  # Only "simple" fully implemented

    # Auto-export settings
    export:
      enabled: true
      formats: ["json", "csv"]
      output_dir: "./metrics"
      auto_export_interval_minutes: 60
      experiment_name: "my-experiment"
      include_timestamp: true

    # Component-specific metrics (can disable individually)
    components:
      core_framework: true
      monitoring_loop: true
      event_bus: true
      # ... etc

## Hot Reload of Strategy Parameters and Resilience

When Polaris is started with a `config_path`, it detects configuration file changes and hot-applies parameter updates during the monitoring loop.

What is hot-reloaded:
- Threshold strategy: `cooldown_seconds` and per-metric `thresholds.{metric}.{high|low}`
- LLM reasoning: `temperature`, `system_description`, `system_prompt`, `per_system_prompts`, and LLM `resilience` (when using the resilient client)
- Hybrid strategy: `selection_mode`, `min_confidence`, and sub-strategy parameters (index-matched), including resilience for LLM sub-strategies
- Multi-agent strategy: `temperature`, `system_description`, per-agent `temperature`/`system_prompt`/`max_tokens`, and per-agent `resilience` (when using the resilient client)

Limitations:
- Changing the strategy type (e.g., threshold → hybrid) requires a restart
- Hybrid sub-strategy updates are index-based; to change counts or order, restart with the new configuration
- Resilience updates require the LLM client to be the resilient wrapper; otherwise the update is skipped with a warning

Best practices:
- Keep resilience and strategy parameters in the main config file passed to Polaris
- Avoid frequent file writes; changes are detected each iteration of the monitoring loop
```

## Hot Reload of Strategy Parameters and Resilience

When Polaris is started with a `config_path`, it detects configuration file changes and hot-applies parameter updates during the monitoring loop.

What is hot-reloaded:

- Threshold strategy: `cooldown_seconds` and per-metric `thresholds.{metric}.{high|low}`
- LLM reasoning: `temperature`, `system_description`, `system_prompt`, `per_system_prompts`, and LLM `resilience` (when using the resilient client)
- Hybrid strategy: `selection_mode`, `min_confidence`, and sub-strategy parameters (index-matched), including resilience for LLM sub-strategies

Limitations:

- Changing the strategy type (e.g., threshold → hybrid) requires a restart
- Hybrid sub-strategy updates are index-based; to change counts or order, restart with the new configuration
- Resilience updates require the LLM client to be the resilient wrapper; otherwise the update is skipped with a warning

Best practices:

- Keep resilience and strategy parameters in the main config file passed to Polaris
- Avoid frequent file writes; changes are detected each iteration of the monitoring loop

---

## Configuration Examples

### Example 1: Minimal Development Setup

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: 5

monitoring:
  interval_seconds: 30

strategy:
  type: "threshold"
  params:
    thresholds:
      cpu_usage: { high: 80.0, low: 20.0 }
    cooldown_seconds: 60

observability:
  logging:
    type: "human"
    level: "DEBUG"
```

**Use Case**: Getting started, local testing, development

### Example 2: Production with LLM Strategy

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
    connection:
      host: "swim.example.com"
      port: 4242

monitoring:
  interval_seconds: 60

strategy:
  type: "llm_reasoning"
  params:
    provider: "openai"
    system_description: "Production SWIM web cluster"
    adaptation_goals: "Maintain 99.9% uptime with <100ms latency"
    temperature: 0.1
    resilience:
      rps: 2
      burst: 4
      max_retries: 4

observability:
  logging:
    type: "structured"
    level: "INFO"
  metrics:
    enabled: true
    export:
      enabled: true
      output_dir: "./metrics"
      auto_export_interval_minutes: 60
      experiment_name: "production-swim-v1"
```

**Use Case**: Production deployments, critical systems, LLM-based decisions

### Example 3: Advanced with Meta-Learning

```yaml
systems:
  - id: "swim"
    connector_type: "swim"
  - id: "wildfire"
    connector_type: "wildfire"
    connection:
      base_url: "http://wildfire.example.com"
      timeout: 15.0

strategy:
  type: "hybrid"
  params:
    selection_mode: "confidence"
    min_confidence: 0.7
    strategies:
      - type: "threshold"
        priority: 0.5
        params:
          thresholds:
            cpu_usage: { high: 80.0, low: 20.0 }
      - type: "llm_reasoning"
        priority: 0.8
        params:
          provider: "openai"
          system_description: "Hybrid managed systems"
          temperature: 0.2

world_model:
  type: "statistical"
  statistical:
    window_size: 200
    use_kalman: true

meta_learner:
  enabled: true
  type: "llm"
  analysis_interval_hours: 2.0
  transparency:
    enabled: true
    output_path: "./logs/meta_learning_updates.jsonl"
  llm:
    provider: "openai"
    temperature: 0.1
    auto_apply: false

observability:
  logging:
    type: "structured"
    level: "INFO"
  metrics:
    enabled: true
    export:
      enabled: true
      output_dir: "./metrics"
      auto_export_interval_minutes: 30
      include_timestamp: true
```

**Use Case**: Multi-system management, advanced ML-based adaptation, continuous learning

---

## Configuration Validation Checklist

### Before Starting ✓

- [ ] Python 3.11+
- [ ] `pyyaml` package installed
- [ ] Configuration file created or `config/default.yaml` available
- [ ] API credentials set (if using LLM strategies)

### Configuration Completeness Checklist

#### 1. Systems Configuration

- [ ] At least one system defined in `systems` array
- [ ] Each system has required fields:
  - [ ] `id`: Unique system identifier (e.g., "swim", "wildfire-sim")
  - [ ] `connector_type`: one of "swim", "wildfire", "suave", "kubernetes" (or another registered connector type)
  - [ ] `enabled`: true or false (boolean)
  - [ ] `connection`: Object with connector-specific params

**SWIM System:**

- [ ] `connection.host`: Hostname or IP (or string "localhost")
- [ ] `connection.port`: Integer between 1-65535
- [ ] `monitoring.collection_interval`: Positive integer (seconds)

**Wildfire System:**

- [ ] Either `connection.base_url` OR both `connection.host` + `connection.port`
- [ ] `connection.timeout`: Positive float (seconds)
- [ ] `connection.session_id`: Optional string identifier
- [ ] `monitoring.collection_interval`: Positive integer (seconds)

#### 2. Monitoring Configuration

- [ ] `monitoring.interval_seconds`: Positive integer (default: 30)
- [ ] Value is reasonable (not too small, e.g., >= 1)

#### 3. Adaptation Strategy Configuration

- [ ] `strategy.type`: One of: "threshold", "llm_reasoning", "hybrid", "agentic_llm", "thread_agentic", "multi_agent"
- [ ] `strategy.params` block present and valid

**Threshold Strategy:**

- [ ] `strategy.params.thresholds`: Object with metric mappings
- [ ] Each metric has `high` and/or `low` values (floats/ints)
- [ ] `high` > `low` (if both present)
- [ ] `strategy.params.cooldown_seconds`: Non-negative integer

**LLM Reasoning Strategy:**

- [ ] `strategy.params.provider`: one of "google", "openai", "openrouter", "groq", "ollama"
- [ ] `strategy.params.system_description`: Non-empty string
- [ ] `strategy.params.adaptation_goals`: Non-empty string
- [ ] `strategy.params.temperature`: Float between 0.0-1.0
- [ ] API credentials available: `GOOGLE_API_KEY` or `OPENAI_API_KEY` env var set

**Hybrid Strategy:**

- [ ] `strategy.params.strategies`: Non-empty array
- [ ] Each sub-strategy has `type` field
- [ ] `strategy.params.selection_mode`: "first", "priority", or "confidence"
- [ ] `strategy.params.min_confidence`: Float between 0.0-1.0

**Agentic LLM Strategy:**

- [ ] `strategy.params.provider`: one of "google", "openai", "openrouter", "groq", "ollama"
- [ ] `strategy.params.steps_limit`: Positive integer (default: 3)
- [ ] `strategy.params.temperature`: Float between 0.0-2.0
- [ ] `strategy.params.system_prompt`: Optional system prompt template
- [ ] `strategy.params.per_system_prompts`: Optional per-system prompt overrides
- [ ] `strategy.params.tools.enabled`: Array of registered tool names (non-empty)
- [ ] `strategy.params.native_tools`: Optional OpenAI-format function definitions list
- [ ] `strategy.params.max_tool_result_chars`: Integer >= 1
- [ ] `strategy.params.native_tools_unsupported_policy`: one of `skip_cycle`, `json_fallback`, `strict_fail`
- [ ] Native tools are provider-agnostic in config; each provider converts internally
- [ ] If native tool response has `tool_calls=None`, strategy falls back to JSON text parsing
- [ ] If a native tool name matches a registered Polaris tool, it must also be enabled under `strategy.params.tools`
- [ ] If provider does not implement native tools (`NotImplementedError`), behavior follows `native_tools_unsupported_policy`

**THREAD Agentic Strategy:**

- [ ] `strategy.params.provider`: one of "google", "openai", "openrouter", "groq", "ollama"
- [ ] `strategy.params.steps_limit`: Positive integer (default: 4)
- [ ] `strategy.params.max_thread_depth`: Non-negative integer (default: 3)
- [ ] `strategy.params.max_total_threads`: Positive integer (default: 16)
- [ ] `strategy.params.child_timeout_seconds`: Positive number
- [ ] `strategy.params.max_tool_result_chars`: Integer >= 1
- [ ] `strategy.params.tools.enabled`: Array of registered tool names (non-empty)

#### 4. World Model Configuration (Optional)

- [ ] If present: `world_model.type`: "statistical"
- [ ] `world_model.statistical.window_size`: Positive integer
- [ ] `world_model.statistical.use_kalman`: boolean

#### 5. Knowledge Store Configuration (Optional)

- [ ] If present: `knowledge_store.type`: "memory" or "sqlite"
- [ ] `knowledge_store.memory.max_states_per_system`: Positive integer

#### 6. Meta-Learner Configuration (Optional)

- [ ] If `meta_learner.enabled: true`:
  - [ ] `meta_learner.type`: "statistical" or "llm"
  - [ ] `meta_learner.analysis_interval_hours`: Positive float
  - [ ] `meta_learner.transparency.enabled`: boolean
  - [ ] `meta_learner.transparency.output_path`: Writable file path
  - [ ] If type="llm": API credentials available

#### 7. Observability Configuration (Optional)

**Logging:**

- [ ] `observability.logging.type`: "human" or "structured"
- [ ] `observability.logging.level`: "DEBUG", "INFO", "WARNING", or "ERROR"
- [ ] `observability.logging.console`: boolean
- [ ] If `file: true`: `observability.logging.file_path` is valid directory path

**Metrics:**

- [ ] `observability.metrics.enabled`: boolean
- [ ] If `export.enabled: true`:
  - [ ] `observability.metrics.export.output_dir`: Valid directory path
  - [ ] Directory is writable
  - [ ] `observability.metrics.export.formats`: Array with "json" and/or "csv"

### Syntax & Format Validation

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('config/default.yaml'))"
echo "✓ YAML is valid" || echo "✗ YAML syntax error"
```

### Type Validation

- [ ] Strings are quoted if needed
- [ ] Numbers are not quoted
- [ ] Booleans are `true` or `false` (lowercase, unquoted)
- [ ] Arrays use `- item` format
- [ ] Objects use `key: value` pairs with proper indentation

### Runtime Validation

```python
from polaris.infrastructure.config import load_config

try:
    config = load_config("config/default.yaml")
    print(f"✓ Config loaded successfully")
    print(f"  Systems: {len(config.systems)}")
    print(f"  Strategy: {config.strategy.type}")
except Exception as e:
    print(f"✗ Config loading failed: {e}")
```

---

## Complete Parameter Reference

### Systems

Configure the managed systems to monitor and adapt.

| Field                                      | Type    | Default       | Description                                     |
| ------------------------------------------ | ------- | ------------- | ----------------------------------------------- |
| `systems[].id`                             | string  | -             | Unique system identifier (required)             |
| `systems[].connector_type`                 | string  | -             | Registered connector type (built-ins: `swim`, `wildfire`, `suave`, `kubernetes`) |
| `systems[].enabled`                        | boolean | `true`        | Enable/disable monitoring for this system       |
| `systems[].connection.host`                | string  | `localhost`   | Server host name or IP (SWIM/Wildfire)          |
| `systems[].connection.port`                | integer | `4242`/`5000` | Server port (1-65535)                           |
| `systems[].connection.base_url`            | string  | -             | Full base URL (Wildfire, overrides host/port)   |
| `systems[].connection.timeout`             | float   | `10.0`        | Request timeout in seconds (Wildfire)           |
| `systems[].connection.session_id`          | string  | -             | Optional session ID (Wildfire)                  |
| `systems[].monitoring.collection_interval` | integer | -             | Collection interval in seconds                  |
| `systems[].monitoring.connector_timeout_seconds` | float | -      | Per-system telemetry/adaptation timeout (seconds) |

### Monitoring

| Field                         | Type    | Default | Description                         |
| ----------------------------- | ------- | ------- | ----------------------------------- |
| `monitoring.interval_seconds` | integer | `30`    | Monitoring loop interval in seconds |
| `monitoring.connector_timeout_seconds` | float | `30` | Global telemetry/adaptation timeout per system cycle |

### Strategy

| Field           | Type   | Default     | Description                                                                                           |
| --------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------- |
| `strategy.type` | string | `threshold` | Strategy type: `threshold`, `llm_reasoning`, `hybrid`, `agentic_llm`, `thread_agentic`, `multi_agent` |

**Threshold Strategy Parameters:**

| Field                              | Type    | Default | Description                                     |
| ---------------------------------- | ------- | ------- | ----------------------------------------------- |
| `strategy.params.thresholds`       | object  | -       | Metric thresholds mapping (high/low per metric) |
| `strategy.params.cooldown_seconds` | integer | `60`    | Minimum seconds between successive adaptations  |
| `strategy.params.enabled`          | boolean | `true`  | Enable/disable threshold strategy               |

**LLM Reasoning Strategy Parameters:**

| Field                                    | Type    | Default  | Description                                        |
| ---------------------------------------- | ------- | -------- | -------------------------------------------------- |
| `strategy.params.provider`               | string  | `google` | LLM provider: google/openai/openrouter/groq/ollama |
| `strategy.params.system_description`     | string  | -        | Description of the managed system                  |
| `strategy.params.adaptation_goals`       | string  | -        | Adaptation objectives                              |
| `strategy.params.temperature`            | float   | `0.1`    | Model creativity (0.0-1.0)                         |
| `strategy.params.system_prompt`          | string  | -        | Custom system prompt                               |
| `strategy.params.per_system_prompts`     | object  | -        | Per-system prompt overrides                        |
| `strategy.params.resilience.rps`         | float   | `2.0`    | Requests per second                                |
| `strategy.params.resilience.burst`       | integer | `4`      | Max burst size                                     |
| `strategy.params.resilience.concurrency` | integer | `4`      | Max concurrent requests                            |
| `strategy.params.resilience.max_retries` | integer | `4`      | Retry attempts                                     |

**Hybrid Strategy Parameters:**

| Field                            | Type   | Default      | Description                                    |
| -------------------------------- | ------ | ------------ | ---------------------------------------------- |
| `strategy.params.selection_mode` | string | `confidence` | Selection mode: first, priority, or confidence |
| `strategy.params.min_confidence` | float  | `0.7`        | Min confidence threshold (0.0-1.0)             |
| `strategy.params.strategies`     | array  | -            | Array of sub-strategies                        |

**Agentic LLM Strategy Parameters:**

| Field                                            | Type         | Default      | Description                                                               |
| ------------------------------------------------ | ------------ | ------------ | ------------------------------------------------------------------------- |
| `strategy.params.provider`                       | string       | `google`     | LLM provider: google/openai/openrouter/groq/ollama                        |
| `strategy.params.steps_limit`                    | integer      | `3`          | Maximum reasoning steps                                                   |
| `strategy.params.temperature`                    | float        | `0.1`        | Model creativity                                                          |
| `strategy.params.system_prompt`                  | string       | -            | Custom system prompt template (supports `{system_id}`, `{allowed_tools}`) |
| `strategy.params.per_system_prompts`             | object       | -            | Per-system prompt overrides keyed by `systems[].id`                       |
| `strategy.params.tools`                          | object/array | -            | Tool allow-list (`{enabled: [...]}` or direct list), validated against registered tool names |
| `strategy.params.max_tool_result_chars`          | integer      | `1200`       | Max serialized tool payload injected into model context                    |
| `strategy.params.native_tools`                   | array        | -            | Optional OpenAI-format native tool definitions (`type:function`)          |
| `strategy.params.native_tools_unsupported_policy` | string      | `skip_cycle` | Behavior when provider lacks native tools: `skip_cycle`, `json_fallback`, `strict_fail` |

Notes for `strategy.params.native_tools`:

- The YAML shape stays OpenAI-compatible regardless of provider.
- Providers (`google`, `openai`, `openrouter`, `groq`, `ollama`) convert internally to native tool-call formats.
- Configured tool names in `strategy.params.tools` are validated against the registered tool factory names.
- If a `native_tools[].function.name` matches a registered Polaris tool, it must also be enabled in `strategy.params.tools`.
- Strategy behavior with native tools:
  - `tool_calls` is `None` -> warning + JSON text fallback parsing.
  - provider raises `NotImplementedError` -> follows `native_tools_unsupported_policy`.

**THREAD Agentic Strategy Parameters:**

| Field                                   | Type    | Default     | Description                                                       |
| --------------------------------------- | ------- | ----------- | ----------------------------------------------------------------- |
| `strategy.params.provider`              | string  | `google`    | LLM provider: google/openai/openrouter/groq/ollama                |
| `strategy.params.steps_limit`           | integer | `4`         | Maximum reasoning steps per thread                                |
| `strategy.params.temperature`           | float   | `0.1`       | Model creativity                                                  |
| `strategy.params.max_thread_depth`      | integer | `3`         | Maximum recursive child depth                                     |
| `strategy.params.max_total_threads`     | integer | `16`        | Global thread budget per assessment                               |
| `strategy.params.child_timeout_seconds` | float   | `20.0`      | Timeout for each spawned child thread                             |
| `strategy.params.max_repeated_spawns`   | integer | `2`         | Maximum repeated identical spawn signatures                       |
| `strategy.params.max_tool_result_chars` | integer | `1200`      | Max serialized tool payload injected into model context           |
| `strategy.params.phi_mode`              | string  | `last_line` | Parent-to-child context mapping mode (`last_line`/`recent_lines`) |
| `strategy.params.phi_max_lines`         | integer | `6`         | Number of parent lines used when `phi_mode=recent_lines`          |
| `strategy.params.listen_token`          | string  | `=>`        | Child feedback framing prefix                                     |
| `strategy.params.return_token`          | string  | `<=`        | Child feedback framing suffix                                     |
| `strategy.params.tools.enabled`         | array   | -           | Enabled tools array (validated against registered tool names)     |

**Multi-Agent Strategy Parameters:**

| Field                                         | Type    | Default            | Description                                             |
| --------------------------------------------- | ------- | ------------------ | ------------------------------------------------------- |
| `strategy.params.provider`                    | string  | `google`           | Shared default LLM provider                             |
| `strategy.params.temperature`                 | float   | `0.1`              | Shared default sampling temperature                     |
| `strategy.params.steps_limit`                 | integer | `3`                | Shared default max reasoning steps per agent stage      |
| `strategy.params.system_description`          | string  | `Managed system`   | System description embedded in default agent prompts    |
| `strategy.params.max_tool_result_chars`       | integer | `1200`             | Max serialized tool payload injected into model context |
| `strategy.params.tools`                       | object/array | -             | Shared tool allow-list (`{enabled: [...]}` or direct list) |
| `strategy.params.resilience.*`                | object  | -                  | Shared LLM resilience settings (see resilience section) |
| `strategy.params.diagnostician`               | object  | -                  | Per-agent override for Diagnostician role               |
| `strategy.params.diagnostician.provider`      | string  | inherits shared    | LLM provider for the Diagnostician agent                |
| `strategy.params.diagnostician.temperature`   | float   | inherits shared    | Sampling temperature for the Diagnostician agent        |
| `strategy.params.diagnostician.system_prompt` | string  | built-in           | Custom system prompt for Diagnostician                  |
| `strategy.params.diagnostician.max_tokens`    | integer | `1024`             | Max tokens for Diagnostician responses                  |
| `strategy.params.diagnostician.steps_limit`   | integer | inherits shared    | Max reasoning steps for Diagnostician                   |
| `strategy.params.diagnostician.tools`         | array   | inherits shared    | Available tools for Diagnostician                       |
| `strategy.params.diagnostician.resilience.*`  | object  | inherits shared    | Per-agent LLM resilience override                       |
| `strategy.params.planner`                     | object  | -                  | Per-agent override for Planner role                     |
| `strategy.params.planner.provider`            | string  | inherits shared    | LLM provider for the Planner agent                      |
| `strategy.params.planner.temperature`         | float   | inherits shared    | Sampling temperature for the Planner agent              |
| `strategy.params.planner.system_prompt`       | string  | built-in           | Custom system prompt for Planner                        |
| `strategy.params.planner.max_tokens`          | integer | `1500`             | Max tokens for Planner responses                        |
| `strategy.params.planner.steps_limit`         | integer | inherits shared    | Max reasoning steps for Planner                         |
| `strategy.params.planner.tools`               | array   | inherits shared    | Available tools for Planner                             |
| `strategy.params.planner.resilience.*`        | object  | inherits shared    | Per-agent LLM resilience override                       |
| `strategy.params.validator`                   | object  | -                  | Per-agent override for SafetyValidator role             |
| `strategy.params.validator.provider`          | string  | inherits shared    | LLM provider for the SafetyValidator agent              |
| `strategy.params.validator.temperature`       | float   | inherits shared    | Sampling temperature for the SafetyValidator agent      |
| `strategy.params.validator.system_prompt`     | string  | built-in           | Custom system prompt for SafetyValidator                |
| `strategy.params.validator.max_tokens`        | integer | `1500`             | Max tokens for SafetyValidator responses                |
| `strategy.params.validator.steps_limit`       | integer | inherits shared    | Max reasoning steps for SafetyValidator                 |
| `strategy.params.validator.tools`             | array   | inherits shared    | Available tools for SafetyValidator                     |
| `strategy.params.validator.resilience.*`      | object  | inherits shared    | Per-agent LLM resilience override                       |

### World Model & Knowledge Store

| Field                                          | Type    | Default       | Description                       |
| ---------------------------------------------- | ------- | ------------- | --------------------------------- |
| `world_model.type`                             | string  | `statistical` | World model type                  |
| `world_model.statistical.window_size`          | integer | `100`         | Recent samples to keep per metric |
| `world_model.statistical.use_kalman`           | boolean | `true`        | Enable Kalman filtering           |
| `knowledge_store.type`                         | string  | `memory`      | Knowledge store type              |
| `knowledge_store.memory.max_states_per_system` | integer | `1000`        | Max states per system             |

### Meta-Learner

| Field                                   | Type    | Default                              | Description                              |
| --------------------------------------- | ------- | ------------------------------------ | ---------------------------------------- |
| `meta_learner.enabled`                  | boolean | `true`                               | Enable/disable meta-learning             |
| `meta_learner.type`                     | string  | `statistical`                        | Type: statistical or llm                 |
| `meta_learner.analysis_interval_hours`  | float   | `1.0`                                | Analysis interval in hours               |
| `meta_learner.conservative_mode`        | boolean | `true`                               | Conservative mode (cautious adjustments) |
| `meta_learner.transparency.enabled`     | boolean | `true`                               | Persist per-cycle update transparency    |
| `meta_learner.transparency.output_path` | string  | `./logs/meta_learning_updates.jsonl` | JSONL file for transparency records      |
| `meta_learner.llm.provider`             | string  | `google`                             | LLM provider (if type=llm)               |
| `meta_learner.llm.temperature`          | float   | `0.1`                                | Model creativity                         |
| `meta_learner.llm.auto_apply`           | boolean | `false`                              | Auto-apply proposals                     |
| `meta_learner.llm.resilience.*`         | object  | -                                    | LLM resilience settings                  |

### Observability

**Logging:**

| Field                              | Type    | Default              | Description                        |
| ---------------------------------- | ------- | -------------------- | ---------------------------------- |
| `observability.logging.type`       | string  | `human`              | Type: human or structured          |
| `observability.logging.level`      | string  | `INFO`               | Level: DEBUG, INFO, WARNING, ERROR |
| `observability.logging.console`    | boolean | `true`               | Log to console                     |
| `observability.logging.file`       | boolean | `true`               | Log to file                        |
| `observability.logging.file_path`  | string  | `./logs/polaris.log` | Log file path                      |
| `observability.logging.use_colors` | boolean | `true`               | Colorized output                   |

**Metrics:**

| Field                                                       | Type    | Default           | Description                            |
| ----------------------------------------------------------- | ------- | ----------------- | -------------------------------------- |
| `observability.metrics.enabled`                             | boolean | `true`            | Enable metrics collection              |
| `observability.metrics.collector_type`                      | string  | `simple`          | Collector: simple, prometheus, datadog |
| `observability.metrics.simple.histogram_max_values`         | integer | `1000`            | Histogram max values                   |
| `observability.metrics.export.enabled`                      | boolean | `false`           | Enable metrics export                  |
| `observability.metrics.export.formats`                      | array   | `["json", "csv"]` | Export formats                         |
| `observability.metrics.export.output_dir`                   | string  | `./metrics`       | Export directory                       |
| `observability.metrics.export.auto_export_interval_minutes` | integer | `60`              | Auto-export interval                   |
| `observability.metrics.export.experiment_name`              | string  | -                 | Experiment name                        |
| `observability.metrics.export.include_timestamp`            | boolean | `true`            | Include timestamp                      |
| `observability.metrics.components.*`                        | boolean | `true`            | Per-component metrics control          |

---

## Configuration Variable Mapping

### Where Configuration is Loaded

| Config Path                       | Loaded By                                     | File                              | Line    |
| --------------------------------- | --------------------------------------------- | --------------------------------- | ------- |
| `systems`                         | [config.py](polaris/infrastructure/config.py) | SystemConfig.**init**             | 23      |
| `monitoring.interval_seconds`     | [polaris.py](polaris/core/polaris.py)         | **init**                          | 153-159 |
| `strategy.type`                   | [config.py](polaris/infrastructure/config.py) | StrategyConfig                    | 166-172 |
| `strategy.params` (threshold)     | [factories.py](polaris/core/factories.py)     | \_threshold_factory               | 121-130 |
| `strategy.params` (llm_reasoning) | [factories.py](polaris/core/factories.py)     | \_llm_reasoning_factory           | 143-161 |
| `strategy.params` (hybrid)        | [factories.py](polaris/core/factories.py)     | \_hybrid_factory                  | 167-224 |
| `strategy.params` (agentic_llm)   | [factories.py](polaris/core/factories.py)     | \_agentic_llm_factory             | 231-282 |
| `observability.logging`           | [polaris.py](polaris/core/polaris.py)         | \_create_logger_from_config       | 185-195 |
| `observability.metrics`           | [polaris.py](polaris/core/polaris.py)         | \_create_metrics_from_config      | 206-225 |
| `meta_learner`                    | [polaris.py](polaris/core/polaris.py)         | \_create_meta_learner_from_config | 327-395 |

### Validation Rules

| Validation                  | File                                          | Line         |
| --------------------------- | --------------------------------------------- | ------------ |
| System ID not empty         | [config.py](polaris/infrastructure/config.py) | 28-29        |
| Connector type registered   | [config.py](polaris/infrastructure/config.py) | 31-35        |
| Port 1-65535                | [config.py](polaris/infrastructure/config.py) | 40-42, 47-49 |
| Threshold high > low        | [config.py](polaris/infrastructure/config.py) | 75-77        |
| Cooldown non-negative       | [config.py](polaris/infrastructure/config.py) | 79-81        |
| Hybrid selection_mode valid | [config.py](polaris/infrastructure/config.py) | 95-97        |
| Min confidence 0.0-1.0      | [config.py](polaris/infrastructure/config.py) | 99-104       |

### CLI Overrides

| CLI Argument            | Config Key                                | Code                                       |
| ----------------------- | ----------------------------------------- | ------------------------------------------ |
| `--monitoring-interval` | `monitoring.interval_seconds`             | [polaris.py](polaris/core/polaris.py#L162) |
| `--log-format`          | `observability.logging.type`              | [polaris.py](polaris/core/polaris.py#L200) |
| `--log-level`           | `observability.logging.level`             | [polaris.py](polaris/core/polaris.py#L201) |
| `--log-file`            | `observability.logging.file_path`         | [polaris.py](polaris/core/polaris.py#L205) |
| `--metrics-export-dir`  | `observability.metrics.export.output_dir` | [polaris.py](polaris/core/polaris.py#L244) |

---

## Troubleshooting Guide

### YAML Syntax Issues

**Issue**: "YAML syntax error" or configuration fails to load

**Solutions**:

1. Use 2-space indentation (never tabs)
2. Check for trailing colons after keys
3. Ensure quotes are balanced
4. Validate with: `python3 -c "import yaml; yaml.safe_load(open('config/default.yaml'))"`

**Example errors**:

```yaml
# ❌ Wrong: trailing colon
strategy:
  type: "threshold":

# ✓ Correct
strategy:
  type: "threshold"
```

### Port Configuration

**Issue**: "Port must be between 1 and 65535"

**Solution**:

- Port must be integer (not quoted string)
- Value must be 1-65535

**Example**:

```yaml
# ❌ Wrong
port: "4242"  # String!

# ✓ Correct
port: 4242    # Integer
```

### Threshold Configuration

**Issue**: "High threshold must be greater than low threshold"

**Solution**:

- For each metric, ensure `high > low`
- Example: `{ high: 80.0, low: 20.0 }` ✓

```yaml
# ❌ Wrong
cpu_usage:
  high: 20.0
  low: 80.0

# ✓ Correct
cpu_usage:
  high: 80.0
  low: 20.0
```

### API Credentials

**Issue**: "API credentials not found" or "GOOGLE_API_KEY not set"

**Solution**:

```bash
# For Google
export GOOGLE_API_KEY="your-key-here"

# OR for OpenAI
export OPENAI_API_KEY="your-key-here"

# Verify:
echo $GOOGLE_API_KEY
```

### Environment Variables

**Issue**: "Environment variable 'SWIM_HOST' not found"

**Solution**:

- Export the variable before running Polaris
- Or remove `${VARIABLE}` syntax from config if not needed

```bash
export SWIM_HOST="localhost"
export SWIM_PORT="4242"
```

### Temperature Validation

**Issue**: "Temperature outside valid range"

**Solution**:

- Temperature must be between 0.0 and 1.0
- 0.0 = deterministic (repeatable)
- 1.0 = creative (variable)
- Recommended: 0.1 for production

### Confidence Range

**Issue**: "min_confidence must be between 0.0 and 1.0"

**Solution**:

```yaml
strategy:
  type: "hybrid"
  params:
    min_confidence: 0.7 # Must be 0.0-1.0
```

### File Paths

**Issue**: "Directory does not exist" for logs or metrics

**Solution**:

```bash
# Create directories
mkdir -p ./logs
mkdir -p ./metrics

# Ensure permissions (user can write)
ls -ld ./logs ./metrics
```

### Common Configuration Patterns

#### Development (Threshold)

- Use simple threshold strategy
- Enable DEBUG logging
- Local connections (localhost)
- Small monitoring intervals (10-30 seconds)

#### Production (LLM-based)

- Use LLM reasoning strategy
- Enable structured logging
- Remote connections with timeouts
- Metrics export enabled
- Meta-learning enabled

#### Multi-system

- Multiple entries in `systems` array
- Hybrid strategy for strict multi-strategy orchestration
- Per-system prompts for specialized handling

---

## Advanced Configuration Topics

### Registering Custom Connectors

```python
from polaris.core.factories import register_connector_factory

def my_connector_factory(system_cfg, logger, metrics):
    return MyConnector(
        base_url=system_cfg.connection.get("base_url"),
        logger=logger,
        metrics=metrics,
    )

register_connector_factory("my_connector", my_connector_factory)
```

Use in config:

```yaml
systems:
  - id: "my-system"
    connector_type: "my_connector"
    connection:
      base_url: "http://localhost:1234"
```

### Registering Custom Strategies

```python
from polaris.core.factories import register_strategy_factory

def my_strategy_factory(strategy_cfg, logger, metrics, knowledge_store, world_model, registry):
    return MyStrategy(logger=logger, metrics=metrics)

register_strategy_factory("my_strategy", my_strategy_factory)
```

Use in config:

```yaml
strategy:
  type: "my_strategy"
```

### Environment Variable Substitution

Use `${VARIABLE_NAME}` syntax in configuration:

```yaml
systems:
  - id: "production"
    connection:
      host: "${SWIM_HOST}"
      port: ${SWIM_PORT}
```

Export variables before loading:

```bash
export SWIM_HOST="swim.example.com"
export SWIM_PORT="4242"
```

### Performance Tuning

**Monitoring Interval**:

- Too frequent (< 5s): May overload servers
- Too infrequent (> 300s): May miss important changes
- Recommended: 10-60 seconds

**World Model Window Size**:

- Too small (< 10): Limited trend data
- Too large (> 1000): High memory usage
- Recommended: 100-500

**Knowledge Store Max States**:

- Too small (< 100): Limited history
- Too large (> 10000): High memory usage
- Recommended: 500-2000

**Metrics Histogram Max**:

- Too small (< 100): Coarse data
- Too large (> 10000): High memory
- Recommended: 1000

---

## Security Best Practices

### API Keys

- **Never commit API keys** in configuration files
- **Use environment variables** for all credentials
- **Restrict file permissions**: `chmod 600 config/default.yaml`

### Configuration Files

```bash
# Secure configuration
chmod 600 config/default.yaml

# If config contains secrets
chmod 600 config/secrets.yaml
```

### Production Deployment

1. Use structured logging (enables log analysis)
2. Enable metrics export for monitoring
3. Set reasonable monitoring intervals
4. Use conservative mode for meta-learner
5. Test configuration before deployment

```bash
# Validate before deploying
python3 -c "from polaris.infrastructure.config import load_config; load_config('config/default.yaml')"
```

---

## Configuration Loading Order

1. Load YAML file from disk
2. Substitute environment variables (`${VAR_NAME}` syntax)
3. Validate configuration structure
4. Register custom factories (if any)
5. Apply CLI overrides
6. Create Polaris instance with validated config

---

## See Also

- [config/default.yaml](config/default.yaml) - Complete example configuration
- [examples/config_usage.py](examples/config_usage.py) - Basic usage example
- [examples/config_validation.py](examples/config_validation.py) - Validation testing
- [examples/llm_powered.py](examples/llm_powered.py) - LLM strategy example
- [README.md](README.md) - Project overview
