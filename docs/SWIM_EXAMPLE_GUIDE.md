# SWIM End-to-End Usage Guide


## 0. What is SWIM?

**SWIM (Simulated Web Infrastructure Manager)** simulates a web server pool that serves a
mix of basic and optional content under time-varying load. The managing system operator
configures Polaris to automatically keep the server pool healthy.

The server pool has up to **3 servers**. When load rises, servers are added or optional
content is reduced. When load drops, the pool scales back down to avoid wasting resources.

---

## 1. Prerequisites

- A running SWIM endpoint reachable at host and port in config.
- Python 3.11+.
- Polaris installed in the current environment.

Recommended install (repo root):

```bash
pip install -c requirements/constraints.txt -e .
```

Default SWIM connection expected by Polaris:

- host: `localhost`
- port: `4242`

Reference config files:

- Minimal baseline: [config/default.yaml](../config/default.yaml)
- SWIM native-tool-calling config: [config/swim.yaml](../config/swim.yaml)
- Full hybrid experiment: [config/swim_thread_threshold_hybrid_experiment.yaml](../config/swim_thread_threshold_hybrid_experiment.yaml)

---

## 2. SWIM Metrics and Actions

Understanding the SWIM interface is essential before writing a config.

### Available Metrics

| Metric | Unit | Description |
|---|---|---|
| `server_count` | int | Total servers in the pool |
| `active_servers` | int | Currently active servers |
| `max_servers` | int | Upper bound on server count (typically 3) |
| `dimmer` | float 0–1 | Optional content ratio (1.0 = full service) |
| `basic_response_time` | ms | Latency for basic requests |
| `optional_response_time` | ms | Latency for optional requests |
| `basic_throughput` | req/s | Throughput for basic service |
| `optional_throughput` | req/s | Throughput for optional service |
| `average_response_time` | ms | Weighted average response time — **primary SLA metric** |
| `average_utilization` | float 0–1 | Server utilization ratio. >0.80 = stressed; <0.30 = underutilised |

### Supported Actions

Use these **canonical action names** in all configs and LLM prompts:

| Action | Parameters | Effect |
|---|---|---|
| `scale_up` | none | Add one server to the pool (fails if already at `max_servers`) |
| `scale_down` | none | Remove one server (fails if only 1 server remains) |
| `set_dimmer` | `{"value": 0.0–1.0}` | Set the optional-content ratio |

> **Note on action names:** The underlying SWIM TCP protocol uses `add_server` and
> `remove_server`. Polaris normalizes these automatically — always use `scale_up` and
> `scale_down` in your YAML config and LLM prompts. The aliases `add_server`,
> `remove_server`, and `adjust_qos` are also accepted but `scale_up`/`scale_down`/`set_dimmer`
> are the canonical names.

### Quick-Reference Constraints

| Fact | Value |
|---|---|
| Max servers | 3 |
| Min servers | 1 |
| Dimmer range | 0.0 – 1.0 |
| Minimum safe dimmer | 0.10 (never set below this) |
| High utilization threshold | > 0.80 |
| Low utilization threshold | < 0.30 |
| Recommended monitoring interval | 5–30 s |
| Recommended threshold cooldown | 30–90 s |
| Recommended hybrid cooldown | 60–120 s |

---

## 3. The Task

Write a YAML configuration file that connects Polaris to a SWIM instance and configures
it to **autonomously manage the server pool**.

### Scenario

- SWIM is running on `localhost:4242`
- The SLA target is: **`average_response_time` must stay below 800 ms**
- Over-provisioning should be avoided (3 servers should not remain active when 1 would suffice)
- Monitoring data should be collected every **5 seconds**

### Requirements

The configuration file must include:

1. **System connection** — connect to SWIM on `localhost:4242` with 5 s collection interval
2. **Monitoring interval** — how often Polaris evaluates the system (recommend: 10–30 s)
3. **Adaptation strategy** — the main deliverable:
   - A **hybrid strategy** combining:
     - A **threshold stage** — fast, deterministic; fires immediately when thresholds are clearly breached
     - An **agentic LLM stage** — intelligent, trend-aware; handles nuance the threshold misses (pre-emptive scale-up, scale-down decisions, dimmer recovery)
   - For the agentic stage:
     - Include a `system_prompt` instructing the LLM on metrics, available actions, and safety constraints
     - Enable **native tool calling** via a `native_tools` block that defines all SWIM adaptation
       actions and Polaris analysis tools as OpenAI-format function schemas
     - Enable the corresponding Polaris built-in tools in `tools.enabled`
4. **Observability** — human-readable logging and metrics export enabled
5. *(Optional bonus)* **World model** and/or **meta-learner** configuration

### Constraints

- Use **only** SWIM-supported metrics in thresholds (see table above)
- Use **only** the canonical SWIM actions: `scale_up`, `scale_down`, `set_dimmer`
- For the agentic LLM stage, choose any supported provider with available credentials (`google`, `openai`, `openrouter`, `groq`, `ollama`)
- The `native_tools` block must include every Polaris tool listed in `tools.enabled`
- The config must pass `polaris doctor --config <config_file.yaml>` without errors

### Scoring

| Criterion | Points |
|---|---|
| Config loads without errors (`polaris doctor`) | 20 |
| SWIM connection correctly configured | 10 |
| Threshold stage uses SWIM-appropriate metrics with correct `action_templates` | 20 |
| Agentic LLM stage present and structurally valid | 15 |
| `native_tools` block present with correct function schemas | 15 |
| Safety constraints in `system_prompt` (anti-oscillation, bounds checking) | 15 |
| Observability configured (logging + metrics export) | 5 |

**Bonus (up to 10 extra points):** world model block, meta-learner block, well-justified
cooldown comments, or a creative strategy composition.

---

## 4. Validate Before Run

Always run doctor first to catch config errors before starting a session:

```bash
polaris doctor --config config/my_swim.yaml
```

What doctor checks:

- Python runtime compatibility.
- YAML/config schema validity (strict `extra='forbid'` Pydantic models).
- Missing `${ENV_VAR}` substitutions in config.
- Optional UI dependency status (`rich`).
- LLM dependency and credential checks if LLM strategy sections are active.

---

## 5. Run Modes

Standard run:

```bash
polaris --config config/my_swim.yaml
```

Dry run (evaluate decisions without executing actions — useful for validating a config):

```bash
polaris --config config/my_swim.yaml --dry-run
```

Dashboard run (requires `rich`):

```bash
polaris --dashboard --config config/my_swim.yaml
```

Split-screen dashboard + interactive:

```bash
polaris --both --config config/my_swim.yaml
```

---

## 6. Strategy Configurations for SWIM

> **Recommended pattern: hybrid with threshold as the first stage and an agentic LLM as
> the second.** The threshold provides a fast deterministic safety net; the agentic branch
> handles nuanced trend-aware decisions and scale-down recovery.

### 6.1 Minimal Threshold-Only Config (Baseline Reference)

Use this as a baseline smoke test or for understanding the structure. For the study task,
use the hybrid configs below.

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
        high: {"type": "scale_up", "parameters": {}}
        low:  {"type": "scale_down", "parameters": {}}
      average_utilization:
        high: {"type": "set_dimmer", "parameters": {"value": 0.5}}
    cooldown_seconds: 60

observability:
  logging:
    type: "human"
    level: "INFO"
    console: true
  metrics:
    enabled: true
    collector_type: "simple"
```

> **`action_templates` is required.**
> The threshold strategy cannot translate a breached threshold into an action unless you
> tell it what action to emit. The `default` key acts as a fallback for any metric that
> doesn't have its own entry. Omitting this block causes a `ValueError` at the first
> threshold crossing.

#### `action_templates` structure

```yaml
action_templates:
  default:                                  # fallback for any metric
    high: {"type": "scale_up",   "parameters": {}}
    low:  {"type": "scale_down", "parameters": {}}
  average_utilization:                      # metric-specific override
    high: {"type": "set_dimmer", "parameters": {"value": 0.5}}
```

Resolution order when a threshold fires:
1. Per-metric template (`action_templates.<metric>.<high|low>`)
2. Default fallback (`action_templates.default.<high|low>`)
3. `ValueError` — configuration incomplete

---

### 6.2 Recommended: Hybrid Threshold + Agentic LLM with Native Tool Calling

This is the recommended configuration for SWIM experiments. The threshold guard handles
clear SLA breaches immediately; the agentic branch uses **native tool calling** to make
precise, structured decisions — the LLM calls predefined functions instead of emitting
free-form JSON text, which improves reliability and schema compliance.

The full ready-to-run version is available at [config/swim.yaml](../config/swim.yaml).
The abbreviated form below shows the key structural differences from a plain agentic config.

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

monitoring:
  interval_seconds: 10

strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 90
    strategies:
      # Stage 1: fast deterministic guard — fires immediately on SLA breach
      - type: "threshold"
        priority: 1.0
        params:
          thresholds:
            average_response_time:
              high: 750.0
            average_utilization:
              high: 0.85
          action_templates:
            default:
              high: {"type": "scale_up", "parameters": {}}
            average_utilization:
              high: {"type": "set_dimmer", "parameters": {"value": 0.5}}
          cooldown_seconds: 45

      # Stage 2: agentic LLM with native tool calling
      - type: "agentic_llm"
        priority: 0.6
        params:
          provider: "groq"          # or "openai", "google", "openrouter", "ollama"
          temperature: 0.05
          steps_limit: 3
          decision_cooldown_seconds: 20
          native_tools_unsupported_policy: skip_cycle

          system_prompt: |
            You are the adaptation controller for SWIM, a simulated web server pool.

            ## Metrics available each cycle
            - average_response_time: weighted average latency (ms). SLA target: <= 750 ms.
            - average_utilization: server load ratio [0.0, 1.0].
                > 0.80 = stressed; < 0.30 = idle / safe to scale down.
            - server_count: total servers (1–3).
            - dimmer: optional-content ratio [0.0, 1.0]. 1.0 = full service.

            ## Safety rules (hard constraints)
            - NEVER scale_down when average_response_time > 400 ms.
            - NEVER scale_down when average_utilization > 0.60.
            - NEVER set dimmer below 0.10.
            - NEVER scale_up when server_count is already 3.
            - NEVER scale_down when server_count is already 1.
            - Propose at most ONE action per cycle.

            ## Response instructions
            Use the provided functions to respond. Call get_recent_states,
            summarize_metric_trends, or get_action_history to gather evidence first.
            Then call scale_up, scale_down, set_dimmer, or no_adaptation.

          # Polaris built-in analysis tools — must also appear in native_tools
          tools:
            enabled:
              - get_recent_states
              - summarize_metric_trends
              - get_action_history

          # Native tool calling (OpenAI function-calling format).
          # The LLM receives these function definitions and must respond by
          # calling one of them.  Providers convert this format internally;
          # strategy logic stays provider-agnostic.
          native_tools:
            - type: "function"
              function:
                name: "get_recent_states"
                description: >
                  Query recent system states from the Polaris knowledge store.
                parameters:
                  type: "object"
                  properties:
                    window_seconds:
                      type: "integer"
                      minimum: 1
                      maximum: 3600
                      description: "Lookback window in seconds."
                    limit:
                      type: "integer"
                      minimum: 1
                      maximum: 200
                      description: "Maximum number of recent states to return."

            - type: "function"
              function:
                name: "summarize_metric_trends"
                description: >
                  Summarize recent trends for one metric.
                parameters:
                  type: "object"
                  properties:
                    metric:
                      type: "string"
                      description: "Metric name to summarize."
                    window_seconds:
                      type: "integer"
                      minimum: 1
                      maximum: 3600
                      description: "Lookback window in seconds."
                  required:
                    - "metric"

            - type: "function"
              function:
                name: "get_action_history"
                description: >
                  Query historical adaptation actions from the Polaris knowledge store.
                parameters:
                  type: "object"
                  properties:
                    window_seconds:
                      type: "integer"
                      minimum: 1
                      maximum: 2592000
                      description: "Lookback window in seconds."
                    limit:
                      type: "integer"
                      minimum: 1
                      maximum: 500
                      description: "Maximum number of actions to return."

            - type: "function"
              function:
                name: "scale_up"
                description: >
                  Add one server to the SWIM pool. Use when response time is rising
                  toward 750 ms or utilization is high.
                parameters:
                  type: "object"
                  properties:
                    reasoning:
                      type: "string"
                      description: "Why scaling up is needed."
                  required:
                    - "reasoning"

            - type: "function"
              function:
                name: "scale_down"
                description: >
                  Remove one server. Only when utilization < 0.30 and
                  response time < 400 ms (sustained).
                parameters:
                  type: "object"
                  properties:
                    reasoning:
                      type: "string"
                      description: "Why scaling down is safe."
                  required:
                    - "reasoning"

            - type: "function"
              function:
                name: "set_dimmer"
                description: >
                  Set the optional-content ratio. Reduce to shed load;
                  restore gradually (max +0.2 per step). Never below 0.10.
                parameters:
                  type: "object"
                  properties:
                    value:
                      type: "number"
                      minimum: 0.10
                      maximum: 1.0
                      description: "Target dimmer value [0.10, 1.0]."
                    reasoning:
                      type: "string"
                      description: "Why this dimmer value is appropriate."
                  required:
                    - "value"
                    - "reasoning"

            - type: "function"
              function:
                name: "no_adaptation"
                description: >
                  Signal that no adaptation is needed this cycle.
                parameters:
                  type: "object"
                  properties:
                    reasoning:
                      type: "string"
                      description: "Why no changes are needed."
                  required:
                    - "reasoning"

          resilience:
            rps: 1
            burst: 4
            max_retries: 3
            base_backoff_ms: 200
            max_backoff_ms: 3000

world_model:
  type: "statistical"
  statistical:
    window_size: 100
    use_kalman: true

knowledge_store:
  type: "memory"
  memory:
    max_states_per_system: 1000

observability:
  logging:
    type: "human"
    level: "INFO"
    console: true
    file: true
    file_path: "./logs/swim_hybrid.log"
  metrics:
    enabled: true
    collector_type: "simple"
    export:
      enabled: true
      formats: ["json", "csv"]
      output_dir: "./metrics/swim"
      auto_export_interval_minutes: 5
```

**Required env var for the `groq` provider:**

```bash
export GROQ_API_KEY="your-key-here"
```

For other providers: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`.

> **Why native tool calling?**
> In JSON-text mode the LLM must produce a correctly-formatted JSON object that Polaris
> then parses. Schema errors cause the cycle to fail.  With native tool calling, the
> provider API enforces the schema before the response reaches Polaris — the LLM must
> call one of the declared functions, which eliminates free-form hallucination of action
> names and parameters.
>
> **Fallback policy:** `native_tools_unsupported_policy: skip_cycle` means that if the
> chosen provider does not support function calling, the agentic stage is silently skipped
> for that cycle and the threshold guard continues protecting the system.

---

### 6.2.1 Native Tool Calling — Key Rules

| Rule | Detail |
|---|---|
| Every tool in `tools.enabled` must also appear in `native_tools` | Polaris validates this at startup |
| Adaptation functions (`scale_up`, `scale_down`, `set_dimmer`, `no_adaptation`) are **not** in `tools.enabled` | They are SWIM-specific and go directly in `native_tools` only |
| `system_prompt` replaces `per_system_prompts` in native-tool-calling configs | The prompt instructs the LLM to use the declared functions rather than emit JSON text |
| `native_tools_unsupported_policy` | Controls fallback when provider lacks function-calling support: `skip_cycle` (default), `json_fallback`, or `strict_fail` |

---

### 6.3 Alternative: Hybrid Threshold + Multi-Agent

Use this variant when you want stronger governance — three specialized agents
(Diagnostician → Planner → SafetyValidator) collaborate before any action is taken.

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 1.0
        params:
          thresholds:
            average_response_time:
              high: 750.0
          action_templates:
            default:
              high: {"type": "scale_up", "parameters": {}}
          cooldown_seconds: 45

      - type: "multi_agent"
        priority: 0.6
        params:
          provider: "google"
          temperature: 0.1
          steps_limit: 3
          system_description: "SWIM web server pool managing response time and utilization"
          diagnostician:
            temperature: 0.0
            steps_limit: 4
          planner:
            temperature: 0.2
          validator:
            temperature: 0.0
```

---

### 6.4 Alternative: Hybrid Threshold + Thread Agentic

Use this variant for recursive, decomposition-heavy analysis. The thread strategy can
spawn child reasoning threads for sub-problems.

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 90
    strategies:
      - type: "threshold"
        priority: 1.0
        params:
          thresholds:
            average_response_time:
              high: 750.0
          action_templates:
            default:
              high: {"type": "scale_up", "parameters": {}}
          cooldown_seconds: 45

      - type: "thread_agentic"
        priority: 0.6
        params:
          provider: "google"
          temperature: 0.1
          steps_limit: 4
          max_thread_depth: 2
          max_total_threads: 8
          child_timeout_seconds: 15.0
          assessment_cooldown_seconds: 60
          tools:
            enabled:
              - get_recent_states
              - summarize_metric_trends
              - predict_outcome
              - get_action_history
              - list_supported_actions
```

---

## 7. Monitoring and Export for Experiments

Enable auto-export in config:

```yaml
observability:
  metrics:
    enabled: true
    collector_type: "simple"
    export:
      enabled: true
      formats: ["json", "csv"]
      output_dir: "./metrics"
      auto_export_interval_minutes: 5
      experiment_name: "swim_run_01"
```

Or override at CLI:

```bash
polaris --config config/my_swim.yaml --metrics-export ./metrics --metrics-format both --metrics-experiment swim_run_01
```

---

## 8. Common Issues and Mistakes to Avoid

### Missing or wrong `action_templates`

**Error:** `ValueError: no action_template configured for metric='average_response_time' threshold_type='high'`

**Fix:** Add an `action_templates` block to the threshold strategy. Include a `default`
fallback at minimum.

### Using metrics SWIM doesn't expose

Using `cpu_usage`, `memory_usage`, or other generic metrics in thresholds will never fire
because SWIM does not report them. Use only the metrics listed in section 2.

### LLM provider credentials not set

When using `provider: google`, `GOOGLE_API_KEY` must be set. Run
`polaris doctor --config <file>` to check. The doctor command reports missing credentials
before the run starts.

### `high` ≤ `low` in thresholds

Setting `high: 0.30, low: 0.85` is a validation error. High must be strictly greater
than low when both are present.

### Missing `selection_mode` in hybrid strategy

Use `selection_mode: first` (threshold fires first, agentic fires only if threshold
doesn't) or `selection_mode: confidence` for weighted selection.

### Missing `tools.enabled` in the agentic stage

Without this, the agent cannot call any tools. Add at least `get_recent_states`
and `get_action_history`.

### `native_tools` entry missing for a tool in `tools.enabled`

**Error:** Polaris validates at startup that every name in `tools.enabled` also has a
corresponding entry in `native_tools`. If you add `summarize_metric_trends` to
`tools.enabled` but forget to add its function schema in `native_tools`, the strategy
will refuse to start.

**Fix:** Keep both lists in sync. The quickest check:
```bash
polaris doctor --config config/swim.yaml
```

### Using `per_system_prompts` instead of `system_prompt` in native-tool-calling configs

When `native_tools` is present, use the top-level `system_prompt` key (as in
`config/swim.yaml`). The `per_system_prompts.swim` key is for JSON-text-mode agentic
configs; mixing them can lead to the wrong prompt being used.

### SWIM unreachable

Verify host/port in config. Verify the SWIM process is running and listening. Run
`polaris doctor` to validate connectivity config before starting.

### Too many adaptations

Increase `cooldown_seconds`. Narrow threshold ranges. Add explicit anti-oscillation
constraints to the agentic strategy's `system_prompt` or `per_system_prompts`.

---

## 9. Documentation Reading Order

1. This file (`docs/SWIM_EXAMPLE_GUIDE.md`) — start here
2. `docs/STRATEGIES_DETAILED.md` — per-strategy deep dive and hybrid composition patterns
3. `CONFIGURATION.md` — complete config parameter reference
4. `README.md` — framework overview and quick-start
