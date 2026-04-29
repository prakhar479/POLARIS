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
   - For the agentic stage, include a `per_system_prompts.swim` block instructing the LLM on:
     - What metrics to watch
     - What actions are available and when to use them
     - Safety constraints (e.g. don't scale down when response time is still high)
4. **Observability** — human-readable logging and metrics export enabled
5. *(Optional bonus)* **World model** and/or **meta-learner** configuration

### Constraints

- Use **only** SWIM-supported metrics in thresholds (see table above)
- Use **only** the canonical SWIM actions: `scale_up`, `scale_down`, `set_dimmer`
- For the agentic LLM stage, choose any supported provider with available credentials (`google`, `openai`, `openrouter`, `groq`, `ollama`)
- The config must pass `polaris doctor --config <config_file.yaml>` without errors

### Scoring

| Criterion | Points |
|---|---|
| Config loads without errors (`polaris doctor`) | 20 |
| SWIM connection correctly configured | 10 |
| Threshold stage uses SWIM-appropriate metrics with correct `action_templates` | 20 |
| Agentic LLM stage present and structurally valid | 20 |
| Agentic prompt (`per_system_prompts.swim`) captures SWIM domain semantics | 15 |
| Safety constraints in the prompt (anti-oscillation, bounds checking) | 10 |
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

### 6.2 Recommended: Hybrid Threshold + Agentic LLM 

This is the recommended configuration for SWIM experiments. The threshold guard handles
clear SLA breaches immediately; the agentic branch handles pre-emptive trend detection,
scale-down, and dimmer recovery.

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

      # Stage 2: agentic LLM — handles nuance the threshold misses
      - type: "agentic_llm"
        priority: 0.6
        params:
          provider: "google"          # or "openai", "openrouter", "groq", "ollama"
          temperature: 0.1
          steps_limit: 3
          per_system_prompts:
            swim: |
              You are an intelligent adaptation controller for the SWIM web server pool.

              SWIM metrics (collected every 5 s):
              - average_response_time: weighted avg response time (ms). SLA target: <= 750 ms.
              - average_utilization: server utilization [0.0, 1.0]. > 0.80 = high, < 0.30 = low.
              - server_count: total servers (max 3).
              - dimmer: optional-content ratio [0.0, 1.0]. 1.0 = full service.

              Available actions: scale_up, scale_down, set_dimmer (params: {"value": 0.0–1.0}).

              Role (the threshold guard handles clear SLA breaches):
              - Detect pre-SLA trends (response time rising toward 750 ms).
              - Handle scale-down when load is sustainably low (utilization < 0.30 for 3+ windows).
              - Restore dimmer gradually (increase by at most 0.2 per step) when load stabilises.

              Safety rules:
              - NEVER scale_down when average_response_time > 400 ms.
              - NEVER scale_down when average_utilization > 0.60.
              - NEVER set dimmer below 0.10.
              - Propose at most ONE action per decision.

              Use tools to inspect trends before deciding. Reply in strict JSON only.
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
            max_retries: 4
            base_backoff_ms: 200
            max_backoff_ms: 4000

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
      output_dir: "./metrics"
      auto_export_interval_minutes: 5
```

**Required env var for the `google` provider:**

```bash
export GOOGLE_API_KEY="your-key-here"
```

For other providers: `OPENAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`.

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

Without this, the agent cannot call any tools. Add at least `list_supported_actions`
and `get_recent_states`.

### SWIM unreachable

Verify host/port in config. Verify the SWIM process is running and listening. Run
`polaris doctor` to validate connectivity config before starting.

### Too many adaptations

Increase `cooldown_seconds`. Narrow threshold ranges. Add explicit anti-oscillation
constraints to the agentic strategy's `per_system_prompts`.

---

## 9. Documentation Reading Order

1. This file (`docs/SWIM_EXAMPLE_GUIDE.md`) — start here
2. `docs/STRATEGIES_DETAILED.md` — per-strategy deep dive and hybrid composition patterns
3. `CONFIGURATION.md` — complete config parameter reference
4. `README.md` — framework overview and quick-start
