# SWIM End-to-End Usage Guide

This guide is a walkthrough for running Polaris against the SWIM exemplar.

## 1. What You Need

- A running SWIM endpoint reachable at host and port in config.
- Python 3.11+.
- Polaris installed in the current environment.

Recommended install (repo root):

```bash
pip install -c requirements/constraints.txt -e .
```

Default SWIM connection expected by Polaris:

- host: localhost
- port: 4242

Reference config file:

- [config/default.yaml](../config/default.yaml)

## 2. Minimal SWIM Configuration

Use this as a minimal starting point:

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

## 3. Validate Before Run

Always run doctor first:

```bash
python -m polaris.cli doctor --config config/default.yaml
```

What doctor checks for SWIM runs:

- Python runtime compatibility.
- YAML/config schema validity.
- Missing environment variables used in config.
- Optional UI dependency status (rich).
- LLM dependency and credential checks if LLM strategy sections are active.

## 4. Run Modes

Standard run:

```bash
python -m polaris.cli --config config/default.yaml
```

Dashboard run:

```bash
python -m polaris.cli --dashboard --config config/default.yaml
```

Interactive CLI run:

```bash
python -m polaris.cli --interactive --config config/default.yaml
```

Split-screen dashboard + interactive run:

```bash
python -m polaris.cli --both --config config/default.yaml
```

Dry run (no action execution):

```bash
python -m polaris.cli --config config/default.yaml --dry-run
```

## 5. SWIM Metrics and Actions

Typical SWIM telemetry exposed through connector collection:

- server_count
- active_servers
- max_servers
- dimmer
- basic_response_time
- basic_throughput
- optional_response_time
- optional_throughput
- average_response_time
- average_utilization

Canonical SWIM actions:

- scale_up
- scale_down
- set_dimmer

Action aliases are normalized internally; for example:

- add_server -> scale_up
- remove_server -> scale_down
- adjust_qos -> set_dimmer

## 6. Recommended Strategy Pattern: Hybrid with Threshold First

For current production guidance, use `hybrid` with `threshold` as the first reactive stage.
For initial smoke testing, you can still run threshold-only briefly.

Example hybrid policy for SWIM (threshold-first):

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 0.9
        params:
          thresholds:
            average_response_time:
              high: 750.0
            average_utilization:
              high: 0.8
              low: 0.3
          cooldown_seconds: 45
      - type: "agentic_llm"
        priority: 0.6
        params:
          provider: "google"
          temperature: 0.1
          steps_limit: 3
```

Operational interpretation:

- If response time rises above high threshold, adaptation is proposed.
- If utilization is high, scale-up pressure increases.
- If utilization is sustainably low, scale-down may be proposed.
- Cooldown prevents adaptation thrashing.

How Polaris chooses the action when a threshold is breached:

- Threshold strategy resolves action templates for the breached metric/direction (`high` or `low`).
- Resolution order is metric-specific template first, then `default` template fallback.
- The strategy emits the first breached metric action in that assessment cycle, then applies cooldown.
- If a breach occurs but no matching action template exists, it is treated as a configuration error.

## 7. Moving to Agentic LLM on SWIM

Once threshold behavior is stable, move to agentic strategy.

Minimal agentic snippet:

```yaml
strategy:
  type: "agentic_llm"
  params:
    provider: "google"
    temperature: 0.1
    steps_limit: 3
    tools:
      enabled:
        - get_recent_states
        - summarize_metric_trends
        - predict_outcome
        - get_action_history
        - list_supported_actions
```

Recommended prompt guidance for SWIM:

- State hard constraints for server bounds.
- Require non-oscillatory behavior.
- Encourage trend-aware decisions over single-point decisions.

## 8. Monitoring and Export for Experiments

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
```

Or override at CLI:

```bash
python -m polaris.cli --config config/default.yaml --metrics-export ./metrics --metrics-format both --metrics-experiment swim_run_01
```

## 9. Common SWIM Issues

SWIM unreachable:

- Verify host/port in config.
- Verify SWIM process is running and listening.

No adaptation events:

- Check thresholds are not too permissive.
- Use interactive mode and inspect current metrics.

Too many adaptations:

- Increase cooldown.
- Narrow threshold ranges.
- Add policy constraints in strategy prompt (for LLM-based strategies).

## 10. Suggested Workflow for New Users

1. Start SWIM service.
2. Run doctor on default config.
3. Start with hybrid strategy where threshold is the first stage.
4. Confirm stable action execution.
5. Enable dashboard and metric export.
6. Tune secondary hybrid branch (`agentic_llm`, `multi_agent`, or `thread_agentic`) for advanced behavior.
