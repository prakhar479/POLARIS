# MonitorAdapter Configuration Guide

This document describes all configuration options for the `MonitorAdapter` in `polaris_refactored/src/adapters/monitor_adapter.py`.

The adapter reads configuration from `AdapterConfiguration.config` (a dictionary). The following sections outline supported keys and their expected values.

## Top-level keys under `config`

- `collection_mode` (string)
  - Values: `"pull"`, `"push"`, `"hybrid"`
  - Default: `"pull"`
  - Behavior:
    - `pull`: The adapter periodically polls targets using a selected strategy.
    - `push`: The adapter subscribes to connectors that support `subscribe_telemetry(handler)` to receive telemetry.
    - `hybrid`: Both pull and push mechanisms are enabled.

- `default_strategy` (string)
  - Values: any registered strategy name. Common examples:
    - `"direct_connector"`, `"polling_direct_connector"`, `"batch_direct_connector"`
    - If retry is enabled, wrapped names like `"retrying_direct_connector"`, `"retrying_polling_direct_connector"` are available.
  - Default: `"direct_connector"` (if present) or the first registered strategy.
  - Notes: If set to an unknown strategy name, a warning is logged and the adapter falls back to its internal default.

- `strategies` (object)
  - Container for strategy-specific configuration. Known keys:

  - `strategies.retrying` (object)
    - `enabled` (bool; default: `false`)
    - `max_retries` (int; default: `3`)
    - `backoff_base` (float; default: `0.5` seconds)
    - `backoff_factor` (float; default: `2.0`)
    - `max_backoff` (float; default: `10.0` seconds)
    - `jitter` (float; default: `0.1`) — proportion of backoff used for jitter range.
    - Effect: When enabled, the adapter registers retrying variants for the base strategies.
      For example: `retrying_direct_connector`, `retrying_polling_direct_connector`, `retrying_batch_direct_connector`.

  - `strategies.batch_direct_connector` (object)
    - `batch_size` (int; default: `5`)
    - Effect: Customizes the batch size used by the `BatchCollectionStrategy` that wraps the `direct_connector` strategy.

- `monitoring_targets` (array of objects)
  - Each object is a `MonitoringTarget` description with fields:
    - `system_id` (string; required)
    - `connector_type` (string; required)
    - `collection_interval` (float; default: `30.0` seconds)
    - `enabled` (bool; default: `true`)
    - `config` (object; default: `{}`) — per-target overrides (see below).

## Per-target `config` fields

- `collection_strategy` (string)
  - Preferred strategy name for this target. The adapter selects this if available and supported; otherwise warns and falls back.

- `min_interval` (float)
  - Default: `max(1.0, collection_interval / 2)`
  - Lower bound for the adaptive next-interval computation within the pull loop.

- `max_interval` (float)
  - Default: `collection_interval * 4`
  - Upper bound for the adaptive next-interval computation within the pull loop.

- `success_adjustment` (float)
  - Default: `1.0`
  - If the last collection succeeded, next interval = `max(min_interval, collection_interval * success_adjustment)`.

- `failure_backoff` (float)
  - Default: `2.0`
  - If the last collection failed, next interval = `min(max_interval, collection_interval * failure_backoff)`.

## Telemetry Metadata

The adapter publishes `TelemetryEvent` instances with metadata tags:
- `system_id`: the target system ID
- `collection_strategy`: the actual strategy used (`CollectionResult.strategy_name` or `"push"` for push handlers)
- `collection_success`: `"true"` or `"false"`
- `collection_duration_seconds`: duration as a string (seconds)

## Post-initialization validation

After strategies are initialized, the adapter checks all targets for a specified `collection_strategy` and logs a warning if it is unknown. The adapter will then fall back to a default or the first supporting strategy.

## Example configuration (YAML-style)

```yaml
config:
  collection_mode: "hybrid"
  default_strategy: "retrying_polling_direct_connector"
  strategies:
    retrying:
      enabled: true
      max_retries: 3
      backoff_base: 0.5
      backoff_factor: 2.0
      max_backoff: 10.0
      jitter: 0.1
    batch_direct_connector:
      batch_size: 10
  monitoring_targets:
    - system_id: "service-A"
      connector_type: "k8s"
      collection_interval: 15.0
      enabled: true
      config:
        collection_strategy: "retrying_direct_connector"
        min_interval: 5.0
        max_interval: 60.0
        success_adjustment: 1.0
        failure_backoff: 2.0
    - system_id: "service-B"
      connector_type: "swim"
      collection_interval: 10.0
      enabled: true
      config:
        collection_strategy: "batch_direct_connector"
```
