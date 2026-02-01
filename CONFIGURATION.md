# Polaris Configuration Guide

This document provides comprehensive information about configuring the Polaris framework.

## Configuration File Structure

Polaris uses YAML configuration files with the following structure:

```yaml
# Core monitoring settings
monitoring:
  interval_seconds: 30  # How often to run monitoring loop

# Managed systems
systems:
  - id: "system-name"
    connector_type: "swim"  # Currently supported: swim
    enabled: true
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: 5

# Adaptation strategy
strategy:
  type: "threshold"  # Options: threshold, llm_reasoning, hybrid
  
  # Threshold strategy configuration
  threshold:
    thresholds:
      cpu_usage:
        high: 80.0
        low: 20.0
      memory_usage:
        high: 85.0
        low: 25.0
    cooldown_seconds: 60
    enabled: true
  
  # LLM reasoning strategy configuration
  llm_reasoning:
    system_description: "Web application server pool"
    adaptation_goals: "Maintain performance with minimal resource usage"
    temperature: 0.1

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
knowledge:
  type: "memory"
  memory:
    max_states_per_system: 1000

# Meta-learner configuration
meta_learner:
  enabled: true
  type: "statistical"
  analysis_interval_hours: 1.0
  conservative_mode: true

  # When a statistical world model is provided to Polaris, the
  # StatisticalMetaLearner automatically consumes its insights via
  # world_model.get_insights(). The meta-learner aggregates basic
  # uncertainty information (e.g., average metric std and regime
  # estimates) into analysis.insights["world_model_uncertainty"], and
  # uses this to slightly adjust proposal confidence (more cautious
  # when variability is high).

# Observability configuration
observability:
  logging:
    type: "human"  # Options: structured, human
    level: "INFO"  # DEBUG, INFO, WARNING, ERROR
    console: true
    file: true
    file_path: "./logs/polaris.log"
    use_colors: true
  
  metrics:
    enabled: true
    collector_type: "simple"  # Options: simple, prometheus, datadog
    
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
- `connector_type` must be supported ("swim" currently)
- `port` must be a valid integer between 1-65535

### Strategy Configuration Validation
- `type` must be one of: "threshold", "llm_reasoning", "hybrid"
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
  threshold:
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

# Override monitoring interval
polaris --config config.yaml --monitoring-interval 60
```

### Available CLI Overrides

| CLI Argument | Config Section | Description |
|--------------|----------------|-------------|
| `--log-level` | `observability.logging.level` | Log level |
| `--log-format` | `observability.logging.type` | Log format |
| `--export-logs` | `observability.logging.file_path` | Log file path |
| `--metrics-export` | `observability.metrics.export.output_dir` | Metrics export directory |
| `--disable-metrics` | `observability.metrics.enabled` | Disable metrics |
| `--monitoring-interval` | `monitoring.interval_seconds` | Monitoring loop interval |

## Strategy Configuration

### Threshold Strategy

```yaml
strategy:
  type: "threshold"
  threshold:
    thresholds:
      cpu_usage:
        high: 80.0    # Trigger scale-up at 80% CPU
        low: 20.0     # Trigger scale-down at 20% CPU
      memory_usage:
        high: 85.0
        low: 25.0
      response_time:
        high: 500.0   # milliseconds
    cooldown_seconds: 60  # Wait 60s between adaptations
    enabled: true
```

### LLM Reasoning Strategy

```yaml
strategy:
  type: "llm_reasoning"
  llm_reasoning:
    system_description: "SWIM web application server pool"
    adaptation_goals: "Maintain performance with minimal resource usage"
    temperature: 0.1  # Lower = more deterministic
```

**Requirements**: 
- Set `GOOGLE_API_KEY` or `OPENAI_API_KEY` environment variable
- Install LLM client libraries: `pip install google-generativeai` or `pip install openai`

### LLM Resilience (Retries, Rate Limiting, Key Rotation)

You can enable a resilience layer for LLM calls that adds retries with exponential backoff, async rate limiting, optional concurrency caps, and API key rotation.

Enable via strategy config (recommended):

```yaml
strategy:
  type: "llm_reasoning"
  llm_reasoning:
    temperature: 0.1
    resilience:
      rps: 2              # tokens refill rate per second for token-bucket
      burst: 4            # bucket capacity for short bursts
      concurrency: 4      # max concurrent LLM requests
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
      collection_interval: 5  # seconds
```

## Observability Configuration

### Logging

```yaml
observability:
  logging:
    type: "human"        # "structured" for JSON, "human" for readable
    level: "INFO"        # DEBUG, INFO, WARNING, ERROR
    console: true        # Log to console
    file: true          # Log to file
    file_path: "./logs/polaris.log"
    use_colors: true    # Colored output (console only)
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
- LLM reasoning: `temperature`, `system_description`, and LLM `resilience` (when using the resilient client)
- Hybrid strategy: `selection_mode`, `min_confidence`, and sub-strategy parameters (index-matched), including resilience for LLM sub-strategies

Limitations:
- Changing the strategy type (e.g., threshold → hybrid) requires a restart
- Hybrid sub-strategy updates are index-based; to change counts or order, restart with the new configuration
- Resilience updates require the LLM client to be the resilient wrapper; otherwise the update is skipped with a warning

Best practices:
- Keep resilience and strategy parameters in the main config file passed to Polaris
- Avoid frequent file writes; changes are detected each iteration of the monitoring loop
```

## Configuration Loading Order

1. Load YAML file
2. Substitute environment variables (`${VAR}` syntax)
3. Validate configuration structure
4. Apply CLI overrides
5. Create Polaris instance

## Error Handling

The framework provides clear error messages for configuration issues:

```
ValueError: Unsupported connector type 'invalid'. Supported: ['swim']
ValueError: Environment variable 'MISSING_VAR' not found. Please set it or remove ${MISSING_VAR} from config.
ValueError: High threshold must be greater than low threshold for 'cpu_usage'
```

## Best Practices

1. **Use environment variables** for sensitive data (API keys, passwords)
2. **Validate configuration** using the provided validation example
3. **Start with default.yaml** and customize as needed
4. **Use CLI overrides** for development and testing
5. **Enable structured logging** for production environments
6. **Configure metrics export** for analysis and debugging

## Example Configurations

See the `examples/` directory for complete configuration examples:

- `examples/config_usage.py` - Basic configuration usage
- `examples/config_validation.py` - Configuration validation testing
- `examples/llm_powered.py` - LLM strategy configuration
- `config/default.yaml` - Complete example configuration

## Troubleshooting

### Common Issues

1. **"Environment variable not found"**
   - Set the required environment variable or remove `${VAR}` from config

2. **"Unsupported connector type"**
   - Currently only "swim" connector is supported

3. **"LLM strategy not available"**
   - Install required packages: `pip install google-generativeai`
   - Set API key environment variable

4. **"Config file not found"**
   - Check file path and ensure file exists
   - Use absolute path if needed

### Debug Configuration Loading

Enable debug logging to see configuration loading details:

```bash
polaris --config config.yaml --log-level DEBUG
```

This will show:
- Configuration file loading
- Environment variable substitution
- Validation results
- Component creation details