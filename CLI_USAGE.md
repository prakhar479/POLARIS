# Polaris CLI Usage Guide

## Overview

The Polaris CLI provides command-line interface for managing and monitoring the Polaris self-adaptive systems framework.

## Installation

```bash
pip install rich  # Required for enhanced features
```

## Basic Usage

```bash
python -m polaris.cli [OPTIONS]

# Diagnostics
python -m polaris.cli doctor [--config config/default.yaml]
```

## Command Options

### Core Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to configuration file (YAML) |
| `--version` | `-v` | Show version and exit |

### Interface Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dashboard` | `-d` | Launch interactive dashboard |
| `--interactive` | `-i` | Launch interactive CLI in single-process mode |
| `--both` | `-b` | Run dashboard + interactive CLI together (split-screen mode) |
| `--no-clear` | `-` | Do not clear the terminal when launching the dashboard |

### Logging Options

| Option | Description |
|--------|-------------|
| `--export-logs FILE` | Export logs to specified file |
| `--log-format {structured,human}` | Log format type |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Log level |

### Metrics Options

| Option | Description |
|--------|-------------|
| `--metrics-export DIR` | Export metrics to directory |
| `--metrics-format {json,csv,both}` | Metrics export format |
| `--metrics-experiment NAME` | Experiment name for metrics |
| `--disable-metrics` | Disable metrics collection |
| `--auto-export-metrics MINUTES` | Auto-export interval |

## Usage Examples

### Standard Operation
```bash
# Basic framework execution
python -m polaris.cli --config config/default.yaml

# With logging
python -m polaris.cli --config config/default.yaml --export-logs polaris.log
```

### Dashboard Mode
```bash
# Launch dashboard
python -m polaris.cli --dashboard --config config/default.yaml

# Dashboard with metrics export
python -m polaris.cli --dashboard --config config/default.yaml --metrics-export ./metrics
```

### Interactive CLI Mode
```bash
# Launch interactive CLI (single-process mode)
python -m polaris.cli --interactive --config config/default.yaml

# Dashboard + interactive CLI in one split-screen terminal
python -m polaris.cli --both --config config/default.yaml
```

### Doctor Diagnostics
```bash
# Run config/env/optional-dependency diagnostics
python -m polaris.cli doctor --config config/default.yaml

# Treat warnings as failures
python -m polaris.cli doctor --config config/default.yaml --strict
```

## Dashboard Features

The dashboard provides real-time monitoring with:

- Connected systems status
- Current metrics with trend indicators
- Recent adaptation events
- Strategy performance information
- System metrics (monitoring, telemetry, adaptations)
- Latest summarized logs panel (time, level, component, short message)

Navigation: Use Ctrl+C to exit.

## Interactive CLI Commands

The interactive CLI provides the following commands:

| Command | Usage | Description |
|---------|-------|-------------|
| `status` | `status` | Show system status |
| `systems` | `systems` | List connected systems |
| `metrics` | `metrics [component]` | Show metrics (optionally filtered) |
| `knowledge` | `knowledge <system_id> [hours]` | Query knowledge base |
| `worldmodel` | `worldmodel [system_id]` | Query world model insights |
| `predict` | `predict <system_id> <action> [params]` | Predict action outcome |
| `export` | `export <file> [format]` | Export metrics to file |
| `history` | `history [N]` | Show recent command history |
| `clear` | `clear` | Clear screen |
| `help` | `help` | Show available commands |
| `quit` | `quit` or `exit` | Exit CLI |

Interactive CLI UX features:
- Aliases: `h` (help), `q` (quit), `wm` (worldmodel), `ks` (knowledge), `st` (status)
- Repeat previous command: `!!`
- Tab completion for key commands/arguments

## Configuration

Polaris requires a YAML configuration file. Example structure:

```yaml
observability:
  logging:
    type: "structured"
    level: "INFO"
  metrics:
    enabled: true
    collector_type: "simple"

systems:
  - id: "system1"
    connector_type: "swim"
    enabled: true
```

Connector types (`systems[].connector_type`) and strategy types (`strategy.type`) must match types registered in Polaris' factory registries.
Built-in types are registered automatically; custom types can be registered via the factory registration APIs.

See `CONFIGURATION.md` for the registration pattern and examples.

## Exit Codes

- `0`: Success
- `1`: Error occurred
- Ctrl+C: Graceful shutdown

## Requirements

- Python 3.8+
- `rich` library (for dashboard and interactive CLI)
- Valid YAML configuration file

## Notes

- `polaris doctor` checks runtime, config validity, environment variables, and optional dependencies.
- `--both` runs a clean split-screen mode: live dashboard on top, interactive command pane at bottom.
- Split-screen mode includes Up/Down history navigation and Tab command completion.
- Dashboard updates in real-time (1-second intervals)
- Dashboard log panel shows a concise, human-readable summary of recent logs; full raw logs remain available via log files/exports.
- All CLI options can be combined as needed
- Metrics collection can be disabled for minimal overhead
