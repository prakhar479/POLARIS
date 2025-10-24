# POLARIS Integration with Switch System

This directory contains the POLARIS plugin for integrating with the Switch YOLO model switching system for conducting adaptation experiments.

## Overview

This integration connects POLARIS's adaptive control framework with the Switch exemplar, enabling:
- **Monitoring**: Collect telemetry metrics from Switch system (processing time, confidence, CPU usage, etc.)
- **Adaptation**: Execute model switching actions based on POLARIS reasoning
- **Experimentation**: Compare POLARIS adaptation strategies against Switch's native MAPE-K approach

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     POLARIS Framework                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Digital Twin │  │   Reasoner   │  │ Verification │     │
│  │  & World     │  │   (Kernel/   │  │   Adapter    │     │
│  │   Model      │  │   Agentic)   │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         │     NATS Message Bus                │              │
│         │  ┌───────────────────────────────┐ │              │
│         └──┤ Telemetry │ Actions │ Results ├─┘              │
│            └───────┬───────────┬───────────┘                │
│                    │           │                             │
│         ┌──────────▼───┐   ┌──▼────────────┐               │
│         │   Monitor    │   │   Execution   │               │
│         │   Adapter    │   │   Adapter     │               │
│         └──────┬───────┘   └──┬────────────┘               │
└────────────────┼──────────────┼─────────────────────────────┘
                 │              │
                 │  HTTP REST   │
                 │              │
         ┌───────▼──────────────▼────────┐
         │    Switch System Plugin       │
         │  ┌──────────────────────────┐ │
         │  │ SwitchSystemConnector    │ │
         │  │  - HTTP Client (aiohttp) │ │
         │  │  - File I/O (CSV)        │ │
         │  └──────────────────────────┘ │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │        Switch System            │
         │  ┌──────────┐  ┌─────────────┐ │
         │  │ Node.py  │  │ process.py  │ │
         │  │ (FastAPI)│  │  (YOLOv5)   │ │
         │  └────┬─────┘  └──────┬──────┘ │
         │       │               │         │
         │  ┌────▼───────────────▼──────┐ │
         │  │  Elasticsearch + Kibana   │ │
         │  └───────────────────────────┘ │
         └─────────────────────────────────┘
```

## Plugin Structure

```
switch_plugin/
├── config.yaml                    # Plugin configuration
├── connector.py                   # SwitchSystemConnector implementation
├── run_monitor_adapter.py         # Launch monitor adapter
├── run_execution_adapter.py       # Launch execution adapter
└── README.md                      # This file
```

## Prerequisites

### 1. Switch System Setup
The Switch system must be running with backend and database:

```bash
# In switch directory
cd polaris_poc/extern/switch

# Start Elasticsearch and Kibana
docker-compose up -d

# Verify services are running
curl http://localhost:9200/     # Elasticsearch
curl http://localhost:5601/     # Kibana

# Start Switch backend
cd NAVIE
python3 Node.py
```

The backend will be available at `http://localhost:3001`.

### 2. NATS Server
POLARIS requires a NATS server for message bus communication:

```bash
# Install NATS server (if not already installed)
# Option 1: Using Docker
docker run -d --name nats -p 4222:4222 nats:latest

# Option 2: Using package manager
# On macOS:
brew install nats-server
nats-server

# On Linux:
# Download from https://nats.io/download/
```

Verify NATS is running at `nats://localhost:4222`.

### 3. Python Dependencies
Ensure all POLARIS dependencies are installed:

```bash
cd polaris_poc/src
pip install -r requirements.txt  # If requirements file exists

# Or install key dependencies manually:
pip install aiohttp nats-py pyyaml jsonschema
```

## Configuration

### Plugin Configuration (`config.yaml`)

The plugin is pre-configured for local development. Key settings:

- **Connection**: HTTP to localhost:3001 (Switch backend)
- **Monitoring interval**: 5 seconds
- **Metrics collected**: image_processing_time, confidence, utility, cpu_usage, detection_boxes, total_processed, current_model
- **Available actions**: Model switching (yolov5n/s/m/l/x), knowledge updates, processing control

### Framework Configuration

Verify NATS URL in `polaris_poc/src/config/polaris_config.yaml`:

```yaml
nats:
  url: "nats://localhost:4222"
```

## Running the Integration

### Step 1: Start Switch System

```bash
# Terminal 1: Start Elasticsearch/Kibana (if not running)
cd polaris_poc/extern/switch
docker-compose up

# Terminal 2: Start Switch backend
cd polaris_poc/extern/switch/NAVIE
python3 Node.py
```

### Step 2: Start NATS Server

```bash
# Terminal 3: Start NATS
docker run --rm -p 4222:4222 nats:latest
# Or if installed locally:
nats-server
```

### Step 3: Start POLARIS Monitor Adapter

```bash
# Terminal 4: Launch monitor adapter
cd polaris_poc/extern/switch_plugin
python3 run_monitor_adapter.py
```

Expected output:
```
INFO - Starting Switch Monitor Adapter
INFO - POLARIS config: .../polaris_config.yaml
INFO - Plugin directory: .../switch_plugin
INFO - Switch system connector initialized
INFO - Monitor adapter started successfully
INFO - Collecting metrics every 5.0 seconds
```

### Step 4: Start POLARIS Execution Adapter

```bash
# Terminal 5: Launch execution adapter
cd polaris_poc/extern/switch_plugin
python3 run_execution_adapter.py
```

Expected output:
```
INFO - Starting Switch Execution Adapter
INFO - POLARIS config: .../polaris_config.yaml
INFO - Plugin directory: .../switch_plugin
INFO - Switch system connector initialized
INFO - Execution adapter started successfully
INFO - Listening for control actions from POLARIS
```

### Step 5: Start Switch Processing

Once adapters are running, start the Switch image processing:

```bash
# Use Switch web interface at http://localhost:3000
# OR use API directly:
curl -X POST http://localhost:3001/execute-python-script
```

## Monitoring Telemetry

The Monitor Adapter publishes telemetry to these NATS subjects:

- `polaris.telemetry.events.stream` - Individual metric events (if streaming enabled)
- `polaris.telemetry.events.batch` - Batched metric events
- `polaris.telemetry.events.snapshots` - System state snapshots

You can subscribe to these subjects to observe telemetry:

```bash
# Install NATS CLI
nats sub "polaris.telemetry.events.>"
```

## Executing Adaptation Actions

### Option 1: Via POLARIS Digital Twin / Kernel

Once the full POLARIS stack is running (Digital Twin, Kernel, World Model), actions will be automatically triggered based on reasoning.

### Option 2: Manual Action Publishing (for testing)

You can manually publish actions to test the Execution Adapter:

```python
import asyncio
import json
from nats.aio.client import Client as NATS

async def send_action():
    nc = NATS()
    await nc.connect("nats://localhost:4222")
    
    # Switch to yolov5m model
    action = {
        "action_id": "test-action-001",
        "action_type": "SWITCH_MODEL",
        "system_name": "switch_yolo",
        "parameters": {"model": "yolov5m"},
        "timestamp": "2024-01-01T00:00:00Z",
        "source": "manual_test"
    }
    
    await nc.publish("polaris.execution.actions", json.dumps(action).encode())
    await nc.close()

asyncio.run(send_action())
```

### Available Actions

1. **SWITCH_MODEL**: Switch to specific YOLO model
   - Parameters: `{"model": "yolov5n|s|m|l|x"}`
   
2. **SWITCH_MODEL_YOLOV5N/S/M/L/X**: Predefined model switches
   - No parameters (model is default)

3. **UPDATE_KNOWLEDGE**: Update adaptation thresholds
   - Parameters: `{yolov5nLower, yolov5nUpper, ...}` for all models

4. **START_PROCESSING**: Start image processing
5. **STOP_PROCESSING**: Stop image processing
6. **RESTART_PROCESSING**: Restart image processing

## Experiment Scenarios

### Scenario 1: Compare POLARIS vs Native MAPE-K

1. **Baseline**: Run Switch with native NAIVE MAPE-K
   - Use Switch web interface, select "NAIVE" approach
   - Collect metrics via Switch's Elasticsearch

2. **POLARIS Adaptation**: Run with POLARIS control
   - Disable Switch's native MAPE-K (select a fixed model)
   - Let POLARIS adapters control model switching
   - Compare adaptation decisions and performance

### Scenario 2: Different World Models

Test POLARIS with different reasoning approaches:

1. **Statistical World Model**: Use time-series analysis for predictions
2. **Bayesian World Model**: Probabilistic reasoning for adaptations
3. **Gemini World Model**: LLM-based reasoning (requires API key)

Configure in `polaris_config.yaml`:
```yaml
digital_twin:
  world_model:
    implementation: "bayesian"  # or "statistical", "gemini"
```

### Scenario 3: Input Rate Variations

Test adaptation under different load conditions:

1. Create different inter-arrival rate files (CSV)
2. Upload via Switch interface
3. Observe how POLARIS adapts model selection
4. Compare metrics: utility, confidence, processing time

## Metrics to Collect

### From Switch System
- `image_processing_time`: Model latency (seconds)
- `confidence`: Detection confidence score
- `utility`: Combined performance metric
- `cpu_usage`: Resource utilization (%)
- `detection_boxes`: Objects detected per image
- `total_processed`: Total images processed

### From POLARIS
- Adaptation frequency (actions per minute)
- Action latency (decision to execution time)
- Reasoning overhead (World Model query time)
- Telemetry collection performance

## Troubleshooting

### Connection Issues

**Problem**: Monitor adapter can't connect to Switch
```
ConnectionError: Cannot connect to Switch system at localhost:3001
```

**Solution**:
- Verify Switch backend is running: `curl http://localhost:3001/api/latest_metrics_data`
- Check host/port in `config.yaml`
- Ensure no firewall blocking port 3001

### NATS Connection Failed

**Problem**: Adapters can't connect to NATS
```
Error: Failed to connect to NATS server
```

**Solution**:
- Verify NATS is running: `netstat -an | grep 4222`
- Check NATS URL in `polaris_config.yaml`
- Try `nats-server -DV` for verbose debugging

### File Path Issues

**Problem**: Connector can't read model.csv or other files
```
FileNotFoundError: model.csv not found
```

**Solution**:
- Check file paths in `config.yaml` under `switch_system`
- Ensure paths are relative to plugin directory
- Verify Switch system has created the CSV files

### Metric Collection Errors

**Problem**: No metrics being collected
```
WARNING: All metrics collection failed
```

**Solution**:
- Ensure Switch processing has started (images being processed)
- Check Switch backend logs for errors
- Verify Elasticsearch is storing data: `curl http://localhost:9200/final_metrics_data/_count`

## Data Export and Analysis

### From Switch
```bash
cd polaris_poc/extern/switch/NAVIE

# Export metrics
python3 get_data.py

# Metrics saved to: Exported_metrics/exported-data-metrics_<id>.csv
# Logs saved to: Exported_logs/exported-data-logs_<id>.json
```

### From POLARIS

Telemetry is published to NATS. To persist data:
1. Start POLARIS Knowledge Base service to store telemetry
2. Query via Digital Twin gRPC API
3. Or subscribe and save NATS messages manually

## Next Steps

1. **Integrate Digital Twin**: Start the full POLARIS stack with Digital Twin and World Model
2. **Configure Policies**: Set up adaptation policies and constraints
3. **Run Experiments**: Compare adaptation strategies systematically
4. **Analyze Results**: Use collected metrics to evaluate POLARIS effectiveness

## File References

- **Plugin config**: `config.yaml`
- **Connector**: `connector.py`
- **Framework config**: `../../src/config/polaris_config.yaml`
- **Switch backend**: `../switch/NAVIE/Node.py`
- **Switch processing**: `../switch/NAVIE/process.py`
- **Switch MAPE-K**: `../switch/NAVIE/{monitor,Analyzer,Planner,Execute}.py`

## Support

For issues or questions:
1. Check POLARIS documentation
2. Review Switch README: `../switch/README.md`
3. Examine adapter logs for detailed error messages
4. Verify all prerequisites are properly installed and running
