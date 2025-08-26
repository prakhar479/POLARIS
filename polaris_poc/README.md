# POLARIS - Plugin-Driven Adaptation Framework

POLARIS is a modular, extensible framework for monitoring and controlling self-adaptive systems. It uses a plugin-driven architecture that allows easy integration with different managed systems while providing a consistent interface for adaptation logic.

## 🏗️ Architecture Overview

POLARIS follows a clean separation between the **core framework** and **managed system plugins**:

- **Core Framework**: Generic adapters, NATS communication, configuration management, data models
- **Managed System Plugins**: System-specific connectors and configurations
- **Observability Tools**: Real-time monitoring and debugging utilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POLARIS Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Monitor Adapter    │  Execution Adapter  │  Digital Twin    │   Tools      │
│  ┌─────────────────┐│  ┌─────────────────┐│  ┌─────────────┐│  ┌─────────┐  │
│  │ Metric Collection││  │ Action Execution││  │ NATS Ingestion│  │NATS Spy │  │
│  │ Telemetry Batch ││  │ Result Publishing││  │ gRPC Service │  │Debugger │  │
│  │ NATS Publishing ││  │ Queue Management││  │ World Model  │  │Validator│  │
│  └─────────────────┘│  └─────────────────┘│  │ Query/Sim/Diag│  └─────────┘  │
│                     │                     │  └─────────────┘│              │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Plugin Interface                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  SWIM Plugin        │    Custom Plugin        │   Future    │              │
│  ┌─────────────────┐│  ┌─────────────────────┐│  ┌─────────┐│              │
│  │ TCP Connector   ││  │ HTTP Connector      ││  │   ...   ││              │
│  │ Config Schema   ││  │ Config Schema       ││  │         ││              │
│  │ Retry Logic     ││  │ Auth Handling       ││  │         ││              │
│  └─────────────────┘│  └─────────────────────┘│  └─────────┘│              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NATS Server (included in `bin/`)
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start NATS server
./bin/nats-server

# Validate configuration
python src/scripts/start_component.py monitor --plugin-dir extern --validate-only

# Start monitor adapter
python src/scripts/start_component.py monitor --plugin-dir extern

# Start execution adapter (in another terminal)
python src/scripts/start_component.py execution --plugin-dir extern

# Start Digital Twin (in another terminal)
python src/scripts/start_component.py digital-twin
```

### Monitor NATS Messages
```bash
# Monitor all POLARIS messages
python src/scripts/nats_spy.py

# Monitor only telemetry
python src/scripts/nats_spy.py --preset telemetry

# Monitor with full message content
python src/scripts/nats_spy.py --show-data
```

## 📁 Directory Structure

```
polaris_poc/
├── bin/                          # Executables (NATS server)
├── extern/                       # Managed system plugins
│   ├── config.yaml              # SWIM plugin configuration
│   ├── connector.py             # SWIM TCP connector
│   └── README.md                # Plugin development guide
├── src/
│   ├── config/                  # Framework configuration
│   │   ├── managed_system.schema.json
│   │   └── polaris_config.yaml
│   ├── polaris/
│   │   ├── adapters/            # Core adapters
│   │   │   ├── base.py         # Base classes and interfaces
│   │   │   ├── monitor.py      # Generic monitor adapter
│   │   │   └── execution.py    # Generic execution adapter
│   │   ├── agents/              # Digital Twin agents
│   │   │   └── digital_twin_agent.py  # Main Digital Twin agent
│   │   ├── services/            # gRPC services
│   │   │   └── digital_twin_service.py  # Digital Twin gRPC service
│   │   ├── proto/               # Protocol buffer definitions
│   │   │   ├── digital_twin_pb2.py     # Generated protobuf code
│   │   │   ├── digital_twin_pb2_grpc.py
│   │   │   └── wrappers.py              # Protobuf helpers
│   │   ├── common/              # Shared utilities
│   │   │   ├── config.py       # Configuration management
│   │   │   ├── nats_client.py  # NATS communication
│   │   │   ├── digital_twin_config.py  # Digital Twin configuration
│   │   │   └── digital_twin_logging.py # Digital Twin logging
│   │   └── models/              # Data models
│   │       ├── actions.py      # Control actions and results
│   │       ├── telemetry.py    # Telemetry events
│   │       ├── digital_twin_events.py  # Digital Twin events
│   │       └── world_model.py  # World Model interface
│   └── scripts/                 # Utility scripts
│       ├── start_component.py  # Adapter/Kernel/DT/KV entry point
│       └── nats_spy.py         # NATS message monitor
├── scripts/                     # Additional scripts
│   ├── run_tests.py                   # Test runner
│   ├── build.py                       # Build helpers
│   ├── generate_proto.py              # Generate protobuf stubs
│   └── setup_test_env.py              # Test env setup
├── docs/                        # Documentation
│   └── digital_twin_integration.md  # Digital Twin integration guide
├── tests/                        # Test suite
└── requirements.txt
```

## Setup Instructions
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Place your NATS server executable in the `/bin` directory.
5. Clone the SWIM project into the `/extern` directory.
6. Fill in your OpenAI API key in the `.env` file.


## 🔧 Core Components

### Monitor Adapter
- **Purpose**: Collects metrics from managed systems and publishes telemetry
- **Features**: 
  - Plugin-driven metric collection
  - Derived metric calculations
  - Batch and streaming telemetry publishing
  - Configurable collection strategies
  - Error handling and retry logic

### Execution Adapter
- **Purpose**: Executes control actions on managed systems
- **Features**:
  - Action validation and precondition checking
  - Parameter type and range validation
  - Concurrent execution control
  - Result publishing and metrics
  - Queue management with throttling

### Digital Twin Component
- **Purpose**: Provides intelligent system modeling and predictive capabilities
- **Features**:
  - **NATS Ingestion**: Automatically processes telemetry and execution events
  - **gRPC Services**: Query, Simulation, Diagnosis, and Management APIs
  - **World Model**: Pluggable AI/ML implementations (Mock, Gemini LLM, Statistical)
  - **Real-time Processing**: Batch processing with configurable timeouts
  - **Health Monitoring**: Comprehensive health checks and metrics

### Plugin System
- **Purpose**: Encapsulates system-specific logic
- **Features**:
  - Declarative configuration via YAML
  - JSON schema validation
  - Dynamic connector loading
  - Standardized interface
  - Easy extensibility

## 🔌 Plugin Development

### Creating a New Plugin

1. **Create Plugin Directory**
   ```bash
   mkdir my_system_plugin
   cd my_system_plugin
   touch __init__.py
   ```

2. **Define Configuration** (`config.yaml`)
   ```yaml
   system_name: "my_system"
   implementation:
     connector_class: "connector.MySystemConnector"
   connection:
     protocol: "http"
     host: "localhost"
     port: 8080
   monitoring:
     metrics:
       - name: "status"
         command: "GET /health"
         unit: "boolean"
   execution:
     actions:
       - type: "RESTART"
         command: "POST /restart"
   ```

3. **Implement Connector** (`connector.py`)
   ```python
   from polaris.adapters.base import ManagedSystemConnector
   
   class MySystemConnector(ManagedSystemConnector):
       async def connect(self):
           # Implementation here
           pass
       
       async def execute_command(self, command, params=None):
           # Implementation here
           pass
   ```

4. **Test Plugin**
   ```bash
   python src/scripts/start_component.py monitor --plugin-dir my_system_plugin --validate-only
   ```

See `extern/README.md` for detailed plugin development guide.

## 🛠️ Development Tools

### Configuration Validation
```bash
# Validate plugin configuration
python src/scripts/start_component.py monitor --plugin-dir extern --validate-only

# Dry run (initialize but don't start)
python src/scripts/start_component.py monitor --plugin-dir extern --dry-run
```

### NATS Message Monitoring
```bash
# Monitor all messages
python src/scripts/nats_spy.py

# Monitor specific subjects
python src/scripts/nats_spy.py --subjects "polaris.telemetry.>" "polaris.execution.>"

# Show full message content
python src/scripts/nats_spy.py --show-data

# Use presets
python src/scripts/nats_spy.py --preset telemetry
python src/scripts/nats_spy.py --preset execution
python src/scripts/nats_spy.py --preset results
```

### Debug Logging
```bash
# Enable debug logging
python src/scripts/start_component.py monitor --plugin-dir extern --log-level DEBUG
```

## 📊 SWIM Plugin Example

The included SWIM plugin demonstrates a complete implementation:

### Metrics Collected
- Server counts (active, max, total)
- Response times (basic, optional, average)
- Throughput (basic, optional)
- Arrival rate and utilization

### Actions Supported
- `ADD_SERVER` - Add server with capacity checks
- `REMOVE_SERVER` - Remove server with minimum checks
- `SET_DIMMER` - Adjust QoS (0.0-1.0 range)

### Key Features
- TCP socket communication with retry logic
- Parameter validation and precondition checking
- Derived metric calculations
- Comprehensive error handling

## 🔍 Monitoring and Observability

### NATS Subjects
- `polaris.telemetry.events.stream` - Individual telemetry events
- `polaris.telemetry.events.batch` - Batched telemetry events
- `polaris.execution.actions` - Control actions to execute
- `polaris.execution.results` - Action execution results
- `polaris.execution.metrics` - Execution performance metrics
- `polaris.digitaltwin.calibrate` - Model calibration feedback
- `polaris.digitaltwin.errors` - Digital Twin error messages

### Message Flow
```
Monitor Adapter → NATS → Digital Twin → gRPC Clients
     ↓              ↑         ↓
Telemetry Events    │    World Model Updates
                    │         ↓
Execution Adapter ←─┘    Query/Simulation/Diagnosis
     ↓
Action Results → NATS → Digital Twin
```

### Digital Twin gRPC Services
- **Query Service** (`:50051`): Current and historical system state queries
- **Simulation Service**: Predictive "what-if" analysis and forecasting
- **Diagnosis Service**: Root cause analysis and anomaly investigation
- **Management Service**: Health checks, metrics, and lifecycle management

## 🤖 Digital Twin Usage

### Start components via dispatcher
```bash
# Monitor adapter (requires plugin dir)
python src/scripts/start_component.py monitor --plugin-dir extern

# Execution adapter (requires plugin dir)
python src/scripts/start_component.py execution --plugin-dir extern

# Digital Twin agent
python src/scripts/start_component.py digital-twin --world-model mock

# Knowledge Base service
python src/scripts/start_component.py knowledge-base

# Kernel
python src/scripts/start_component.py kernel
```

CLI options (common):
- `--config <path>`: Framework config (default: `src/config/polaris_config.yaml`)
- `--log-level <LEVEL>`: DEBUG | INFO | WARNING | ERROR
- `--validate-only`: Load and validate configuration, then exit
- `--dry-run`: Initialize without connecting to external systems

Component-specific:
- Monitor/Execution: `--plugin-dir <dir>` points to managed system plugin (e.g., `extern/`)
- Digital Twin: `--world-model <mock|gemini|...>`, `--health-check`

### Starting the Digital Twin
```bash
# Start with default configuration
python src/scripts/start_component.py digital-twin --world-model mock

# Health Check
python src/scripts/start_component.py digital-twin --health-check
```

### gRPC Client Examples
```python
import grpc
from polaris.proto import digital_twin_pb2, digital_twin_pb2_grpc

# Connect to Digital Twin
channel = grpc.insecure_channel('localhost:50051')
stub = digital_twin_pb2_grpc.DigitalTwinStub(channel)

# Query current system state
query = digital_twin_pb2.QueryRequest(
    query_type="current_state",
    query_content="What is the current CPU usage?"
)
response = stub.Query(query)

# Run simulation
simulation = digital_twin_pb2.SimulationRequest(
    simulation_type="what_if",
    actions=[...],  # Define actions
    horizon_minutes=60
)
sim_response = stub.Simulate(simulation)
```

### Integration Verification
```bash
# Run full test suite
python scripts/run_tests.py

# Run only non-async tests (fast sanity check)
python scripts/run_tests.py --non-async

# Or run pytest directly
python -m pytest tests/ -v
```

## 🧪 Testing

### Integration Tests
```bash
# Run full test suite
python scripts/run_tests.py

# Run only non-async tests (fast sanity check)
python scripts/run_tests.py --non-async

# Test specific components
python -m pytest tests/ -v
```

### Manual Testing
```bash
# Start all components
python src/scripts/start_component.py monitor --plugin-dir extern &
python src/scripts/start_component.py execution --plugin-dir extern &
python src/scripts/start_component.py digital-twin &

# Monitor messages
python src/scripts/nats_spy.py --preset all

# Send test action (requires NATS client)
nats pub polaris.execution.actions '{"action_type":"SET_DIMMER","params":{"value":0.8}}'

# Test Digital Twin gRPC
grpcurl -plaintext localhost:50051 polaris.digitaltwin.DigitalTwin/Query
```

## 🔧 Configuration

### Framework Configuration (`src/config/polaris_config.yaml`)
- NATS connection settings
- Telemetry batching parameters
- Logging configuration
- Default timeouts and retries

### Plugin Configuration (`extern/config.yaml`)
- System identification and metadata
- Connection parameters
- Metric definitions and collection strategies
- Action definitions and validation rules
- Execution constraints

## 🚨 Troubleshooting

### Common Issues

1. **Plugin Not Found**
   - Check plugin directory path
   - Ensure `__init__.py` exists
   - Verify connector class path

2. **Configuration Validation Errors**
   - Use `--validate-only` flag
   - Check against JSON schema
   - Review error messages for specific issues

3. **Connection Failures**
   - Verify managed system is running
   - Check connection parameters
   - Review connector implementation

4. **NATS Connection Issues**
   - Ensure NATS server is running
   - Check NATS URL configuration
   - Verify network connectivity

### Debug Steps
1. Enable debug logging (`--log-level DEBUG`)
2. Use validation mode (`--validate-only`)
3. Try dry run mode (`--dry-run`)
4. Monitor NATS messages (`nats_spy.py`)
5. Check connector health methods

## 🔧 Implementation Details

### Core Components

#### 1. Kernel
- **Fast Controller** (`src/polaris/controllers/fast_controller.py`)
  - Implements threshold-based control logic
  - Handles rapid response to critical system states
  - Processes telemetry data and generates immediate control actions

- **State Management** (`src/polaris/kernel/kernel.py`)
  - Maintains system state in memory
  - Implements state transition logic
  - Enforces safety constraints and invariants

#### 2. Digital Twin
- **World Model** (`src/polaris/models/world_model.py`)
  - Abstract interface for system representation
  - Supports multiple implementations (e.g., mock, Gemini-based)
  - Handles state updates and queries

- **Knowledge Base** (`src/polaris/knowledge_base/`)
  - Stores system models and historical data
  - Implements knowledge graph for relationship modeling
  - Provides query interface for system state and history

#### 3. Adapters
- **Monitor Adapter** (`src/polaris/adapters/monitor.py`)
  - Collects and processes telemetry data
  - Implements batching and rate limiting
  - Supports multiple data collection strategies

- **Execution Adapter** (`src/polaris/adapters/execution.py`)
  - Executes control actions on managed systems
  - Implements action queuing and retry logic
  - Validates action parameters and preconditions

### Data Flow

1. **Telemetry Collection**
   ```python
   # Monitor Adapter (simplified)
   async def collect_metrics(self):
       while self.running:
           metrics = await self.connector.collect_metrics()
           events = self._process_metrics(metrics)
           await self._publish_events(events)
           await asyncio.sleep(self.collection_interval)
   ```

2. **Event Processing**
   ```python
   # Digital Twin (simplified)
   async def process_telemetry(self, event):
       self.world_model.update_state(event)
       if self._requires_adaptation(event):
           action = await self._determine_adaptation(event)
           if action:
               await self._execute_adaptation(action)
   ```

3. **Action Execution**
   ```python
   # Execution Adapter (simplified)
   async def execute_action(self, action):
       try:
           result = await self.connector.execute_action(action)
           await self._publish_result(action.id, result)
       except Exception as e:
           await self._handle_error(action.id, str(e))
   ```

### Extension Points

1. **Adding a New Managed System**
   - Implement `ManagedSystemConnector` interface
   - Create configuration schema
   - Register with the plugin system

2. **Custom Control Logic**
   - Extend `BaseController` class
   - Implement custom decision logic
   - Register with the control plane

3. **World Model Implementations**
   - Extend `WorldModel` base class
   - Implement required methods
   - Register with `WorldModelFactory`

### Configuration

Example configuration for a managed system:

```yaml
managed_system:
  name: "example_system"
  type: "http"
  connection:
    base_url: "http://example.com/api"
    timeout: 5.0
  monitoring:
    interval: 5.0
    metrics:
      - name: "cpu_usage"
        path: "/metrics/cpu"
        type: "gauge"
  actions:
    - name: "scale_out"
      method: "POST"
      path: "/scale"
      parameters:
        count: "integer"
```

## 🚀 Getting Started
### Prerequisites
- Python 3.8+
- NATS Server (included in `bin/`)
- Required Python packages (see `requirements.txt`)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start NATS server
./bin/nats-server

# Validate configuration
python src/scripts/start_component.py monitor --plugin-dir extern --validate-only

# Start monitor adapter
python src/scripts/start_component.py monitor --plugin-dir extern

# Start execution adapter (in another terminal)
python src/scripts/start_component.py execution --plugin-dir extern

# Start Digital Twin (in another terminal)
python src/scripts/start_component.py digital-twin
```

### Monitor NATS Messages
```bash
# Monitor all POLARIS messages
python src/scripts/nats_spy.py

# Monitor only telemetry
python src/scripts/nats_spy.py --preset telemetry

# Monitor with full message content
python src/scripts/nats_spy.py --show-data
