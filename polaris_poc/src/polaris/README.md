# POLARIS Core Components

## Quick Start

### Start NATS server
```bash
./bin/nats-server
```

### Start Components (Clean Architecture)

```bash
# External adapters (require plugin)
python src/scripts/start_component.py monitor --plugin-dir extern
python src/scripts/start_component.py execution --plugin-dir extern

# Internal adapters (can work standalone)
python src/scripts/start_component.py verification
python src/scripts/start_component.py verification --plugin-dir extern  # Optional

# Core services
python src/scripts/start_component.py kernel
python src/scripts/start_component.py digital-twin
python src/scripts/start_component.py knowledge-base
```

### Validation
```bash
# Validate any component configuration
python src/scripts/start_component.py <component> --validate-only --log-level DEBUG
```

## Architecture Overview

POLARIS uses a clean, single-inheritance architecture with clear separation of concerns:

### Core Base Classes

#### `BaseComponent`
- **Purpose**: Common functionality for all POLARIS components
- **Features**:
  - NATS client management and communication
  - Configuration loading and management
  - Async lifecycle management (start/stop)
  - Task management and cleanup
  - Structured logging
- **Usage**: Base class for all adapters and services

#### `ManagedSystemConnector` (Abstract Interface)
- **Purpose**: Standardized interface for connecting to external managed systems
- **Required Methods**:
  - `connect()` / `disconnect()`: Connection lifecycle
  - `execute_command(command, params)`: Execute commands on managed system
  - `health_check()`: Verify system health
- **Features**:
  - Timeout and retry configuration
  - Connection validation
  - Structured error handling

### Adapter Types

#### `ExternalAdapter` (extends `BaseComponent`)
- **Purpose**: Adapters that interface with external managed systems
- **Requirements**: 
  - Must have a plugin directory with connector implementation
  - Plugin must implement `ManagedSystemConnector` interface
- **Features**:
  - Dynamic connector loading from plugins
  - Plugin configuration validation
  - Managed system connection lifecycle
  - Schema-based validation
- **Examples**: `MonitorAdapter`, `ExecutionAdapter`

#### `InternalAdapter` (extends `BaseComponent`)
- **Purpose**: Framework-internal adapters that don't require external connections
- **Features**:
  - Optional plugin configuration support
  - Built-in framework defaults
  - Can work standalone or with plugin enhancements
  - Flexible configuration loading
- **Examples**: `VerificationAdapter`

### Inheritance Hierarchy

```
BaseComponent (abstract)
├── ExternalAdapter (abstract)
│   ├── MonitorAdapter
│   └── ExecutionAdapter
└── InternalAdapter (abstract)
    └── VerificationAdapter

ManagedSystemConnector (interface)
└── [Plugin-specific implementations]
    ├── SWIMConnector
    ├── HTTPConnector
    └── [Custom connectors...]
```

## Component Details

### Monitor Adapter (`ExternalAdapter`)
- **Purpose**: Collects metrics from managed systems and publishes telemetry
- **Plugin Requirements**: 
  - Connector implementing `ManagedSystemConnector`
  - Monitoring configuration with metrics definitions
- **Features**:
  - Configurable metric collection strategies (parallel/sequential)
  - Derived metric calculations using safe evaluation
  - Batch and streaming telemetry publishing
  - Knowledge base integration for telemetry storage
  - Comprehensive error handling with configurable strategies
  - Performance metrics and monitoring

### Execution Adapter (`ExternalAdapter`)
- **Purpose**: Executes control actions on managed systems
- **Plugin Requirements**:
  - Connector implementing `ManagedSystemConnector`
  - Execution configuration with action definitions
- **Features**:
  - Parameter validation with type checking and constraints
  - Precondition evaluation with mathematical operations
  - Concurrent execution control with throttling
  - Action queuing with configurable constraints
  - Comprehensive result reporting and metrics
  - Expression evaluation for preconditions and parameters

### Verification Adapter (`InternalAdapter`)
- **Purpose**: Validates control actions before execution
- **Plugin Requirements**: Optional (has built-in defaults)
- **Features**:
  - Safety constraint checking with configurable rules
  - Organizational policy enforcement
  - Digital Twin integration for predictive verification
  - Multi-level verification (basic, policy, formal, comprehensive)
  - Violation reporting with severity levels and recommendations
  - Built-in framework defaults for standalone operation

## Configuration Architecture

### Framework Configuration (`polaris_config.yaml`)
```yaml
nats:
  url: "nats://localhost:4222"
  
telemetry:
  stream_subject: "polaris.telemetry.events.stream"
  batch_subject: "polaris.telemetry.events.batch"
  batch_size: 100
  
execution:
  action_subject: "polaris.execution.actions"
  result_subject: "polaris.execution.results"
  
verification:
  input_subject: "polaris.verification.requests"
  output_subject: "polaris.verification.results"
  default_timeout_sec: 30
```

### Plugin Configuration (`extern/config.yaml`)
```yaml
system_name: "swim_system"
implementation:
  connector_class: "connector.SWIMConnector"
  
connection:
  protocol: "tcp"
  host: "localhost"
  port: 4242
  
monitoring:
  interval: 5.0
  metrics:
    - name: "server_count"
      command: "get_server_count"
      type: "integer"
      unit: "count"
  
execution:
  actions:
    - type: "ADD_SERVER"
      command: "add_server {count}"
      parameters:
        - name: "count"
          type: "integer"
          required: true
          validation:
            min: 1
            max: 10
      preconditions:
        - check: "get_server_count < get_max_servers"
          message: "Cannot exceed maximum server limit"
```

## Plugin Development Guide

### 1. Create Plugin Structure
```bash
mkdir my_system_plugin
cd my_system_plugin
touch __init__.py config.yaml connector.py
```

### 2. Implement Connector
```python
from polaris.adapters.core import ManagedSystemConnector
import asyncio
import logging

class MySystemConnector(ManagedSystemConnector):
    def __init__(self, system_config: Dict[str, Any], logger: logging.Logger):
        super().__init__(system_config, logger)
        self.client = None
    
    async def connect(self) -> None:
        """Establish connection to managed system."""
        host = self.connection_config.get("host", "localhost")
        port = self.connection_config.get("port", 8080)
        
        # Initialize your client here
        self.client = MySystemClient(host, port)
        await self.client.connect()
        
        self.logger.info(f"Connected to {host}:{port}")
    
    async def disconnect(self) -> None:
        """Disconnect from managed system."""
        if self.client:
            await self.client.disconnect()
            self.client = None
    
    async def execute_command(
        self, 
        command_template: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute command on managed system."""
        if not self.client:
            raise RuntimeError("Not connected to managed system")
        
        # Format command with parameters
        if params:
            command = command_template.format(**params)
        else:
            command = command_template
        
        # Execute command and return result
        result = await self.client.execute(command)
        return str(result)
    
    async def health_check(self) -> bool:
        """Check system health."""
        try:
            if not self.client:
                return False
            return await self.client.ping()
        except Exception:
            return False
```

### 3. Define Configuration
```yaml
system_name: "my_system"
implementation:
  connector_class: "connector.MySystemConnector"
  timeout: 30.0
  max_retries: 3

connection:
  host: "localhost"
  port: 8080
  protocol: "http"

monitoring:
  interval: 10.0
  strategies:
    parallel_collection: true
    max_concurrent: 3
  metrics:
    - name: "cpu_usage"
      command: "GET /metrics/cpu"
      type: "float"
      unit: "percent"
      category: "performance"

execution:
  constraints:
    min_interval: 1.0
    max_concurrent: 2
  actions:
    - type: "SCALE_UP"
      command: "POST /scale?instances={count}"
      parameters:
        - name: "count"
          type: "integer"
          required: true
          validation:
            min: 1
            max: 5
      preconditions:
        - check: "cpu_usage > 80"
          message: "CPU usage must be above 80% to scale up"
```

### 4. Test Plugin
```bash
# Validate configuration
python src/scripts/start_component.py monitor --plugin-dir my_system_plugin --validate-only

# Test monitor adapter
python src/scripts/start_component.py monitor --plugin-dir my_system_plugin --dry-run

# Test execution adapter  
python src/scripts/start_component.py execution --plugin-dir my_system_plugin --dry-run
```

## Advanced Features

### Expression Evaluation in Execution Adapter
The execution adapter supports sophisticated expression evaluation for preconditions:

```yaml
preconditions:
  # Mathematical comparisons
  - check: "get_server_count + params.count <= get_max_servers"
    message: "Would exceed maximum server limit"
  
  # String operations
  - check: "get_status == 'running'"
    message: "System must be running"
  
  # List operations
  - check: "params.server_type in ['compute', 'storage', 'network']"
    message: "Invalid server type"
  
  # Complex expressions
  - check: "get_utilization < 0.9 and get_response_time < 1.0"
    message: "System performance constraints not met"
```

### Verification Adapter Configuration
```yaml
verification:
  constraints:
    - id: "resource_limit"
      type: "resource"
      condition: "params.get('count', 1) + current_state.get('active_servers', 0) <= current_state.get('max_servers', 10)"
      severity: "critical"
      description: "Cannot exceed maximum resource limits"
      suggested_fix: "Reduce the number of resources requested"
  
  policies:
    - id: "approval_policy"
      name: "Change Approval Policy"
      rules:
        - id: "high_impact_approval"
          condition: "action_type in ['REMOVE_SERVER', 'RESTART'] and 'approved_by' in params"
          severity: "medium"
          description: "High-impact actions require approval"
          suggested_fix: "Add 'approved_by' parameter with approver name"
  
  settings:
    default_timeout_sec: 30
    enable_digital_twin: true
    enable_formal_verification: false
```

## Error Handling and Monitoring

### Structured Logging
All components use structured logging with contextual information:

```python
self.logger.info(
    "Action execution completed",
    extra={
        "action_id": action.action_id,
        "action_type": action.action_type,
        "success": result.success,
        "duration_sec": result.duration_sec
    }
)
```

### Metrics Publishing
Components publish metrics to NATS for monitoring:

```python
await self._publish_metric("action_duration", {
    "action_id": action.action_id,
    "action_type": action.action_type,
    "duration_sec": duration,
    "success": success
})
```

### Health Checks
All adapters support health checking through their connectors:

```bash
# Check adapter health
python src/scripts/start_component.py monitor --plugin-dir extern --health-check
```

## Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run adapter-specific tests
python -m pytest tests/test_verification_adapter.py -v
```

### Integration Tests
```bash
# Start test environment
python scripts/setup_test_env.py

# Run integration tests
python scripts/run_tests.py --integration
```

This architecture provides a clean, extensible foundation for building self-adaptive systems with POLARIS, supporting both simple and complex managed system integrations while maintaining consistency and reliability.