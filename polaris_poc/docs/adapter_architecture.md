# POLARIS Adapter Architecture

## Overview

POLARIS implements a clean, single-inheritance architecture for adapters that provides clear separation of concerns while maintaining flexibility and extensibility. The architecture is built around three core abstractions that handle different aspects of system adaptation.

## Core Architecture Principles

### 1. Single Inheritance Hierarchy
- **Clean separation**: Each class has a single, well-defined responsibility
- **No multiple inheritance**: Avoids complexity and diamond problems
- **Interface-based design**: Uses abstract base classes to define contracts

### 2. Plugin-Driven Extensibility
- **External systems**: Integrated through standardized connector interfaces
- **Configuration-driven**: Behavior controlled through YAML configuration
- **Dynamic loading**: Connectors loaded at runtime based on configuration

### 3. Event-Driven Communication
- **NATS messaging**: All inter-component communication through NATS
- **Structured events**: JSON-based event schemas with validation
- **Async/await**: Non-blocking, concurrent processing throughout

## Class Hierarchy

```
BaseComponent (abstract)
├── ExternalAdapter (abstract)
│   ├── MonitorAdapter
│   └── ExecutionAdapter
└── InternalAdapter (abstract)
    └── VerificationAdapter

ManagedSystemConnector (interface)
└── [Plugin implementations]
    ├── SWIMConnector
    ├── HTTPConnector
    └── [Custom connectors...]
```

## Core Classes

### BaseComponent

**Purpose**: Provides common functionality for all POLARIS components.

**Responsibilities**:
- NATS client management and lifecycle
- Configuration loading and validation
- Async component lifecycle (start/stop)
- Task management and cleanup
- Structured logging with context

**Key Methods**:
```python
async def start() -> None:
    """Start the component and its processing."""

async def stop() -> None:
    """Stop the component gracefully."""

async def _start_processing(self) -> None:
    """Abstract: Component-specific startup logic."""

async def _stop_processing(self) -> None:
    """Abstract: Component-specific shutdown logic."""
```

**Usage Pattern**:
```python
async with component:
    # Component automatically starts and stops
    await asyncio.sleep(60)
```

### ManagedSystemConnector

**Purpose**: Abstract interface for connecting to external managed systems.

**Responsibilities**:
- Connection lifecycle management
- Command execution on managed systems
- Health monitoring and validation
- Error handling and retry logic

**Required Methods**:
```python
async def connect(self) -> None:
    """Establish connection to the managed system."""

async def disconnect(self) -> None:
    """Disconnect from the managed system."""

async def execute_command(
    self, 
    command_template: str, 
    params: Optional[Dict[str, Any]] = None
) -> str:
    """Execute a command on the managed system."""

async def health_check(self) -> bool:
    """Check if the managed system is healthy."""
```

**Configuration Support**:
```python
def get_timeout(self) -> float:
    """Get configured timeout for operations."""

def get_max_retries(self) -> int:
    """Get configured maximum retries."""
```

### ExternalAdapter

**Purpose**: Base class for adapters that interface with external managed systems.

**Characteristics**:
- **Requires plugin**: Must have a plugin directory with connector implementation
- **Connector loading**: Dynamically loads and instantiates system-specific connectors
- **Schema validation**: Validates plugin configuration against JSON schemas
- **Connection management**: Handles connector lifecycle

**Plugin Requirements**:
1. **Connector Implementation**: Class implementing `ManagedSystemConnector`
2. **Configuration File**: YAML file with system-specific settings
3. **Schema Compliance**: Configuration must validate against schema

**Examples**: `MonitorAdapter`, `ExecutionAdapter`

### InternalAdapter

**Purpose**: Base class for framework-internal adapters.

**Characteristics**:
- **Optional plugin**: Can work standalone or with plugin enhancements
- **Built-in defaults**: Has framework defaults for standalone operation
- **Flexible configuration**: Supports both plugin and framework configuration
- **No external connections**: Works with framework internals only

**Configuration Priority**:
1. Plugin configuration (if provided)
2. Framework configuration
3. Built-in defaults

**Examples**: `VerificationAdapter`

## Adapter Implementations

### MonitorAdapter (ExternalAdapter)

**Purpose**: Collects metrics from managed systems and publishes telemetry.

**Key Features**:
- **Configurable collection**: Parallel or sequential metric collection
- **Derived metrics**: Calculate metrics from base measurements
- **Batch publishing**: Efficient telemetry event batching
- **Knowledge base integration**: Store telemetry for historical analysis
- **Error handling**: Configurable strategies (skip, default, fail)

**Configuration Example**:
```yaml
monitoring:
  interval: 5.0
  strategies:
    parallel_collection: true
    max_concurrent: 3
    error_handling: "skip"
  metrics:
    - name: "cpu_usage"
      command: "get_cpu_usage"
      type: "float"
      unit: "percent"
      category: "performance"
  derived_metrics:
    - name: "cpu_efficiency"
      formula: "cpu_usage / max_cpu_capacity"
      unit: "ratio"
```

**Event Flow**:
```
Managed System → Connector → MonitorAdapter → NATS → Digital Twin
                                    ↓
                              Knowledge Base
```

### ExecutionAdapter (ExternalAdapter)

**Purpose**: Executes control actions on managed systems.

**Key Features**:
- **Par