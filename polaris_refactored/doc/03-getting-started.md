# Getting Started

## Prerequisites

Before installing and running POLARIS, ensure you have the following prerequisites:

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free disk space

### Required Dependencies

- **NATS Server**: For message bus functionality
- **Docker** (optional): For containerized deployment
- **Git**: For source code management

### Python Dependencies

POLARIS uses the following key Python packages:
- `asyncio`: Asynchronous programming support
- `nats-py`: NATS client library
- `pydantic`: Data validation and settings management
- `pytest`: Testing framework
- `typing-extensions`: Enhanced type hints

## Installation

### Option 1: Development Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd polaris_refactored
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -e .  # Install POLARIS in development mode
```

4. **Install NATS Server**
```bash
# On macOS with Homebrew
brew install nats-server

# On Ubuntu/Debian
wget https://github.com/nats-io/nats-server/releases/download/v2.9.0/nats-server-v2.9.0-linux-amd64.zip
unzip nats-server-v2.9.0-linux-amd64.zip
sudo mv nats-server /usr/local/bin/

# On Windows
# Download from https://github.com/nats-io/nats-server/releases
```

### Option 2: Docker Installation

1. **Build Docker Image**
```bash
docker build -t polaris:latest .
```

2. **Run with Docker Compose**
```bash
docker-compose up -d
```

## Quick Start

### 1. Start NATS Server

```bash
# Start NATS server with default configuration
nats-server

# Or with custom configuration
nats-server -c nats-server.conf
```

### 2. Basic Configuration

Create a configuration file `config.yaml`:

```yaml
# Basic POLARIS configuration
framework:
  name: "polaris-demo"
  version: "2.0.0"

# Message bus configuration
message_bus:
  nats:
    servers:
      - "nats://localhost:4222"
    name: "polaris-client"

# Data storage configuration
data_storage:
  backends:
    default:
      type: "in_memory"

# Logging configuration
logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "console"
    - type: "file"
      path: "logs/polaris.log"

# Managed systems configuration
managed_systems:
  - system_id: "demo-system"
    connector_type: "demo"
    config:
      endpoint: "http://localhost:8080"
      collection_interval: 30
```

### 3. Create a Simple Managed System Connector

Create `plugins/demo/connector.py`:

```python
from typing import Dict, List
from datetime import datetime, timezone

from polaris_refactored.src.domain.interfaces import ManagedSystemConnector
from polaris_refactored.src.domain.models import (
    SystemState, AdaptationAction, ExecutionResult, 
    MetricValue, HealthStatus, ExecutionStatus
)

class DemoConnector(ManagedSystemConnector):
    """Demo connector for getting started with POLARIS."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = "demo-system"
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the demo system."""
        print(f"Connecting to demo system at {self.config.get('endpoint')}")
        self._connected = True
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from the demo system."""
        print("Disconnecting from demo system")
        self._connected = False
        return True
    
    async def get_system_id(self) -> str:
        """Get the system identifier."""
        return self.system_id
    
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect metrics from the demo system."""
        # Simulate some metrics
        import random
        return {
            "cpu_usage": MetricValue("cpu_usage", random.uniform(0.1, 0.9), "ratio"),
            "memory_usage": MetricValue("memory_usage", random.uniform(0.2, 0.8), "ratio"),
            "response_time": MetricValue("response_time", random.uniform(10, 100), "ms"),
        }
    
    async def get_system_state(self) -> SystemState:
        """Get the current system state."""
        metrics = await self.collect_metrics()
        
        # Determine health based on metrics
        cpu = metrics["cpu_usage"].value
        health = HealthStatus.HEALTHY
        if cpu > 0.8:
            health = HealthStatus.WARNING
        elif cpu > 0.9:
            health = HealthStatus.CRITICAL
        
        return SystemState(
            system_id=self.system_id,
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            health_status=health
        )
    
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action."""
        print(f"Executing action: {action.action_type} with parameters: {action.parameters}")
        
        # Simulate action execution
        import asyncio
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={"message": f"Successfully executed {action.action_type}"}
        )
    
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if an action can be executed."""
        # Simple validation - check if action type is supported
        supported_actions = await self.get_supported_actions()
        return action.action_type in supported_actions
    
    async def get_supported_actions(self) -> List[str]:
        """Get supported action types."""
        return ["scale_out", "scale_in", "restart", "tune_parameters"]
```

Create `plugins/demo/plugin.yaml`:

```yaml
name: "demo"
version: "1.0.0"
description: "Demo connector for POLARIS getting started"
connector_class: "connector.DemoConnector"
supported_systems:
  - "demo-system"
configuration_schema:
  type: "object"
  properties:
    endpoint:
      type: "string"
      description: "Demo system endpoint URL"
    collection_interval:
      type: "integer"
      description: "Metric collection interval in seconds"
      default: 30
  required:
    - "endpoint"
```

### 4. Run POLARIS

Create `run_demo.py`:

```python
import asyncio
import logging
from pathlib import Path

from polaris_refactored.src.framework.polaris_framework import create_polaris_framework

async def main():
    """Run POLARIS demo."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start POLARIS framework
    framework = await create_polaris_framework("config.yaml")
    
    try:
        print("Starting POLARIS framework...")
        await framework.start()
        
        print("POLARIS is running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down POLARIS...")
        await framework.stop()
        print("POLARIS stopped.")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the demo:

```bash
python run_demo.py
```

## Understanding the Output

When POLARIS starts, you should see output similar to:

```
2024-01-15 10:30:00 - polaris.framework - INFO - Starting POLARIS framework...
2024-01-15 10:30:00 - polaris.infrastructure.message_bus - INFO - Connected to NATS servers: ['nats://localhost:4222']
2024-01-15 10:30:00 - polaris.framework.plugin_management - INFO - Discovered plugin: demo
2024-01-15 10:30:00 - polaris.adapters.monitor_adapter - INFO - Starting monitoring for system: demo-system
2024-01-15 10:30:00 - polaris.framework - INFO - POLARIS framework started successfully
POLARIS is running. Press Ctrl+C to stop.
```

## Next Steps

### 1. Explore the System

- **Monitor Logs**: Watch the log output to see telemetry collection and processing
- **Check Metrics**: If you have Prometheus configured, view the metrics dashboard
- **Trigger Adaptations**: Modify the demo connector to generate high CPU usage and observe adaptations

### 2. Add More Managed Systems

Create additional connectors for your actual systems:

```python
# Example: Database connector
class DatabaseConnector(ManagedSystemConnector):
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        # Collect database-specific metrics
        return {
            "connection_count": MetricValue("connection_count", 45, "count"),
            "query_latency": MetricValue("query_latency", 12.5, "ms"),
            "cpu_usage": MetricValue("cpu_usage", 0.65, "ratio"),
        }
```

### 3. Configure Advanced Features

- **World Models**: Enable predictive capabilities
- **Learning Engine**: Configure pattern learning
- **Custom Strategies**: Implement domain-specific control strategies
- **Observability**: Set up Prometheus and Grafana for monitoring

### 4. Production Deployment

- **Security**: Configure authentication and authorization
- **Scaling**: Set up horizontal scaling with load balancers
- **Monitoring**: Implement comprehensive monitoring and alerting
- **Backup**: Configure data backup and disaster recovery

## Common Issues and Solutions

### NATS Connection Issues

**Problem**: Cannot connect to NATS server
```
ERROR - Failed to connect to NATS: [Errno 61] Connection refused
```

**Solution**:
1. Ensure NATS server is running: `nats-server`
2. Check the server address in configuration
3. Verify firewall settings

### Plugin Loading Issues

**Problem**: Plugin not found or failed to load
```
ERROR - Failed to load plugin: demo
```

**Solution**:
1. Check plugin directory structure
2. Verify `plugin.yaml` syntax
3. Ensure connector class is properly defined
4. Check Python import paths

### Configuration Validation Errors

**Problem**: Configuration validation fails
```
ERROR - Configuration validation failed: 'endpoint' is a required property
```

**Solution**:
1. Review configuration file syntax
2. Check required properties in plugin schema
3. Validate YAML/JSON format
4. Ensure all required sections are present

## Getting Help

- **Documentation**: Refer to the comprehensive documentation in the `doc/` directory
- **Examples**: Check the `examples/` directory for more complex scenarios
- **Tests**: Review test files for usage patterns and examples
- **Issues**: Report bugs and request features through the issue tracker

## What's Next?

Now that you have POLARIS running, explore these topics:

1. **[System Architecture](./04-system-architecture.md)**: Understand the system design
2. **[Plugin Development](./18-plugin-development.md)**: Create custom connectors
3. **[Configuration Management](./19-configuration-management.md)**: Advanced configuration
4. **[Monitoring & Telemetry](./13-monitoring-telemetry.md)**: Set up comprehensive monitoring
5. **[Deployment Guide](./26-deployment-guide.md)**: Production deployment strategies

---

*Continue to [System Architecture](./04-system-architecture.md) →*