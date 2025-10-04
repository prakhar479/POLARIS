# Digital Twin Integration Troubleshooting Guide

This guide helps diagnose and fix issues with the Digital Twin integration in the Agentic Reasoner.

## Common Issues and Solutions

### 1. "Digital Twin client not connected" Error

**Symptoms:**
- Agentic reasoner logs show "Digital Twin client not connected"
- Tool calls to digital twin fail with connection errors

**Causes & Solutions:**

#### A. Digital Twin Service Not Running
```bash
# Check if digital twin is running
ps aux | grep digital-twin

# Start digital twin if not running
python src/scripts/start_component.py digital-twin
```

#### B. Wrong gRPC Address Configuration
Check your configuration file (`src/config/polaris_config.yaml` or `src/config/agentic_reasoner_config.yaml`):

```yaml
digital_twin:
  grpc:
    host: "0.0.0.0"  # or "localhost" or "127.0.0.1"
    port: 50051
```

#### C. Port Already in Use
```bash
# Check if port 50051 is in use
netstat -an | grep 50051
# or
lsof -i :50051

# If in use, either:
# 1. Stop the process using the port
# 2. Change the port in configuration
```

### 2. gRPC Connection Timeout

**Symptoms:**
- Connection attempts timeout
- "Failed to connect to Digital Twin" errors

**Solutions:**

#### A. Increase Connection Timeout
In your reasoner configuration:
```yaml
agentic_reasoner:
  tools:
    dt_query_timeout: 60.0  # Increase from default 45.0
```

#### B. Check Network Connectivity
```bash
# Test if the port is reachable
telnet localhost 50051
# or
nc -zv localhost 50051
```

#### C. Firewall Issues
```bash
# On Linux, check if port is blocked
sudo ufw status
sudo iptables -L

# Allow port if needed
sudo ufw allow 50051
```

### 3. "No response from digital twin" Error

**Symptoms:**
- Connection succeeds but operations return no response
- Digital twin responds with empty or null responses

**Solutions:**

#### A. Check Digital Twin Logs
Look for errors in the digital twin service logs:
```bash
# If running with logging to file
tail -f logs/digital_twin.log

# If running in console, check the terminal output
```

#### B. Verify World Model Configuration
Check if the world model is properly configured:
```yaml
digital_twin:
  world_model:
    implementation: "mock"  # or "gemini", "statistical", etc.
    config_path: "mock_world_model.yaml"
```

#### C. Test Digital Twin Directly
Use the diagnostic script:
```bash
python tests/test_digital_twin_connection.py
```

### 4. Proto/gRPC Version Mismatch

**Symptoms:**
- Import errors related to `digital_twin_pb2` or `digital_twin_pb2_grpc`
- "Method not implemented" errors

**Solutions:**

#### A. Regenerate Proto Files
```bash
# Navigate to proto directory
cd src/polaris/proto

# Regenerate proto files
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. digital_twin.proto
```

#### B. Check gRPC Version
```bash
pip list | grep grpc
# Ensure compatible versions are installed
```

### 5. World Model Initialization Errors

**Symptoms:**
- Digital twin starts but world model fails to initialize
- "World Model not available" errors

**Solutions:**

#### A. Check World Model Dependencies
For Gemini world model:
```bash
# Ensure API key is set
echo $API_KEY

# Check if google-generativeai is installed
pip list | grep google-generativeai
```

#### B. Use Mock World Model for Testing
Temporarily switch to mock world model:
```yaml
digital_twin:
  world_model:
    implementation: "mock"
    config_path: "mock_world_model.yaml"
```

## Diagnostic Steps

### Step 1: Basic Connectivity Test
```bash
# Run the diagnostic script
python tests/test_digital_twin_connection.py
```

### Step 2: Check Component Status
```bash
# Check if all required components are running
ps aux | grep -E "(nats-server|digital-twin|knowledge-base)"
```

### Step 3: Verify Configuration
```bash
# Validate configuration files
python scripts/validate_configs.py
```

### Step 4: Check Logs
```bash
# Check digital twin logs
tail -f logs/digital_twin.log

# Check agentic reasoner logs
# (logs appear in the terminal where you started the reasoner)
```

### Step 5: Test Individual Components

#### Test Digital Twin Standalone
```bash
# Start only digital twin
python src/scripts/start_component.py digital-twin --log-level DEBUG
```

#### Test Agentic Reasoner Without Digital Twin
Temporarily disable digital twin in configuration:
```yaml
agentic_reasoner:
  tools:
    digital_twin_enabled: false
```

## Configuration Examples

### Working Configuration (Local Development)
```yaml
# polaris_config.yaml
digital_twin:
  grpc:
    host: "localhost"
    port: 50051
    max_workers: 10
  world_model:
    implementation: "mock"
    config_path: "mock_world_model.yaml"

# agentic_reasoner_config.yaml
agentic_reasoner:
  tools:
    digital_twin_enabled: true
    dt_query_timeout: 45.0
```

### Working Configuration (Docker/Container)
```yaml
# polaris_config.yaml
digital_twin:
  grpc:
    host: "0.0.0.0"  # Accept connections from any interface
    port: 50051
  world_model:
    implementation: "mock"

# agentic_reasoner_config.yaml
agentic_reasoner:
  tools:
    digital_twin_enabled: true
    dt_query_timeout: 60.0  # Longer timeout for container environments
```

## Quick Fixes

### Fix 1: Reset Digital Twin Connection
```python
# In Python console or script
import asyncio
from polaris.agents.reasoner_agent import GRPCDigitalTwinClient

async def reset_connection():
    client = GRPCDigitalTwinClient("localhost:50051")
    await client.connect()
    print("Connection successful!")
    await client.disconnect()

asyncio.run(reset_connection())
```

### Fix 2: Disable Digital Twin Temporarily
In `agentic_reasoner.py`, modify the initialization:
```python
# Temporarily disable digital twin
dt_interface = None  # Instead of agent.dt_query
```

### Fix 3: Use Mock Digital Twin
Create a mock digital twin interface for testing:
```python
class MockDigitalTwin:
    async def connect(self): pass
    async def disconnect(self): pass
    async def query(self, query):
        from polaris.agents.reasoner_agent import DTResponse
        return DTResponse(success=True, result="Mock response", confidence=0.8)
    # ... implement other methods
```

## Getting Help

If none of these solutions work:

1. **Enable Debug Logging**: Start components with `--log-level DEBUG`
2. **Check System Resources**: Ensure sufficient memory and CPU
3. **Test Network**: Verify no network issues between components
4. **Simplify Configuration**: Use minimal configuration for testing
5. **Check Dependencies**: Ensure all required packages are installed

## Prevention

To avoid future issues:

1. **Use Health Checks**: Implement regular health checks for digital twin
2. **Monitor Resources**: Monitor system resources and network connectivity
3. **Version Control**: Keep track of working configurations
4. **Testing**: Regularly run diagnostic tests
5. **Documentation**: Document any custom configurations or modifications