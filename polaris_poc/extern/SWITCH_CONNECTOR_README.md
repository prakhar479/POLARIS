# Switch System Connector for POLARIS

This connector enables POLARIS to manage and adapt the Switch YOLO model switching system for computer vision applications.

## Overview

The Switch system is a self-adaptive computer vision system that can dynamically switch between different YOLO models (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x) based on system conditions and performance requirements. The connector provides a bridge between POLARIS and the Switch system's REST API.

## Architecture

```
POLARIS Framework
       ↓
Switch Connector (switch_connector.py)
       ↓ HTTP/REST API
Switch System (Node.js + Python)
       ↓
YOLO Models (Computer Vision Processing)
```

## Features

### Core Functionality
- **Model Switching**: Dynamically switch between YOLO models
- **Performance Monitoring**: Real-time metrics collection
- **Health Monitoring**: System health checks and status monitoring
- **Knowledge Base Management**: Update model switching thresholds
- **Processing Control**: Start/stop/restart image processing pipeline

### Supported Operations
1. `switch_model` - Switch to a specific YOLO model
2. `get_current_model` - Get the currently active model
3. `get_metrics` - Retrieve system performance metrics
4. `get_latest_metrics` - Get the most recent metrics
5. `get_latest_logs` - Get the most recent log data
6. `update_knowledge` - Update model switching thresholds
7. `start_processing` - Start image processing
8. `stop_processing` - Stop image processing
9. `restart_processing` - Restart image processing

### Available YOLO Models
- **yolov5n** (Nano): Fastest, lowest accuracy, minimal resources
- **yolov5s** (Small): Fast, low accuracy, low resources
- **yolov5m** (Medium): Balanced speed/accuracy, medium resources
- **yolov5l** (Large): Slow, high accuracy, high resources
- **yolov5x** (Extra Large): Slowest, highest accuracy, maximum resources

## Configuration

### System Configuration (`switch_system_config.yaml`)

```yaml
system_name: "switch_yolo_system"
connection:
  host: "localhost"
  port: 3001
  protocol: "http"

implementation:
  connector_class: "SwitchSystemConnector"
  timeout: 30.0
  max_retries: 3

switch_system:
  model_file_path: "model.csv"
  knowledge_file_path: "knowledge.csv"
  metrics_file_path: "metrics.csv"
```

### Key Configuration Sections

1. **Connection**: HTTP endpoint configuration
2. **Implementation**: Connector behavior settings
3. **Switch System**: File paths and system-specific settings
4. **Monitoring**: Metrics collection configuration
5. **Adaptation**: Thresholds and strategies
6. **Knowledge Base**: Model switching logic
7. **Verification**: Action validation constraints

## Usage

### Basic Usage

```python
import asyncio
from switch_connector import SwitchSystemConnector

async def main():
    # Load configuration
    config = load_config("switch_system_config.yaml")
    
    # Create connector
    connector = SwitchSystemConnector(config, logger)
    
    # Connect to system
    await connector.connect()
    
    # Switch model
    await connector.execute_command("switch_model", {"model": "yolov5s"})
    
    # Get metrics
    metrics = await connector.get_performance_metrics()
    
    # Disconnect
    await connector.disconnect()

asyncio.run(main())
```

### Advanced Usage

```python
# Get comprehensive system state
state = await connector.get_system_state()
print(f"Current model: {state['current_model']}")
print(f"Metrics: {state['metrics']}")

# Optimal model selection based on input rate
optimal_model = await connector.switch_to_optimal_model(input_rate=5.0)

# Update knowledge base thresholds
thresholds = {
    "yolov5nLower": "0.0", "yolov5nUpper": "2.0",
    "yolov5sLower": "2.0", "yolov5sUpper": "5.0",
    # ... more thresholds
}
await connector.execute_command("update_knowledge", thresholds)
```

## Integration with POLARIS

### Monitor Adapter Integration

The connector works with POLARIS Monitor Adapter to collect telemetry:

```python
# In monitor adapter
metrics = await connector.get_performance_metrics()
telemetry_event = TelemetryEvent(
    timestamp=time.time(),
    source="switch_system",
    metrics={
        "response_time": metrics["image_processing_time"],
        "confidence": metrics["confidence"],
        "utility": metrics["utility"],
        "cpu_usage": metrics["cpu_usage"]
    }
)
```

### Execution Adapter Integration

The connector works with POLARIS Execution Adapter to perform adaptations:

```python
# In execution adapter
async def execute_adaptation(action: ControlAction):
    if action.action_type == "switch_model":
        result = await connector.execute_command(
            "switch_model", 
            {"model": action.parameters["target_model"]}
        )
    elif action.action_type == "update_thresholds":
        result = await connector.execute_command(
            "update_knowledge",
            action.parameters["thresholds"]
        )
```

### Verification Adapter Integration

The connector supports action verification:

```python
# Verify model switch action
def verify_model_switch(action):
    target_model = action.parameters.get("target_model")
    if target_model not in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
        return VerificationResult(approved=False, reason="Invalid model")
    return VerificationResult(approved=True)
```

## Metrics and Monitoring

### Key Performance Metrics

- **image_processing_time**: Time to process one image (seconds)
- **confidence**: Average confidence score of detections
- **utility**: Calculated utility value based on response time and confidence
- **cpu_usage**: CPU utilization percentage
- **detection_boxes**: Number of objects detected
- **total_processed**: Total images processed

### Health Monitoring

The connector provides health check capabilities:

```python
# Check system health
is_healthy = await connector.health_check()

# Monitor continuously
while True:
    health_status = await connector.health_check()
    if not health_status:
        logger.warning("Switch system is unhealthy")
    await asyncio.sleep(30)
```

## Error Handling

The connector implements robust error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Timeout Handling**: Configurable timeouts for all operations
- **Validation**: Parameter validation before sending commands
- **Graceful Degradation**: Fallback behaviors for failed operations

## Testing

### Running the Demo

```bash
# Basic functionality demo
python switch_demo.py --mode basic

# Health monitoring demo
python switch_demo.py --mode health
```

### Unit Testing

```python
# Test model switching
async def test_model_switch():
    connector = SwitchSystemConnector(config, logger)
    await connector.connect()
    
    result = await connector.execute_command("switch_model", {"model": "yolov5s"})
    assert "Successfully switched" in result
    
    current = await connector.execute_command("get_current_model")
    assert current == "yolov5s"
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify Switch system is running on correct host/port
   - Check network connectivity
   - Ensure API endpoints are accessible

2. **Model Switch Failed**
   - Verify model name is valid
   - Check file permissions for model.csv
   - Ensure Switch system is in correct state

3. **Metrics Not Available**
   - Verify Elasticsearch is running (if used)
   - Check metrics collection is enabled
   - Ensure processing pipeline is active

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Connection Pooling**: Uses aiohttp session for efficient HTTP connections
- **Async Operations**: All operations are asynchronous for better performance
- **Caching**: Optional caching of frequently accessed data
- **Batch Operations**: Support for batch processing where applicable

## Security Considerations

- **Input Validation**: All parameters are validated before use
- **Error Sanitization**: Sensitive information is not exposed in error messages
- **Connection Security**: Support for HTTPS connections
- **Access Control**: Integration with POLARIS security framework

## Future Enhancements

1. **WebSocket Support**: Real-time bidirectional communication
2. **Advanced Caching**: Intelligent caching strategies
3. **Load Balancing**: Support for multiple Switch system instances
4. **Enhanced Metrics**: Additional performance and quality metrics
5. **Auto-Discovery**: Automatic discovery of Switch system capabilities

## Dependencies

- `aiohttp`: Async HTTP client
- `asyncio`: Async programming support
- `pathlib`: Path handling
- `csv`: CSV file operations
- `json`: JSON data handling
- `logging`: Structured logging

## License

This connector is part of the POLARIS framework and follows the same licensing terms.