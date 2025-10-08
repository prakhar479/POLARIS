# Gemini World Model for POLARIS Digital Twin

The Gemini World Model is a clean, focused implementation that uses Google's Gemini LLM to provide intelligent system analysis, predictions, and diagnostics with meta-learning capabilities.

## Overview

The Gemini World Model provides:

- **Intelligent State Analysis**: Uses LLM reasoning to understand system state
- **Predictive Simulation**: Forecasts future system behavior based on proposed actions
- **Root Cause Diagnosis**: Analyzes anomalies and identifies likely causes
- **Meta-Learning**: Continuously improves accuracy through calibration feedback
- **Conversation Memory**: Maintains context across interactions for better analysis

## Key Features

### 🧠 LLM-Powered Analysis
- Real Google Gemini API integration
- Intelligent interpretation of system metrics
- Natural language understanding of queries
- Context-aware responses

### 📈 Meta-Learning Capabilities
- Learns from prediction accuracy feedback
- Adapts confidence scoring over time
- Improves analysis quality through experience
- Exponential moving average for accuracy tracking

### 🔧 Clean Architecture
- Focused, maintainable codebase
- Proper error handling and retries
- Rate limiting and concurrency control
- Modular design for easy extension

## Configuration

### Basic Configuration (`gemini_world_model.yaml`)

```yaml
implementation: "gemini"

config:
  # API Configuration
  api_key_env: "GEMINI_API_KEY"
  model: "gemini-1.5-flash"
  temperature: 0.7
  max_tokens: 2048
  
  # Performance Settings
  concurrent_requests: 5
  request_timeout_sec: 30
  retry_attempts: 3
  
  # Meta-Learning Settings
  max_conversation_memory: 50
  max_history_events: 1000
  learning_rate: 0.1
  confidence_threshold: 0.6
```

### Environment Variables

```bash
# Required: Your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Optional: Override model selection
export GEMINI_MODEL="gemini-1.5-pro"  # For more advanced reasoning
```

## Usage

### 1. Setup and Initialization

```python
from polaris.models.gemini_world_model import GeminiWorldModel

config = {
    "api_key_env": "GEMINI_API_KEY",
    "model": "gemini-1.5-flash",
    "temperature": 0.7,
    "max_tokens": 1024
}

model = GeminiWorldModel(config, logger)
await model.initialize()
```

### 2. Query System State

```python
from polaris.models.world_model import QueryRequest

query = QueryRequest(
    query_id="query-1",
    query_type="current_state",
    query_content="What is the current system performance status?",
    parameters={"format": "summary"}
)

response = await model.query_state(query)
print(f"Result: {response.result}")
print(f"Confidence: {response.confidence}")
```

### 3. Run Simulations

```python
from polaris.models.world_model import SimulationRequest

simulation = SimulationRequest(
    simulation_id="sim-1",
    simulation_type="what_if",
    actions=[{"action_type": "ADD_SERVER", "parameters": {"count": 1}}],
    horizon_minutes=30
)

response = await model.simulate(simulation)
print(f"Future states: {len(response.future_states)}")
print(f"Confidence: {response.confidence}")
```

### 4. Diagnose Issues

```python
from polaris.models.world_model import DiagnosisRequest

diagnosis = DiagnosisRequest(
    diagnosis_id="diag-1",
    anomaly_description="High response time detected in web service",
    context={"service": "web-api", "severity": "high"}
)

response = await model.diagnose(diagnosis)
print(f"Hypotheses: {response.hypotheses}")
print(f"Causal chain: {response.causal_chain}")
```

### 5. Meta-Learning with Calibration

```python
from polaris.models.digital_twin_events import CalibrationEvent

# After making a prediction, provide feedback on accuracy
calibration = CalibrationEvent(
    calibration_id="cal-1",
    prediction_id="pred-1",
    predicted_outcome={"cpu_usage": 80.0},
    actual_outcome={"cpu_usage": 78.5},
    accuracy_metrics={"absolute_error": 1.5, "accuracy": 0.85}
)

await model.calibrate(calibration)
```

## Integration with Agentic Reasoner

The Gemini World Model works seamlessly with the Agentic Reasoner:

```bash
# 1. Start Digital Twin with Gemini World Model
python src/scripts/start_component.py digital-twin

# 2. Start Agentic Reasoner
python src/scripts/start_component.py agentic-reasoner

# 3. The agentic reasoner will now use the Gemini-powered Digital Twin
```

## Testing

### Unit Tests
```bash
# Test the Gemini World Model implementation
python tests/test_gemini_world_model.py
```

### Meta-Learning Demo
```bash
# Demonstrate meta-learning capabilities
python examples/gemini_meta_learning_demo.py
```

### Integration Test
```bash
# Test with agentic reasoner
python tests/test_digital_twin_connection.py
```

## Performance Considerations

### API Usage
- **Rate Limiting**: Built-in rate limiting prevents API quota exhaustion
- **Concurrency**: Configurable concurrent request limits
- **Retries**: Automatic retry with exponential backoff
- **Timeouts**: Configurable timeouts prevent hanging requests

### Cost Optimization
- **Model Selection**: Use `gemini-1.5-flash` for cost-effectiveness
- **Token Limits**: Configure `max_tokens` to control response length
- **Temperature**: Lower temperature (0.3-0.5) for more focused responses
- **Batch Processing**: Group related queries when possible

### Memory Management
- **Conversation Memory**: Limited to configurable size (default 50 messages)
- **Event History**: Limited to configurable size (default 1000 events)
- **State Cleanup**: Automatic cleanup of old data

## Meta-Learning Features

### Accuracy Tracking
- Exponential moving average of prediction accuracy
- Per-prediction confidence scoring
- Historical accuracy trends

### Adaptive Confidence
- Confidence scores adjust based on past performance
- Lower confidence for areas with poor historical accuracy
- Higher confidence for well-understood patterns

### Learning Rate Configuration
```yaml
config:
  learning_rate: 0.1  # How quickly to adapt (0.0-1.0)
  confidence_threshold: 0.6  # Minimum confidence for reliable predictions
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

2. **API Rate Limits**
   - Reduce `concurrent_requests` in configuration
   - Increase `request_timeout_sec`
   - Check your API quota

3. **Empty Responses**
   - Check API key validity
   - Verify internet connectivity
   - Review Gemini API status

4. **JSON Parsing Errors**
   - The model handles both JSON and text responses
   - Parsing failures fall back to text interpretation
   - Check prompt clarity for better JSON responses

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### 1. API Key Security
- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly
- Monitor API usage

### 2. Prompt Engineering
- Be specific in queries
- Provide clear context
- Request structured responses when needed
- Include confidence requirements

### 3. Meta-Learning
- Provide regular calibration feedback
- Monitor accuracy trends
- Adjust learning rate based on system stability
- Use accuracy metrics to guide decision making

### 4. Performance Optimization
- Use appropriate model for your use case
- Configure reasonable token limits
- Monitor API costs
- Implement caching for repeated queries

## Integration Examples

### With Monitor Adapter
The Gemini World Model automatically receives telemetry updates from monitor adapters and uses them to build system understanding.

### With Execution Adapter
Execution results are fed back to the model for learning and improving future predictions.

### With Meta-Learner Agent
The Meta-Learner Agent can provide calibration feedback to continuously improve the Gemini World Model's accuracy.

## Future Enhancements

Potential improvements:
1. **Vector Embeddings**: Add semantic search capabilities
2. **Fine-tuning**: Custom model fine-tuning for domain-specific knowledge
3. **Multi-modal**: Support for logs, metrics, and other data types
4. **Ensemble Methods**: Combine multiple models for better accuracy
5. **Advanced Meta-Learning**: More sophisticated learning algorithms

The Gemini World Model provides a solid foundation for intelligent system analysis while maintaining simplicity and extensibility.