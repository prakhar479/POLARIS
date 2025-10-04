# Agentic Reasoner for POLARIS

The Agentic Reasoner is an autonomous adaptation decision-making component that can dynamically decide which tools to use (Knowledge Base queries, Digital Twin interactions) and make informed adaptation decisions based on its analysis.

## Overview

Unlike the traditional LLM reasoner that follows a fixed workflow, the Agentic Reasoner acts as an autonomous agent that:

1. **Analyzes** the current system situation
2. **Decides** what additional information it needs
3. **Uses tools** dynamically to gather that information
4. **Synthesizes** all information to make adaptation decisions
5. **Generates** appropriate control actions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telemetry     │───▶│  Agentic LLM     │───▶│  Control        │
│   Data          │    │  Reasoner        │    │  Actions        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │     Tools        │
                    │                  │
                    │ ┌──────────────┐ │
                    │ │ Knowledge    │ │
                    │ │ Base Query   │ │
                    │ └──────────────┘ │
                    │                  │
                    │ ┌──────────────┐ │
                    │ │ Digital Twin │ │
                    │ │ Interface    │ │
                    │ └──────────────┘ │
                    └──────────────────┘
```

## Key Features

### 🤖 Autonomous Decision Making
- The LLM acts as the agent's reasoning engine
- Dynamically decides when and how to use available tools
- No fixed workflow - adapts reasoning process to the situation

### 🔧 Dynamic Tool Usage
- **Knowledge Base Tool**: Query historical data, observations, and telemetry
- **Digital Twin Tool**: Query system state, run simulations, perform diagnostics
- Tools are used based on the agent's assessment of what information is needed

### 🧠 Multi-Step Reasoning
- Can perform multiple reasoning iterations
- Each iteration can involve tool calls and analysis
- Builds up understanding progressively

### 📊 Context-Aware Analysis
- Considers current system state, historical trends, and predictions
- Adapts reasoning depth based on situation complexity
- Handles both routine and anomalous situations

## Available Tools

### Knowledge Base Tool (`query_knowledge_base`)

Query the knowledge base for historical data and observations.

**Parameters:**
- `query_type`: Type of query (`structured`, `natural_language`, `recent_observations`, `raw_telemetry`)
- `data_types`: Types of data to query (for structured queries)
- `filters`: Filters to apply to the query
- `query_text`: Natural language query text
- `limit`: Maximum number of results to return

**Example Usage:**
```
TOOL_CALL: query_knowledge_base
PARAMETERS: {"query_type": "structured", "data_types": ["observation"], "filters": {"source": "swim_snapshotter"}, "limit": 50}
```

### Digital Twin Tool (`query_digital_twin`)

Interact with the digital twin for system analysis and predictions.

**Parameters:**
- `operation`: Type of operation (`query`, `simulate`, `diagnose`)
- `query_type`: Type of query (for query operations)
- `query_content`: Content of the query
- `simulation_type`: Type of simulation (for simulate operations)
- `actions`: Actions to simulate
- `horizon_minutes`: Simulation horizon in minutes
- `anomaly_description`: Description of anomaly (for diagnose operations)

**Example Usage:**
```
TOOL_CALL: query_digital_twin
PARAMETERS: {"operation": "simulate", "simulation_type": "forecast", "actions": [{"action_type": "ADD_SERVER", "params": {"count": 1}}], "horizon_minutes": 30}
```

## Configuration

The agentic reasoner uses the configuration file `src/config/agentic_reasoner_config.yaml`:

```yaml
agentic_reasoner:
  llm:
    model: "gpt-oss:20b"
    base_url: "http://10.10.16.46:11435"
    max_tokens: 1024
    temperature: 0.3
    timeout: 600.0
    max_retries: 3

  tools:
    max_tool_calls: 5
    knowledge_base_enabled: true
    digital_twin_enabled: true
    kb_query_timeout: 30.0
    dt_query_timeout: 45.0

  decision_making:
    confidence_thresholds:
      high_impact_actions: 0.8
      medium_impact_actions: 0.6
      low_impact_actions: 0.4
    
    constraints:
      max_response_time_ms: 1000
      min_utilization: 0.5
      max_utilization: 0.85
      min_servers: 1
      max_servers: 10
```

## Usage

### Starting the Agentic Reasoner

```bash
# Start the agentic reasoner
python src/scripts/start_component.py agentic-reasoner

# With debug logging
python src/scripts/start_component.py agentic-reasoner --log-level DEBUG

# Validate configuration only
python src/scripts/start_component.py agentic-reasoner --validate-only

# Dry run (initialize but don't start processing)
python src/scripts/start_component.py agentic-reasoner --dry-run
```

### Prerequisites

Before starting the agentic reasoner, ensure these components are running:

1. **NATS Server**:
   ```bash
   nats-server
   ```

2. **Knowledge Base Service**:
   ```bash
   python src/scripts/start_component.py knowledge-base
   ```

3. **Digital Twin** (optional but recommended):
   ```bash
   python src/scripts/start_component.py digital-twin
   ```

### Running the Demo

```bash
# Run the comprehensive demo
python examples/agentic_reasoner_demo.py
```

The demo includes several scenarios:
- **High Response Time Crisis**: Tests emergency response capabilities
- **Underutilized System**: Tests cost optimization decisions
- **Optimal System**: Tests steady-state behavior
- **Borderline Performance**: Tests complex analysis capabilities
- **System Anomaly**: Tests diagnostic capabilities

## Example Reasoning Session

Here's an example of how the agentic reasoner processes a high response time scenario:

### Input Telemetry:
```json
{
  "events": [
    {"name": "average_response_time", "value": 2.5},
    {"name": "utilization", "value": 0.95},
    {"name": "active_servers", "value": 2},
    {"name": "dimmer", "value": 1.0}
  ]
}
```

### Agent's Reasoning Process:

1. **Initial Analysis**: "High response time (2.5s) with very high utilization (95%) - critical situation"

2. **Tool Usage Decision**: "Need historical data to understand trends"
   ```
   TOOL_CALL: query_knowledge_base
   PARAMETERS: {"query_type": "structured", "data_types": ["observation"], "limit": 20}
   ```

3. **Trend Analysis**: "Response time has been increasing over the last 10 minutes"

4. **Prediction Check**: "Let me simulate adding a server to see the impact"
   ```
   TOOL_CALL: query_digital_twin
   PARAMETERS: {"operation": "simulate", "actions": [{"action_type": "ADD_SERVER"}], "horizon_minutes": 15}
   ```

5. **Final Decision**: Based on analysis and simulation results
   ```json
   {
     "action_type": "ADD_SERVER",
     "source": "agentic_reasoner",
     "params": {"server_type": "compute", "count": 1},
     "priority": "high",
     "reasoning": "Critical response time with simulation showing 40% improvement"
   }
   ```

## Comparison with Traditional LLM Reasoner

| Aspect | Traditional LLM Reasoner | Agentic LLM Reasoner |
|--------|-------------------------|---------------------|
| **Workflow** | Fixed, predefined steps | Dynamic, adaptive |
| **Tool Usage** | Always queries KB | Uses tools as needed |
| **Reasoning Depth** | Single iteration | Multiple iterations |
| **Context Building** | Static context builder | Dynamic information gathering |
| **Decision Process** | Template-based prompts | Autonomous reasoning |
| **Adaptability** | Limited to prompt changes | Fully adaptive behavior |

## Monitoring and Debugging

### Logs to Monitor

1. **Tool Usage Patterns**: Which tools are being used and when
2. **Reasoning Iterations**: How many steps the agent takes
3. **Decision Confidence**: How confident the agent is in its decisions
4. **Tool Success Rates**: Whether tool calls are successful

### Debug Mode

Enable debug mode in the configuration:

```yaml
development:
  enable_debug_mode: true
  log_tool_calls: true
  log_llm_conversations: true
  save_reasoning_traces: true
```

### Performance Metrics

The agentic reasoner tracks:
- Reasoning time per session
- Tool calls per session
- Decision confidence scores
- Action success rates

## Best Practices

### 1. Tool Configuration
- Enable both KB and DT tools for best results
- Set appropriate timeouts based on your system
- Limit max_tool_calls to prevent excessive iterations

### 2. LLM Configuration
- Use higher temperature (0.3-0.5) for more creative tool usage
- Increase max_tokens for complex reasoning scenarios
- Adjust timeout based on your LLM response times

### 3. Monitoring
- Monitor tool usage patterns to optimize configuration
- Track decision confidence to identify uncertain scenarios
- Watch for excessive tool calls indicating confusion

### 4. Safety
- Enable constraint validation
- Set appropriate confidence thresholds
- Consider enabling action simulation for critical decisions

## Troubleshooting

### Common Issues

1. **Agent not using tools**: Check tool configuration and LLM prompt understanding
2. **Excessive tool calls**: Reduce max_tool_calls or improve LLM instructions
3. **Poor decisions**: Review tool results and LLM reasoning logs
4. **Timeouts**: Increase tool timeouts or LLM timeout settings

### Debug Steps

1. Enable debug logging
2. Check tool availability and responses
3. Review LLM conversation logs
4. Verify NATS connectivity
5. Test individual tools separately

## Future Enhancements

Potential improvements for the agentic reasoner:

1. **Learning from Experience**: Remember successful tool usage patterns
2. **Multi-Agent Collaboration**: Coordinate with other reasoning agents
3. **Advanced Tool Orchestration**: Chain tool calls more intelligently
4. **Confidence-Based Tool Selection**: Use tools based on uncertainty levels
5. **Custom Tool Development**: Add domain-specific tools for specialized analysis

## Contributing

To extend the agentic reasoner:

1. **Add New Tools**: Implement the `AgenticTool` interface
2. **Improve Reasoning**: Enhance the system prompt and reasoning logic
3. **Add Metrics**: Implement additional monitoring capabilities
4. **Optimize Performance**: Improve tool call efficiency and caching

See the source code in `src/polaris/agents/agentic_reasoner.py` for implementation details.