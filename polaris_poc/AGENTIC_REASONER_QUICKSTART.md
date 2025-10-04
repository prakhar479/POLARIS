# Agentic Reasoner Quick Start Guide

This guide will help you quickly set up and test the new Agentic Reasoner in the POLARIS system.

## What is the Agentic Reasoner?

The Agentic Reasoner is an autonomous adaptation decision-making component that can:
- **Dynamically decide** which tools to use (Knowledge Base, Digital Twin)
- **Autonomously gather** additional information as needed
- **Make informed decisions** based on comprehensive analysis
- **Adapt its reasoning process** to different scenarios

Unlike the traditional LLM reasoner that follows a fixed workflow, the Agentic Reasoner acts as an intelligent agent that decides its own reasoning process.

## Quick Setup

### 1. Prerequisites

Ensure you have the following running:

```bash
# 1. Start NATS server
nats-server

# 2. Start Knowledge Base (in separate terminal)
cd polaris_poc
python src/scripts/start_component.py knowledge-base

# 3. Start Digital Twin (in separate terminal) - Optional but recommended
python src/scripts/start_component.py digital-twin
```

### 2. Start the Agentic Reasoner

```bash
# Start the agentic reasoner
python src/scripts/start_component.py agentic-reasoner

# Or with debug logging to see tool usage
python src/scripts/start_component.py agentic-reasoner --log-level DEBUG
```

You should see output like:
```
🚀 Starting Agentic Reasoner agent...
🤖 This reasoner can autonomously use tools (KB, Digital Twin) to make decisions
✅ Agentic Reasoner agent started successfully
🔧 Available tools: Knowledge Base queries, Digital Twin interactions
📡 Agentic Reasoner is running - press Ctrl+C to stop
```

### 3. Test with Demo

In another terminal, run the demo:

```bash
python examples/agentic_reasoner_demo.py
```

This will run 5 different scenarios showing how the agent autonomously uses tools.

## Quick Test

### Option 1: Run Unit Tests

```bash
# Test the core functionality without full infrastructure
python tests/test_agentic_reasoner.py
```

### Option 2: Manual Test via NATS

You can send a reasoning request directly:

```python
import asyncio
import json
from polaris.common.nats_client import NATSClient

async def test_agentic_reasoner():
    client = NATSClient("nats://localhost:4222")
    await client.connect()
    
    request = {
        "session_id": "test-123",
        "reasoning_type": "decision",
        "events": [
            {"name": "average_response_time", "value": 2.0},  # High!
            {"name": "utilization", "value": 0.95},
            {"name": "active_servers", "value": 2}
        ]
    }
    
    response = await client.request_json(
        "polaris.reasoner.kernel.requests",
        request,
        timeout=60.0
    )
    
    print("Response:", json.dumps(response, indent=2))
    await client.close()

# Run the test
asyncio.run(test_agentic_reasoner())
```

## Expected Behavior

### Scenario 1: High Response Time
**Input**: Response time = 2.0s, Utilization = 95%

**Expected Agent Behavior**:
1. Analyze current situation → "Critical performance issue"
2. Query Knowledge Base → "Check historical trends"
3. Possibly query Digital Twin → "Simulate adding server"
4. Decision → "ADD_SERVER with high priority"

### Scenario 2: Underutilized System
**Input**: Response time = 0.3s, Utilization = 25%

**Expected Agent Behavior**:
1. Analyze current situation → "System over-provisioned"
2. Query Knowledge Base → "Check if this is a trend"
3. Decision → "REMOVE_SERVER with medium priority"

### Scenario 3: Optimal Performance
**Input**: Response time = 0.8s, Utilization = 65%

**Expected Agent Behavior**:
1. Analyze current situation → "System performing well"
2. Minimal tool usage → "Quick verification"
3. Decision → "NO_ACTION"

## Key Differences from Traditional Reasoner

| Aspect | Traditional LLM Reasoner | Agentic LLM Reasoner |
|--------|-------------------------|---------------------|
| **Tool Usage** | Always queries KB first | Uses tools as needed |
| **Reasoning** | Fixed prompt template | Dynamic reasoning process |
| **Iterations** | Single pass | Multiple iterations possible |
| **Adaptability** | Static workflow | Adapts to situation |

## Monitoring the Agent

### 1. Check Logs

Look for these log patterns:

```
# Tool usage
INFO - Executing tool: query_knowledge_base
INFO - Added 1 tool results to conversation

# Reasoning iterations
INFO - Reasoning iteration 1
INFO - Reasoning iteration 2

# Final decisions
INFO - Final decision reached
```

### 2. Debug Mode

Enable detailed logging in the config:

```yaml
development:
  enable_debug_mode: true
  log_tool_calls: true
  log_llm_conversations: true
```

## Configuration

The agentic reasoner uses `src/config/agentic_reasoner_config.yaml`:

```yaml
agentic_reasoner:
  tools:
    max_tool_calls: 5  # Max tools per reasoning session
    knowledge_base_enabled: true
    digital_twin_enabled: true
  
  llm:
    model: "gpt-oss:20b"
    temperature: 0.3  # Higher for more creative tool usage
    max_tokens: 1024
```

## Troubleshooting

### Issue: Agent not using tools
**Solution**: 
- Check tool configuration is enabled
- Verify KB/DT services are running
- Increase LLM temperature for more exploration

### Issue: Too many tool calls
**Solution**:
- Reduce `max_tool_calls` in config
- Check LLM is understanding tool results
- Review system prompt clarity

### Issue: Poor decisions
**Solution**:
- Enable debug logging to see reasoning process
- Check tool results are meaningful
- Verify LLM model is appropriate

### Issue: Timeouts
**Solution**:
- Increase tool timeouts in config
- Check NATS connectivity
- Verify LLM endpoint is responsive

## Next Steps

1. **Experiment with Scenarios**: Try different telemetry patterns
2. **Monitor Tool Usage**: See which tools are most useful
3. **Tune Configuration**: Adjust based on your system's needs
4. **Add Custom Tools**: Extend with domain-specific tools
5. **Compare Performance**: Test against traditional reasoner

## Advanced Usage

### Custom Tool Development

Create your own tools by extending `AgenticTool`:

```python
class CustomAnalyticsTool(AgenticTool):
    def __init__(self):
        super().__init__(
            name="custom_analytics",
            description="Perform custom analytics",
            parameters={"analysis_type": {"type": "string"}}
        )
    
    async def execute(self, **kwargs):
        # Your custom logic here
        return {"success": True, "result": "analysis_complete"}
```

### Integration with Existing Systems

The agentic reasoner is plug-and-play compatible with the existing POLARIS architecture:

- Uses same NATS subjects
- Produces same action format
- Integrates with existing monitoring
- Compatible with verification adapter

## Support

For issues or questions:
1. Check the logs for error messages
2. Run the unit tests to verify functionality
3. Review the full documentation in `docs/AGENTIC_REASONER.md`
4. Check the source code in `src/polaris/agents/agentic_reasoner.py`

Happy reasoning! 🤖