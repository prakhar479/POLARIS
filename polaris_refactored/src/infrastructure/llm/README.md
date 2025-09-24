# LLM Integration Infrastructure

This module provides comprehensive LLM integration capabilities for the POLARIS adaptation framework, enabling intelligent reasoning, agentic interactions, and tool-based problem solving.

## Overview

The LLM integration infrastructure consists of six main components:

1. **LLM Client Abstraction** - Unified interface for different LLM providers
2. **Prompt Management** - Template system for structured LLM interactions
3. **Response Parser** - JSON schema validation and structured output parsing
4. **Tool Registry** - Agentic tool system for LLM function calling
5. **Conversation Manager** - Multi-turn conversation and agentic workflow management
6. **Caching System** - In-memory caching for LLM responses and tool results

## Components

### LLM Client (`client.py`)

Provides a unified interface for different LLM providers with support for:
- Multiple providers (OpenAI, Anthropic, Mock)
- Function calling capabilities
- Streaming responses
- Comprehensive error handling and retry logic
- Async/await support

```python
from polaris_refactored.src.infrastructure.llm import LLMConfiguration, OpenAIClient

config = LLMConfiguration(
    provider=LLMProvider.OPENAI,
    api_endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4"
)

async with OpenAIClient(config) as client:
    response = await client.generate_response(request)
```

### Prompt Manager (`prompt_manager.py`)

Manages prompt templates and conversation flows:
- Template rendering with variable substitution
- Default templates for common POLARIS use cases
- Conversation flow definitions for agentic interactions
- YAML-based template loading

```python
from polaris_refactored.src.infrastructure.llm import PromptManager

prompt_manager = PromptManager()
rendered = prompt_manager.render_template(
    "system_reasoning",
    system_id="web-server-01",
    current_metrics={"cpu": 85},
    health_status="warning",
    available_actions=["scale_up", "restart"]
)
```

### Response Parser (`response_parser.py`)

Parses and validates LLM responses:
- JSON schema validation
- Function call parsing
- Structured data extraction
- Error handling for malformed responses

```python
from polaris_refactored.src.infrastructure.llm import ResponseParser

parser = ResponseParser()
parsed = parser.parse_reasoning_response(llm_response)
if parsed["validation_passed"]:
    recommendations = parsed["parsed_json"]["recommendations"]
```

### Tool Registry (`tool_registry.py`)

Manages tools for agentic LLM interactions:
- Tool registration and discovery
- Schema generation for different LLM providers
- Parallel and sequential tool execution
- Execution history and statistics

```python
from polaris_refactored.src.infrastructure.llm import ToolRegistry, BaseTool

class CustomTool(BaseTool):
    def get_schema(self):
        return ToolSchema(name="custom", description="Custom tool", parameters={})
    
    async def execute(self, parameters):
        return {"result": "success"}

registry = ToolRegistry()
registry.register_tool(CustomTool())
result = await registry.execute_tool("custom", {})
```

### Conversation Manager (`conversation_manager.py`)

Manages conversations and agentic workflows:
- Multi-turn conversation tracking
- Agentic conversation execution with tool usage
- Context management and iteration control
- Conversation persistence and cleanup

```python
from polaris_refactored.src.infrastructure.llm import ConversationManager

manager = ConversationManager(llm_client, tool_registry, response_parser)

# Create agentic conversation
conv = manager.create_agentic_conversation(
    objective="Analyze system performance",
    available_tools=["get_metrics", "analyze_trends"],
    max_iterations=10
)

# Run to completion
result = await manager.run_agentic_conversation(conv.conversation_id)
```

### LLM Cache (`cache.py`)

Provides caching for LLM responses and tool results:
- In-memory caching with TTL support
- Request hashing for cache keys
- Cache statistics and monitoring
- Automatic cleanup and eviction

```python
from polaris_refactored.src.infrastructure.llm import LLMCache

cache = LLMCache(max_size=1000, default_ttl=300)

# Cache automatically used by conversation manager
cached_response = cache.get_llm_response(request)
if cached_response is None:
    response = await llm_client.generate_response(request)
    cache.cache_llm_response(request, response)
```

## Configuration

The LLM integration system uses Pydantic models for configuration validation:

```python
from polaris_refactored.src.infrastructure.llm import LLMConfiguration, LLMProvider

config = LLMConfiguration(
    provider=LLMProvider.OPENAI,
    api_endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4",
    max_tokens=2000,
    temperature=0.1,
    timeout=60.0,
    max_retries=3,
    cache_ttl=300
)
```

## Default Prompt Templates

The system includes several default prompt templates:

- **system_reasoning**: For system analysis and adaptation recommendations
- **world_model_update**: For updating digital twin world models
- **adaptation_impact_simulation**: For simulating adaptation impacts
- **agentic_reasoning**: For agentic tool-based reasoning

## Error Handling

Comprehensive exception hierarchy extends the POLARIS exception system:

- `LLMIntegrationError`: Base exception for LLM integration issues
- `LLMAPIError`: API communication errors
- `LLMResponseParsingError`: Response parsing and validation errors
- `LLMToolError`: Tool execution errors
- `LLMConfigurationError`: Configuration and setup errors

## Usage Examples

### Basic LLM Interaction

```python
import asyncio
from polaris_refactored.src.infrastructure.llm import *

async def basic_example():
    config = LLMConfiguration(
        provider=LLMProvider.MOCK,
        api_endpoint="http://localhost:8000",
        model_name="mock-model"
    )
    
    async with MockLLMClient(config) as client:
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hello!")],
            model_name="mock-model"
        )
        
        response = await client.generate_response(request)
        print(f"Response: {response.content}")

asyncio.run(basic_example())
```

### Agentic Reasoning with Tools

```python
async def agentic_example():
    # Set up components
    client = MockLLMClient()
    tool_registry = ToolRegistry()
    tool_registry.register_tool(EchoTool())
    response_parser = ResponseParser()
    
    manager = ConversationManager(client, tool_registry, response_parser)
    
    # Create agentic conversation
    conv = manager.create_agentic_conversation(
        objective="Echo a message",
        available_tools=["echo"],
        max_iterations=5
    )
    
    # Set mock response to use tool
    client.set_mock_responses([
        '{"action": "use_tool", "tool_name": "echo", "parameters": {"message": "Hello World"}}',
        '{"action": "final_answer", "result": {"echoed": "Hello World"}, "confidence": 1.0}'
    ])
    
    # Run conversation
    result = await manager.run_agentic_conversation(
        conv.conversation_id,
        "Please echo 'Hello World'"
    )
    
    print(f"Final result: {result.final_result}")

asyncio.run(agentic_example())
```

## Integration with POLARIS Framework

The LLM integration infrastructure is designed to seamlessly integrate with the existing POLARIS framework:

1. **Configuration**: Uses the existing POLARIS configuration system
2. **Observability**: Integrates with POLARIS logging, metrics, and tracing
3. **Error Handling**: Extends the POLARIS exception hierarchy
4. **Async Patterns**: Follows existing async/await patterns
5. **Dependency Injection**: Compatible with POLARIS DI container

## Testing

Comprehensive test suite covers all components:

```bash
python -m pytest tests/test_llm_integration.py -v
```

The test suite includes:
- Unit tests for all components
- Integration tests for complete workflows
- Mock implementations for testing without external dependencies
- Error handling and edge case testing

## Performance Considerations

- **Caching**: Automatic caching of LLM responses and tool results
- **Async Operations**: Non-blocking I/O for all LLM interactions
- **Connection Pooling**: Efficient HTTP connection management
- **Resource Management**: Automatic cleanup and resource lifecycle management

## Security Considerations

- **API Key Management**: Secure handling of API credentials
- **Input Validation**: Comprehensive validation of all inputs
- **Error Sanitization**: Safe error messages without sensitive data exposure
- **Rate Limiting**: Built-in retry logic with exponential backoff

## Monitoring and Observability

The system provides comprehensive monitoring capabilities:

- **Metrics**: API call counts, latency, cache hit rates, tool execution times
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing for complex workflows
- **Health Checks**: Component health monitoring and status reporting

## Future Enhancements

Planned enhancements include:

1. **Additional Providers**: Support for more LLM providers (Azure OpenAI, Google PaLM, etc.)
2. **Advanced Caching**: Persistent caching with Redis/database backends
3. **Streaming Support**: Full streaming response support for all providers
4. **Fine-tuning Integration**: Support for custom fine-tuned models
5. **Advanced Tool System**: More sophisticated tool orchestration and workflows