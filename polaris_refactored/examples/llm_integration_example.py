"""
Example demonstrating the LLM integration infrastructure.

This example shows how to use the various components of the LLM integration
system including clients, prompt management, tool registry, and conversation management.
"""

import asyncio
from pathlib import Path

from polaris_refactored.src.infrastructure.llm import (
    LLMConfiguration, LLMProvider, MockLLMClient,
    PromptManager, ResponseParser, ToolRegistry, ConversationManager,
    LLMCache, BaseTool, ToolSchema, Message, MessageRole, LLMRequest
)


class SystemMetricsTool(BaseTool):
    """Example tool for retrieving system metrics."""
    
    def __init__(self):
        super().__init__("get_system_metrics", "Retrieve current system metrics")
        self.mock_metrics = {
            "cpu_usage": 75.5,
            "memory_usage": 68.2,
            "disk_usage": 45.0,
            "network_io": 1024.5
        }
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "system_id": {
                    "type": "string",
                    "description": "ID of the system to get metrics for"
                },
                "metric_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of metrics to retrieve (optional)"
                }
            },
            required=["system_id"]
        )
    
    async def execute(self, parameters):
        system_id = parameters["system_id"]
        metric_types = parameters.get("metric_types", list(self.mock_metrics.keys()))
        
        # Simulate async operation
        await asyncio.sleep(0.1)
        
        filtered_metrics = {
            k: v for k, v in self.mock_metrics.items()
            if k in metric_types
        }
        
        return {
            "system_id": system_id,
            "metrics": filtered_metrics,
            "timestamp": "2024-01-01T12:00:00Z"
        }


async def demonstrate_basic_llm_client():
    """Demonstrate basic LLM client usage."""
    print("=== Basic LLM Client Demo ===")
    
    # Create mock LLM client
    config = LLMConfiguration(
        provider=LLMProvider.MOCK,
        api_endpoint="http://localhost:8000",
        model_name="mock-gpt-4"
    )
    
    async with MockLLMClient(config) as client:
        # Create a simple request
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="What is the capital of France?")
            ],
            model_name="mock-gpt-4"
        )
        
        # Generate response
        response = await client.generate_response(request)
        
        print(f"Request: {request.messages[-1].content}")
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        print()


async def demonstrate_prompt_management():
    """Demonstrate prompt management system."""
    print("=== Prompt Management Demo ===")
    
    prompt_manager = PromptManager()
    
    # Render a system reasoning prompt
    rendered_prompt = prompt_manager.render_template(
        "system_reasoning",
        system_id="web-server-01",
        current_metrics={"cpu_usage": 85, "memory_usage": 70, "response_time": 250},
        health_status="warning",
        available_actions=["scale_up", "restart_service", "clear_cache"]
    )
    
    print("Rendered System Reasoning Prompt:")
    print(rendered_prompt[:500] + "..." if len(rendered_prompt) > 500 else rendered_prompt)
    print()


async def demonstrate_tool_system():
    """Demonstrate tool registry and execution."""
    print("=== Tool System Demo ===")
    
    # Create tool registry and register custom tool
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SystemMetricsTool())
    
    print(f"Available tools: {tool_registry.list_tools()}")
    
    # Execute system metrics tool
    result = await tool_registry.execute_tool(
        "get_system_metrics",
        {
            "system_id": "web-server-01",
            "metric_types": ["cpu_usage", "memory_usage"]
        }
    )
    
    print(f"Tool execution result:")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")
    print(f"  Execution time: {result.execution_time:.3f}s")
    print()


async def demonstrate_conversation_management():
    """Demonstrate conversation management with agentic capabilities."""
    print("=== Conversation Management Demo ===")
    
    # Set up components
    client = MockLLMClient()
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SystemMetricsTool())
    response_parser = ResponseParser()
    
    conversation_manager = ConversationManager(client, tool_registry, response_parser)
    
    # Create a regular conversation
    conv = conversation_manager.create_conversation()
    
    # Send a message and get response
    response_msg = await conversation_manager.send_message(
        conv.conversation_id,
        "Hello! Can you help me analyze system performance?",
        generate_response=True
    )
    
    print(f"User: Hello! Can you help me analyze system performance?")
    print(f"Assistant: {response_msg.content}")
    print()
    
    # Create an agentic conversation
    print("Creating agentic conversation...")
    agentic_conv = conversation_manager.create_agentic_conversation(
        objective="Analyze system metrics and provide recommendations",
        available_tools=["get_system_metrics"],
        max_iterations=3
    )
    
    # Set up mock responses for agentic flow
    client.set_mock_responses([
        '{"action": "use_tool", "tool_name": "get_system_metrics", "parameters": {"system_id": "web-server-01"}}',
        '{"action": "final_answer", "result": {"analysis": "System is under moderate load", "recommendations": ["Monitor CPU usage", "Consider scaling if trend continues"]}, "reasoning": "Based on current metrics", "confidence": 0.8}'
    ])
    
    # Run agentic conversation
    result_conv = await conversation_manager.run_agentic_conversation(
        agentic_conv.conversation_id,
        "Please analyze the metrics for web-server-01 and provide recommendations"
    )
    
    print(f"Agentic conversation completed: {result_conv.is_complete}")
    print(f"Iterations: {result_conv.current_iteration}")
    print(f"Final result: {result_conv.final_result}")
    print()


async def demonstrate_caching():
    """Demonstrate LLM response caching."""
    print("=== Caching Demo ===")
    
    cache = LLMCache(max_size=5, default_ttl=300)
    
    # Create a request
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
        model_name="mock-model"
    )
    
    # Check cache miss
    cached_response = cache.get_llm_response(request)
    print(f"Cache miss (expected): {cached_response is None}")
    
    # Generate and cache response
    client = MockLLMClient()
    response = await client.generate_response(request)
    cache.cache_llm_response(request, response)
    
    # Check cache hit
    cached_response = cache.get_llm_response(request)
    print(f"Cache hit: {cached_response is not None}")
    print(f"Cached content: {cached_response.content}")
    
    # Show cache statistics
    stats = cache.get_statistics()
    print(f"Cache statistics: {stats}")
    print()


async def demonstrate_response_parsing():
    """Demonstrate response parsing capabilities."""
    print("=== Response Parsing Demo ===")
    
    response_parser = ResponseParser()
    
    # Create a mock response with JSON content
    json_content = '''
    {
        "analysis": "System CPU usage is at 85%, which is above the recommended threshold of 80%",
        "recommendations": [
            {
                "action_type": "scale_up",
                "parameters": {"instances": 2},
                "rationale": "Increase capacity to handle current load",
                "confidence": 0.9,
                "risk_level": "low"
            }
        ],
        "reasoning_steps": [
            "Analyzed current CPU metrics",
            "Compared against thresholds",
            "Determined scaling is appropriate"
        ]
    }
    '''
    
    from polaris_refactored.src.infrastructure.llm.models import LLMResponse
    
    response = LLMResponse(
        content=json_content,
        model_name="mock-model",
        usage={"total_tokens": 150}
    )
    
    # Parse as reasoning response
    parsed = response_parser.parse_reasoning_response(response)
    
    print(f"Parsing successful: {parsed['validation_passed']}")
    if parsed['validation_passed']:
        json_data = parsed['parsed_json']
        print(f"Analysis: {json_data['analysis']}")
        print(f"Number of recommendations: {len(json_data['recommendations'])}")
        print(f"First recommendation: {json_data['recommendations'][0]['action_type']}")
    print()


async def main():
    """Run all demonstrations."""
    print("LLM Integration Infrastructure Demo")
    print("=" * 50)
    print()
    
    await demonstrate_basic_llm_client()
    await demonstrate_prompt_management()
    await demonstrate_tool_system()
    await demonstrate_conversation_management()
    await demonstrate_caching()
    await demonstrate_response_parsing()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())