"""
Test suite for LLM integration infrastructure.

Tests all components of the LLM integration system including clients,
prompt management, response parsing, tool registry, conversation management,
and caching.
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from polaris_refactored.src.infrastructure.llm import (
    LLMClient, MockLLMClient, OpenAIClient, AnthropicClient,
    PromptManager, ResponseParser, ToolRegistry, ConversationManager,
    LLMCache, BaseTool, ToolSchema, ToolResult,
    LLMConfiguration, LLMProvider, LLMRequest, LLMResponse,
    Message, MessageRole, FunctionCall, ToolCall, AgenticConversation,
    LLMIntegrationError, LLMAPIError, LLMResponseParsingError, LLMToolError
)


class TestLLMConfiguration:
    """Test LLM configuration validation."""
    
    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = LLMConfiguration(
            provider=LLMProvider.OPENAI,
            api_endpoint="https://api.openai.com/v1",
            api_key="test-key",
            model_name="gpt-4"
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.api_endpoint == "https://api.openai.com/v1"
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 1000  # default
        assert config.temperature == 0.1  # default
    
    def test_invalid_endpoint(self):
        """Test invalid API endpoint validation."""
        with pytest.raises(ValueError, match="API endpoint must be a valid HTTP/HTTPS URL"):
            LLMConfiguration(
                provider=LLMProvider.OPENAI,
                api_endpoint="invalid-url",
                model_name="gpt-4"
            )


class TestMockLLMClient:
    """Test mock LLM client implementation."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return MockLLMClient()
    
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_client):
        """Test mock response generation."""
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model_name="mock-model"
        )
        
        response = await mock_client.generate_response(request)
        
        assert isinstance(response, LLMResponse)
        assert response.content in mock_client.mock_responses
        assert response.model_name == "mock-model"
        assert response.usage["total_tokens"] == 30
    
    @pytest.mark.asyncio
    async def test_function_calling_support(self, mock_client):
        """Test function calling with mock client."""
        tools = [{
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
        
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Use the test function")],
            model_name="mock-model",
            tools=tools
        )
        
        response = await mock_client.generate_response(request)
        
        assert len(response.function_calls) == 1
        assert response.function_calls[0].name == "test_function"
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, mock_client):
        """Test streaming response generation."""
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model_name="mock-model"
        )
        
        chunks = []
        async for chunk in mock_client.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks).strip()
        assert full_response in mock_client.mock_responses


class TestPromptManager:
    """Test prompt management system."""
    
    @pytest.fixture
    def prompt_manager(self):
        """Create prompt manager."""
        return PromptManager()
    
    def test_default_templates_loaded(self, prompt_manager):
        """Test that default templates are loaded."""
        templates = prompt_manager.list_templates()
        
        assert "system_reasoning" in templates
        assert "world_model_update" in templates
        assert "adaptation_impact_simulation" in templates
        assert "agentic_reasoning" in templates
    
    def test_render_template(self, prompt_manager):
        """Test template rendering."""
        rendered = prompt_manager.render_template(
            "system_reasoning",
            system_id="test-system",
            current_metrics={"cpu": 80},
            health_status="warning",
            available_actions=["scale_up", "scale_down"]
        )
        
        assert "test-system" in rendered
        assert "cpu" in rendered
        assert "warning" in rendered
    
    def test_missing_template_variable(self, prompt_manager):
        """Test error handling for missing template variables."""
        with pytest.raises(Exception):  # Should raise LLMConfigurationError
            prompt_manager.render_template("system_reasoning", system_id="test")
    
    def test_create_messages(self, prompt_manager):
        """Test message creation from templates."""
        system_msg = prompt_manager.create_system_message(
            "system_reasoning",
            system_id="test",
            current_metrics={},
            health_status="healthy",
            available_actions=[]
        )
        
        assert system_msg.role == MessageRole.SYSTEM
        assert "test" in system_msg.content
        assert system_msg.metadata["template"] == "system_reasoning"


class TestResponseParser:
    """Test response parsing system."""
    
    @pytest.fixture
    def response_parser(self):
        """Create response parser."""
        return ResponseParser()
    
    def test_json_extraction(self, response_parser):
        """Test JSON extraction from text."""
        text = 'Here is the result: {"analysis": "test", "recommendations": []}'
        
        json_data = response_parser.function_parser.extract_json_from_text(text)
        
        assert json_data is not None
        assert json_data["analysis"] == "test"
        assert json_data["recommendations"] == []
    
    def test_json_in_code_block(self, response_parser):
        """Test JSON extraction from code blocks."""
        text = '''
        Here is the analysis:
        ```json
        {"analysis": "test", "recommendations": []}
        ```
        '''
        
        json_data = response_parser.function_parser.extract_json_from_text(text)
        
        assert json_data is not None
        assert json_data["analysis"] == "test"
    
    def test_parse_reasoning_response(self, response_parser):
        """Test parsing reasoning response."""
        response_content = '''
        {
            "analysis": "System is under high load",
            "recommendations": [
                {
                    "action_type": "scale_up",
                    "parameters": {"instances": 2},
                    "rationale": "Increase capacity",
                    "confidence": 0.85,
                    "risk_level": "low"
                }
            ],
            "reasoning_steps": ["step1", "step2"]
        }
        '''
        
        response = LLMResponse(
            content=response_content,
            model_name="test-model",
            usage={"total_tokens": 100}
        )
        
        parsed = response_parser.parse_reasoning_response(response)
        
        assert parsed["validation_passed"] is True
        assert parsed["parsed_json"]["analysis"] == "System is under high load"
        assert len(parsed["parsed_json"]["recommendations"]) == 1


class TestToolSystem:
    """Test tool registry and execution system."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create tool registry with default tools."""
        from polaris_refactored.src.infrastructure.llm.tool_registry import create_default_registry
        return create_default_registry()
    
    def test_tool_registration(self, tool_registry):
        """Test tool registration."""
        tools = tool_registry.list_tools()
        
        assert "echo" in tools
        assert "calculator" in tools
    
    @pytest.mark.asyncio
    async def test_echo_tool_execution(self, tool_registry):
        """Test echo tool execution."""
        result = await tool_registry.execute_tool(
            "echo",
            {"message": "Hello, World!"}
        )
        
        assert result.success is True
        assert result.result["echoed_message"] == "Hello, World!"
        assert "timestamp" in result.result
    
    @pytest.mark.asyncio
    async def test_calculator_tool_execution(self, tool_registry):
        """Test calculator tool execution."""
        result = await tool_registry.execute_tool(
            "calculator",
            {"operation": "add", "a": 5, "b": 3}
        )
        
        assert result.success is True
        assert result.result["result"] == 8
        assert result.result["operation"] == "add"
    
    @pytest.mark.asyncio
    async def test_calculator_division_by_zero(self, tool_registry):
        """Test calculator error handling."""
        result = await tool_registry.execute_tool(
            "calculator",
            {"operation": "divide", "a": 5, "b": 0}
        )
        
        assert result.success is False
        assert "Division by zero" in result.error_message
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self, tool_registry):
        """Test execution of unknown tool."""
        result = await tool_registry.execute_tool(
            "unknown_tool",
            {}
        )
        
        assert result.success is False
        assert "Tool not found" in result.error_message
    
    def test_tool_schemas(self, tool_registry):
        """Test tool schema generation."""
        schemas = tool_registry.get_tool_schemas("openai")
        
        assert len(schemas) >= 2
        
        # Check echo tool schema
        echo_schema = next(s for s in schemas if s["function"]["name"] == "echo")
        assert echo_schema["type"] == "function"
        assert "message" in echo_schema["function"]["parameters"]["properties"]


class TestConversationManager:
    """Test conversation management system."""
    
    @pytest.fixture
    def conversation_manager(self):
        """Create conversation manager with mocks."""
        mock_client = MockLLMClient()
        from polaris_refactored.src.infrastructure.llm.tool_registry import create_default_registry
        tool_registry = create_default_registry()
        response_parser = ResponseParser()
        
        return ConversationManager(mock_client, tool_registry, response_parser)
    
    def test_create_conversation(self, conversation_manager):
        """Test conversation creation."""
        conv = conversation_manager.create_conversation()
        
        assert conv.conversation_id is not None
        assert len(conv.messages) == 0
        assert conv.max_messages == 100
    
    @pytest.mark.asyncio
    async def test_send_message(self, conversation_manager):
        """Test sending message to conversation."""
        conv = conversation_manager.create_conversation()
        
        response_msg = await conversation_manager.send_message(
            conv.conversation_id,
            "Hello, how are you?",
            generate_response=True
        )
        
        assert response_msg is not None
        assert response_msg.role == MessageRole.ASSISTANT
        assert len(conv.messages) == 2  # User message + assistant response
    
    def test_create_agentic_conversation(self, conversation_manager):
        """Test agentic conversation creation."""
        conv = conversation_manager.create_agentic_conversation(
            objective="Test objective",
            available_tools=["echo", "calculator"],
            max_iterations=5
        )
        
        assert conv.conversation_id is not None
        assert conv.context["objective"] == "Test objective"
        assert conv.context["available_tools"] == ["echo", "calculator"]
        assert conv.max_iterations == 5
    
    @pytest.mark.asyncio
    async def test_agentic_conversation_execution(self, conversation_manager):
        """Test running agentic conversation."""
        conv = conversation_manager.create_agentic_conversation(
            objective="Calculate 5 + 3",
            available_tools=["calculator"],
            max_iterations=3
        )
        
        # Mock the LLM client to return tool usage
        conversation_manager.llm_client.set_mock_responses([
            '{"action": "use_tool", "tool_name": "calculator", "parameters": {"operation": "add", "a": 5, "b": 3}}',
            '{"action": "final_answer", "result": {"answer": 8}, "reasoning": "Used calculator to add 5 + 3", "confidence": 1.0}'
        ])
        
        result_conv = await conversation_manager.run_agentic_conversation(
            conv.conversation_id,
            "Please calculate 5 + 3"
        )
        
        assert result_conv.is_complete
        assert result_conv.final_result is not None


class TestLLMCache:
    """Test LLM caching system."""
    
    @pytest.fixture
    def llm_cache(self):
        """Create LLM cache."""
        return LLMCache(max_size=10, default_ttl=60)
    
    def test_cache_llm_response(self, llm_cache):
        """Test caching LLM response."""
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model_name="test-model"
        )
        
        response = LLMResponse(
            content="Hi there!",
            model_name="test-model",
            usage={"total_tokens": 10}
        )
        
        # Cache the response
        llm_cache.cache_llm_response(request, response)
        
        # Retrieve from cache
        cached_response = llm_cache.get_llm_response(request)
        
        assert cached_response is not None
        assert cached_response.content == "Hi there!"
    
    def test_cache_miss(self, llm_cache):
        """Test cache miss."""
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hello")],
            model_name="test-model"
        )
        
        cached_response = llm_cache.get_llm_response(request)
        
        assert cached_response is None
    
    def test_cache_tool_result(self, llm_cache):
        """Test caching tool result."""
        result = ToolResult(
            call_id="test-call",
            tool_name="echo",
            success=True,
            result={"message": "Hello"}
        )
        
        # Cache the result
        llm_cache.cache_tool_result("echo", {"message": "Hello"}, result)
        
        # Retrieve from cache
        cached_result = llm_cache.get_tool_result("echo", {"message": "Hello"})
        
        assert cached_result is not None
        assert cached_result.result["message"] == "Hello"
    
    def test_cache_statistics(self, llm_cache):
        """Test cache statistics."""
        stats = llm_cache.get_statistics()
        
        assert "llm_cache" in stats
        assert "tool_cache" in stats
        assert "total_size" in stats
        assert stats["llm_cache"]["size"] == 0
        assert stats["tool_cache"]["size"] == 0


class TestIntegration:
    """Integration tests for the complete LLM system."""
    
    @pytest.fixture
    def llm_system(self):
        """Create complete LLM system."""
        config = LLMConfiguration(
            provider=LLMProvider.MOCK,
            api_endpoint="http://localhost:8000",
            model_name="mock-model"
        )
        
        client = MockLLMClient(config)
        prompt_manager = PromptManager()
        response_parser = ResponseParser()
        from polaris_refactored.src.infrastructure.llm.tool_registry import create_default_registry
        tool_registry = create_default_registry()
        conversation_manager = ConversationManager(client, tool_registry, response_parser)
        cache = LLMCache()
        
        return {
            "client": client,
            "prompt_manager": prompt_manager,
            "response_parser": response_parser,
            "tool_registry": tool_registry,
            "conversation_manager": conversation_manager,
            "cache": cache
        }
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_flow(self, llm_system):
        """Test complete reasoning flow with all components."""
        # Create reasoning prompt
        prompt = llm_system["prompt_manager"].render_template(
            "system_reasoning",
            system_id="test-system",
            current_metrics={"cpu": 85, "memory": 70},
            health_status="warning",
            available_actions=["scale_up", "restart"]
        )
        
        # Create LLM request
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content=prompt)],
            model_name="mock-model"
        )
        
        # Generate response
        response = await llm_system["client"].generate_response(request)
        
        # Parse response
        parsed = llm_system["response_parser"].parse_response(response)
        
        assert response.content is not None
        assert parsed["content"] == response.content
    
    @pytest.mark.asyncio
    async def test_agentic_workflow_with_tools(self, llm_system):
        """Test complete agentic workflow with tool usage."""
        # Set up mock responses for agentic flow
        llm_system["client"].set_mock_responses([
            '{"action": "use_tool", "tool_name": "calculator", "parameters": {"operation": "multiply", "a": 6, "b": 7}}',
            '{"action": "final_answer", "result": {"calculation": 42}, "reasoning": "Calculated 6 * 7 = 42", "confidence": 1.0}'
        ])
        
        # Create agentic conversation
        conv = llm_system["conversation_manager"].create_agentic_conversation(
            objective="Calculate 6 * 7",
            available_tools=["calculator"],
            max_iterations=5
        )
        
        # Run conversation
        result = await llm_system["conversation_manager"].run_agentic_conversation(
            conv.conversation_id,
            "Please calculate 6 multiplied by 7"
        )
        
        assert result.is_complete
        assert result.final_result is not None
        assert len(result.messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])