"""
LLM Client Abstraction

Provides a unified interface for different LLM providers with support for
function calling, streaming, and comprehensive error handling.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
import aiohttp
from datetime import datetime, timezone

from .models import (
    LLMConfiguration, LLMProvider, LLMRequest, LLMResponse, 
    Message, MessageRole, FunctionCall
)
from .exceptions import (
    LLMAPIError, LLMTimeoutError, LLMRateLimitError, 
    LLMConfigurationError, LLMResponseParsingError
)


class LLMClient(ABC):
    """Abstract base class for LLM clients with provider abstraction."""
    
    def __init__(self, config: LLMConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @abstractmethod
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Check if the client supports function calling."""
        pass
    
    @abstractmethod
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        pass
    
    async def _make_request(
        self, 
        method: str, 
        url: str, 
        headers: Dict[str, str], 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        await self._ensure_session()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return response_data
                    elif response.status == 429:
                        retry_after = int(response.headers.get('retry-after', 60))
                        raise LLMRateLimitError(
                            f"Rate limit exceeded",
                            retry_after=retry_after,
                            provider=self.get_provider().value
                        )
                    elif response.status >= 400:
                        error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                        raise LLMAPIError(
                            f"API error: {error_msg}",
                            status_code=response.status,
                            api_endpoint=url,
                            provider=self.get_provider().value
                        )
                        
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries:
                    raise LLMTimeoutError(
                        f"Request timed out after {self.config.timeout} seconds",
                        timeout_seconds=self.config.timeout,
                        provider=self.get_provider().value
                    )
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries:
                    raise LLMAPIError(
                        f"Client error: {str(e)}",
                        provider=self.get_provider().value,
                        cause=e
                    )
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise LLMAPIError(
            f"Max retries ({self.config.max_retries}) exceeded",
            provider=self.get_provider().value
        )


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, config: LLMConfiguration):
        if config.provider != LLMProvider.OPENAI:
            raise LLMConfigurationError(
                "OpenAIClient requires provider to be OPENAI",
                config_key="provider",
                config_value=config.provider.value
            )
        
        if not config.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is required",
                config_key="api_key"
            )
        
        super().__init__(config)
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.OPENAI
    
    def supports_function_calling(self) -> bool:
        """OpenAI supports function calling."""
        return True
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to OpenAI format
        messages = []
        for msg in request.messages:
            openai_msg = {
                "role": msg.role.value,
                "content": msg.content
            }
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.parameters)
                        }
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            messages.append(openai_msg)
        
        data = {
            "model": request.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if request.tools and self.config.enable_function_calling:
            data["tools"] = request.tools
            if request.tool_choice:
                data["tool_choice"] = request.tool_choice
        
        try:
            response_data = await self._make_request(
                method="POST",
                url=f"{self.config.api_endpoint}/chat/completions",
                headers=headers,
                data=data
            )
            
            return self._parse_openai_response(response_data, request.request_id)
            
        except Exception as e:
            if isinstance(e, (LLMAPIError, LLMTimeoutError, LLMRateLimitError)):
                raise
            raise LLMAPIError(
                f"Unexpected error: {str(e)}",
                provider=self.get_provider().value,
                cause=e
            )
    
    def _parse_openai_response(self, response_data: Dict[str, Any], request_id: str) -> LLMResponse:
        """Parse OpenAI API response."""
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            
            content = message.get("content", "")
            function_calls = []
            
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if tool_call["type"] == "function":
                        func = tool_call["function"]
                        function_calls.append(FunctionCall(
                            name=func["name"],
                            arguments=json.loads(func["arguments"]),
                            call_id=tool_call["id"]
                        ))
            
            return LLMResponse(
                content=content,
                model_name=response_data["model"],
                usage=response_data["usage"],
                function_calls=function_calls,
                finish_reason=choice["finish_reason"],
                request_id=request_id,
                metadata={"raw_response": response_data}
            )
            
        except (KeyError, json.JSONDecodeError) as e:
            raise LLMResponseParsingError(
                f"Failed to parse OpenAI response: {str(e)}",
                response_content=str(response_data),
                expected_format="OpenAI chat completion format",
                provider=self.get_provider().value,
                cause=e
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response (basic implementation)."""
        # For now, return non-streaming response as single chunk
        response = await self.generate_response(request)
        yield response.content


class AnthropicClient(LLMClient):
    """Anthropic API client implementation."""
    
    def __init__(self, config: LLMConfiguration):
        if config.provider != LLMProvider.ANTHROPIC:
            raise LLMConfigurationError(
                "AnthropicClient requires provider to be ANTHROPIC",
                config_key="provider",
                config_value=config.provider.value
            )
        
        if not config.api_key:
            raise LLMConfigurationError(
                "Anthropic API key is required",
                config_key="api_key"
            )
        
        super().__init__(config)
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.ANTHROPIC
    
    def supports_function_calling(self) -> bool:
        """Anthropic supports tool calling."""
        return True
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages to Anthropic format
        messages = []
        system_message = None
        
        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        data = {
            "model": request.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        if system_message:
            data["system"] = system_message
        
        if request.tools and self.config.enable_function_calling:
            data["tools"] = request.tools
        
        try:
            response_data = await self._make_request(
                method="POST",
                url=f"{self.config.api_endpoint}/messages",
                headers=headers,
                data=data
            )
            
            return self._parse_anthropic_response(response_data, request.request_id)
            
        except Exception as e:
            if isinstance(e, (LLMAPIError, LLMTimeoutError, LLMRateLimitError)):
                raise
            raise LLMAPIError(
                f"Unexpected error: {str(e)}",
                provider=self.get_provider().value,
                cause=e
            )
    
    def _parse_anthropic_response(self, response_data: Dict[str, Any], request_id: str) -> LLMResponse:
        """Parse Anthropic API response."""
        try:
            content_blocks = response_data["content"]
            content = ""
            function_calls = []
            
            for block in content_blocks:
                if block["type"] == "text":
                    content += block["text"]
                elif block["type"] == "tool_use":
                    function_calls.append(FunctionCall(
                        name=block["name"],
                        arguments=block["input"],
                        call_id=block["id"]
                    ))
            
            return LLMResponse(
                content=content,
                model_name=response_data["model"],
                usage=response_data["usage"],
                function_calls=function_calls,
                finish_reason=response_data["stop_reason"],
                request_id=request_id,
                metadata={"raw_response": response_data}
            )
            
        except KeyError as e:
            raise LLMResponseParsingError(
                f"Failed to parse Anthropic response: {str(e)}",
                response_content=str(response_data),
                expected_format="Anthropic messages format",
                provider=self.get_provider().value,
                cause=e
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response (basic implementation)."""
        # For now, return non-streaming response as single chunk
        response = await self.generate_response(request)
        yield response.content


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and development."""
    
    def __init__(self, config: Optional[LLMConfiguration] = None):
        if config is None:
            config = LLMConfiguration(
                provider=LLMProvider.MOCK,
                api_endpoint="http://localhost:8000",
                model_name="mock-model"
            )
        super().__init__(config)
        self.mock_responses: List[str] = [
            "This is a mock response from the LLM.",
            "Another mock response for testing purposes.",
            "Mock LLM is working correctly."
        ]
        self.response_index = 0
    
    def get_provider(self) -> LLMProvider:
        """Get the provider type."""
        return LLMProvider.MOCK
    
    def supports_function_calling(self) -> bool:
        """Mock client supports function calling."""
        return True
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response."""
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        content = self.mock_responses[self.response_index % len(self.mock_responses)]
        self.response_index += 1
        
        # Mock function calls if tools are provided
        function_calls = []
        if request.tools and len(request.tools) > 0:
            tool = request.tools[0]
            if tool.get("type") == "function":
                func_name = tool["function"]["name"]
                function_calls.append(FunctionCall(
                    name=func_name,
                    arguments={"mock": "parameters"}
                ))
        
        return LLMResponse(
            content=content,
            model_name=request.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            function_calls=function_calls,
            finish_reason="stop",
            request_id=request.request_id,
            metadata={"mock": True}
        )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        content = self.mock_responses[self.response_index % len(self.mock_responses)]
        self.response_index += 1
        
        # Simulate streaming by yielding words
        words = content.split()
        for word in words:
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield word + " "
    
    def set_mock_responses(self, responses: List[str]) -> None:
        """Set custom mock responses."""
        self.mock_responses = responses
        self.response_index = 0