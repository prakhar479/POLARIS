"""
LLM Integration Exception Hierarchy

Extends the POLARIS exception hierarchy with LLM-specific error types
for comprehensive error handling and diagnostics.
"""

from typing import Dict, Any, Optional, List
from ..exceptions import PolarisException


class LLMIntegrationError(PolarisException):
    """Base exception for LLM integration issues."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if provider:
            context['provider'] = provider
        if model_name:
            context['model_name'] = model_name
        
        if 'error_code' not in kwargs:
            kwargs['error_code'] = "LLM_INTEGRATION_ERROR"
        super().__init__(
            message=message,
            context=context,
            **kwargs
        )


class LLMAPIError(LLMIntegrationError):
    """LLM API communication errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        api_endpoint: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if status_code:
            context['status_code'] = status_code
        if api_endpoint:
            context['api_endpoint'] = api_endpoint
        if request_id:
            context['request_id'] = request_id
        
        super().__init__(
            message=message,
            error_code="LLM_API_ERROR",
            context=context,
            **kwargs
        )


class LLMResponseParsingError(LLMIntegrationError):
    """LLM response parsing and validation errors."""
    
    def __init__(
        self,
        message: str,
        response_content: Optional[str] = None,
        expected_format: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if response_content:
            context['response_content'] = response_content[:500]  # Truncate for logging
        if expected_format:
            context['expected_format'] = expected_format
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message=message,
            error_code="LLM_RESPONSE_PARSING_ERROR",
            context=context,
            **kwargs
        )


class LLMToolError(LLMIntegrationError):
    """LLM tool execution and management errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if tool_name:
            context['tool_name'] = tool_name
        if tool_call_id:
            context['tool_call_id'] = tool_call_id
        if tool_parameters:
            context['tool_parameters'] = tool_parameters
        
        super().__init__(
            message=message,
            error_code="LLM_TOOL_ERROR",
            context=context,
            **kwargs
        )


class LLMConfigurationError(LLMIntegrationError):
    """LLM configuration and setup errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value:
            context['config_value'] = config_value
        
        super().__init__(
            message=message,
            error_code="LLM_CONFIGURATION_ERROR",
            context=context,
            **kwargs
        )


class LLMTimeoutError(LLMAPIError):
    """LLM API timeout errors."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        
        super().__init__(
            message=message,
            error_code="LLM_TIMEOUT_ERROR",
            context=context,
            **kwargs
        )


class LLMRateLimitError(LLMAPIError):
    """LLM API rate limit errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if retry_after:
            context['retry_after'] = retry_after
        
        super().__init__(
            message=message,
            error_code="LLM_RATE_LIMIT_ERROR",
            context=context,
            **kwargs
        )