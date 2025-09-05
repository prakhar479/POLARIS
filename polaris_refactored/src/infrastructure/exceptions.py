"""
Structured Exception Hierarchy

Provides a comprehensive exception hierarchy with contextual information
for better error handling and diagnostics.
"""

from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime


class PolarisException(Exception):
    """
    Base exception class for all POLARIS-specific exceptions.
    
    Provides structured error information including error codes,
    context data, and correlation IDs for tracing.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(PolarisException):
    """Raised when configuration-related errors occur."""
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to remove from kwargs
        if config_path:
            context['config_path'] = config_path
        if validation_errors:
            context['validation_errors'] = validation_errors
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs
        )


class ConnectorError(PolarisException):
    """Raised when managed system connector errors occur."""
    
    def __init__(
        self,
        message: str,
        system_id: Optional[str] = None,
        connector_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})  # Use pop to remove from kwargs
        if system_id:
            context['system_id'] = system_id
        if connector_type:
            context['connector_type'] = connector_type
        
        super().__init__(
            message=message,
            error_code="CONNECTOR_ERROR",
            context=context,
            **kwargs
        )


class AdaptationError(PolarisException):
    """Raised when adaptation execution errors occur."""
    
    def __init__(
        self,
        message: str,
        action_id: Optional[str] = None,
        system_id: Optional[str] = None,
        action_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if action_id:
            context['action_id'] = action_id
        if system_id:
            context['system_id'] = system_id
        if action_type:
            context['action_type'] = action_type
        
        super().__init__(
            message=message,
            error_code="ADAPTATION_ERROR",
            context=context,
            **kwargs
        )


class WorldModelError(PolarisException):
    """Raised when world model errors occur."""
    
    def __init__(
        self,
        message: str,
        model_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if model_type:
            context['model_type'] = model_type
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="WORLD_MODEL_ERROR",
            context=context,
            **kwargs
        )


class EventBusError(PolarisException):
    """Raised when event bus errors occur."""
    
    def __init__(
        self,
        message: str,
        event_type: Optional[str] = None,
        topic: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if event_type:
            context['event_type'] = event_type
        if topic:
            context['topic'] = topic
        
        super().__init__(
            message=message,
            error_code="EVENT_BUS_ERROR",
            context=context,
            **kwargs
        )


class DataStoreError(PolarisException):
    """Raised when data storage errors occur."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        entity_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if operation:
            context['operation'] = operation
        if entity_type:
            context['entity_type'] = entity_type
        
        super().__init__(
            message=message,
            error_code="DATA_STORE_ERROR",
            context=context,
            **kwargs
        )


class PluginError(PolarisException):
    """Raised when plugin-related errors occur."""
    
    def __init__(
        self,
        message: str,
        plugin_id: Optional[str] = None,
        plugin_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if plugin_id:
            context['plugin_id'] = plugin_id
        if plugin_type:
            context['plugin_type'] = plugin_type
        
        super().__init__(
            message=message,
            error_code="PLUGIN_ERROR",
            context=context,
            **kwargs
        )