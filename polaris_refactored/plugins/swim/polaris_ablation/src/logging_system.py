"""
Comprehensive Logging System for SWIM POLARIS Adaptation System

Provides structured logging with correlation IDs, component-specific loggers,
log aggregation, filtering, and integration with observability systems.
"""

import logging
import logging.handlers
import json
import uuid
import threading
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import contextvars
from contextlib import contextmanager


class LogLevel(Enum):
    """Log levels with numeric values."""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class LogFormat(Enum):
    """Supported log formats."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogContext:
    """Context information for log entries."""
    correlation_id: str
    component: str
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record):
        # Get correlation ID from context
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id
        
        # Get additional context
        context = get_log_context()
        if context:
            record.component = context.component
            record.operation = context.operation or ""
            record.trace_id = context.trace_id or ""
            record.span_id = context.span_id or ""
        else:
            record.component = getattr(record, 'component', 'unknown')
            record.operation = ""
            record.trace_id = ""
            record.span_id = ""
        
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', ''),
            'component': getattr(record, 'component', ''),
            'operation': getattr(record, 'operation', ''),
            'trace_id': getattr(record, 'trace_id', ''),
            'span_id': getattr(record, 'span_id', ''),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for human-readable logs with context."""
    
    def __init__(self):
        super().__init__()
    
    def format(self, record):
        # Base format
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        correlation_id = getattr(record, 'correlation_id', '')[:8]  # Short correlation ID
        component = getattr(record, 'component', 'unknown')
        operation = getattr(record, 'operation', '')
        
        # Build the log line
        parts = [
            f"{timestamp}",
            f"[{record.levelname:8}]",
            f"[{correlation_id}]" if correlation_id else "[--------]",
            f"[{component:15}]",
            f"[{operation:20}]" if operation else "[" + " " * 20 + "]",
            f"{record.name}:{record.lineno}",
            f"- {record.getMessage()}"
        ]
        
        log_line = " ".join(parts)
        
        # Add exception information if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


# Context variables for correlation tracking
_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')
_log_context_var: contextvars.ContextVar[Optional[LogContext]] = contextvars.ContextVar('log_context', default=None)


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    return _correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    _correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_log_context() -> Optional[LogContext]:
    """Get current log context."""
    return _log_context_var.get()


def set_log_context(context: LogContext) -> None:
    """Set log context."""
    _log_context_var.set(context)


@contextmanager
def log_context(component: str, 
                operation: Optional[str] = None,
                correlation_id: Optional[str] = None,
                **kwargs):
    """Context manager for setting log context.
    
    Args:
        component: Component name
        operation: Operation name
        correlation_id: Correlation ID (generates new if None)
        **kwargs: Additional context metadata
    """
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    context = LogContext(
        correlation_id=correlation_id,
        component=component,
        operation=operation,
        metadata=kwargs
    )
    
    # Set context variables
    old_correlation_id = get_correlation_id()
    old_context = get_log_context()
    
    set_correlation_id(correlation_id)
    set_log_context(context)
    
    try:
        yield context
    finally:
        # Restore previous context
        set_correlation_id(old_correlation_id)
        set_log_context(old_context)


class ComponentLogger:
    """Component-specific logger with context management."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(component_name)
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log message with component context."""
        extra = kwargs.pop('extra', {})
        extra['component'] = self.component_name
        
        # Add operation context if available
        context = get_log_context()
        if context and context.operation:
            extra['operation'] = context.operation
        
        self.logger.log(level, message, *args, extra=extra, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, *args, **kwargs)


class LogAggregator:
    """Aggregates and filters log entries for analysis."""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def add_entry(self, record: logging.LogRecord):
        """Add log entry to aggregator."""
        entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', ''),
            'component': getattr(record, 'component', ''),
            'operation': getattr(record, 'operation', ''),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        with self.lock:
            self.entries.append(entry)
            if len(self.entries) > self.max_entries:
                self.entries.pop(0)  # Remove oldest entry
    
    def get_entries(self, 
                   component: Optional[str] = None,
                   level: Optional[str] = None,
                   correlation_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get filtered log entries.
        
        Args:
            component: Filter by component
            level: Filter by log level
            correlation_id: Filter by correlation ID
            limit: Maximum number of entries to return
            
        Returns:
            List of filtered log entries
        """
        with self.lock:
            filtered_entries = self.entries.copy()
        
        # Apply filters
        if component:
            filtered_entries = [e for e in filtered_entries if e['component'] == component]
        
        if level:
            filtered_entries = [e for e in filtered_entries if e['level'] == level]
        
        if correlation_id:
            filtered_entries = [e for e in filtered_entries if e['correlation_id'] == correlation_id]
        
        # Sort by timestamp (newest first)
        filtered_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        if limit:
            filtered_entries = filtered_entries[:limit]
        
        return filtered_entries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.lock:
            entries = self.entries.copy()
        
        if not entries:
            return {}
        
        # Count by level
        level_counts = {}
        component_counts = {}
        
        for entry in entries:
            level = entry['level']
            component = entry['component']
            
            level_counts[level] = level_counts.get(level, 0) + 1
            component_counts[component] = component_counts.get(component, 0) + 1
        
        return {
            'total_entries': len(entries),
            'level_distribution': level_counts,
            'component_distribution': component_counts,
            'oldest_entry': min(entries, key=lambda x: x['timestamp'])['timestamp'].isoformat(),
            'newest_entry': max(entries, key=lambda x: x['timestamp'])['timestamp'].isoformat()
        }


class LogAggregatorHandler(logging.Handler):
    """Handler that sends log records to aggregator."""
    
    def __init__(self, aggregator: LogAggregator):
        super().__init__()
        self.aggregator = aggregator
    
    def emit(self, record):
        try:
            self.aggregator.add_entry(record)
        except Exception:
            self.handleError(record)


class SwimPolarisLoggingSystem:
    """
    Comprehensive logging system for SWIM POLARIS adaptation system.
    
    Provides structured logging, correlation tracking, component-specific loggers,
    and integration with observability systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the logging system.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self.component_loggers: Dict[str, ComponentLogger] = {}
        self.aggregator = LogAggregator(max_entries=config.get('aggregator_max_entries', 10000))
        self.root_logger = logging.getLogger()
        
        # Configure logging
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure the logging system."""
        # Clear existing handlers
        self.root_logger.handlers.clear()
        
        # Set root logger level
        level_str = self.config.get('level', 'INFO')
        level = getattr(logging, level_str.upper(), logging.INFO)
        self.root_logger.setLevel(level)
        
        # Add correlation ID filter to all handlers
        correlation_filter = CorrelationIdFilter()
        
        # Configure handlers
        handlers_config = self.config.get('handlers', [])
        for handler_config in handlers_config:
            handler = self._create_handler(handler_config)
            if handler:
                handler.addFilter(correlation_filter)
                self.root_logger.addHandler(handler)
        
        # Add aggregator handler
        aggregator_handler = LogAggregatorHandler(self.aggregator)
        aggregator_handler.addFilter(correlation_filter)
        self.root_logger.addHandler(aggregator_handler)
    
    def _create_handler(self, handler_config: Dict[str, Any]) -> Optional[logging.Handler]:
        """Create a log handler from configuration."""
        handler_type = handler_config.get('type', 'console')
        
        if handler_type == 'console':
            handler = logging.StreamHandler()
        
        elif handler_type == 'file':
            log_path = Path(handler_config['path'])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            max_size = handler_config.get('max_size_mb', 100) * 1024 * 1024
            backup_count = handler_config.get('backup_count', 5)
            
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_size,
                backupCount=backup_count
            )
        
        elif handler_type == 'syslog':
            facility = handler_config.get('facility', 'local0')
            handler = logging.handlers.SysLogHandler(facility=facility)
        
        else:
            return None
        
        # Set handler level
        handler_level_str = handler_config.get('level', 'INFO')
        handler_level = getattr(logging, handler_level_str.upper(), logging.INFO)
        handler.setLevel(handler_level)
        
        # Set formatter
        format_type = handler_config.get('format', self.config.get('format', 'simple'))
        formatter = self._create_formatter(format_type, handler_config)
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_formatter(self, format_type: str, handler_config: Dict[str, Any]) -> logging.Formatter:
        """Create a log formatter."""
        if format_type == 'json':
            return JSONFormatter()
        
        elif format_type == 'structured':
            return StructuredFormatter()
        
        elif format_type == 'detailed':
            format_str = handler_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s')
            return logging.Formatter(format_str)
        
        else:  # simple
            return logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    
    def get_component_logger(self, component_name: str) -> ComponentLogger:
        """Get or create a component-specific logger.
        
        Args:
            component_name: Name of the component
            
        Returns:
            ComponentLogger instance
        """
        if component_name not in self.component_loggers:
            logger = logging.getLogger(f"swim_polaris.{component_name}")
            self.component_loggers[component_name] = ComponentLogger(component_name, logger)
        
        return self.component_loggers[component_name]
    
    def get_aggregated_logs(self, **filters) -> List[Dict[str, Any]]:
        """Get aggregated log entries with filters."""
        return self.aggregator.get_entries(**filters)
    
    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics."""
        return self.aggregator.get_statistics()
    
    def export_logs(self, output_path: str, **filters) -> None:
        """Export logs to file.
        
        Args:
            output_path: Path to output file
            **filters: Filters to apply
        """
        entries = self.get_aggregated_logs(**filters)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for entry in entries:
                # Convert datetime to string for JSON serialization
                entry_copy = entry.copy()
                entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
                f.write(json.dumps(entry_copy) + '\n')
    
    def cleanup(self):
        """Clean up logging resources."""
        # Close all handlers
        for handler in self.root_logger.handlers[:]:
            handler.close()
            self.root_logger.removeHandler(handler)
        
        # Clear component loggers
        self.component_loggers.clear()


# Convenience functions for common logging patterns

def get_swim_logger(component_name: str) -> ComponentLogger:
    """Get a SWIM POLARIS component logger.
    
    Args:
        component_name: Name of the component
        
    Returns:
        ComponentLogger instance
    """
    # This assumes a global logging system instance
    # In practice, this would be injected or retrieved from a registry
    return ComponentLogger(component_name)


@contextmanager
def operation_context(operation_name: str, component: str = "unknown"):
    """Context manager for operation logging.
    
    Args:
        operation_name: Name of the operation
        component: Component performing the operation
    """
    with log_context(component=component, operation=operation_name) as ctx:
        logger = get_swim_logger(component)
        logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield ctx
            logger.info(f"Completed operation: {operation_name}")
        except Exception as e:
            logger.error(f"Failed operation: {operation_name}", exc_info=True)
            raise


async def async_operation_context(operation_name: str, component: str = "unknown"):
    """Async context manager for operation logging."""
    with log_context(component=component, operation=operation_name) as ctx:
        logger = get_swim_logger(component)
        logger.info(f"Starting async operation: {operation_name}")
        
        try:
            yield ctx
            logger.info(f"Completed async operation: {operation_name}")
        except Exception as e:
            logger.error(f"Failed async operation: {operation_name}", exc_info=True)
            raise