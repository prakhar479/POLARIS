"""
Structured Logging System for POLARIS

Provides structured JSON logging with correlation IDs, adaptation context management,
and configurable formatters and handlers for different output destinations.
"""

import json
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union
from contextvars import ContextVar

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
adaptation_id_var: ContextVar[Optional[str]] = ContextVar('adaptation_id', default=None)
system_id_var: ContextVar[Optional[str]] = ContextVar('system_id', default=None)


class LogLevel(Enum):
    """Log levels for POLARIS logging system"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormatter(ABC):
    """Abstract base class for log formatters"""
    
    @abstractmethod
    def format(self, record: Dict[str, Any]) -> str:
        """Format a log record into a string"""
        pass


class JSONLogFormatter(LogFormatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as JSON string"""
        return json.dumps(record, default=str, ensure_ascii=False)


class HumanReadableFormatter(LogFormatter):
    """Human-readable formatter for development/debugging"""
    
    def format(self, record: Dict[str, Any]) -> str:
        """Format log record as human-readable string"""
        timestamp = record.get('timestamp', '')
        level = record.get('level', '')
        message = record.get('message', '')
        correlation_id = record.get('correlation_id', '')
        
        base_msg = f"[{timestamp}] {level}: {message}"
        
        if correlation_id:
            base_msg += f" [correlation_id={correlation_id}]"
            
        if record.get('extra'):
            extra_str = ', '.join(f"{k}={v}" for k, v in record['extra'].items())
            base_msg += f" [{extra_str}]"
            
        return base_msg


class LogHandler(ABC):
    """Abstract base class for log handlers"""
    
    def __init__(self, formatter: LogFormatter):
        self.formatter = formatter
    
    @abstractmethod
    def emit(self, record: Dict[str, Any]) -> None:
        """Emit a log record"""
        pass


class ConsoleLogHandler(LogHandler):
    """Console log handler that writes to stdout/stderr"""
    
    def __init__(self, formatter: LogFormatter, stream: TextIO = sys.stdout):
        super().__init__(formatter)
        self.stream = stream
    
    def emit(self, record: Dict[str, Any]) -> None:
        """Write log record to console"""
        formatted_message = self.formatter.format(record)
        self.stream.write(formatted_message + '\n')
        self.stream.flush()


class FileLogHandler(LogHandler):
    """File log handler that writes to a file"""
    
    def __init__(self, formatter: LogFormatter, file_path: Union[str, Path]):
        super().__init__(formatter)
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record: Dict[str, Any]) -> None:
        """Write log record to file"""
        formatted_message = self.formatter.format(record)
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')


class PolarisLogger:
    """
    Structured logger for POLARIS with correlation ID support and adaptation context management.
    
    Features:
    - Structured JSON logging
    - Correlation ID tracking across components
    - Adaptation context management
    - Multiple output handlers (console, file, etc.)
    - Configurable log levels and formatters
    """
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.handlers: list[LogHandler] = []
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler"""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def set_level(self, level: LogLevel) -> None:
        """Set the logging level"""
        self.level = level
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on current level"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        return level_order[level] >= level_order[self.level]
    
    def _create_log_record(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a structured log record"""
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level.value,
            'logger': self.name,
            'message': message,
            'correlation_id': correlation_id_var.get(),
            'adaptation_id': adaptation_id_var.get(),
            'system_id': system_id_var.get(),
        }
        
        if extra:
            record['extra'] = extra
            
        # Remove None values to keep logs clean
        return {k: v for k, v in record.items() if v is not None}
    
    def _log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method"""
        if not self._should_log(level):
            return
            
        record = self._create_log_record(level, message, extra)
        
        for handler in self.handlers:
            try:
                handler.emit(record)
            except Exception as e:
                # Fallback to stderr if handler fails
                sys.stderr.write(f"Logging handler failed: {e}\n")
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        self._log(LogLevel.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: Optional[Exception] = None) -> None:
        """Log error message with optional exception info"""
        if exc_info:
            if extra is None:
                extra = {}
            extra['exception'] = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'module': type(exc_info).__module__
            }
        self._log(LogLevel.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: Optional[Exception] = None) -> None:
        """Log critical message with optional exception info"""
        if exc_info:
            if extra is None:
                extra = {}
            extra['exception'] = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'module': type(exc_info).__module__
            }
        self._log(LogLevel.CRITICAL, message, extra)
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID tracking"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        token = correlation_id_var.set(correlation_id)
        try:
            yield correlation_id
        finally:
            correlation_id_var.reset(token)
    
    @contextmanager
    def adaptation_context(self, adaptation_id: str, system_id: Optional[str] = None):
        """Context manager for adaptation context tracking"""
        adaptation_token = adaptation_id_var.set(adaptation_id)
        system_token = None
        
        if system_id:
            system_token = system_id_var.set(system_id)
        
        try:
            yield adaptation_id
        finally:
            adaptation_id_var.reset(adaptation_token)
            if system_token:
                system_id_var.reset(system_token)
    
    @contextmanager
    def system_context(self, system_id: str):
        """Context manager for system context tracking"""
        token = system_id_var.set(system_id)
        try:
            yield system_id
        finally:
            system_id_var.reset(token)


# Global logger registry
_loggers: Dict[str, PolarisLogger] = {}


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> PolarisLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = PolarisLogger(name, level)
    return _loggers[name]


def configure_default_logging(
    level: LogLevel = LogLevel.INFO,
    use_json: bool = True,
    log_file: Optional[Union[str, Path]] = None
) -> None:
    """Configure default logging for POLARIS"""
    
    # Choose formatter
    formatter = JSONLogFormatter() if use_json else HumanReadableFormatter()
    
    # Configure root logger
    root_logger = get_logger("polaris")
    root_logger.set_level(level)
    
    # Add console handler
    console_handler = ConsoleLogHandler(formatter)
    root_logger.add_handler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = FileLogHandler(formatter, log_file)
        root_logger.add_handler(file_handler)


# Convenience function to get current correlation ID
def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context"""
    return correlation_id_var.get()


def get_adaptation_id() -> Optional[str]:
    """Get the current adaptation ID from context"""
    return adaptation_id_var.get()


def get_system_id() -> Optional[str]:
    """Get the current system ID from context"""
    return system_id_var.get()