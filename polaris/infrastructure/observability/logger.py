"""Logger implementations for Polaris."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from polaris.abstractions.observability import Logger as LoggerInterface


class StructuredLogger(LoggerInterface):
    """Structured logger for POLARIS framework.

    Logs in JSON format for easy parsing and analysis.
    """

    def __init__(
        self,
        name: str = "polaris",
        level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
    ) -> None:
        """Initialize structured JSON logger with name and output configuration."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()

        # JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "component": record.name,
                    "message": record.getMessage(),
                }

                # Add extra context if available
                if hasattr(record, "context"):
                    log_data["context"] = record.context

                return json.dumps(log_data)

        formatter = JSONFormatter()

        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra={"context": context} if context else {})

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra={"context": context} if context else {})

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra={"context": context} if context else {})

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra={"context": context} if context else {})


class HumanReadableLogger(LoggerInterface):
    """
    Human-readable logger for Polaris.

    Provides colorized, formatted output optimized for human consumption.
    """

    def __init__(
        self,
        name: str = "polaris",
        level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
        use_colors: bool = True,
    ) -> None:
        """Initialize human-readable logger with color and formatting options."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()
        self.use_colors = use_colors

        # Color codes for different log levels
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "RESET": "\033[0m",  # Reset
        }

        # Human-readable formatter
        class HumanFormatter(logging.Formatter):
            def __init__(self, use_colors: bool = True) -> None:
                self.use_colors = use_colors
                # Color codes for different log levels
                self.colors = {
                    "DEBUG": "\033[36m",  # Cyan
                    "INFO": "\033[32m",  # Green
                    "WARNING": "\033[33m",  # Yellow
                    "ERROR": "\033[31m",  # Red
                    "RESET": "\033[0m",  # Reset
                }
                super().__init__()

            def format(self, record: logging.LogRecord) -> str:
                # Format timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Get level with color
                level = record.levelname
                if self.use_colors:
                    color = self.colors.get(level, "")
                    reset = self.colors["RESET"]
                    level = f"{color}{level: <7}{reset}"
                else:
                    level = f"{level: <7}"

                # Format component name
                component = record.name.split(".")[-1]  # Get last part of name
                component = f"[{component}]"

                # Base message
                message = record.getMessage()

                # Add context if available
                context_str = ""
                if hasattr(record, "context") and record.context:
                    context_parts = []
                    for key, value in record.context.items():
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, indent=None)
                        context_parts.append(f"{key}={value}")

                    if context_parts:
                        context_str = f" | {', '.join(context_parts)}"

                return f"{timestamp} {level} {component: <12} {message}{context_str}"

        formatter = HumanFormatter(use_colors=self.use_colors and console)

        # Console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler (without colors)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = HumanFormatter(use_colors=False)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra={"context": context} if context else {})

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra={"context": context} if context else {})

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra={"context": context} if context else {})

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra={"context": context} if context else {})


def create_logger(
    logger_type: str = "structured",
    name: str = "polaris",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    **kwargs: Any,
) -> LoggerInterface:
    """Create logger instances with specified configuration.

    Args:
        logger_type: Type of logger ("structured" or "human")
        name: Logger name
        level: Log level
        log_file: Optional log file path
        console: Whether to output to console
        **kwargs: Additional logger-specific arguments

    Returns:
        Logger instance
    """
    if logger_type.lower() == "human":
        return HumanReadableLogger(
            name=name, level=level, log_file=log_file, console=console, **kwargs
        )
    else:
        return StructuredLogger(name=name, level=level, log_file=log_file, console=console)
