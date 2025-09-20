"""
Test fixtures for POLARIS logging system.

This module provides pytest fixtures for testing logging functionality
with proper setup, teardown, and assertion capabilities.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from io import StringIO

from polaris_refactored.src.infrastructure.observability.factory import (
    LoggerFactory, get_polaris_logger, configure_logging, reset_logging,
    TemporaryLoggingConfig
)
from polaris_refactored.src.infrastructure.observability.logging import (
    PolarisLogger, LogLevel, LogHandler, LogFormatter, JSONLogFormatter, 
    HumanReadableFormatter
)

from polaris_refactored.src.framework.configuration.models import LoggingConfiguration


class TestLogHandler(LogHandler):
    """Test log handler that captures log records for assertions."""
    
    def __init__(self, formatter: LogFormatter):
        super().__init__(formatter)
        self.records: List[Dict[str, Any]] = []
    
    def emit(self, record: Dict[str, Any]) -> None:
        """Capture log record for testing."""
        self.records.append(record.copy())
    
    def clear(self) -> None:
        """Clear captured records."""
        self.records.clear()
    
    def get_records(self, level: Optional[LogLevel] = None) -> List[Dict[str, Any]]:
        """Get captured records, optionally filtered by level."""
        if level is None:
            return self.records.copy()
        return [r for r in self.records if r.get('level') == level.value]
    
    def has_record_with_message(self, message: str) -> bool:
        """Check if any record contains the specified message."""
        return any(message in r.get('message', '') for r in self.records)
    
    def has_record_with_extra(self, key: str, value: Any) -> bool:
        """Check if any record has the specified extra field."""
        return any(
            r.get('extra', {}).get(key) == value 
            for r in self.records
        )


class LogCapture:
    """Context manager for capturing logs during tests."""
    
    def __init__(self, logger: PolarisLogger, level: LogLevel = LogLevel.DEBUG):
        self.logger = logger
        self.level = level
        self.handler = TestLogHandler(JSONLogFormatter())
        self.original_level = None
        self.original_handlers = []
    
    def __enter__(self):
        # Store original state
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()
        
        # Configure for capture
        self.logger.set_level(self.level)
        self.logger.handlers.clear()
        self.logger.add_handler(self.handler)
        
        return self.handler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        self.logger.handlers.clear()
        for handler in self.original_handlers:
            self.logger.add_handler(handler)
        if self.original_level:
            self.logger.set_level(self.original_level)


@pytest.fixture
def test_logging_config():
    """Provide a test logging configuration."""
    return LoggingConfiguration(
        level="DEBUG",
        format="json",
        output="console",
        file_path=None,
        max_file_size=1048576,
        backup_count=3
    )


@pytest.fixture
def file_logging_config():
    """Provide a file-based logging configuration."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        log_file = f.name
    
    config = LoggingConfiguration(
        level="INFO",
        format="json",
        output="file",
        file_path=log_file,
        max_file_size=1048576,
        backup_count=3
    )
    
    yield config, log_file
    
    # Cleanup
    try:
        Path(log_file).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def logger_factory():
    """Provide a clean logger factory for testing."""
    factory = LoggerFactory.get_instance()
    factory.reset()
    yield factory
    factory.reset()


@pytest.fixture
def configured_logger_factory(logger_factory, test_logging_config):
    """Provide a configured logger factory."""
    logger_factory.configure(test_logging_config)
    yield logger_factory
    logger_factory.reset()


@pytest.fixture
def test_logger(configured_logger_factory):
    """Provide a test logger instance."""
    return configured_logger_factory.get_logger("test")


@pytest.fixture
def capture_logs():
    """Provide a log capture context manager factory."""
    def _capture_logs(logger: PolarisLogger, level: LogLevel = LogLevel.DEBUG):
        return LogCapture(logger, level)
    return _capture_logs


@pytest.fixture
def mock_log_handler():
    """Provide a mock log handler for testing."""
    handler = Mock(spec=LogHandler)
    handler.emit = Mock()
    return handler


@pytest.fixture
def temporary_log_config():
    """Provide a temporary logging configuration context manager."""
    def _temp_config(config):
        return TemporaryLoggingConfig(config)
    return _temp_config


@pytest.fixture
def log_file_reader():
    """Provide a utility for reading and parsing log files."""
    def _read_log_file(file_path: str, format_type: str = "json") -> List[Dict[str, Any]]:
        """Read and parse log file contents."""
        records = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if format_type == "json":
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                    else:
                        # For text format, create a simple record
                        records.append({"message": line})
        except FileNotFoundError:
            pass
        
        return records
    
    return _read_log_file


@pytest.fixture
def correlation_id_context():
    """Provide correlation ID context for testing."""
    def _with_correlation_id(logger: PolarisLogger, correlation_id: str):
        return logger.correlation_context(correlation_id)
    return _with_correlation_id


@pytest.fixture
def adaptation_context():
    """Provide adaptation context for testing."""
    def _with_adaptation_context(logger: PolarisLogger, adaptation_id: str, system_id: str = None):
        return logger.adaptation_context(adaptation_id, system_id)
    return _with_adaptation_context


@pytest.fixture
def system_context():
    """Provide system context for testing."""
    def _with_system_context(logger: PolarisLogger, system_id: str):
        return logger.system_context(system_id)
    return _with_system_context


# Assertion helpers
class LogAssertions:
    """Helper class for log-related assertions."""
    
    @staticmethod
    def assert_log_record_exists(
        records: List[Dict[str, Any]], 
        level: str, 
        message: str = None,
        extra_fields: Dict[str, Any] = None
    ):
        """Assert that a log record with specified criteria exists."""
        matching_records = [r for r in records if r.get('level') == level]
        
        if message:
            matching_records = [r for r in matching_records if message in r.get('message', '')]
        
        if extra_fields:
            for key, value in extra_fields.items():
                matching_records = [
                    r for r in matching_records 
                    if r.get('extra', {}).get(key) == value
                ]
        
        assert len(matching_records) > 0, f"No log record found matching criteria: level={level}, message={message}, extra={extra_fields}"
    
    @staticmethod
    def assert_correlation_id_present(records: List[Dict[str, Any]], correlation_id: str):
        """Assert that records contain the specified correlation ID."""
        matching_records = [r for r in records if r.get('correlation_id') == correlation_id]
        assert len(matching_records) > 0, f"No records found with correlation_id={correlation_id}"
    
    @staticmethod
    def assert_log_level_filtering(records: List[Dict[str, Any]], min_level: LogLevel):
        """Assert that all records are at or above the minimum level."""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        
        min_level_value = level_order[min_level]
        
        for record in records:
            record_level = LogLevel(record.get('level', 'INFO'))
            record_level_value = level_order[record_level]
            assert record_level_value >= min_level_value, f"Record level {record_level} is below minimum {min_level}"
    
    @staticmethod
    def assert_json_format(records: List[Dict[str, Any]]):
        """Assert that all records are properly formatted JSON."""
        required_fields = ['timestamp', 'level', 'logger', 'message']
        
        for record in records:
            for field in required_fields:
                assert field in record, f"Required field '{field}' missing from record: {record}"
    
    @staticmethod
    def assert_no_errors_logged(records: List[Dict[str, Any]]):
        """Assert that no ERROR or CRITICAL level logs were recorded."""
        error_records = [r for r in records if r.get('level') in ['ERROR', 'CRITICAL']]
        assert len(error_records) == 0, f"Unexpected error logs found: {error_records}"


@pytest.fixture
def log_assertions():
    """Provide log assertion helpers."""
    return LogAssertions


# Integration test fixtures
@pytest.fixture
def integration_logging_setup():
    """Set up logging for integration tests."""
    # Reset any existing configuration
    reset_logging()
    
    # Configure with test settings
    config = LoggingConfiguration(
        level="DEBUG",
        format="json",
        output="console"
    )
    configure_logging(config)
    
    yield
    
    # Cleanup
    reset_logging()


@pytest.fixture
def performance_logging_setup():
    """Set up logging for performance tests with minimal overhead."""
    reset_logging()
    
    config = LoggingConfiguration(
        level="WARNING",  # Minimal logging for performance
        format="json",
        output="console"
    )
    configure_logging(config)
    
    yield
    
    reset_logging()


# Mock fixtures for external dependencies
@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing."""
    with patch('pathlib.Path.mkdir'), \
         patch('builtins.open', create=True) as mock_open:
        mock_file = StringIO()
        mock_open.return_value.__enter__.return_value = mock_file
        yield mock_file


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent timestamps in tests."""
    with patch('polaris_refactored.src.infrastructure.observability.logging.datetime') as mock_dt:
        mock_dt.utcnow.return_value.isoformat.return_value = "2023-01-01T00:00:00.000000"
        yield mock_dt