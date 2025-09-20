"""
Comprehensive tests for the POLARIS Logger Factory.

Tests cover factory functionality, configuration management, logger creation,
and integration with the framework configuration system.
"""

import pytest
import tempfile
import threading
import time
from flaky import flaky

from pathlib import Path
from unittest.mock import Mock, patch

from polaris_refactored.src.infrastructure.observability.factory import (
    LoggerFactory, configure_logging, get_polaris_logger, is_logging_configured,
    get_logging_configuration, reset_logging, get_framework_logger,
    get_infrastructure_logger, get_adapter_logger, get_digital_twin_logger,
    get_control_logger, get_test_logger, TemporaryLoggingConfig
)
from polaris_refactored.src.infrastructure.observability.logging import (
    PolarisLogger, LogLevel
)

from polaris_refactored.src.framework.configuration.models import LoggingConfiguration



class TestLoggerFactory:
    """Test LoggerFactory core functionality."""
    
    def test_singleton_pattern(self):
        """Test that LoggerFactory follows singleton pattern."""
        factory1 = LoggerFactory.get_instance()
        factory2 = LoggerFactory.get_instance()
        
        assert factory1 is factory2
        assert id(factory1) == id(factory2)
    
    def test_reset_instance(self):
        """Test factory instance reset."""
        factory1 = LoggerFactory.get_instance()
        LoggerFactory.reset_instance()
        factory2 = LoggerFactory.get_instance()
        
        assert factory1 is not factory2
    
    def test_initial_state(self, logger_factory):
        """Test factory initial state."""
        assert not logger_factory.is_configured()
        assert logger_factory.get_configuration() is None
        assert logger_factory.get_logger_names() == []
    
    def test_configure_factory(self, logger_factory, test_logging_config):
        """Test factory configuration."""
        logger_factory.configure(test_logging_config)
        
        assert logger_factory.is_configured()
        assert logger_factory.get_configuration() == test_logging_config
    
    def test_get_logger_without_configuration(self, logger_factory):
        """Test getting logger without prior configuration."""
        logger = logger_factory.get_logger("test")
        
        assert isinstance(logger, PolarisLogger)
        assert logger.name == "test"
        assert logger.level == LogLevel.INFO  # Default level
    
    def test_get_logger_with_configuration(self, configured_logger_factory):
        """Test getting logger with configuration."""
        logger = configured_logger_factory.get_logger("test")
        
        assert isinstance(logger, PolarisLogger)
        assert logger.name == "test"
        assert logger.level == LogLevel.DEBUG
    
    def test_logger_caching(self, configured_logger_factory):
        """Test that loggers are cached properly."""
        logger1 = configured_logger_factory.get_logger("test")
        logger2 = configured_logger_factory.get_logger("test")
        
        assert logger1 is logger2
        assert "test" in configured_logger_factory.get_logger_names()
    
    def test_logger_level_override(self, configured_logger_factory):
        """Test logger creation with level override."""
        logger = configured_logger_factory.get_logger("test", LogLevel.ERROR)
        
        assert logger.level == LogLevel.ERROR
    
    def test_logger_config_override(self, logger_factory):
        """Test logger creation with config override."""
        
        override_config = LoggingConfiguration(
            level="WARNING",
            format="text",
            output="console"
        )
        
        logger = logger_factory.get_logger("test", config_override=override_config)
        
        assert logger.level == LogLevel.WARNING
    
    def test_reconfigure_existing_loggers(self, logger_factory, test_logging_config):
        """Test that existing loggers are reconfigured when factory is reconfigured."""
        # Create logger before configuration
        logger = logger_factory.get_logger("test")
        original_level = logger.level
        
        # Configure factory
        logger_factory.configure(test_logging_config)
        
        # Logger should be reconfigured
        assert logger.level == LogLevel.DEBUG
        assert logger.level != original_level
    
    def test_thread_safety(self, logger_factory, test_logging_config):
        """Test factory thread safety."""
        results = []
        
        def create_logger(name):
            logger = logger_factory.get_logger(f"test_{name}")
            results.append(logger)
        
        # Configure factory
        logger_factory.configure(test_logging_config)
        
        # Create loggers concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_logger, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All loggers should be created successfully
        assert len(results) == 10
        assert len(set(id(logger) for logger in results)) == 10  # All unique instances
    
    def test_reset_factory(self, configured_logger_factory):
        """Test factory reset functionality."""
        # Create some loggers
        logger1 = configured_logger_factory.get_logger("test1")
        logger2 = configured_logger_factory.get_logger("test2")
        
        assert len(configured_logger_factory.get_logger_names()) == 2
        assert configured_logger_factory.is_configured()
        
        # Reset factory
        configured_logger_factory.reset()
        
        assert len(configured_logger_factory.get_logger_names()) == 0
        assert not configured_logger_factory.is_configured()
        assert configured_logger_factory.get_configuration() is None


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_configure_logging(self, test_logging_config):
        """Test global configure_logging function."""
        reset_logging()
        
        assert not is_logging_configured()
        
        configure_logging(test_logging_config)
        
        assert is_logging_configured()
        assert get_logging_configuration() == test_logging_config
        
        reset_logging()
    
    def test_get_polaris_logger(self, test_logging_config):
        """Test global get_polaris_logger function."""
        reset_logging()
        configure_logging(test_logging_config)
        
        logger = get_polaris_logger("test")
        
        assert isinstance(logger, PolarisLogger)
        assert logger.name == "test"
        assert logger.level == LogLevel.DEBUG
        
        reset_logging()
    
    def test_component_logger_helpers(self, test_logging_config):
        """Test component-specific logger helper functions."""
        reset_logging()
        configure_logging(test_logging_config)
        
        framework_logger = get_framework_logger("test")
        infrastructure_logger = get_infrastructure_logger("test")
        adapter_logger = get_adapter_logger("test")
        digital_twin_logger = get_digital_twin_logger("test")
        control_logger = get_control_logger("test")
        test_logger = get_test_logger("test")
        
        assert framework_logger.name == "polaris.framework.test"
        assert infrastructure_logger.name == "polaris.infrastructure.test"
        assert adapter_logger.name == "polaris.adapter.test"
        assert digital_twin_logger.name == "polaris.digital_twin.test"
        assert control_logger.name == "polaris.control.test"
        assert test_logger.name == "polaris.test.test"
        
        reset_logging()


class TestTemporaryLoggingConfig:
    """Test temporary logging configuration context manager."""
    
    def test_temporary_config_context(self, test_logging_config):
        """Test temporary configuration context manager."""
        
        reset_logging()
        
        # Set initial configuration
        initial_config = LoggingConfiguration(
            level="INFO",
            format="text",
            output="console"
        )
        configure_logging(initial_config)
        
        assert get_logging_configuration().level == "INFO"
        
        # Use temporary configuration
        temp_config = LoggingConfiguration(
            level="DEBUG",
            format="json",
            output="console"
        )
        
        with TemporaryLoggingConfig(temp_config):
            assert get_logging_configuration().level == "DEBUG"
            assert get_logging_configuration().format == "json"
        
        # Should revert to original
        assert get_logging_configuration().level == "INFO"
        assert get_logging_configuration().format == "text"
        
        reset_logging()
    
    def test_temporary_config_exception_handling(self, test_logging_config):
        """Test temporary config handles exceptions properly."""
        
        reset_logging()
        configure_logging(test_logging_config)
        
        original_config = get_logging_configuration()
        
        temp_config = LoggingConfiguration(
            level="ERROR",
            format="text",
            output="console"
        )
        
        try:
            with TemporaryLoggingConfig(temp_config):
                assert get_logging_configuration().level == "ERROR"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still revert to original
        assert get_logging_configuration().level == original_config.level
        
        reset_logging()


class TestFileLogging:
    """Test file logging functionality."""
    
    def test_file_logging_configuration(self, file_logging_config):
        """Test file logging configuration."""
        config, log_file = file_logging_config
        
        reset_logging()
        configure_logging(config)
        
        logger = get_polaris_logger("test")
        
        # Log some messages
        logger.info("Test message 1")
        logger.warning("Test message 2")
        
        # Check file exists and has content
        log_path = Path(log_file)
        assert log_path.exists()
        
        content = log_path.read_text()
        assert "Test message 1" in content
        assert "Test message 2" in content
        
        reset_logging()
    
    def test_file_and_console_logging(self, file_logging_config):
        """Test combined file and console logging."""
        config, log_file = file_logging_config
        config.output = "both"
        
        reset_logging()
        configure_logging(config)
        
        logger = get_polaris_logger("test")
        
        with patch('sys.stdout') as mock_stdout:
            logger.info("Test message")
            
            # Should write to both file and console
            log_path = Path(log_file)
            assert log_path.exists()
            assert "Test message" in log_path.read_text()
        
        reset_logging()


class TestLoggingIntegration:
    """Test logging integration with other components."""
    
    def test_correlation_id_propagation(self, configured_logger_factory, capture_logs):
        """Test correlation ID propagation in logs."""
        logger = configured_logger_factory.get_logger("test")
        
        with capture_logs(logger) as handler:
            with logger.correlation_context("test-correlation-123"):
                logger.info("Test message with correlation")
        
        records = handler.get_records()
        assert len(records) == 1
        assert records[0]['correlation_id'] == "test-correlation-123"
    
    def test_adaptation_context_propagation(self, configured_logger_factory, capture_logs):
        """Test adaptation context propagation in logs."""
        logger = configured_logger_factory.get_logger("test")
        
        with capture_logs(logger) as handler:
            with logger.adaptation_context("adaptation-456", "system-789"):
                logger.info("Test message with adaptation context")
        
        records = handler.get_records()
        assert len(records) == 1
        assert records[0]['adaptation_id'] == "adaptation-456"
        assert records[0]['system_id'] == "system-789"
    
    def test_structured_logging_with_extra(self, configured_logger_factory, capture_logs):
        """Test structured logging with extra fields."""
        logger = configured_logger_factory.get_logger("test")
        
        with capture_logs(logger) as handler:
            logger.info("Test message", extra={
                "component": "test_component",
                "operation": "test_operation",
                "duration_ms": 123
            })
        
        records = handler.get_records()
        assert len(records) == 1
        
        extra = records[0]['extra']
        assert extra['component'] == "test_component"
        assert extra['operation'] == "test_operation"
        assert extra['duration_ms'] == 123
    
    def test_error_logging_with_exception(self, configured_logger_factory, capture_logs):
        """Test error logging with exception information."""
        logger = configured_logger_factory.get_logger("test")
        
        with capture_logs(logger) as handler:
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                logger.error("An error occurred", exc_info=e)
        
        records = handler.get_records(LogLevel.ERROR)
        assert len(records) == 1
        
        extra = records[0]['extra']
        assert 'exception' in extra
        assert extra['exception']['type'] == "ValueError"
        assert extra['exception']['message'] == "Test exception"


class TestLoggingPerformance:
    """Test logging performance characteristics."""
    
    def test_logger_creation_performance(self, configured_logger_factory):
        """Test logger creation performance."""
        start_time = time.time()
        
        # Create many loggers
        loggers = []
        for i in range(1000):
            logger = configured_logger_factory.get_logger(f"test_{i}")
            loggers.append(logger)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should create 1000 loggers in reasonable time (< 1 second)
        assert duration < 1.0
        assert len(loggers) == 1000
    
    @flaky(max_runs=3, min_passes=1)
    def test_logging_overhead(self, configured_logger_factory):
        """Test logging overhead with disabled levels."""
        logger = configured_logger_factory.get_logger("test")
        logger.set_level(LogLevel.ERROR)  # Disable DEBUG/INFO/WARNING
        
        start_time = time.time()
        
        # Log many messages that should be filtered out
        for i in range(10000):
            logger.debug(f"Debug message {i}")
            logger.info(f"Info message {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be very fast since messages are filtered out
        assert duration < 0.1  # Less than 100ms for 20k filtered messages


class TestErrorHandling:
    """Test error handling in logging system."""
    
    def test_invalid_log_level(self, logger_factory):
        """Test handling of invalid log levels."""
        
        with pytest.raises(ValueError):
            config = LoggingConfiguration(
                level="INVALID_LEVEL",
                format="json",
                output="console"
            )
    
    def test_missing_file_path_for_file_output(self):
        """Test validation of file path for file output."""
        
        with pytest.raises(ValueError, match="file_path is required"):
            LoggingConfiguration(
                level="INFO",
                format="json",
                output="file",
                file_path=None
            )
    
    def test_handler_failure_resilience(self, configured_logger_factory):
        """Test that logger continues working when a handler fails."""
        logger = configured_logger_factory.get_logger("test")
        
        # Add a mock handler that always fails
        failing_handler = Mock()
        failing_handler.emit.side_effect = Exception("Handler failed")
        logger.add_handler(failing_handler)
        
        # Should not raise exception
        logger.info("Test message")
        
        # Handler should have been called
        failing_handler.emit.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])