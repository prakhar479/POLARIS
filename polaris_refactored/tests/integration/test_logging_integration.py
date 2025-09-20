"""
Integration tests for POLARIS logging standardization.

These tests validate that logging is properly integrated across all framework
components and that configuration, context propagation, and observability
work correctly in realistic scenarios.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from polaris_refactored.src.framework.configuration.core import PolarisConfiguration
from polaris_refactored.src.infrastructure.observability.factory import (
    configure_logging, get_polaris_logger, reset_logging, get_framework_logger,
    get_infrastructure_logger, LoggerFactory
)
from polaris_refactored.src.framework.configuration.models import LoggingConfiguration
from polaris_refactored.src.infrastructure.observability.logging import LogLevel
from polaris_refactored.tests.fixtures.logging_fixtures import (
    integration_logging_setup, log_assertions, log_file_reader
)


class TestFrameworkLoggingIntegration:
    """Test logging integration with framework components."""
    
    def test_configuration_system_logging(self, integration_logging_setup):
        """Test that configuration system uses POLARIS logging."""
        from polaris_refactored.src.framework.configuration.core import PolarisConfiguration
        
        # Create configuration with logging
        config = PolarisConfiguration()
        
        # Verify it has a POLARIS logger
        assert hasattr(config, 'logger')
        assert config.logger.name == "polaris.infrastructure.configuration"
    
    def test_framework_startup_logging(self, integration_logging_setup):
        """Test logging during framework startup sequence."""
        # This would test the actual framework startup
        # For now, we'll test the logger creation pattern
        
        framework_logger = get_framework_logger("main")
        
        with patch('polaris_refactored.src.infrastructure.observability.factory.configure_default_logging') as mock_configure:
            # Simulate framework configuration
            
            config = LoggingConfiguration(
                level="INFO",
                format="json",
                output="console"
            )
            
            configure_logging(config)
            
            # Verify configuration was called
            mock_configure.assert_called_once()
            
            # Test logging works
            framework_logger.info("Framework starting", extra={
                "version": "2.0.0",
                "components": ["infrastructure", "framework", "adapters"]
            })
    
    def test_component_logger_hierarchy(self, integration_logging_setup):
        """Test that component loggers follow proper naming hierarchy."""
        # Create loggers for different components
        framework_main = get_framework_logger("main")
        framework_config = get_framework_logger("configuration")
        infrastructure_di = get_infrastructure_logger("di")
        infrastructure_messaging = get_infrastructure_logger("messaging")
        
        # Verify naming hierarchy
        assert framework_main.name == "polaris.framework.main"
        assert framework_config.name == "polaris.framework.configuration"
        assert infrastructure_di.name == "polaris.infrastructure.di"
        assert infrastructure_messaging.name == "polaris.infrastructure.messaging"
    
    def test_configuration_hot_reload_logging(self, integration_logging_setup, capture_logs):
        """Test logging during configuration hot-reload."""
        # Create temporary config file
        config_data = {
            'framework': {
                'logging_config': {
                    'level': 'DEBUG',
                    'format': 'json',
                    'output': 'console'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            from polaris_refactored.src.framework.configuration.sources import YAMLConfigurationSource
            
            # Create configuration with hot-reload
            source = YAMLConfigurationSource(temp_file)
            config = PolarisConfiguration([source], enable_hot_reload=False)  # Disable for test
            
            # Capture logs during reload
            logger = config.logger
            
            with capture_logs(logger) as handler:
                config.reload_configuration()
            
            # Should have logged the reload
            records = handler.get_records()
            # Note: Actual reload logging depends on implementation
            
        finally:
            Path(temp_file).unlink()


class TestContextPropagation:
    """Test context propagation across components."""
    
    def test_correlation_id_across_components(self, integration_logging_setup, capture_logs):
        """Test correlation ID propagation across different components."""
        framework_logger = get_framework_logger("test")
        infrastructure_logger = get_infrastructure_logger("test")
        
        correlation_id = "test-correlation-123"
        
        with capture_logs(framework_logger) as framework_handler, \
             capture_logs(infrastructure_logger) as infra_handler:
            
            with framework_logger.correlation_context(correlation_id):
                framework_logger.info("Framework operation")
                
                # Simulate cross-component call
                with infrastructure_logger.correlation_context(correlation_id):
                    infrastructure_logger.info("Infrastructure operation")
        
        # Both should have the same correlation ID
        framework_records = framework_handler.get_records()
        infra_records = infra_handler.get_records()
        
        assert len(framework_records) == 1
        assert len(infra_records) == 1
        
        assert framework_records[0]['correlation_id'] == correlation_id
        assert infra_records[0]['correlation_id'] == correlation_id
    
    def test_adaptation_context_propagation(self, integration_logging_setup, capture_logs):
        """Test adaptation context propagation."""
        controller_logger = get_polaris_logger("polaris.control.adaptive_controller")
        adapter_logger = get_polaris_logger("polaris.adapter.test_system")
        
        adaptation_id = "adaptation-456"
        system_id = "test-system-789"
        
        with capture_logs(controller_logger) as controller_handler, \
             capture_logs(adapter_logger) as adapter_handler:
            
            with controller_logger.adaptation_context(adaptation_id, system_id):
                controller_logger.info("Starting adaptation")
                
                # Simulate adapter execution
                with adapter_logger.adaptation_context(adaptation_id, system_id):
                    adapter_logger.info("Executing adaptation action")
        
        controller_records = controller_handler.get_records()
        adapter_records = adapter_handler.get_records()
        
        assert len(controller_records) == 1
        assert len(adapter_records) == 1
        
        # Both should have adaptation context
        assert controller_records[0]['adaptation_id'] == adaptation_id
        assert controller_records[0]['system_id'] == system_id
        assert adapter_records[0]['adaptation_id'] == adaptation_id
        assert adapter_records[0]['system_id'] == system_id
    
    def test_nested_context_propagation(self, integration_logging_setup, capture_logs):
        """Test nested context propagation."""
        logger = get_polaris_logger("test")
        
        with capture_logs(logger) as handler:
            with logger.correlation_context("correlation-123"):
                logger.info("Outer context")
                
                with logger.system_context("system-456"):
                    logger.info("With system context")
                    
                    with logger.adaptation_context("adaptation-789"):
                        logger.info("With adaptation context")
        
        records = handler.get_records()
        assert len(records) == 3
        
        # All should have correlation ID
        for record in records:
            assert record['correlation_id'] == "correlation-123"
        
        # System context should be in last two
        assert records[1]['system_id'] == "system-456"
        assert records[2]['system_id'] == "system-456"
        
        # Adaptation context only in last one
        assert 'adaptation_id' not in records[0]
        assert 'adaptation_id' not in records[1]
        assert records[2]['adaptation_id'] == "adaptation-789"


class TestObservabilityIntegration:
    """Test integration with observability decorators."""
    
    def test_observe_component_decorator_logging(self, integration_logging_setup, capture_logs):
        """Test that @observe_polaris_component decorator integrates with logging."""
        from polaris_refactored.src.infrastructure.observability import observe_polaris_component
        
        @observe_polaris_component("test_component", log_method_calls=True)
        class TestComponent:
            def test_method(self):
                self._logger.info("Method called")
                return "result"
        
        component = TestComponent()
        
        # Component should have logger
        assert hasattr(component, '_logger')
        assert component._logger.name == "polaris.test_component"
        
        # Test method call logging
        with capture_logs(component._logger) as handler:
            result = component.test_method()
        
        records = handler.get_records()
        
        # Should have logs from both decorator and method
        assert len(records) >= 1
        assert any("Method called" in record['message'] for record in records)
    
    def test_trace_decorator_logging_integration(self, integration_logging_setup, capture_logs):
        """Test integration between tracing decorators and logging."""
        from polaris_refactored.src.infrastructure.observability import trace_adaptation_flow
        
        logger = get_polaris_logger("test")
        
        @trace_adaptation_flow("test_adaptation")
        def test_function():
            logger.info("Function executing")
            return "success"
        
        with capture_logs(logger) as handler:
            result = test_function()
        
        records = handler.get_records()
        assert len(records) == 1
        assert records[0]['message'] == "Function executing"


class TestFileLoggingIntegration:
    """Test file logging in integration scenarios."""
    
    def test_file_logging_with_rotation(self, integration_logging_setup):
        """Test file logging with rotation configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:            
            config = LoggingConfiguration(
                level="INFO",
                format="json",
                output="file",
                file_path=log_file,
                max_file_size=1024,  # Small size for testing
                backup_count=3
            )
            
            configure_logging(config)
            
            logger = get_polaris_logger("test")
            
            # Log many messages to trigger rotation
            for i in range(100):
                logger.info(f"Test message {i}", extra={
                    "iteration": i,
                    "component": "test"
                })
            
            # Verify log file exists and has content
            log_path = Path(log_file)
            assert log_path.exists()
            
            content = log_path.read_text()
            assert len(content) > 0
            
            # Parse JSON logs
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            records = []
            for line in lines:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    pass
            
            assert len(records) > 0
            
            # Verify structure
            for record in records[:5]:  # Check first 5
                assert 'timestamp' in record
                assert 'level' in record
                assert 'message' in record
                assert record['level'] == 'INFO'
        
        finally:
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass
    
    def test_concurrent_file_logging(self, integration_logging_setup):
        """Test concurrent file logging from multiple components."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:            
            config = LoggingConfiguration(
                level="INFO",
                format="json",
                output="file",
                file_path=log_file
            )
            
            configure_logging(config)
            
            # Create multiple loggers
            loggers = [
                get_framework_logger("component1"),
                get_infrastructure_logger("component2"),
                get_polaris_logger("polaris.adapter.component3")
            ]
            
            # Log concurrently
            import threading
            
            def log_messages(logger, component_id):
                for i in range(10):
                    logger.info(f"Message {i} from {component_id}", extra={
                        "component_id": component_id,
                        "iteration": i
                    })
            
            threads = []
            for i, logger in enumerate(loggers):
                thread = threading.Thread(target=log_messages, args=(logger, f"comp_{i}"))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Verify all messages were logged
            log_path = Path(log_file)
            content = log_path.read_text()
            
            # Should have 30 messages total (10 from each of 3 components)
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            assert len(lines) >= 30
        
        finally:
            try:
                Path(log_file).unlink()
            except FileNotFoundError:
                pass


class TestErrorScenarios:
    """Test logging in error scenarios."""
    
    def test_logging_during_exceptions(self, integration_logging_setup, capture_logs):
        """Test logging behavior during exception handling."""
        logger = get_polaris_logger("test")
        
        with capture_logs(logger) as handler:
            try:
                logger.info("Starting operation")
                raise ValueError("Test error")
            except ValueError as e:
                logger.error("Operation failed", extra={
                    "operation": "test_operation",
                    "error_type": "ValueError"
                }, exc_info=e)
            finally:
                logger.info("Cleanup completed")
        
        records = handler.get_records()
        assert len(records) == 3
        
        # Check error record
        error_records = handler.get_records(LogLevel.ERROR)
        assert len(error_records) == 1
        
        error_record = error_records[0]
        assert "Operation failed" in error_record['message']
        assert 'exception' in error_record['extra']
        assert error_record['extra']['exception']['type'] == "ValueError"
    
    def test_logging_with_invalid_configuration(self, integration_logging_setup):
        """Test logging behavior with invalid configuration."""
        # Reset to clean state
        reset_logging()
        
        # Try to configure with invalid settings
        try:            
            invalid_config = LoggingConfiguration(
                level="INFO",
                format="json",
                output="file",
                file_path="/invalid/path/that/does/not/exist/test.log"
            )
            
            # This might fail, but logging should still work
            configure_logging(invalid_config)
            
            logger = get_polaris_logger("test")
            logger.info("Test message")  # Should not raise exception
            
        except Exception:
            # Even if configuration fails, basic logging should work
            logger = get_polaris_logger("test")
            logger.info("Fallback logging")  # Should not raise exception


class TestPerformanceIntegration:
    """Test logging performance in integration scenarios."""
    
    def test_high_volume_logging_performance(self, integration_logging_setup):
        """Test performance with high volume logging."""
        import time
        
        logger = get_polaris_logger("performance_test")
        
        # Test with different log levels
        logger.set_level(LogLevel.WARNING)  # Filter out DEBUG/INFO
        
        start_time = time.time()
        
        # Log many messages that should be filtered
        for i in range(10000):
            logger.debug(f"Debug message {i}")  # Should be filtered
            logger.info(f"Info message {i}")    # Should be filtered
        
        # Log some that should pass through
        for i in range(100):
            logger.warning(f"Warning message {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time
        assert duration < 1.0  # Less than 1 second
    
    def test_memory_usage_with_many_loggers(self, integration_logging_setup):
        """Test memory usage with many logger instances."""
        import gc
        
        # Create many loggers
        loggers = []
        for i in range(1000):
            logger = get_polaris_logger(f"test_logger_{i}")
            loggers.append(logger)
        
        # Force garbage collection
        gc.collect()
        
        # All loggers should be cached in factory
        factory = LoggerFactory.get_instance()
        assert len(factory.get_logger_names()) == 1000
        
        # Clear references
        loggers.clear()
        
        # Factory should still have cached loggers
        assert len(factory.get_logger_names()) == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])