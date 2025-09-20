"""
Integration tests to validate logging migration across all POLARIS components.

These tests ensure that all components have been properly migrated to use the
centralized POLARIS logging system and that logging works consistently across
the entire framework.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from polaris_refactored.src.infrastructure.observability.factory import (
    configure_logging, get_polaris_logger, reset_logging, LoggerFactory
)
from polaris_refactored.tests.fixtures.logging_fixtures import (
    integration_logging_setup, configured_logger_factory
)


class TestFrameworkComponentLogging:
    """Test logging integration in framework components."""
    
    def test_plugin_registry_logging(self, integration_logging_setup, capture_logs):
        """Test that plugin registry uses POLARIS logging."""
        from polaris_refactored.src.framework.plugin_management.plugin_registry import PolarisPluginRegistry
        
        registry = PolarisPluginRegistry()
        
        # Verify it has POLARIS logger
        assert hasattr(registry, 'logger')
        assert registry.logger.name == "polaris.framework.plugin_registry"
        
        # Test logging works
        with capture_logs(registry.logger) as handler:
            registry.logger.info("Test plugin registry logging", extra={
                "component": "plugin_registry",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test plugin registry logging" in records[0]['message']
        assert records[0]['extra']['component'] == "plugin_registry"
    
    def test_event_bus_logging(self, integration_logging_setup, capture_logs):
        """Test that event bus uses POLARIS logging."""
        from polaris_refactored.src.framework.events import PolarisEventBus
        
        event_bus = PolarisEventBus()
        
        # Verify it has POLARIS logger
        assert hasattr(event_bus, 'logger')
        assert event_bus.logger.name == "polaris.framework.event_bus"
        
        # Test logging works
        with capture_logs(event_bus.logger) as handler:
            event_bus.logger.info("Test event bus logging", extra={
                "component": "event_bus",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test event bus logging" in records[0]['message']
        assert records[0]['extra']['component'] == "event_bus"
    
    def test_configuration_system_logging(self, integration_logging_setup, capture_logs):
        """Test that configuration system uses POLARIS logging."""
        from polaris_refactored.src.framework.configuration.core import PolarisConfiguration
        
        config = PolarisConfiguration()
        
        # Verify it has POLARIS logger
        assert hasattr(config, 'logger')
        assert config.logger.name == "polaris.infrastructure.configuration"
        
        # Test logging works
        with capture_logs(config.logger) as handler:
            config.logger.info("Test configuration logging", extra={
                "component": "configuration",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test configuration logging" in records[0]['message']
        assert records[0]['extra']['component'] == "configuration"


class TestDigitalTwinComponentLogging:
    """Test logging integration in digital twin components."""
    
    def test_knowledge_base_logging(self, integration_logging_setup, capture_logs):
        """Test that knowledge base uses POLARIS logging."""
        from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
        from polaris_refactored.src.infrastructure.data_storage import PolarisDataStore
        
        # Mock data store
        mock_data_store = Mock(spec=PolarisDataStore)
        
        kb = PolarisKnowledgeBase(mock_data_store)
        
        # Verify it has POLARIS logger
        assert hasattr(kb, 'logger')
        assert kb.logger.name == "polaris.digital_twin.knowledge_base"
        
        # Test logging works
        with capture_logs(kb.logger) as handler:
            kb.logger.info("Test knowledge base logging", extra={
                "component": "knowledge_base",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test knowledge base logging" in records[0]['message']
        assert records[0]['extra']['component'] == "knowledge_base"
    
    def test_learning_engine_logging(self, integration_logging_setup, capture_logs):
        """Test that learning engine uses POLARIS logging."""
        from polaris_refactored.src.digital_twin.learning_engine import PolarisLearningEngine
        
        engine = PolarisLearningEngine()
        
        # Verify it has POLARIS logger
        assert hasattr(engine, 'logger')
        assert engine.logger.name == "polaris.digital_twin.learning_engine"
        
        # Test logging works
        with capture_logs(engine.logger) as handler:
            engine.logger.info("Test learning engine logging", extra={
                "component": "learning_engine",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test learning engine logging" in records[0]['message']
        assert records[0]['extra']['component'] == "learning_engine"
    
    def test_world_model_logging(self, integration_logging_setup):
        """Test that world model components use POLARIS logging."""
        from polaris_refactored.src.digital_twin.world_model import (
            CompositeWorldModel, StatisticalWorldModel, MLWorldModel
        )
        
        # Test CompositeWorldModel
        composite = CompositeWorldModel([])
        assert hasattr(composite, 'logger')
        assert composite.logger.name == "polaris.digital_twin.composite_world_model"
        
        # Test StatisticalWorldModel
        statistical = StatisticalWorldModel()
        assert hasattr(statistical, 'logger')
        assert statistical.logger.name == "polaris.digital_twin.statistical_world_model"
        
        # Test MLWorldModel
        ml_model = MLWorldModel()
        assert hasattr(ml_model, 'logger')
        assert ml_model.logger.name == "polaris.digital_twin.ml_world_model"


class TestControlReasoningComponentLogging:
    """Test logging integration in control and reasoning components."""
    
    def test_reasoning_engine_logging(self, integration_logging_setup, capture_logs):
        """Test that reasoning engine uses POLARIS logging."""
        from polaris_refactored.src.control_reasoning.reasoning_engine import PolarisReasoningEngine
        
        engine = PolarisReasoningEngine()
        
        # Verify it has POLARIS logger
        assert hasattr(engine, 'logger')
        assert engine.logger.name == "polaris.control.reasoning_engine"
        
        # Test logging works
        with capture_logs(engine.logger) as handler:
            engine.logger.info("Test reasoning engine logging", extra={
                "component": "reasoning_engine",
                "test": True
            })
        
        records = handler.get_records()
        assert len(records) == 1
        assert "Test reasoning engine logging" in records[0]['message']
        assert records[0]['extra']['component'] == "reasoning_engine"


class TestLoggingConsistency:
    """Test logging consistency across components."""
    
    def test_all_components_use_polaris_logging(self, integration_logging_setup, capture_logs):
        """Test that all major components use POLARIS logging consistently."""
        components_to_test = [
            # Framework components
            ("polaris_refactored.src.framework.plugin_management.plugin_registry", "PolarisPluginRegistry"),
            ("polaris_refactored.src.framework.events", "PolarisEventBus"),
            ("polaris_refactored.src.framework.configuration.core", "PolarisConfiguration"),
            
            # Digital twin components
            ("polaris_refactored.src.digital_twin.learning_engine", "PolarisLearningEngine"),
            ("polaris_refactored.src.digital_twin.world_model", "StatisticalWorldModel"),
            ("polaris_refactored.src.control_reasoning.reasoning_engine", "PolarisReasoningEngine"),
        ]
        
        for module_path, class_name in components_to_test:
            # Import the module and class
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Create instance (with mocks if needed)
            if class_name == "PolarisKnowledgeBase":
                from polaris_refactored.src.infrastructure.data_storage import PolarisDataStore
                mock_data_store = Mock(spec=PolarisDataStore)
                instance = component_class(mock_data_store)
            else:
                instance = component_class()
            
            # Verify it has a POLARIS logger
            assert hasattr(instance, 'logger'), f"{class_name} should have a logger attribute"
            assert hasattr(instance.logger, 'name'), f"{class_name} logger should have a name"
            assert instance.logger.name.startswith("polaris."), f"{class_name} should use POLARIS logger naming"
    
    def test_logging_context_propagation(self, integration_logging_setup, capture_logs):
        """Test that logging context propagates correctly across components."""
        from polaris_refactored.src.framework.events import PolarisEventBus
        from polaris_refactored.src.digital_twin.learning_engine import PolarisLearningEngine
        
        event_bus = PolarisEventBus()
        learning_engine = PolarisLearningEngine()
        
        correlation_id = "test-correlation-123"
        
        # Test context propagation
        with capture_logs(event_bus.logger) as event_handler, \
             capture_logs(learning_engine.logger) as learning_handler:
            
            with event_bus.logger.correlation_context(correlation_id):
                event_bus.logger.info("Event bus operation")
                
                with learning_engine.logger.correlation_context(correlation_id):
                    learning_engine.logger.info("Learning engine operation")
        
        # Both should have the same correlation ID
        event_records = event_handler.get_records()
        learning_records = learning_handler.get_records()
        
        assert len(event_records) == 1
        assert len(learning_records) == 1
        
        assert event_records[0]['correlation_id'] == correlation_id
        assert learning_records[0]['correlation_id'] == correlation_id
    
    def test_structured_logging_consistency(self, integration_logging_setup, capture_logs):
        """Test that all components use structured logging consistently."""
        from polaris_refactored.src.framework.events import PolarisEventBus
        from polaris_refactored.src.digital_twin.learning_engine import PolarisLearningEngine
        from polaris_refactored.src.control_reasoning.reasoning_engine import PolarisReasoningEngine
        
        components = [
            PolarisEventBus(),
            PolarisLearningEngine(),
            PolarisReasoningEngine()
        ]
        
        for component in components:
            with capture_logs(component.logger) as handler:
                component.logger.info("Test structured logging", extra={
                    "component": component.__class__.__name__,
                    "test_field": "test_value",
                    "numeric_field": 123
                })
            
            records = handler.get_records()
            assert len(records) == 1
            
            record = records[0]
            
            # Verify standard fields
            assert 'timestamp' in record
            assert 'level' in record
            assert 'logger' in record
            assert 'message' in record
            
            # Verify extra fields
            assert 'extra' in record
            assert record['extra']['component'] == component.__class__.__name__
            assert record['extra']['test_field'] == "test_value"
            assert record['extra']['numeric_field'] == 123
    
    def test_error_logging_consistency(self, integration_logging_setup, capture_logs):
        """Test that error logging is consistent across components."""
        from polaris_refactored.src.framework.events import PolarisEventBus
        from polaris_refactored.src.digital_twin.learning_engine import PolarisLearningEngine
        
        components = [
            PolarisEventBus(),
            PolarisLearningEngine()
        ]
        
        for component in components:
            with capture_logs(component.logger) as handler:
                try:
                    raise ValueError("Test error")
                except ValueError as e:
                    component.logger.error("Test error occurred", extra={
                        "component": component.__class__.__name__,
                        "operation": "test_operation"
                    }, exc_info=e)
            
            records = handler.get_records()
            assert len(records) == 1
            
            record = records[0]
            
            # Verify error logging structure
            assert record['level'] == 'ERROR'
            assert 'Test error occurred' in record['message']
            assert 'extra' in record
            assert record['extra']['component'] == component.__class__.__name__
            assert record['extra']['operation'] == "test_operation"
            
            # Verify exception info is captured
            assert 'exception' in record['extra']
            assert record['extra']['exception']['type'] == "ValueError"
            assert record['extra']['exception']['message'] == "Test error"


class TestLoggingPerformance:
    """Test logging performance after migration."""
    
    def test_logging_performance_impact(self, integration_logging_setup, capture_logs):
        """Test that logging migration doesn't significantly impact performance."""
        import time
        from polaris_refactored.src.framework.events import PolarisEventBus
        
        event_bus = PolarisEventBus()
        
        # Test with logging disabled (high log level)
        event_bus.logger.set_level(event_bus.logger.level.__class__.ERROR)
        
        start_time = time.time()
        for i in range(1000):
            event_bus.logger.debug(f"Debug message {i}")
            event_bus.logger.info(f"Info message {i}")
        end_time = time.time()
        
        disabled_duration = end_time - start_time
        
        # Test with logging enabled
        event_bus.logger.set_level(event_bus.logger.level.__class__.DEBUG)
        
        with capture_logs(event_bus.logger) as handler:
            start_time = time.time()
            for i in range(100):  # Fewer iterations to avoid overwhelming test
                event_bus.logger.debug(f"Debug message {i}")
                event_bus.logger.info(f"Info message {i}")
            end_time = time.time()
        
        enabled_duration = end_time - start_time
        
        # Verify logs were captured
        records = handler.get_records()
        assert len(records) == 200  # 100 debug + 100 info
        
        # Performance should be reasonable (this is a rough check)
        # Disabled logging should be very fast
        assert disabled_duration < 0.1  # Less than 100ms for 2000 filtered messages
        
        # Enabled logging should still be reasonable for 200 messages
        assert enabled_duration < 1.0  # Less than 1 second for 200 messages


if __name__ == '__main__':
    pytest.main([__file__, '-v'])