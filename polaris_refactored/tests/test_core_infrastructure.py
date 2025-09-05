"""
Test Core Infrastructure and Layered Architecture

Tests for the basic infrastructure and dependency injection system.
"""

import pytest
import asyncio
from datetime import datetime

from ..src.infrastructure.di import DIContainer, Injectable, inject
from ..src.infrastructure.exceptions import PolarisException, ConfigurationError
from ..src.domain.models import SystemState, MetricValue, HealthStatus
from ..src.framework import PolarisFramework


class TestDependencyInjection:
    """Test the dependency injection container."""
    
    def test_container_creation(self):
        """Test creating a DI container."""
        container = DIContainer()
        assert container is not None
    
    def test_singleton_registration(self):
        """Test registering and resolving singletons."""
        container = DIContainer()
        
        class TestService(Injectable):
            def __init__(self):
                self.value = "test"
        
        container.register_singleton(TestService, TestService)
        
        # Resolve twice and verify it's the same instance
        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)
        
        assert service1 is service2
        assert service1.value == "test"
    
    def test_transient_registration(self):
        """Test registering and resolving transients."""
        container = DIContainer()
        
        class TestService(Injectable):
            def __init__(self):
                self.value = "test"
        
        container.register_transient(TestService, TestService)
        
        # Resolve twice and verify they are different instances
        service1 = container.resolve(TestService)
        service2 = container.resolve(TestService)
        
        assert service1 is not service2
        assert service1.value == service2.value == "test"
    
    def test_factory_registration(self):
        """Test registering and resolving with factory functions."""
        container = DIContainer()
        
        class TestService(Injectable):
            def __init__(self, value: str):
                self.value = value
        
        def factory():
            return TestService("factory_created")
        
        container.register_factory(TestService, factory)
        
        service = container.resolve(TestService)
        assert service.value == "factory_created"
    
    def test_dependency_injection(self):
        """Test automatic dependency injection."""
        container = DIContainer()
        
        class DatabaseService(Injectable):
            def __init__(self):
                self.connected = True
        
        class UserService(Injectable):
            def __init__(self, db: DatabaseService):
                self.db = db
        
        container.register_singleton(DatabaseService, DatabaseService)
        container.register_singleton(UserService, UserService)
        
        user_service = container.resolve(UserService)
        assert user_service.db is not None
        assert user_service.db.connected is True
    
    def test_unregistered_service_error(self):
        """Test error when resolving unregistered service."""
        container = DIContainer()
        
        class UnregisteredService(Injectable):
            pass
        
        with pytest.raises(ValueError, match="Service UnregisteredService is not registered"):
            container.resolve(UnregisteredService)


class TestDomainModels:
    """Test the core domain models."""
    
    def test_metric_value_creation(self):
        """Test creating MetricValue objects."""
        metric = MetricValue(
            name="cpu_usage",
            value=75.5,
            unit="percent"
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "percent"
        assert metric.timestamp is not None
        assert isinstance(metric.tags, dict)
    
    def test_system_state_creation(self):
        """Test creating SystemState objects."""
        metrics = {
            "cpu": MetricValue("cpu_usage", 75.5, "percent"),
            "memory": MetricValue("memory_usage", 60.0, "percent")
        }
        
        state = SystemState(
            system_id="test_system",
            timestamp=datetime.utcnow(),
            metrics=metrics,
            health_status=HealthStatus.HEALTHY
        )
        
        assert state.system_id == "test_system"
        assert len(state.metrics) == 2
        assert state.health_status == HealthStatus.HEALTHY
        assert isinstance(state.metadata, dict)


class TestExceptionHierarchy:
    """Test the structured exception hierarchy."""
    
    def test_polaris_exception_creation(self):
        """Test creating PolarisException."""
        exception = PolarisException(
            message="Test error",
            error_code="TEST_ERROR",
            context={"key": "value"}
        )
        
        assert str(exception) == "Test error"
        assert exception.error_code == "TEST_ERROR"
        assert exception.context["key"] == "value"
        assert exception.correlation_id is not None
        assert exception.timestamp is not None
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            message="Invalid config",
            config_path="/path/to/config.yaml",
            validation_errors=["Missing required field"]
        )
        
        assert error.error_code == "CONFIG_ERROR"
        assert error.context["config_path"] == "/path/to/config.yaml"
        assert "Missing required field" in error.context["validation_errors"]
    
    def test_exception_to_dict(self):
        """Test converting exception to dictionary."""
        exception = PolarisException(
            message="Test error",
            error_code="TEST_ERROR"
        )
        
        error_dict = exception.to_dict()
        
        assert error_dict["error_type"] == "PolarisException"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert "correlation_id" in error_dict
        assert "timestamp" in error_dict


class TestLayeredArchitecture:
    """Test the layered architecture structure."""
    
    def test_layer_imports(self):
        """Test that all layers can be imported."""
        # Framework layer
        from ..src.framework import PolarisFramework
        from ..src.framework.configuration import PolarisConfiguration
        
        # Infrastructure layer
        from ..src.infrastructure import DIContainer, PolarisException
        
        # Domain layer
        from ..src.domain import SystemState, AdaptationAction
        
        # Adapter layer
        from ..src.adapters import PolarisAdapter
        
        # Digital Twin layer
        from ..src.digital_twin import PolarisWorldModel
        
        # Control & Reasoning layer
        from ..src.control_reasoning import PolarisAdaptiveController
        
        # All imports should succeed
        assert True
    
    def test_framework_creation(self):
        """Test creating a basic framework instance."""
        container = DIContainer()
        
        # This will use placeholder implementations for now
        framework = PolarisFramework(
            container=container,
            configuration=None,
            message_bus=None,
            data_store=None,
            plugin_registry=None,
            event_bus=None
        )
        
        assert framework is not None
        assert framework.container is container
        assert not framework.is_running()


@pytest.mark.asyncio
class TestAsyncInfrastructure:
    """Test async infrastructure components."""
    
    async def test_framework_lifecycle_placeholder(self):
        """Test framework lifecycle methods (placeholder test)."""
        container = DIContainer()
        
        framework = PolarisFramework(
            container=container,
            configuration=None,
            message_bus=None,
            data_store=None,
            plugin_registry=None,
            event_bus=None
        )
        
        # These will fail for now since components are None
        # But we can test the basic structure
        assert not framework.is_running()
        
        status = framework.get_status()
        assert "running" in status
        assert status["running"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])