"""
Tests for the POLARIS Base Adapter

Comprehensive tests for the Template Method pattern implementation, lifecycle management,
validation, error handling, and monitoring capabilities.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from ..src.adapters.base_adapter import (
    PolarisAdapter, AdapterState, AdapterHealthStatus, AdapterConfiguration,
    AdapterMetrics, AdapterValidationError, AdapterLifecycleError
)
from ..src.framework.events import PolarisEventBus


class MockPolarisAdapter(PolarisAdapter):
    """Mock adapter for testing base functionality."""
    
    def __init__(self, configuration: AdapterConfiguration, event_bus=None):
        super().__init__(configuration, event_bus)
        
        # Mock control flags
        self.should_fail_validation = False
        self.should_fail_initialization = False
        self.should_fail_start_processing = False
        self.should_fail_stop_processing = False
        self.should_fail_cleanup = False
        self.should_be_unhealthy = False
        
        # Track method calls
        self.validation_called = False
        self.initialization_called = False
        self.start_processing_called = False
        self.stop_processing_called = False
        self.cleanup_called = False
        self.health_check_called = False
    
    async def _validate_configuration(self) -> None:
        """Mock configuration validation."""
        self.validation_called = True
        if self.should_fail_validation:
            raise ValueError("Mock validation failure")
    
    async def _initialize_resources(self) -> None:
        """Mock resource initialization."""
        self.initialization_called = True
        if self.should_fail_initialization:
            raise RuntimeError("Mock initialization failure")
    
    async def _start_processing(self) -> None:
        """Mock processing start."""
        self.start_processing_called = True
        if self.should_fail_start_processing:
            raise RuntimeError("Mock start processing failure")
    
    async def _stop_processing(self) -> None:
        """Mock processing stop."""
        self.stop_processing_called = True
        if self.should_fail_stop_processing:
            raise RuntimeError("Mock stop processing failure")
    
    async def _cleanup_resources(self) -> None:
        """Mock resource cleanup."""
        self.cleanup_called = True
        if self.should_fail_cleanup:
            raise RuntimeError("Mock cleanup failure")
    
    async def _check_health(self) -> AdapterHealthStatus:
        """Mock health check."""
        self.health_check_called = True
        if self.should_be_unhealthy:
            return AdapterHealthStatus.UNHEALTHY
        return AdapterHealthStatus.HEALTHY


class TestAdapterConfiguration:
    """Test the AdapterConfiguration class."""
    
    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = AdapterConfiguration(
            adapter_id="test_adapter",
            adapter_type="test_type",
            enabled=True,
            config={"key": "value"},
            health_check_interval=30
        )
        
        assert config.adapter_id == "test_adapter"
        assert config.adapter_type == "test_type"
        assert config.enabled is True
        assert config.config == {"key": "value"}
        assert config.health_check_interval == 30
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        config = AdapterConfiguration(
            adapter_id="test_adapter",
            adapter_type="test_type"
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_configuration_validation_failures(self):
        """Test configuration validation failures."""
        config = AdapterConfiguration(
            adapter_id="",  # Invalid empty ID
            adapter_type="",  # Invalid empty type
            enabled="not_bool",  # Invalid type
            config="not_dict",  # Invalid type
            health_check_interval=-1,  # Invalid negative value
            max_retries=-1,  # Invalid negative value
            retry_delay=-1,  # Invalid negative value
            timeout=-1  # Invalid negative value
        )
        
        errors = config.validate()
        assert len(errors) > 0
        assert any("adapter_id" in error for error in errors)
        assert any("adapter_type" in error for error in errors)
        assert any("enabled" in error for error in errors)
        assert any("config" in error for error in errors)
        assert any("health_check_interval" in error for error in errors)
        assert any("max_retries" in error for error in errors)
        assert any("retry_delay" in error for error in errors)
        assert any("timeout" in error for error in errors)


class TestAdapterMetrics:
    """Test the AdapterMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = AdapterMetrics()
        
        assert metrics.processed_items == 0
        assert metrics.failed_items == 0
        assert metrics.last_activity is None
        assert metrics.uptime is None
        assert metrics.error_rate == 0.0
        assert metrics.throughput == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = AdapterMetrics()
        
        # No items processed
        assert metrics.success_rate == 0.0
        
        # Some successful items
        metrics.processed_items = 8
        metrics.failed_items = 2
        assert metrics.success_rate == 0.8
        
        # All failed items
        metrics.processed_items = 0
        metrics.failed_items = 10
        assert metrics.success_rate == 0.0
        
        # All successful items
        metrics.processed_items = 10
        metrics.failed_items = 0
        assert metrics.success_rate == 1.0


class TestPolarisAdapter:
    """Test the PolarisAdapter base class."""
    
    @pytest.fixture
    def valid_configuration(self):
        """Create a valid adapter configuration."""
        return AdapterConfiguration(
            adapter_id="test_adapter",
            adapter_type="test_type",
            enabled=True,
            health_check_interval=1  # Short interval for testing
        )
    
    @pytest.fixture
    def mock_adapter(self, valid_configuration):
        """Create a mock adapter for testing."""
        return MockPolarisAdapter(valid_configuration)
    
    @pytest.fixture
    async def event_bus(self):
        """Create an event bus for testing."""
        bus = PolarisEventBus(worker_count=1)
        await bus.start()
        yield bus
        await bus.stop()
    
    def test_adapter_initialization(self, valid_configuration):
        """Test adapter initialization."""
        adapter = MockPolarisAdapter(valid_configuration)
        
        assert adapter.adapter_id == "test_adapter"
        assert adapter.adapter_type == "test_type"
        assert adapter.state == AdapterState.STOPPED
        assert adapter.health_status == AdapterHealthStatus.UNKNOWN
        assert not adapter.is_running()
        assert not adapter.is_healthy()
    
    @pytest.mark.asyncio
    async def test_successful_start_stop_lifecycle(self, mock_adapter):
        """Test successful adapter start and stop lifecycle."""
        # Start adapter
        await mock_adapter.start()
        
        # Verify start sequence
        assert mock_adapter.validation_called
        assert mock_adapter.initialization_called
        assert mock_adapter.start_processing_called
        assert mock_adapter.state == AdapterState.RUNNING
        assert mock_adapter.health_status == AdapterHealthStatus.HEALTHY
        assert mock_adapter.is_running()
        assert mock_adapter.is_healthy()
        
        # Stop adapter
        await mock_adapter.stop()
        
        # Verify stop sequence
        assert mock_adapter.stop_processing_called
        assert mock_adapter.cleanup_called
        assert mock_adapter.state == AdapterState.STOPPED
        assert mock_adapter.health_status == AdapterHealthStatus.UNKNOWN
        assert not mock_adapter.is_running()
    
    @pytest.mark.asyncio
    async def test_start_validation_failure(self, mock_adapter):
        """Test adapter start failure during validation."""
        mock_adapter.should_fail_validation = True
        
        with pytest.raises(AdapterLifecycleError) as exc_info:
            await mock_adapter.start()
        
        assert exc_info.value.adapter_id == "test_adapter"
        assert exc_info.value.state == AdapterState.ERROR
        assert mock_adapter.state == AdapterState.ERROR
        assert mock_adapter.health_status == AdapterHealthStatus.UNHEALTHY
        assert not mock_adapter.is_running()
    
    @pytest.mark.asyncio
    async def test_start_initialization_failure(self, mock_adapter):
        """Test adapter start failure during initialization."""
        mock_adapter.should_fail_initialization = True
        
        with pytest.raises(AdapterLifecycleError):
            await mock_adapter.start()
        
        assert mock_adapter.state == AdapterState.ERROR
        assert mock_adapter.health_status == AdapterHealthStatus.UNHEALTHY
        assert mock_adapter.validation_called
        assert mock_adapter.initialization_called
        assert not mock_adapter.start_processing_called
    
    @pytest.mark.asyncio
    async def test_start_processing_failure(self, mock_adapter):
        """Test adapter start failure during processing start."""
        mock_adapter.should_fail_start_processing = True
        
        with pytest.raises(AdapterLifecycleError):
            await mock_adapter.start()
        
        assert mock_adapter.state == AdapterState.ERROR
        assert mock_adapter.health_status == AdapterHealthStatus.UNHEALTHY
        assert mock_adapter.validation_called
        assert mock_adapter.initialization_called
        assert mock_adapter.start_processing_called
    
    @pytest.mark.asyncio
    async def test_stop_failure(self, mock_adapter):
        """Test adapter stop failure."""
        # Start adapter first
        await mock_adapter.start()
        assert mock_adapter.is_running()
        
        # Make stop processing fail
        mock_adapter.should_fail_stop_processing = True
        
        with pytest.raises(AdapterLifecycleError):
            await mock_adapter.stop()
        
        assert mock_adapter.state == AdapterState.ERROR
        assert mock_adapter.health_status == AdapterHealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_disabled_adapter_start(self, valid_configuration):
        """Test that disabled adapter doesn't start."""
        valid_configuration.enabled = False
        adapter = MockPolarisAdapter(valid_configuration)
        
        await adapter.start()
        
        # Should not have started
        assert not adapter.validation_called
        assert not adapter.initialization_called
        assert not adapter.start_processing_called
        assert adapter.state == AdapterState.STOPPED
    
    @pytest.mark.asyncio
    async def test_restart_functionality(self, mock_adapter):
        """Test adapter restart functionality."""
        # Start adapter
        await mock_adapter.start()
        assert mock_adapter.is_running()
        
        # Restart adapter
        await mock_adapter.restart()
        
        # Should be running again
        assert mock_adapter.is_running()
        assert mock_adapter.state == AdapterState.RUNNING
        assert mock_adapter.stop_processing_called
        assert mock_adapter.cleanup_called
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_adapter):
        """Test health monitoring functionality."""
        # Start adapter
        await mock_adapter.start()
        
        # Wait for health check
        await asyncio.sleep(1.5)  # Health check interval is 1 second
        
        # Health check should have been called
        assert mock_adapter.health_check_called
        assert mock_adapter._last_health_check is not None
        
        # Stop adapter
        await mock_adapter.stop()
    
    @pytest.mark.asyncio
    async def test_health_status_change(self, mock_adapter):
        """Test health status change detection."""
        # Start adapter
        await mock_adapter.start()
        initial_status = mock_adapter.health_status
        
        # Change health status
        mock_adapter.should_be_unhealthy = True
        
        # Wait for health check
        await asyncio.sleep(1.5)
        
        # Status should have changed
        assert mock_adapter.health_status != initial_status
        assert mock_adapter.health_status == AdapterHealthStatus.UNHEALTHY
        
        # Stop adapter
        await mock_adapter.stop()
    
    def test_metrics_update(self, mock_adapter):
        """Test metrics update functionality."""
        # Update metrics
        mock_adapter.update_metrics(processed=10, failed=2)
        
        metrics = mock_adapter.metrics
        assert metrics.processed_items == 10
        assert metrics.failed_items == 2
        assert metrics.error_rate == 2/12  # 2/(10+2) = 2/12 ≈ 0.1667
        assert metrics.last_activity is not None
    
    def test_lifecycle_hooks(self, mock_adapter):
        """Test lifecycle hooks functionality."""
        pre_start_called = False
        post_start_called = False
        pre_stop_called = False
        post_stop_called = False
        
        def pre_start_hook(adapter):
            nonlocal pre_start_called
            pre_start_called = True
        
        def post_start_hook(adapter):
            nonlocal post_start_called
            post_start_called = True
        
        def pre_stop_hook(adapter):
            nonlocal pre_stop_called
            pre_stop_called = True
        
        def post_stop_hook(adapter):
            nonlocal post_stop_called
            post_stop_called = True
        
        # Add hooks
        mock_adapter.add_pre_start_hook(pre_start_hook)
        mock_adapter.add_post_start_hook(post_start_hook)
        mock_adapter.add_pre_stop_hook(pre_stop_hook)
        mock_adapter.add_post_stop_hook(post_stop_hook)
        
        # Test hooks are added
        assert len(mock_adapter._pre_start_hooks) == 1
        assert len(mock_adapter._post_start_hooks) == 1
        assert len(mock_adapter._pre_stop_hooks) == 1
        assert len(mock_adapter._post_stop_hooks) == 1
    
    @pytest.mark.asyncio
    async def test_lifecycle_hooks_execution(self, mock_adapter):
        """Test that lifecycle hooks are executed."""
        hook_calls = []
        
        def pre_start_hook(adapter):
            hook_calls.append("pre_start")
        
        def post_start_hook(adapter):
            hook_calls.append("post_start")
        
        def pre_stop_hook(adapter):
            hook_calls.append("pre_stop")
        
        def post_stop_hook(adapter):
            hook_calls.append("post_stop")
        
        # Add hooks
        mock_adapter.add_pre_start_hook(pre_start_hook)
        mock_adapter.add_post_start_hook(post_start_hook)
        mock_adapter.add_pre_stop_hook(pre_stop_hook)
        mock_adapter.add_post_stop_hook(post_stop_hook)
        
        # Start and stop adapter
        await mock_adapter.start()
        await mock_adapter.stop()
        
        # Verify hooks were called in correct order
        assert "pre_start" in hook_calls
        assert "post_start" in hook_calls
        assert "pre_stop" in hook_calls
        assert "post_stop" in hook_calls
        
        # Verify order
        pre_start_idx = hook_calls.index("pre_start")
        post_start_idx = hook_calls.index("post_start")
        pre_stop_idx = hook_calls.index("pre_stop")
        post_stop_idx = hook_calls.index("post_stop")
        
        assert pre_start_idx < post_start_idx
        assert post_start_idx < pre_stop_idx
        assert pre_stop_idx < post_stop_idx
    
    def test_status_summary(self, mock_adapter):
        """Test status summary generation."""
        summary = mock_adapter.get_status_summary()
        
        assert "adapter_id" in summary
        assert "adapter_type" in summary
        assert "state" in summary
        assert "health_status" in summary
        assert "enabled" in summary
        assert "metrics" in summary
        
        assert summary["adapter_id"] == "test_adapter"
        assert summary["adapter_type"] == "test_type"
        assert summary["state"] == AdapterState.STOPPED.value
        assert summary["health_status"] == AdapterHealthStatus.UNKNOWN.value
    
    @pytest.mark.asyncio
    async def test_event_bus_integration(self, valid_configuration, event_bus):
        """Test integration with event bus."""
        adapter = MockPolarisAdapter(valid_configuration, event_bus)
        
        # Start and stop adapter
        await adapter.start()
        await adapter.stop()
        
        # Events should have been published (logged in this implementation)
        # In a real implementation, we would verify actual events
        assert adapter._start_time is None  # Should be None after stop
    
    @pytest.mark.asyncio
    async def test_concurrent_start_stop(self, mock_adapter):
        """Test concurrent start/stop operations."""
        # Try to start multiple times concurrently
        tasks = [mock_adapter.start() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        # Should only be started once
        assert mock_adapter.is_running()
        assert mock_adapter.state == AdapterState.RUNNING
        
        # Try to stop multiple times concurrently
        tasks = [mock_adapter.stop() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        # Should be stopped
        assert not mock_adapter.is_running()
        assert mock_adapter.state == AdapterState.STOPPED
    
    @pytest.mark.asyncio
    async def test_error_rate_health_impact(self, mock_adapter):
        """Test that high error rate affects health status."""
        # Start adapter
        await mock_adapter.start()
        
        # Simulate high error rate
        mock_adapter.update_metrics(processed=1, failed=9)  # 90% error rate
        
        # Perform health check
        health_status = await mock_adapter._perform_health_check()
        
        # Should be unhealthy due to high error rate
        assert health_status == AdapterHealthStatus.DEGRADED
        
        # Stop adapter
        await mock_adapter.stop()


class TestAdapterExceptions:
    """Test adapter-specific exceptions."""
    
    def test_adapter_validation_error(self):
        """Test AdapterValidationError creation."""
        errors = ["Error 1", "Error 2"]
        exception = AdapterValidationError(
            "Validation failed",
            "test_adapter",
            errors
        )
        
        assert str(exception) == "Validation failed"
        assert exception.adapter_id == "test_adapter"
        assert exception.validation_errors == errors
    
    def test_adapter_lifecycle_error(self):
        """Test AdapterLifecycleError creation."""
        cause = RuntimeError("Original error")
        exception = AdapterLifecycleError(
            "Lifecycle failed",
            "test_adapter",
            AdapterState.ERROR,
            cause
        )
        
        assert str(exception) == "Lifecycle failed"
        assert exception.adapter_id == "test_adapter"
        assert exception.state == AdapterState.ERROR
        assert exception.cause == cause


if __name__ == "__main__":
    pytest.main([__file__, "-v"])