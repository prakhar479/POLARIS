"""
Tests for PID Strategy Integration with Adaptive Controller

Tests the integration of PID reactive strategy with the existing
POLARIS adaptive controller framework.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from polaris_refactored.src.control_reasoning.adaptive_controller import (
    PolarisAdaptiveController, AdaptationNeed
)
from polaris_refactored.src.control_reasoning.pid_reactive_strategy import PIDReactiveStrategy
from polaris_refactored.src.control_reasoning.pid_strategy_factory import PIDStrategyFactory
from polaris_refactored.src.domain.models import SystemState, HealthStatus, MetricValue
from polaris_refactored.src.framework.events import TelemetryEvent


class TestPIDIntegration:
    """Test PID strategy integration with adaptive controller."""
    
    def test_adaptive_controller_with_pid_strategy_enabled(self):
        """Test that adaptive controller can be initialized with PID strategy."""
        # Create controller with PID strategy enabled
        controller = PolarisAdaptiveController(
            enable_pid_strategy=True
        )
        
        # Verify PID strategy is in the strategies list
        pid_strategies = [s for s in controller._control_strategies 
                         if isinstance(s, PIDReactiveStrategy)]
        assert len(pid_strategies) == 1
        
        # Verify the PID strategy has default controllers
        pid_strategy = pid_strategies[0]
        assert len(pid_strategy.pid_controllers) == 2  # CPU and memory by default
        assert "cpu_usage" in pid_strategy.pid_controllers
        assert "memory_usage" in pid_strategy.pid_controllers
    
    def test_adaptive_controller_with_custom_pid_config(self):
        """Test adaptive controller with custom PID configuration."""
        pid_config = {
            "controllers": [
                {
                    "metric_name": "response_time",
                    "setpoint": 100.0,
                    "kp": 0.5,
                    "ki": 0.1,
                    "kd": 0.02,
                    "min_output": -5.0,
                    "max_output": 10.0
                }
            ],
            "action_scaling_factor": 1.5,
            "max_concurrent_actions": 2
        }
        
        controller = PolarisAdaptiveController(
            enable_pid_strategy=True,
            pid_config=pid_config
        )
        
        # Verify custom configuration is applied
        pid_strategies = [s for s in controller._control_strategies 
                         if isinstance(s, PIDReactiveStrategy)]
        assert len(pid_strategies) == 1
        
        pid_strategy = pid_strategies[0]
        assert len(pid_strategy.pid_controllers) == 1
        assert "response_time" in pid_strategy.pid_controllers
        assert pid_strategy.config.action_scaling_factor == 1.5
        assert pid_strategy.config.max_concurrent_actions == 2
    
    @pytest.mark.asyncio
    async def test_pid_strategy_selection_in_controller(self):
        """Test that controller selects PID strategy for reactive scenarios."""
        controller = PolarisAdaptiveController(
            enable_pid_strategy=True
        )
        
        # Create adaptation context
        adaptation_need = AdaptationNeed(
            system_id="test_system",
            is_needed=True,
            reason="High CPU",
            urgency=0.8
        )
        
        current_state = {
            "metrics": {
                "cpu_usage": MetricValue(name="cpu_usage", value=85.0)
            }
        }
        
        # Test strategy selection
        strategy = await controller.select_control_strategy(
            "test_system", 
            {"adaptation_need": adaptation_need, "current_state": current_state}
        )
        
        # Should select PID strategy for reactive scenario
        assert isinstance(strategy, PIDReactiveStrategy)
    
    @pytest.mark.asyncio
    async def test_end_to_end_pid_adaptation_flow(self):
        """Test complete adaptation flow with PID strategy."""
        # Mock dependencies
        mock_event_bus = Mock()
        mock_event_bus.publish_adaptation_needed = AsyncMock()
        
        # Create controller with PID strategy
        controller = PolarisAdaptiveController(
            enable_pid_strategy=True,
            event_bus=mock_event_bus
        )
        
        # Create telemetry with high CPU usage
        system_state = SystemState(
            system_id="test_system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue(name="cpu_usage", value=90.0),
                "memory_usage": MetricValue(name="memory_usage", value=60.0)
            },
            health_status=HealthStatus.WARNING
        )
        
        telemetry = TelemetryEvent(system_state=system_state)
        
        # Process telemetry
        await controller.process_telemetry(telemetry)
        
        # Verify adaptation event was published
        mock_event_bus.publish_adaptation_needed.assert_called_once()
        
        # Get the published event
        call_args = mock_event_bus.publish_adaptation_needed.call_args[0][0]
        assert call_args.system_id == "test_system"
        assert len(call_args.suggested_actions) > 0
        
        # Verify action is related to CPU scaling
        actions = call_args.suggested_actions
        cpu_actions = [a for a in actions if "cpu" in a.parameters.get("reason", "").lower()]
        assert len(cpu_actions) > 0
    
    def test_pid_strategy_factory_integration(self):
        """Test PID strategy factory methods."""
        # Test default strategy creation
        default_strategy = PIDStrategyFactory.create_default_cpu_memory_strategy()
        assert isinstance(default_strategy, PIDReactiveStrategy)
        assert len(default_strategy.pid_controllers) == 2
        
        # Test web service strategy creation
        web_strategy = PIDStrategyFactory.create_web_service_strategy()
        assert isinstance(web_strategy, PIDReactiveStrategy)
        assert len(web_strategy.pid_controllers) == 3  # CPU, memory, response_time
        
        # Test database strategy creation
        db_strategy = PIDStrategyFactory.create_database_strategy()
        assert isinstance(db_strategy, PIDReactiveStrategy)
        assert len(db_strategy.pid_controllers) == 3  # CPU, memory, query_latency
    
    def test_pid_controller_configuration_validation(self):
        """Test PID controller configuration validation."""
        # Test valid configuration
        valid_config = {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 70.0,
                    "kp": 1.0,
                    "ki": 0.1,
                    "kd": 0.05,
                    "min_output": -10.0,
                    "max_output": 10.0
                }
            ]
        }
        
        strategy = PIDStrategyFactory.create_from_config(valid_config)
        assert isinstance(strategy, PIDReactiveStrategy)
        
        # Test invalid configuration (negative gains)
        invalid_config = {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 70.0,
                    "kp": -1.0,  # Invalid negative gain
                    "ki": 0.1,
                    "kd": 0.05,
                    "min_output": -10.0,
                    "max_output": 10.0
                }
            ]
        }
        
        with pytest.raises(ValueError):
            PIDStrategyFactory.create_from_config(invalid_config)
    
    @pytest.mark.asyncio
    async def test_pid_strategy_fallback_behavior(self):
        """Test PID strategy fallback to basic reactive behavior."""
        # Create PID strategy with fallback enabled
        strategy = PIDStrategyFactory.create_default_cpu_memory_strategy()
        assert strategy.config.enable_fallback is True
        
        # Create adaptation need
        adaptation_need = AdaptationNeed(
            system_id="test_system",
            is_needed=True,
            reason="Test",
            urgency=0.5
        )
        
        # Test with valid metrics
        current_state = {
            "metrics": {
                "cpu_usage": MetricValue(name="cpu_usage", value=85.0)
            }
        }
        
        actions = await strategy.generate_actions(
            "test_system", current_state, adaptation_need
        )
        
        # Should generate actions (either PID or fallback)
        assert isinstance(actions, list)
    
    def test_controller_status_and_monitoring(self):
        """Test PID controller status and monitoring capabilities."""
        strategy = PIDStrategyFactory.create_default_cpu_memory_strategy()
        
        # Get controller status
        status = strategy.get_controller_status()
        
        assert status["strategy_type"] == "PIDReactiveStrategy"
        assert status["controller_count"] == 2
        assert "controllers" in status
        assert "cpu_usage" in status["controllers"]
        assert "memory_usage" in status["controllers"]
        
        # Verify controller tuning info
        cpu_info = status["controllers"]["cpu_usage"]
        assert "metric_name" in cpu_info
        assert "setpoint" in cpu_info
        assert "gains" in cpu_info
        assert "output_bounds" in cpu_info
        assert "state" in cpu_info


if __name__ == "__main__":
    pytest.main([__file__])