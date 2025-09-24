"""
PID Reactive Control Strategy Implementation

Implements a PID-based reactive control strategy that extends the existing
ReactiveControlStrategy class. This strategy provides fast, mathematical
control responses to metric deviations using multiple PID controllers.

Key Features:
- Multiple PID controllers for different metrics
- Metric prioritization and action selection logic
- Configuration support for PID parameters
- Integration with existing observability framework
- Fallback to simple reactive behavior when PID fails
"""

import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .adaptive_controller import ReactiveControlStrategy, AdaptationNeed
from .pid_controller import PIDController, PIDConfig, MetricHistoryManager
from ..domain.models import AdaptationAction, MetricValue
from ..infrastructure.observability import get_logger, get_metrics_collector


@dataclass
class PIDReactiveConfig:
    """Configuration for PID Reactive Strategy."""
    controllers: List[PIDConfig] = field(default_factory=list)
    action_scaling_factor: float = 1.0
    max_concurrent_actions: int = 3
    priority_weights: Dict[str, float] = field(default_factory=dict)
    enable_fallback: bool = True
    fallback_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 0.85,
        "memory": 0.85,
        "latency": 0.85,
        "error_rate": 0.1
    })


class PIDReactiveStrategy(ReactiveControlStrategy):
    """
    PID-based Reactive Control Strategy.
    
    Extends ReactiveControlStrategy to provide mathematical PID control for
    fast reactive adaptation. Uses multiple PID controllers to monitor different
    metrics and generate precise adaptation actions based on control theory.
    
    The strategy maintains individual PID controllers for each configured metric
    and combines their outputs to generate adaptation actions. It includes
    metric prioritization, action scaling, and fallback mechanisms.
    """
    
    def __init__(self, config: Optional[PIDReactiveConfig] = None):
        """Initialize PID Reactive Strategy with configuration."""
        super().__init__()
        
        self.config = config or PIDReactiveConfig()
        self.logger = get_logger("polaris.pid_reactive_strategy")
        self.metrics = get_metrics_collector()
        
        # Initialize PID controllers
        self.pid_controllers: Dict[str, PIDController] = {}
        for pid_config in self.config.controllers:
            self.pid_controllers[pid_config.metric_name] = PIDController(pid_config)
        
        # Initialize metric history manager
        self.history_manager = MetricHistoryManager(window_size=100)
        
        # Action generation state
        self.last_action_time: Dict[str, float] = {}
        self.action_cooldown = 5.0  # seconds between actions for same metric
        
        self.logger.info("PID Reactive Strategy initialized", extra={
            "controller_count": len(self.pid_controllers),
            "monitored_metrics": list(self.pid_controllers.keys()),
            "action_scaling_factor": self.config.action_scaling_factor,
            "max_concurrent_actions": self.config.max_concurrent_actions
        })
    
    async def generate_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        """
        Generate adaptation actions using PID control logic.
        
        Args:
            system_id: ID of the system requiring adaptation
            current_state: Current system state including metrics
            adaptation_need: Identified adaptation need
            
        Returns:
            List of adaptation actions based on PID controller outputs
        """
        try:
            actions = await self._generate_pid_actions(system_id, current_state, adaptation_need)
            
            # Log PID action generation
            self.logger.info("PID actions generated", extra={
                "system_id": system_id,
                "action_count": len(actions),
                "adaptation_reason": adaptation_need.reason,
                "urgency": adaptation_need.urgency
            })
            
            # Update metrics
            self.metrics.increment_counter("pid_strategy_actions_generated", {
                "system_id": system_id,
                "action_count": str(len(actions))
            })
            
            return actions
            
        except Exception as e:
            self.logger.error("PID action generation failed", extra={
                "system_id": system_id,
                "error": str(e)
            }, exc_info=e)
            
            # Fallback to parent reactive strategy if enabled
            if self.config.enable_fallback:
                self.logger.info("Falling back to basic reactive strategy", extra={
                    "system_id": system_id
                })
                return await super().generate_actions(system_id, current_state, adaptation_need)
            
            return []
    
    async def _generate_pid_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        """Generate actions using PID controller outputs."""
        current_time = time.time()
        metrics = current_state.get("metrics", {})
        actions: List[AdaptationAction] = []
        
        # Update metric history
        self._update_metric_history(metrics)
        
        # Calculate PID outputs for each controller
        pid_outputs: Dict[str, float] = {}
        for metric_name, controller in self.pid_controllers.items():
            current_value = self._get_metric_value(metrics, metric_name)
            if current_value is not None:
                try:
                    output = controller.calculate_output(current_value, current_time)
                    pid_outputs[metric_name] = output
                    
                    self.logger.debug("PID output calculated", extra={
                        "metric_name": metric_name,
                        "current_value": current_value,
                        "setpoint": controller.config.setpoint,
                        "output": output
                    })
                    
                except Exception as e:
                    self.logger.warning("PID calculation failed for metric", extra={
                        "metric_name": metric_name,
                        "error": str(e)
                    })
        
        # Generate actions based on PID outputs
        prioritized_outputs = self._prioritize_outputs(pid_outputs)
        
        for metric_name, output in prioritized_outputs:
            # Skip if action cooldown hasn't elapsed
            if self._is_action_on_cooldown(metric_name, current_time):
                continue
            
            # Generate action based on output magnitude and direction
            action = self._create_action_from_output(
                system_id, metric_name, output, adaptation_need.urgency
            )
            
            if action:
                actions.append(action)
                self.last_action_time[metric_name] = current_time
                
                # Limit concurrent actions
                if len(actions) >= self.config.max_concurrent_actions:
                    break
        
        return actions
    
    def _update_metric_history(self, metrics: Dict[str, Any]) -> None:
        """Update metric history for all received metrics."""
        for name, value in metrics.items():
            if isinstance(value, MetricValue):
                self.history_manager.add_metric(value)
            else:
                # Convert raw values to MetricValue
                try:
                    metric_value = MetricValue(
                        name=name,
                        value=float(value) if isinstance(value, (int, float)) else value
                    )
                    self.history_manager.add_metric(metric_value)
                except (ValueError, TypeError):
                    self.logger.debug("Skipping non-numeric metric", extra={
                        "metric_name": name,
                        "value_type": type(value).__name__
                    })
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract numeric value from metrics dictionary."""
        # Try exact match first
        if metric_name in metrics:
            return self._extract_numeric_value(metrics[metric_name])
        
        # Try common variations
        variations = [
            f"{metric_name}_usage",
            f"{metric_name}_percent",
            f"{metric_name}_utilization",
            f"avg_{metric_name}",
            f"p95_{metric_name}",
            f"{metric_name}_p95"
        ]
        
        for variation in variations:
            if variation in metrics:
                return self._extract_numeric_value(metrics[variation])
        
        return None
    
    def _extract_numeric_value(self, value: Any) -> Optional[float]:
        """Extract numeric value from various metric value types."""
        try:
            if isinstance(value, MetricValue):
                return float(value.value)
            elif isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    def _prioritize_outputs(self, pid_outputs: Dict[str, float]) -> List[tuple[str, float]]:
        """Prioritize PID outputs based on magnitude and configured weights."""
        weighted_outputs = []
        
        for metric_name, output in pid_outputs.items():
            # Get priority weight (default to 1.0)
            weight = self.config.priority_weights.get(metric_name, 1.0)
            
            # Calculate priority score based on output magnitude and weight
            priority_score = abs(output) * weight
            
            weighted_outputs.append((priority_score, metric_name, output))
        
        # Sort by priority score (descending)
        weighted_outputs.sort(key=lambda x: x[0], reverse=True)
        
        # Return (metric_name, output) tuples
        return [(metric_name, output) for _, metric_name, output in weighted_outputs]
    
    def _is_action_on_cooldown(self, metric_name: str, current_time: float) -> bool:
        """Check if action for metric is on cooldown."""
        last_time = self.last_action_time.get(metric_name, 0)
        return (current_time - last_time) < self.action_cooldown
    
    def _create_action_from_output(
        self, 
        system_id: str, 
        metric_name: str, 
        output: float, 
        urgency: float
    ) -> Optional[AdaptationAction]:
        """Create adaptation action based on PID output."""
        # Skip if output is too small to be meaningful
        if abs(output) < 0.01:
            return None
        
        # Scale output by configuration factor
        scaled_output = output * self.config.action_scaling_factor
        
        # Determine action type and parameters based on metric and output
        action_type, parameters = self._determine_action_parameters(
            metric_name, scaled_output, urgency
        )
        
        if not action_type:
            return None
        
        # Calculate priority based on output magnitude and urgency
        priority = min(5, max(1, int(abs(scaled_output) * 3 + urgency * 2)))
        
        return AdaptationAction(
            action_id="",  # Will be auto-generated
            action_type=action_type,
            target_system=system_id,
            parameters=parameters,
            priority=priority,
            timeout_seconds=60
        )
    
    def _determine_action_parameters(
        self, 
        metric_name: str, 
        output: float, 
        urgency: float
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Determine action type and parameters based on metric and PID output."""
        
        # CPU-related actions
        if "cpu" in metric_name.lower():
            if output > 0:  # Need to reduce CPU usage
                if output > 2.0 or urgency > 0.8:
                    return "scale_out", {
                        "scale_factor": min(3, max(1.5, 1 + output * 0.5)),
                        "reason": f"High CPU usage (PID output: {output:.2f})"
                    }
                else:
                    return "optimize_resources", {
                        "cpu_limit_adjustment": -min(0.3, output * 0.1),
                        "reason": f"CPU optimization (PID output: {output:.2f})"
                    }
            elif output < -1.0:  # CPU usage is well below setpoint
                return "scale_in", {
                    "scale_factor": max(0.5, 1 + output * 0.1),
                    "reason": f"Low CPU usage (PID output: {output:.2f})"
                }
        
        # Memory-related actions
        elif "memory" in metric_name.lower() or "mem" in metric_name.lower():
            if output > 0:  # Need to reduce memory usage
                if output > 2.0 or urgency > 0.8:
                    return "scale_out", {
                        "scale_factor": min(2.5, max(1.3, 1 + output * 0.3)),
                        "reason": f"High memory usage (PID output: {output:.2f})"
                    }
                else:
                    return "optimize_resources", {
                        "memory_limit_adjustment": min(0.5, output * 0.2),
                        "reason": f"Memory optimization (PID output: {output:.2f})"
                    }
        
        # Latency-related actions
        elif "latency" in metric_name.lower() or "response_time" in metric_name.lower():
            if output > 0:  # Need to reduce latency
                if output > 1.5 or urgency > 0.7:
                    return "scale_out", {
                        "scale_factor": min(2.0, max(1.2, 1 + output * 0.4)),
                        "reason": f"High latency (PID output: {output:.2f})"
                    }
                else:
                    return "tune_qos", {
                        "qos_level": "high",
                        "cache_optimization": True,
                        "reason": f"Latency optimization (PID output: {output:.2f})"
                    }
        
        # Error rate actions
        elif "error" in metric_name.lower():
            if output > 0:  # Need to reduce error rate
                return "improve_reliability", {
                    "circuit_breaker_threshold": max(0.01, 0.05 - output * 0.01),
                    "retry_policy": "aggressive",
                    "reason": f"High error rate (PID output: {output:.2f})"
                }
        
        # Generic scaling action for unknown metrics
        elif output > 1.0:
            return "scale_out", {
                "scale_factor": min(2.0, max(1.1, 1 + output * 0.2)),
                "reason": f"Metric {metric_name} deviation (PID output: {output:.2f})"
            }
        
        return None, {}
    
    def add_pid_controller(self, config: PIDConfig) -> None:
        """Add a new PID controller for a metric."""
        controller = PIDController(config)
        self.pid_controllers[config.metric_name] = controller
        
        self.logger.info("PID controller added", extra={
            "metric_name": config.metric_name,
            "setpoint": config.setpoint
        })
    
    def remove_pid_controller(self, metric_name: str) -> bool:
        """Remove PID controller for a metric."""
        if metric_name in self.pid_controllers:
            del self.pid_controllers[metric_name]
            self.logger.info("PID controller removed", extra={
                "metric_name": metric_name
            })
            return True
        return False
    
    def update_controller_config(self, metric_name: str, **kwargs) -> bool:
        """Update configuration for a specific PID controller."""
        if metric_name not in self.pid_controllers:
            return False
        
        controller = self.pid_controllers[metric_name]
        
        # Update setpoint if provided
        if "setpoint" in kwargs:
            controller.update_setpoint(kwargs["setpoint"])
        
        # Update gains if provided
        gains = {}
        for gain in ["kp", "ki", "kd"]:
            if gain in kwargs:
                gains[gain] = kwargs[gain]
        
        if gains:
            controller.update_gains(**gains)
        
        return True
    
    def get_controller_status(self) -> Dict[str, Any]:
        """Get status information for all PID controllers."""
        status = {
            "strategy_type": "PIDReactiveStrategy",
            "controller_count": len(self.pid_controllers),
            "controllers": {}
        }
        
        for metric_name, controller in self.pid_controllers.items():
            status["controllers"][metric_name] = controller.get_tuning_info()
        
        return status
    
    def reset_controllers(self) -> None:
        """Reset all PID controllers."""
        for controller in self.pid_controllers.values():
            controller.reset()
        
        self.last_action_time.clear()
        self.logger.info("All PID controllers reset")