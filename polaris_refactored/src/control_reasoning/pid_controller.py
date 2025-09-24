"""
PID Controller Implementation

Implements a Proportional-Integral-Derivative (PID) controller for reactive control
in the POLARIS adaptation framework. The PID controller provides mathematical
control responses to metric deviations with configurable parameters.

Key Features:
- Proportional, integral, and derivative term calculations
- Sliding window history management for metrics
- Output bounding and saturation handling
- Multiple metric support with individual PID parameters
- Integration with existing observability framework
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timezone

from ..infrastructure.observability import get_logger
from ..domain.models import MetricValue


@dataclass
class PIDConfig:
    """Configuration for a single PID controller."""
    metric_name: str
    setpoint: float
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    min_output: float
    max_output: float
    history_window: int = 10
    sample_time: float = 1.0  # seconds


@dataclass
class PIDState:
    """Internal state for a PID controller."""
    last_error: float = 0.0
    integral_sum: float = 0.0
    last_time: Optional[float] = None
    error_history: Deque[float] = None
    time_history: Deque[float] = None
    
    def __post_init__(self):
        if self.error_history is None:
            object.__setattr__(self, 'error_history', deque())
        if self.time_history is None:
            object.__setattr__(self, 'time_history', deque())


class PIDController:
    """
    PID Controller for reactive adaptation control.
    
    Implements a discrete PID controller that calculates control outputs based on
    the error between a setpoint and the current metric value. The controller
    maintains history for derivative and integral calculations and provides
    output bounding to prevent system instability.
    
    Mathematical Implementation:
    - Proportional term: Kp * error
    - Integral term: Ki * sum(errors) * dt
    - Derivative term: Kd * (error - last_error) / dt
    - Output: P + I + D (bounded by min/max limits)
    """
    
    def __init__(self, config: PIDConfig):
        """Initialize PID controller with configuration."""
        self.config = config
        self.state = PIDState()
        self.logger = get_logger(f"polaris.pid_controller.{config.metric_name}")
        
        # Validate configuration
        if config.kp < 0 or config.ki < 0 or config.kd < 0:
            raise ValueError("PID gains must be non-negative")
        if config.min_output >= config.max_output:
            raise ValueError("min_output must be less than max_output")
        if config.history_window <= 0:
            raise ValueError("history_window must be positive")
        if config.sample_time <= 0:
            raise ValueError("sample_time must be positive")
        
        self.logger.info("PID controller initialized", extra={
            "metric_name": config.metric_name,
            "setpoint": config.setpoint,
            "kp": config.kp,
            "ki": config.ki,
            "kd": config.kd,
            "output_range": f"[{config.min_output}, {config.max_output}]"
        })
    
    def calculate_output(self, current_value: float, current_time: Optional[float] = None) -> float:
        """
        Calculate PID control output for the current metric value.
        
        Args:
            current_value: Current value of the controlled metric
            current_time: Current timestamp (seconds since epoch). If None, uses current time.
            
        Returns:
            Control output bounded by min_output and max_output
        """
        if current_time is None:
            current_time = time.time()

        # Calculate error
        # Use current_value - setpoint so that a positive output means the metric is
        # above the desired setpoint (e.g., high CPU) which aligns with strategy
        # expectations where positive PID output => need to reduce metric (scale_out)
        error = current_value - self.config.setpoint

        # Initialize on first call
        if self.state.last_time is None:
            self.state.last_time = current_time
            self.state.last_error = error
            self._update_history(error, current_time)
            
            # Return proportional term only for first calculation
            # Note: using new error sign convention
            output = self.config.kp * error
            return self._bound_output(output)
        
        # Calculate time delta
        dt = current_time - self.state.last_time
        
        # Skip calculation if time delta is too small or negative
        if dt <= 0:
            self.logger.warning("Invalid time delta for PID calculation", extra={
                "dt": dt,
                "current_time": current_time,
                "last_time": self.state.last_time
            })
            return 0.0
        
        # Only calculate if enough time has passed (respects sample_time)
        if dt < self.config.sample_time:
            return self._get_last_output()
        
        # Proportional term
        proportional = self.config.kp * error
        
        # Integral term (with windup protection)
        self.state.integral_sum += error * dt
        integral = self.config.ki * self.state.integral_sum
        
        # Derivative term
        derivative = 0.0
        if dt > 0:
            derivative = self.config.kd * (error - self.state.last_error) / dt
        
        # Calculate total output
        output = proportional + integral + derivative
        
        # Apply output bounds and handle integral windup
        bounded_output = self._bound_output(output)
        
        # Integral windup protection: reduce integral sum if output is saturated
        if output != bounded_output and self.config.ki > 0:
            # Back-calculate what integral sum should be to stay within bounds
            max_integral = (bounded_output - proportional - derivative) / self.config.ki
            self.state.integral_sum = max_integral
        
        # Update state
        self.state.last_error = error
        self.state.last_time = current_time
        self._update_history(error, current_time)
        
        self.logger.debug("PID calculation completed", extra={
            "metric_name": self.config.metric_name,
            "current_value": current_value,
            "error": error,
            "proportional": proportional,
            "integral": integral,
            "derivative": derivative,
            "output": output,
            "bounded_output": bounded_output,
            "dt": dt
        })
        
        return bounded_output
    
    def _bound_output(self, output: float) -> float:
        """Apply output bounds to prevent system instability."""
        return max(self.config.min_output, min(self.config.max_output, output))
    
    def _get_last_output(self) -> float:
        """Get the last calculated output (for when sample time hasn't elapsed)."""
        # Return current PID calculation without updating state
        error = self.state.last_error
        proportional = self.config.kp * error
        integral = self.config.ki * self.state.integral_sum
        derivative = 0.0  # No derivative when time hasn't changed
        return self._bound_output(proportional + integral + derivative)
    
    def _update_history(self, error: float, timestamp: float) -> None:
        """Update error and time history with sliding window management."""
        self.state.error_history.append(error)
        self.state.time_history.append(timestamp)
        
        # Maintain sliding window
        while len(self.state.error_history) > self.config.history_window:
            self.state.error_history.popleft()
            self.state.time_history.popleft()
    
    def reset(self) -> None:
        """Reset PID controller state."""
        self.state = PIDState()
        self.logger.info("PID controller reset", extra={
            "metric_name": self.config.metric_name
        })
    
    def get_tuning_info(self) -> Dict[str, Any]:
        """Get current tuning information for debugging and monitoring."""
        return {
            "metric_name": self.config.metric_name,
            "setpoint": self.config.setpoint,
            "gains": {
                "kp": self.config.kp,
                "ki": self.config.ki,
                "kd": self.config.kd
            },
            "output_bounds": {
                "min": self.config.min_output,
                "max": self.config.max_output
            },
            "state": {
                "last_error": self.state.last_error,
                "integral_sum": self.state.integral_sum,
                "history_length": len(self.state.error_history)
            }
        }
    
    def update_setpoint(self, new_setpoint: float) -> None:
        """Update the setpoint for the PID controller."""
        old_setpoint = self.config.setpoint
        object.__setattr__(self.config, 'setpoint', new_setpoint)
        
        self.logger.info("PID setpoint updated", extra={
            "metric_name": self.config.metric_name,
            "old_setpoint": old_setpoint,
            "new_setpoint": new_setpoint
        })
    
    def update_gains(self, kp: Optional[float] = None, ki: Optional[float] = None, kd: Optional[float] = None) -> None:
        """Update PID gains."""
        changes = {}
        
        if kp is not None and kp >= 0:
            changes["kp"] = {"old": self.config.kp, "new": kp}
            object.__setattr__(self.config, 'kp', kp)
        
        if ki is not None and ki >= 0:
            changes["ki"] = {"old": self.config.ki, "new": ki}
            object.__setattr__(self.config, 'ki', ki)
            # Reset integral sum when Ki changes to prevent windup
            self.state.integral_sum = 0.0
        
        if kd is not None and kd >= 0:
            changes["kd"] = {"old": self.config.kd, "new": kd}
            object.__setattr__(self.config, 'kd', kd)
        
        if changes:
            self.logger.info("PID gains updated", extra={
                "metric_name": self.config.metric_name,
                "changes": changes
            })


class MetricHistoryManager:
    """
    Manages sliding window history for multiple metrics.
    
    Provides efficient storage and retrieval of metric history for PID controllers
    and other components that need historical data for calculations.
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize history manager with specified window size."""
        self.window_size = window_size
        self.histories: Dict[str, Deque[MetricValue]] = {}
        self.logger = get_logger("polaris.metric_history_manager")
    
    def add_metric(self, metric: MetricValue) -> None:
        """Add a metric value to the history."""
        metric_name = metric.name
        
        if metric_name not in self.histories:
            self.histories[metric_name] = deque(maxlen=self.window_size)
        
        self.histories[metric_name].append(metric)
        
        self.logger.debug("Metric added to history", extra={
            "metric_name": metric_name,
            "value": metric.value,
            "history_length": len(self.histories[metric_name])
        })
    
    def get_history(self, metric_name: str, count: Optional[int] = None) -> List[MetricValue]:
        """Get metric history for a specific metric."""
        if metric_name not in self.histories:
            return []
        
        history = list(self.histories[metric_name])
        
        if count is not None and count > 0:
            return history[-count:]
        
        return history
    
    def get_latest_value(self, metric_name: str) -> Optional[float]:
        """Get the latest numeric value for a metric."""
        history = self.get_history(metric_name, count=1)
        if not history:
            return None
        
        try:
            return float(history[0].value)
        except (ValueError, TypeError):
            return None
    
    def clear_history(self, metric_name: Optional[str] = None) -> None:
        """Clear history for a specific metric or all metrics."""
        if metric_name is not None:
            if metric_name in self.histories:
                self.histories[metric_name].clear()
                self.logger.info("Metric history cleared", extra={"metric_name": metric_name})
        else:
            self.histories.clear()
            self.logger.info("All metric histories cleared")
    
    def get_metrics_with_history(self) -> List[str]:
        """Get list of metrics that have history data."""
        return list(self.histories.keys())