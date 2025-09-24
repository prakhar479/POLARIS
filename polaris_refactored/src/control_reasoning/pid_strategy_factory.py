"""
PID Strategy Factory

Factory functions and configuration support for creating and configuring
PID reactive strategies within the POLARIS adaptive controller.
"""

from typing import Dict, Any, List, Optional
from .pid_reactive_strategy import PIDReactiveStrategy, PIDReactiveConfig
from .pid_controller import PIDConfig
from ..infrastructure.observability import get_logger


class PIDStrategyFactory:
    """Factory for creating configured PID reactive strategies."""
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> PIDReactiveStrategy:
        """
        Create PID reactive strategy from configuration dictionary.
        
        Expected config format:
        {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 70.0,
                    "kp": 1.0,
                    "ki": 0.1,
                    "kd": 0.05,
                    "min_output": -10.0,
                    "max_output": 10.0,
                    "history_window": 10,
                    "sample_time": 1.0
                }
            ],
            "action_scaling_factor": 1.0,
            "max_concurrent_actions": 3,
            "priority_weights": {
                "cpu_usage": 1.0,
                "memory_usage": 0.8,
                "latency": 1.2
            },
            "enable_fallback": true
        }
        """
        logger = get_logger("polaris.pid_strategy_factory")
        
        try:
            # Parse PID controller configurations
            controllers = []
            for controller_config in config.get("controllers", []):
                pid_config = PIDConfig(
                    metric_name=controller_config["metric_name"],
                    setpoint=float(controller_config["setpoint"]),
                    kp=float(controller_config.get("kp", 1.0)),
                    ki=float(controller_config.get("ki", 0.1)),
                    kd=float(controller_config.get("kd", 0.05)),
                    min_output=float(controller_config.get("min_output", -10.0)),
                    max_output=float(controller_config.get("max_output", 10.0)),
                    history_window=int(controller_config.get("history_window", 10)),
                    sample_time=float(controller_config.get("sample_time", 1.0))
                )
                controllers.append(pid_config)
            
            # Parse strategy configuration
            strategy_config = PIDReactiveConfig(
                controllers=controllers,
                action_scaling_factor=float(config.get("action_scaling_factor", 1.0)),
                max_concurrent_actions=int(config.get("max_concurrent_actions", 3)),
                priority_weights=config.get("priority_weights", {}),
                enable_fallback=bool(config.get("enable_fallback", True)),
                fallback_thresholds=config.get("fallback_thresholds", {
                    "cpu": 0.85,
                    "memory": 0.85,
                    "latency": 0.85,
                    "error_rate": 0.1
                })
            )
            
            strategy = PIDReactiveStrategy(strategy_config)
            
            logger.info("PID strategy created from config", extra={
                "controller_count": len(controllers),
                "metrics": [c.metric_name for c in controllers]
            })
            
            return strategy
            
        except Exception as e:
            logger.error("Failed to create PID strategy from config", extra={
                "error": str(e),
                "config": config
            }, exc_info=e)
            raise
    
    @staticmethod
    def create_default_cpu_memory_strategy() -> PIDReactiveStrategy:
        """Create a default PID strategy for CPU and memory monitoring."""
        config = {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 70.0,
                    "kp": 1.0,
                    "ki": 0.1,
                    "kd": 0.05,
                    "min_output": -5.0,
                    "max_output": 10.0,
                    "history_window": 10,
                    "sample_time": 1.0
                },
                {
                    "metric_name": "memory_usage",
                    "setpoint": 75.0,
                    "kp": 0.8,
                    "ki": 0.05,
                    "kd": 0.02,
                    "min_output": -3.0,
                    "max_output": 8.0,
                    "history_window": 10,
                    "sample_time": 1.0
                }
            ],
            "action_scaling_factor": 1.0,
            "max_concurrent_actions": 2,
            "priority_weights": {
                "cpu_usage": 1.0,
                "memory_usage": 0.8
            },
            "enable_fallback": True
        }
        
        return PIDStrategyFactory.create_from_config(config)
    
    @staticmethod
    def create_web_service_strategy() -> PIDReactiveStrategy:
        """Create a PID strategy optimized for web service monitoring."""
        config = {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 65.0,
                    "kp": 1.2,
                    "ki": 0.15,
                    "kd": 0.08,
                    "min_output": -8.0,
                    "max_output": 12.0,
                    "history_window": 15,
                    "sample_time": 0.5
                },
                {
                    "metric_name": "memory_usage",
                    "setpoint": 70.0,
                    "kp": 0.9,
                    "ki": 0.08,
                    "kd": 0.03,
                    "min_output": -5.0,
                    "max_output": 10.0,
                    "history_window": 12,
                    "sample_time": 1.0
                },
                {
                    "metric_name": "response_time",
                    "setpoint": 200.0,  # 200ms
                    "kp": 0.5,
                    "ki": 0.02,
                    "kd": 0.1,
                    "min_output": -3.0,
                    "max_output": 8.0,
                    "history_window": 20,
                    "sample_time": 0.5
                }
            ],
            "action_scaling_factor": 1.2,
            "max_concurrent_actions": 3,
            "priority_weights": {
                "cpu_usage": 1.0,
                "memory_usage": 0.8,
                "response_time": 1.5
            },
            "enable_fallback": True
        }
        
        return PIDStrategyFactory.create_from_config(config)
    
    @staticmethod
    def create_database_strategy() -> PIDReactiveStrategy:
        """Create a PID strategy optimized for database monitoring."""
        config = {
            "controllers": [
                {
                    "metric_name": "cpu_usage",
                    "setpoint": 80.0,  # Databases can handle higher CPU
                    "kp": 0.8,
                    "ki": 0.05,
                    "kd": 0.02,
                    "min_output": -3.0,
                    "max_output": 6.0,
                    "history_window": 8,
                    "sample_time": 2.0
                },
                {
                    "metric_name": "memory_usage",
                    "setpoint": 85.0,  # Databases use memory for caching
                    "kp": 0.6,
                    "ki": 0.03,
                    "kd": 0.01,
                    "min_output": -2.0,
                    "max_output": 5.0,
                    "history_window": 6,
                    "sample_time": 2.0
                },
                {
                    "metric_name": "query_latency",
                    "setpoint": 50.0,  # 50ms for queries
                    "kp": 1.0,
                    "ki": 0.1,
                    "kd": 0.05,
                    "min_output": -4.0,
                    "max_output": 8.0,
                    "history_window": 15,
                    "sample_time": 1.0
                }
            ],
            "action_scaling_factor": 0.8,  # More conservative scaling
            "max_concurrent_actions": 2,
            "priority_weights": {
                "cpu_usage": 0.7,
                "memory_usage": 0.5,
                "query_latency": 1.3
            },
            "enable_fallback": True
        }
        
        return PIDStrategyFactory.create_from_config(config)


def create_pid_strategy_from_system_type(system_type: str) -> Optional[PIDReactiveStrategy]:
    """
    Create a PID strategy based on system type.
    
    Args:
        system_type: Type of system ("web_service", "database", "default", etc.)
        
    Returns:
        Configured PID strategy or None if system type not recognized
    """
    logger = get_logger("polaris.pid_strategy_factory")
    
    try:
        if system_type.lower() in ["web", "web_service", "api", "microservice"]:
            return PIDStrategyFactory.create_web_service_strategy()
        elif system_type.lower() in ["database", "db", "sql", "nosql"]:
            return PIDStrategyFactory.create_database_strategy()
        elif system_type.lower() in ["default", "generic", "basic"]:
            return PIDStrategyFactory.create_default_cpu_memory_strategy()
        else:
            logger.warning("Unknown system type for PID strategy", extra={
                "system_type": system_type
            })
            return None
            
    except Exception as e:
        logger.error("Failed to create PID strategy for system type", extra={
            "system_type": system_type,
            "error": str(e)
        }, exc_info=e)
        return None