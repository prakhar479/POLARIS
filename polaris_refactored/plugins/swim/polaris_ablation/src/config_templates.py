"""
Configuration Templates for SWIM POLARIS Adaptation System

Provides templates and utilities for generating environment-specific
configurations and managing SWIM connection parameters.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SwimConnectionConfig:
    """SWIM connection configuration parameters."""
    host: str = "localhost"
    port: int = 4242
    timeout: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 5.0
    collection_interval: float = 10.0


@dataclass
class AdaptationThresholds:
    """Adaptation threshold configuration."""
    response_time_warning: float = 800.0
    response_time_critical: float = 1200.0
    utilization_low: float = 0.2
    utilization_high: float = 0.8
    utilization_critical: float = 0.95
    dimmer_min: float = 0.1
    dimmer_max: float = 1.0


class ConfigurationTemplateGenerator:
    """
    Generates configuration templates for different environments
    and manages SWIM-specific parameters.
    """
    
    def __init__(self):
        self.environment_defaults = {
            Environment.DEVELOPMENT: {
                "logging_level": "DEBUG",
                "collection_interval": 5.0,
                "mape_k_interval": 15.0,
                "study_duration": 600,
                "max_servers": 5,
                "aggressive_thresholds": True
            },
            Environment.TESTING: {
                "logging_level": "WARNING",
                "collection_interval": 1.0,
                "mape_k_interval": 5.0,
                "study_duration": 60,
                "max_servers": 3,
                "aggressive_thresholds": False
            },
            Environment.STAGING: {
                "logging_level": "INFO",
                "collection_interval": 15.0,
                "mape_k_interval": 30.0,
                "study_duration": 1800,
                "max_servers": 10,
                "aggressive_thresholds": False
            },
            Environment.PRODUCTION: {
                "logging_level": "INFO",
                "collection_interval": 30.0,
                "mape_k_interval": 60.0,
                "study_duration": 7200,
                "max_servers": 20,
                "aggressive_thresholds": False
            }
        }
    
    def generate_base_config(self, 
                           environment: Environment,
                           swim_config: Optional[SwimConnectionConfig] = None,
                           thresholds: Optional[AdaptationThresholds] = None) -> Dict[str, Any]:
        """
        Generate base configuration for specified environment.
        
        Args:
            environment: Target environment
            swim_config: SWIM connection configuration
            thresholds: Adaptation thresholds
            
        Returns:
            Configuration dictionary
        """
        if swim_config is None:
            swim_config = SwimConnectionConfig()
        
        if thresholds is None:
            thresholds = AdaptationThresholds()
        
        env_defaults = self.environment_defaults[environment]
        
        # Adjust thresholds based on environment
        if env_defaults["aggressive_thresholds"]:
            thresholds.response_time_warning *= 0.8
            thresholds.response_time_critical *= 0.8
            thresholds.utilization_high *= 0.9
        
        config = {
            "framework": {
                "name": f"swim-polaris-adaptation-{environment.value}",
                "version": "1.0.0" + ("" if environment == Environment.PRODUCTION else f"-{environment.value}"),
                "description": f"{environment.value.title()} environment for SWIM POLARIS adaptation system"
            },
            
            "logging": {
                "level": env_defaults["logging_level"],
                "format": "json" if environment in [Environment.STAGING, Environment.PRODUCTION] else "detailed",
                "handlers": self._get_logging_handlers(environment)
            },
            
            "message_bus": {
                "nats": self._get_nats_config(environment)
            },
            
            "managed_systems": [{
                "system_id": f"swim-{environment.value}",
                "connector_type": "swim",
                "enabled": True,
                "config": {
                    "system_name": f"swim-{environment.value}",
                    "connection": {
                        "host": swim_config.host,
                        "port": swim_config.port
                    },
                    "implementation": {
                        "timeout": swim_config.timeout,
                        "max_retries": swim_config.max_retries,
                        "retry_base_delay": swim_config.retry_base_delay,
                        "retry_max_delay": swim_config.retry_max_delay,
                        "collection_interval": env_defaults["collection_interval"]
                    }
                }
            }],
            
            "adaptation": {
                "strategies": self._get_adaptation_strategies(environment),
                "thresholds": {
                    "response_time": {
                        "warning": thresholds.response_time_warning,
                        "critical": thresholds.response_time_critical
                    },
                    "server_utilization": {
                        "low": thresholds.utilization_low,
                        "high": thresholds.utilization_high,
                        "critical": thresholds.utilization_critical
                    },
                    "dimmer": {
                        "min": thresholds.dimmer_min,
                        "max": thresholds.dimmer_max
                    }
                },
                "constraints": {
                    "min_servers": 1 if environment != Environment.PRODUCTION else 2,
                    "max_servers": env_defaults["max_servers"],
                    "server_change_limit": 1 if environment == Environment.TESTING else 2,
                    "dimmer_change_limit": 0.1 if environment == Environment.PRODUCTION else 0.2
                }
            },
            
            "digital_twin": self._get_digital_twin_config(environment),
            "control_reasoning": self._get_control_reasoning_config(environment, env_defaults),
            "performance": self._get_performance_config(environment),
            "ablation": self._get_ablation_config(environment, env_defaults)
        }
        
        # Add environment-specific configurations
        if environment == Environment.PRODUCTION:
            config.update(self._get_production_specific_config())
        elif environment == Environment.DEVELOPMENT:
            config.update(self._get_development_specific_config())
        
        return config
    
    def _get_logging_handlers(self, environment: Environment) -> list:
        """Get logging handlers for environment."""
        handlers = [{"type": "console", "level": "INFO"}]
        
        if environment == Environment.TESTING:
            handlers[0]["level"] = "ERROR"
        elif environment == Environment.DEVELOPMENT:
            handlers[0]["level"] = "DEBUG"
        
        # File handler
        log_file = f"logs/swim_polaris_{environment.value}.log"
        handlers.append({
            "type": "file",
            "path": log_file,
            "level": "DEBUG" if environment == Environment.DEVELOPMENT else "INFO",
            "max_size_mb": 10 if environment == Environment.TESTING else 100,
            "backup_count": 1 if environment == Environment.TESTING else 5
        })
        
        # Production gets syslog
        if environment == Environment.PRODUCTION:
            handlers.append({
                "type": "syslog",
                "level": "WARNING",
                "facility": "local0"
            })
        
        return handlers
    
    def _get_nats_config(self, environment: Environment) -> Dict[str, Any]:
        """Get NATS configuration for environment."""
        base_config = {
            "name": f"swim-polaris-{environment.value}-client",
            "connection_timeout": 10.0,
            "reconnect_attempts": 5,
            "reconnect_delay": 2.0
        }
        
        if environment == Environment.PRODUCTION:
            base_config.update({
                "servers": [
                    "nats://nats-1.production:4222",
                    "nats://nats-2.production:4222",
                    "nats://nats-3.production:4222"
                ],
                "connection_timeout": 30.0,
                "reconnect_attempts": 10,
                "reconnect_delay": 5.0,
                "max_reconnect_delay": 60.0,
                "ping_interval": 120.0,
                "max_pings_outstanding": 2
            })
        elif environment == Environment.TESTING:
            base_config.update({
                "servers": ["nats://localhost:4223"],
                "connection_timeout": 2.0,
                "reconnect_attempts": 1,
                "reconnect_delay": 0.5
            })
        else:
            base_config["servers"] = ["nats://localhost:4222"]
        
        return base_config
    
    def _get_adaptation_strategies(self, environment: Environment) -> Dict[str, Any]:
        """Get adaptation strategies configuration."""
        if environment == Environment.TESTING:
            return {
                "reactive": {
                    "enabled": True,
                    "weight": 1.0,
                    "response_time_threshold": 500.0,
                    "utilization_high_threshold": 0.8,
                    "utilization_low_threshold": 0.3
                },
                "predictive": {
                    "enabled": False,
                    "weight": 0.0
                }
            }
        
        return {
            "reactive": {
                "enabled": True,
                "weight": 0.4 if environment != Environment.PRODUCTION else 0.3,
                "response_time_threshold": 800.0 if environment == Environment.DEVELOPMENT else 1000.0,
                "utilization_high_threshold": 0.7 if environment == Environment.DEVELOPMENT else 0.8,
                "utilization_low_threshold": 0.2 if environment == Environment.DEVELOPMENT else 0.25
            },
            "predictive": {
                "enabled": True,
                "weight": 0.6 if environment != Environment.PRODUCTION else 0.7,
                "prediction_window": 120.0 if environment == Environment.DEVELOPMENT else 600.0,
                "trend_analysis_window": 300.0 if environment == Environment.DEVELOPMENT else 1800.0,
                "confidence_threshold": 0.6 if environment == Environment.DEVELOPMENT else 0.8
            }
        }
    
    def _get_digital_twin_config(self, environment: Environment) -> Dict[str, Any]:
        """Get digital twin configuration."""
        config = {
            "world_model": {
                "enabled": True,
                "type": "statistical",
                "update_interval": 15.0 if environment == Environment.DEVELOPMENT else 60.0,
                "prediction_horizon": 120.0 if environment == Environment.DEVELOPMENT else 1800.0,
                "confidence_threshold": 0.6 if environment == Environment.DEVELOPMENT else 0.8
            },
            "knowledge_base": {
                "enabled": True,
                "max_entries": 1000 if environment == Environment.DEVELOPMENT else 10000,
                "similarity_threshold": 0.7 if environment == Environment.DEVELOPMENT else 0.85,
                "cleanup_interval": 1800.0 if environment == Environment.DEVELOPMENT else 7200.0
            },
            "learning_engine": {
                "enabled": environment != Environment.TESTING,
                "learning_rate": 0.02 if environment == Environment.DEVELOPMENT else 0.005,
                "batch_size": 16 if environment == Environment.DEVELOPMENT else 64,
                "update_interval": 30.0 if environment == Environment.DEVELOPMENT else 300.0,
                "pattern_recognition": True
            }
        }
        
        # Production gets composite world model
        if environment == Environment.PRODUCTION:
            config["world_model"].update({
                "type": "composite",
                "models": [
                    {"type": "statistical", "weight": 0.4},
                    {"type": "neural_network", "weight": 0.6}
                ]
            })
            config["knowledge_base"].update({
                "max_entries": 100000,
                "persistence": True,
                "backup_interval": 86400.0
            })
        
        return config
    
    def _get_control_reasoning_config(self, environment: Environment, env_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Get control and reasoning configuration."""
        config = {
            "adaptive_controller": {
                "enabled": True,
                "mape_k_interval": env_defaults["mape_k_interval"],
                "max_concurrent_adaptations": 1 if environment == Environment.TESTING else 2,
                "adaptation_cooldown": 10.0 if environment == Environment.TESTING else 60.0
            },
            "reasoning_engine": {
                "enabled": True,
                "confidence_threshold": 0.8 if environment == Environment.TESTING else 0.6,
                "result_fusion": "first_valid" if environment == Environment.TESTING else "weighted_average"
            }
        }
        
        # Configure reasoning strategies based on environment
        if environment == Environment.TESTING:
            config["reasoning_engine"]["strategies"] = {
                "statistical": {"enabled": True, "weight": 1.0},
                "causal": {"enabled": False, "weight": 0.0},
                "experience_based": {"enabled": False, "weight": 0.0}
            }
        else:
            config["reasoning_engine"]["strategies"] = {
                "statistical": {"enabled": True, "weight": 0.3},
                "causal": {"enabled": True, "weight": 0.4},
                "experience_based": {"enabled": True, "weight": 0.3}
            }
        
        # Production gets additional safety features
        if environment == Environment.PRODUCTION:
            config["adaptive_controller"].update({
                "adaptation_cooldown": 300.0,
                "safety_checks": True
            })
            config["reasoning_engine"].update({
                "confidence_threshold": 0.75,
                "result_fusion": "weighted_consensus",
                "validation_enabled": True
            })
        
        return config
    
    def _get_performance_config(self, environment: Environment) -> Dict[str, Any]:
        """Get performance monitoring configuration."""
        return {
            "metrics_collection": {
                "enabled": True,
                "collection_interval": 1.0 if environment == Environment.TESTING else 30.0,
                "retention_period": 300 if environment == Environment.TESTING else 86400
            },
            "adaptation_tracking": {
                "enabled": True,
                "track_success_rate": True,
                "track_response_time": True,
                "track_impact": environment != Environment.TESTING
            },
            "system_monitoring": {
                "enabled": environment not in [Environment.TESTING],
                "cpu_monitoring": True,
                "memory_monitoring": True,
                "network_monitoring": environment == Environment.PRODUCTION
            }
        }
    
    def _get_ablation_config(self, environment: Environment, env_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Get ablation study configuration."""
        return {
            "description": f"{environment.value.title()} configuration with all components enabled",
            "components": {
                "world_model": True,
                "knowledge_base": True,
                "learning_engine": environment != Environment.TESTING,
                "statistical_reasoning": True,
                "causal_reasoning": environment != Environment.TESTING,
                "experience_based_reasoning": environment != Environment.TESTING,
                "reactive_strategy": True,
                "predictive_strategy": environment != Environment.TESTING
            },
            "study_parameters": {
                "duration": env_defaults["study_duration"],
                "warmup_period": 5 if environment == Environment.TESTING else 300,
                "cooldown_period": 5 if environment == Environment.TESTING else 300,
                "metrics_export_interval": 10 if environment == Environment.TESTING else 60
            }
        }
    
    def _get_production_specific_config(self) -> Dict[str, Any]:
        """Get production-specific configuration additions."""
        return {
            "data_storage": {
                "backends": {
                    "default": {
                        "type": "postgresql",
                        "connection_string": "${SWIM_POLARIS_DB_URL}",
                        "pool_size": 20,
                        "max_overflow": 30
                    },
                    "metrics": {
                        "type": "timeseries",
                        "connection_string": "${SWIM_POLARIS_METRICS_DB_URL}",
                        "retention_hours": 168
                    }
                }
            },
            "security": {
                "enabled": True,
                "authentication": {"enabled": True, "method": "jwt"},
                "authorization": {"enabled": True, "rbac": True},
                "encryption": {"enabled": True, "tls_version": "1.3"}
            },
            "observability": {
                "enable_metrics": True,
                "enable_tracing": True,
                "jaeger_endpoint": "${JAEGER_ENDPOINT}",
                "prometheus_endpoint": "${PROMETHEUS_ENDPOINT}"
            }
        }
    
    def _get_development_specific_config(self) -> Dict[str, Any]:
        """Get development-specific configuration additions."""
        return {
            "data_storage": {
                "backends": {
                    "default": {"type": "in_memory"},
                    "metrics": {"type": "in_memory", "retention_hours": 24}
                }
            }
        }
    
    def save_config_template(self, 
                           config: Dict[str, Any], 
                           output_path: str, 
                           format: str = "yaml") -> None:
        """Save configuration template to file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
            format: Output format ('yaml' or 'json')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                import json
                json.dump(config, f, indent=2)
    
    def generate_all_environments(self, output_dir: str) -> None:
        """Generate configuration templates for all environments.
        
        Args:
            output_dir: Directory to save configuration files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for environment in Environment:
            config = self.generate_base_config(environment)
            config_file = output_path / f"{environment.value}_config.yaml"
            self.save_config_template(config, str(config_file))
            print(f"Generated {environment.value} configuration: {config_file}")


def generate_swim_connection_config(host: str = "localhost", 
                                  port: int = 4242,
                                  environment: str = "development") -> SwimConnectionConfig:
    """Generate SWIM connection configuration for environment.
    
    Args:
        host: SWIM host
        port: SWIM port
        environment: Target environment
        
    Returns:
        SwimConnectionConfig instance
    """
    env = Environment(environment)
    
    if env == Environment.PRODUCTION:
        return SwimConnectionConfig(
            host=host,
            port=port,
            timeout=60.0,
            max_retries=5,
            retry_base_delay=2.0,
            retry_max_delay=30.0,
            collection_interval=30.0
        )
    elif env == Environment.TESTING:
        return SwimConnectionConfig(
            host=host,
            port=port,
            timeout=5.0,
            max_retries=1,
            retry_base_delay=0.1,
            retry_max_delay=0.5,
            collection_interval=1.0
        )
    else:  # Development/Staging
        return SwimConnectionConfig(
            host=host,
            port=port,
            timeout=15.0 if env == Environment.DEVELOPMENT else 30.0,
            max_retries=2 if env == Environment.DEVELOPMENT else 3,
            retry_base_delay=0.5 if env == Environment.DEVELOPMENT else 1.0,
            retry_max_delay=2.0 if env == Environment.DEVELOPMENT else 5.0,
            collection_interval=5.0 if env == Environment.DEVELOPMENT else 15.0
        )