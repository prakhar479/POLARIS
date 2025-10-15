"""
Digital Twin Configuration Management.

This module provides configuration loading, validation, and management
specifically for the Digital Twin component, including World Model
configuration and environment variable overrides.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from .config import ConfigurationManager
from .validation import ConfigurationValidator
from .validation_result import ValidationResult


class DigitalTwinConfigError(Exception):
    """Base exception for Digital Twin configuration errors."""
    pass


class DigitalTwinConfigManager:
    """
    Configuration manager specifically for Digital Twin component.
    
    Handles loading and validation of Digital Twin configuration,
    World Model settings, and environment variable overrides.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize Digital Twin configuration manager.
        
        Args:
            logger: Logger instance for structured logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.framework_config: Dict[str, Any] = {}
        self.world_model_config: Dict[str, Any] = {}
        self.active_model_config: Dict[str, Any] = {}
        
        # Initialize validator
        try:
            self.validator = ConfigurationValidator(self.logger)
            self.logger.info("Configuration validation enabled for Digital Twin")
        except ImportError as e:
            self.logger.error(f"Failed to initialize validator: {e}")
            raise
        
    def load_configuration(
        self,
        framework_config_path: Union[str, Path],
        world_model_config_path: Optional[Union[str, Path]] = None,
        validate_config: bool = True
    ) -> Dict[str, Any]:
        """Load Digital Twin configuration from files with comprehensive validation.
        
        Args:
            framework_config_path: Path to main POLARIS configuration
            world_model_config_path: Path to World Model configuration
            validate_config: Whether to perform validation
            
        Returns:
            Complete Digital Twin configuration
            
        Raises:
            DigitalTwinConfigError: If configuration loading fails
        """
        try:
            # Load framework configuration
            config_manager = ConfigurationManager(self.logger)
            self.framework_config = config_manager.load_framework_config(
                framework_config_path,
                validate_config=validate_config
            )
            
            # Extract Digital Twin specific configuration
            dt_config = self.framework_config.get("digital_twin", {})
            if not dt_config:
                raise DigitalTwinConfigError(
                    "No 'digital_twin' section found in framework configuration"
                )
            
            # Load World Model configuration if path provided
            if world_model_config_path:
                self._load_world_model_config(world_model_config_path, validate_config)
            elif "config_path" in dt_config.get("world_model", {}):
                # Use path from framework config
                wm_config_path = dt_config["world_model"]["config_path"]
                # Resolve relative to framework config directory
                if not Path(wm_config_path).is_absolute():
                    base_dir = Path(framework_config_path).parent
                    wm_config_path = base_dir / wm_config_path
                self._load_world_model_config(wm_config_path, validate_config)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            self.logger.info(
                "Digital Twin configuration loaded successfully",
                extra={
                    "world_model_implementation": dt_config.get("world_model", {}).get("implementation"),
                    "grpc_port": dt_config.get("grpc", {}).get("port"),
                    "nats_subjects": {
                        "update": dt_config.get("nats", {}).get("update_subject"),
                        "calibrate": dt_config.get("nats", {}).get("calibrate_subject")
                    }
                }
            )
            
            return self.get_complete_config()
            
        except Exception as e:
            raise DigitalTwinConfigError(f"Failed to load Digital Twin configuration: {str(e)}") from e
    
    def _load_world_model_config(self, config_path: Union[str, Path], validate_config: bool = True) -> None:
        """Load World Model specific configuration with comprehensive validation.
        
        Args:
            config_path: Path to World Model configuration file
            validate_config: Whether to perform validation
            
        Raises:
            DigitalTwinConfigError: If World Model config loading fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.warning(f"World Model config file not found: {config_path}")
            return
        
        # Perform validation
        if validate_config:
            try:
                validation_result = self.validator.validate_config_file(
                    config_path=config_path,
                    config_type="world_model"
                )
                
                if validation_result.valid:
                    if validation_result.warnings or validation_result.infos:
                        self.logger.info(
                            f"World Model configuration validation passed with {len(validation_result.warnings)} warnings "
                            f"and {len(validation_result.infos)} suggestions"
                        )
                        self.logger.debug(f"Validation report:\n{validation_result.format_report()}")
                else:
                    error_msg = f"World Model configuration validation failed with {len(validation_result.errors)} errors"
                    self.logger.error(error_msg)
                    self.logger.error(f"Validation report:\n{validation_result.format_report()}")
                    raise DigitalTwinConfigError(f"World Model configuration validation failed: {error_msg}")
                    
            except DigitalTwinConfigError:
                raise  # Re-raise validation errors
            except Exception as e:
                self.logger.warning(f"Validation failed, proceeding with basic loading: {e}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.world_model_config = yaml.safe_load(f) or {}
            
            self.logger.info(
                f"World Model configuration loaded from {config_path}",
                extra={
                    "validation_enabled": validate_config
                }
            )
            
        except yaml.YAMLError as e:
            raise DigitalTwinConfigError(f"Failed to parse World Model config: {str(e)}") from e
        except Exception as e:
            raise DigitalTwinConfigError(f"Failed to load World Model config: {str(e)}") from e
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        dt_config = self.framework_config.get("digital_twin", {})
        
        # Override World Model implementation
        if "DIGITAL_TWIN_WORLD_MODEL_IMPLEMENTATION" in os.environ:
            impl = os.environ["DIGITAL_TWIN_WORLD_MODEL_IMPLEMENTATION"]
            dt_config.setdefault("world_model", {})["implementation"] = impl
            self.logger.info(f"World Model implementation overridden to: {impl}")
        
        # Override gRPC port
        if "DIGITAL_TWIN_GRPC_PORT" in os.environ:
            try:
                port = int(os.environ["DIGITAL_TWIN_GRPC_PORT"])
                dt_config.setdefault("grpc", {})["port"] = port
                self.logger.info(f"gRPC port overridden to: {port}")
            except ValueError:
                self.logger.warning("Invalid DIGITAL_TWIN_GRPC_PORT value, ignoring")
        
        # Override log level
        if "DIGITAL_TWIN_LOG_LEVEL" in os.environ:
            log_level = os.environ["DIGITAL_TWIN_LOG_LEVEL"].upper()
            dt_config.setdefault("debugging", {})["log_level"] = log_level
            self.logger.info(f"Log level overridden to: {log_level}")
        
        # Apply World Model specific overrides
        if self.world_model_config and "env_overrides" in self.world_model_config:
            self._apply_world_model_env_overrides()
    
    def _apply_world_model_env_overrides(self) -> None:
        """Apply World Model specific environment variable overrides.
        
        This method applies environment variable overrides to the world model configuration.
        The configuration is expected to have a flat structure with an 'implementation' field
        and a 'config' dictionary for implementation-specific settings.
        """
        # Get the config section where implementation-specific settings are stored
        config_section = self.world_model_config.setdefault("config", {})
        overrides = self.world_model_config.get("env_overrides", {})
        
        # Get the implementation type (e.g., 'gemini', 'mock')
        implementation = self.world_model_config.get("implementation", "").lower()
        
        # Common overrides that apply to all implementations
        if "temperature_override" in overrides and overrides["temperature_override"] in os.environ:
            try:
                temp = float(os.environ[overrides["temperature_override"]])
                config_section["temperature"] = temp
                self.logger.info(f"Temperature overridden to {temp}")
            except ValueError:
                self.logger.warning("Invalid temperature value, ignoring")
        
        # Implementation-specific overrides
        if implementation == "gemini":
            # Override Gemini model
            if "gemini_model_override" in overrides and overrides["gemini_model_override"] in os.environ:
                model = os.environ[overrides["gemini_model_override"]]
                config_section["model"] = model
                self.logger.info(f"Gemini model overridden to: {model}")
            
            # Override API key environment variable
            if "api_key_env_override" in overrides and overrides["api_key_env_override"] in os.environ:
                api_key_env = os.environ[overrides["api_key_env_override"]]
                config_section["api_key_env"] = api_key_env
                self.logger.info(f"API key environment variable overridden to: {api_key_env}")
            
            # Override vector store provider
            if "vector_store_provider_override" in overrides and overrides["vector_store_provider_override"] in os.environ:
                provider = os.environ[overrides["vector_store_provider_override"]]
                vector_config = config_section.setdefault("vector_store", {})
                vector_config["provider"] = provider
                self.logger.info(f"Vector store provider overridden to: {provider}")
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration.
        
        This method validates the Digital Twin configuration, including:
        - Required top-level sections (nats, grpc, world_model)
        - World Model implementation and configuration
        - Implementation-specific configuration (e.g., Gemini settings)
        
        Raises:
            DigitalTwinConfigError: If configuration is invalid
        """
        dt_config = self.framework_config.get("digital_twin", {})
        
        # Validate required sections
        required_sections = ["nats", "grpc", "world_model"]
        for section in required_sections:
            if section not in dt_config:
                raise DigitalTwinConfigError(f"Missing required configuration section: {section}")
        
        # Validate NATS configuration
        nats_config = dt_config["nats"]
        required_nats_fields = ["update_subject", "calibrate_subject", "error_subject", "queue_group"]
        for field in required_nats_fields:
            if field not in nats_config:
                raise DigitalTwinConfigError(f"Missing required NATS field: {field}")
        
        # Validate gRPC configuration
        grpc_config = dt_config["grpc"]
        # Host presence and type
        if "host" not in grpc_config:
            raise DigitalTwinConfigError("Missing required gRPC host")
        if not isinstance(grpc_config["host"], str) or not grpc_config["host"].strip():
            raise DigitalTwinConfigError("Invalid gRPC host: must be a non-empty string")

        # Port presence and range
        if "port" not in grpc_config:
            raise DigitalTwinConfigError("Missing required gRPC port")
        try:
            port = int(grpc_config["port"])
            if port < 1 or port > 65535:
                raise DigitalTwinConfigError(f"Invalid gRPC port: {port}")
        except (ValueError, TypeError) as e:
            raise DigitalTwinConfigError(f"Invalid gRPC port value: {grpc_config['port']}") from e

        # Optional numeric bounds aligned with schema
        if "max_workers" in grpc_config:
            try:
                mw = int(grpc_config["max_workers"])
                if mw < 1:
                    raise DigitalTwinConfigError("gRPC max_workers must be >= 1")
            except (ValueError, TypeError) as e:
                raise DigitalTwinConfigError("gRPC max_workers must be an integer") from e

        if "max_message_size" in grpc_config:
            try:
                mms = int(grpc_config["max_message_size"])
                if mms < 1024:
                    raise DigitalTwinConfigError("gRPC max_message_size must be >= 1024 bytes")
            except (ValueError, TypeError) as e:
                raise DigitalTwinConfigError("gRPC max_message_size must be an integer") from e

        if "keepalive_time_ms" in grpc_config:
            try:
                kat = int(grpc_config["keepalive_time_ms"])
                if kat < 1000:
                    raise DigitalTwinConfigError("gRPC keepalive_time_ms must be >= 1000")
            except (ValueError, TypeError) as e:
                raise DigitalTwinConfigError("gRPC keepalive_time_ms must be an integer") from e

        if "keepalive_timeout_ms" in grpc_config:
            try:
                kao = int(grpc_config["keepalive_timeout_ms"])
                if kao < 1000:
                    raise DigitalTwinConfigError("gRPC keepalive_timeout_ms must be >= 1000")
            except (ValueError, TypeError) as e:
                raise DigitalTwinConfigError("gRPC keepalive_timeout_ms must be an integer") from e
        
        # Validate World Model configuration
        wm_config = dt_config["world_model"]
        if "implementation" not in wm_config:
            raise DigitalTwinConfigError("Missing World Model implementation in digital_twin.world_model")
        
        implementation = wm_config["implementation"].lower()
        
        # Ensure world model config is properly structured
        if not isinstance(self.world_model_config, dict):
            raise DigitalTwinConfigError("World Model configuration must be a dictionary")
        
        # Check implementation matches between framework and world model config
        wm_implementation = self.world_model_config.get("implementation", "").lower()
        if wm_implementation and wm_implementation != implementation:
            self.logger.warning(
                f"Implementation mismatch: framework expects '{implementation}', "
                f"world model config has '{wm_implementation}'"
            )
        
        # Set the implementation if not already set in world model config
        if not wm_implementation:
            self.world_model_config["implementation"] = implementation
            wm_implementation = implementation
        
        # Ensure config section exists
        if "config" not in self.world_model_config:
            self.world_model_config["config"] = {}
        
        # Validate known implementations
        known_implementations = ["mock", "gemini", "bayesian", "statistical", "hybrid"]
        if wm_implementation not in known_implementations:
            self.logger.warning(f"Unknown World Model implementation: {wm_implementation}")
        
        # Validate implementation-specific configuration
        if wm_implementation == "gemini":
            self._validate_gemini_config()
        
        # Additional validation for other implementations can be added here
        elif wm_implementation == "bayesian":
            self._validate_bayesian_config()
        elif wm_implementation == "mock":
            # Basic validation for mock config
            mock_config = self.world_model_config.get("config", {})
            if not isinstance(mock_config, dict):
                raise DigitalTwinConfigError("Mock configuration must be a dictionary")
        
        # Log successful validation
        self.logger.info(
            "Digital Twin configuration validated successfully",
            extra={"implementation": wm_implementation}
        )
    
    def _validate_gemini_config(self) -> None:
        """Validate Gemini LLM specific configuration.
        
        This method validates that the Gemini configuration in the world model's config
        section is valid. The config is expected to be in the 'config' section of the
        world model configuration.
        
        Raises:
            DigitalTwinConfigError: If Gemini configuration is invalid
        """
        # Get the config section where implementation-specific settings are stored
        config_section = self.world_model_config.get("config", {})
        
        # Check required fields
        required_fields = ["api_key_env", "model"]
        for field in required_fields:
            if field not in config_section:
                raise DigitalTwinConfigError(f"Missing required Gemini field in config: {field}")
        
        # Validate API key environment variable exists
        api_key_env = config_section["api_key_env"]
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise DigitalTwinConfigError("Gemini API key environment variable name is required")
            
        if api_key_env not in os.environ:
            raise DigitalTwinConfigError(
                f"Gemini API key environment variable not set: {api_key_env}"
            )
        
        # Validate temperature if specified
        if "temperature" in config_section:
            temp = config_section["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise DigitalTwinConfigError(
                    f"Invalid temperature value in Gemini config: {temp}. Must be between 0 and 2"
                )
        
        # Validate max_tokens if specified
        if "max_tokens" in config_section:
            max_tokens = config_section["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise DigitalTwinConfigError(
                    f"Invalid max_tokens value in Gemini config: {max_tokens}. Must be a positive integer"
                )
        
        # Validate concurrent_requests if specified
        if "concurrent_requests" in config_section:
            concurrent = config_section["concurrent_requests"]
            if not isinstance(concurrent, int) or concurrent <= 0:
                raise DigitalTwinConfigError(
                    f"Invalid concurrent_requests value in Gemini config: {concurrent}. Must be a positive integer"
                )
        
        # Validate vector store config if specified
        if "vector_store" in config_section:
            vector_config = config_section["vector_store"]
            if not isinstance(vector_config, dict):
                raise DigitalTwinConfigError("Vector store config must be a dictionary")
            
            # Validate required vector store fields
            if "provider" not in vector_config:
                raise DigitalTwinConfigError("Vector store configuration is missing 'provider' field")
    
    def _validate_bayesian_config(self) -> None:
        """Validate Bayesian/Kalman Filter specific configuration.
        
        This method validates that the Bayesian configuration in the world model's config
        section is valid. The config is expected to be in the 'config' section of the
        world model configuration.
        
        Raises:
            DigitalTwinConfigError: If Bayesian configuration is invalid
        """
        # Get the config section where implementation-specific settings are stored
        config_section = self.world_model_config.get("config", {})
        
        # Validate prediction_horizon_minutes if specified
        if "prediction_horizon_minutes" in config_section:
            horizon = config_section["prediction_horizon_minutes"]
            if not isinstance(horizon, int) or horizon <= 0 or horizon > 1440:  # Max 24 hours
                raise DigitalTwinConfigError(
                    f"Invalid prediction_horizon_minutes in Bayesian config: {horizon}. "
                    f"Must be a positive integer between 1 and 1440 (24 hours)"
                )
        
        # Validate correlation_threshold if specified
        if "correlation_threshold" in config_section:
            threshold = config_section["correlation_threshold"]
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                raise DigitalTwinConfigError(
                    f"Invalid correlation_threshold in Bayesian config: {threshold}. "
                    f"Must be a number between 0 and 1"
                )
        
        # Validate anomaly_threshold if specified
        if "anomaly_threshold" in config_section:
            threshold = config_section["anomaly_threshold"]
            if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 10:
                raise DigitalTwinConfigError(
                    f"Invalid anomaly_threshold in Bayesian config: {threshold}. "
                    f"Must be a positive number between 0 and 10 (standard deviations)"
                )
        
        # Validate process_noise if specified
        if "process_noise" in config_section:
            noise = config_section["process_noise"]
            if not isinstance(noise, (int, float)) or noise <= 0 or noise > 1:
                raise DigitalTwinConfigError(
                    f"Invalid process_noise in Bayesian config: {noise}. "
                    f"Must be a positive number between 0 and 1"
                )
        
        # Validate measurement_noise if specified
        if "measurement_noise" in config_section:
            noise = config_section["measurement_noise"]
            if not isinstance(noise, (int, float)) or noise <= 0 or noise > 1:
                raise DigitalTwinConfigError(
                    f"Invalid measurement_noise in Bayesian config: {noise}. "
                    f"Must be a positive number between 0 and 1"
                )
        
        # Validate learning_rate if specified
        if "learning_rate" in config_section:
            rate = config_section["learning_rate"]
            if not isinstance(rate, (int, float)) or rate <= 0 or rate > 1:
                raise DigitalTwinConfigError(
                    f"Invalid learning_rate in Bayesian config: {rate}. "
                    f"Must be a positive number between 0 and 1"
                )
        
        # Validate max_history_points if specified
        if "max_history_points" in config_section:
            points = config_section["max_history_points"]
            if not isinstance(points, int) or points <= 0 or points > 100000:
                raise DigitalTwinConfigError(
                    f"Invalid max_history_points in Bayesian config: {points}. "
                    f"Must be a positive integer between 1 and 100000"
                )
        
        # Validate update_interval_seconds if specified
        if "update_interval_seconds" in config_section:
            interval = config_section["update_interval_seconds"]
            if not isinstance(interval, (int, float)) or interval <= 0 or interval > 3600:
                raise DigitalTwinConfigError(
                    f"Invalid update_interval_seconds in Bayesian config: {interval}. "
                    f"Must be a positive number between 0 and 3600 (1 hour)"
                )
    
    def get_complete_config(self) -> Dict[str, Any]:
        """Get the complete Digital Twin configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return {
            "framework": self.framework_config,
            "digital_twin": self.framework_config.get("digital_twin", {}),
            "world_model": self.world_model_config,
            "active_model": self.get_active_model_config()
        }
    
    def get_digital_twin_config(self) -> Dict[str, Any]:
        """Get Digital Twin specific configuration.
        
        Returns:
            Digital Twin configuration section
        """
        return self.framework_config.get("digital_twin", {})
    
    def get_world_model_config(self) -> Dict[str, Any]:
        """Get World Model configuration.
        
        Returns the complete world model configuration including the implementation
        and configuration sections. The structure is:
        {
            "implementation": str,  # e.g., "mock", "gemini"
            "config": Dict[str, Any],  # Implementation-specific configuration
            ...  # Other top-level fields
        }
        
        Returns:
            Dict[str, Any]: World Model configuration dictionary
            
        Note:
            The returned dictionary includes both the implementation type and its
            configuration. For just the implementation-specific config, access the
            'config' key of the returned dictionary.
        """
        # Ensure config section exists
        if "config" not in self.world_model_config:
            self.world_model_config["config"] = {}
            
        return self.world_model_config
    
    def get_active_model_config(self) -> Dict[str, Any]:
        """Get configuration for the active World Model implementation.
        
        Expects world_model_config to be a dictionary with an 'implementation' field
        that matches the requested implementation.
        
        Returns:
            Configuration dictionary if implementation matches, otherwise empty dict
        """
        dt_config = self.framework_config.get("digital_twin", {})
        expected_implementation = dt_config.get("world_model", {}).get("implementation", "mock")
        
        current_implementation = self.world_model_config.get("implementation")
        
        if current_implementation == expected_implementation:
            return self.world_model_config
            
        self.logger.warning(
            f"World Model implementation mismatch. Expected '{expected_implementation}', "
            f"found '{current_implementation}'"
        )
        return {}
    
    def get_nats_config(self) -> Dict[str, Any]:
        """Get NATS configuration for Digital Twin.
        
        Returns:
            NATS configuration dictionary
        """
        return self.get_digital_twin_config().get("nats", {})
    
    def get_grpc_config(self) -> Dict[str, Any]:
        """Get gRPC configuration for Digital Twin.
        
        Returns:
            gRPC configuration dictionary
        """
        return self.get_digital_twin_config().get("grpc", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration for Digital Twin.
        
        Returns:
            Performance configuration dictionary
        """
        return self.get_digital_twin_config().get("performance", {})
    
    def get_debugging_config(self) -> Dict[str, Any]:
        """Get debugging configuration for Digital Twin.
        
        Returns:
            Debugging configuration dictionary
        """
        return self.get_digital_twin_config().get("debugging", {})
    
    def is_model_available(self, model_type: str) -> bool:
        """Check if a specific model type is available and configured.
        
        Args:
            model_type: Model type to check
            
        Returns:
            True if model is available and configured
        """
        # Compare requested type with configured implementation
        impl = self.world_model_config.get("implementation", "").lower()
        if impl != model_type.lower():
            return False

        # Read implementation-specific settings from the 'config' section
        config_section = self.world_model_config.get("config", {})
        if not isinstance(config_section, dict):
            return False

        # Check if model is enabled (default True)
        if not config_section.get("enabled", True):
            return False

        # Model-specific availability checks
        if impl == "gemini":
            # Check if API key environment variable is set
            api_key_env = config_section.get("api_key_env")
            if not api_key_env or api_key_env not in os.environ:
                return False

        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available and configured model types.
        
        Returns:
            List of available model type names
        """
        impl = self.world_model_config.get("implementation")
        if not impl:
            return []
        return [impl] if self.is_model_available(impl) else []
    
    def create_model_config_summary(self) -> Dict[str, Any]:
        """Create a summary of model configuration for logging/debugging.
        
        Returns:
            Configuration summary (with sensitive data masked)
        """
        dt_config = self.get_digital_twin_config()
        active_config = self.get_active_model_config()
        
        summary = {
            "implementation": dt_config.get("world_model", {}).get("implementation"),
            "available_models": self.get_available_models(),
            "nats_subjects": {
                "update": dt_config.get("nats", {}).get("update_subject"),
                "calibrate": dt_config.get("nats", {}).get("calibrate_subject"),
                "error": dt_config.get("nats", {}).get("error_subject")
            },
            "grpc_port": dt_config.get("grpc", {}).get("port"),
            "performance": {
                "max_concurrent_queries": dt_config.get("performance", {}).get("max_concurrent_queries"),
                "query_timeout_sec": dt_config.get("performance", {}).get("query_timeout_sec"),
                "simulation_timeout_sec": dt_config.get("performance", {}).get("simulation_timeout_sec")
            }
        }
        
        # Add model-specific summary (mask sensitive data)
        if active_config:
            cfg = active_config.get("config", {}) if isinstance(active_config, dict) else {}
            if isinstance(cfg, dict):
                if "api_key_env" in cfg:
                    summary["api_key_configured"] = cfg["api_key_env"] in os.environ
                if "model" in cfg:
                    summary["model_version"] = cfg["model"]
                if "temperature" in cfg:
                    summary["temperature"] = cfg["temperature"]
        
        return summary  
  
    def get_configuration_validation_report(
        self,
        framework_config_path: Union[str, Path],
        world_model_config_path: Optional[Union[str, Path]] = None,
        output_format: str = "text"
    ) -> Optional[str]:
        """Get comprehensive validation report for Digital Twin configuration.
        
        Args:
            framework_config_path: Path to framework configuration
            world_model_config_path: Path to world model configuration
            output_format: Output format ('text' or 'json')
            
        Returns:
            Validation report string or None if enhanced validation not available
        """

        
        try:
            # Prepare file configurations for multi-file validation
            file_configs = [
                {
                    'path': framework_config_path,
                    'config_type': 'framework',
                    'schema_path': None  # Will be auto-discovered
                }
            ]
            
            # Add world model config if specified
            if world_model_config_path:
                file_configs.append({
                    'path': world_model_config_path,
                    'config_type': 'world_model',
                    'schema_path': None
                })
            elif self.framework_config and "digital_twin" in self.framework_config:
                # Try to get world model path from framework config
                dt_config = self.framework_config["digital_twin"]
                if "world_model" in dt_config and "config_path" in dt_config["world_model"]:
                    wm_path = dt_config["world_model"]["config_path"]
                    if not Path(wm_path).is_absolute():
                        base_dir = Path(framework_config_path).parent
                        wm_path = base_dir / wm_path
                    
                    if Path(wm_path).exists():
                        file_configs.append({
                            'path': wm_path,
                            'config_type': 'world_model',
                            'schema_path': None
                        })
            
            # Perform multi-file validation
            multi_result = self.validator.validate_multiple_files(file_configs)
            
            if output_format == "json":
                # Convert to JSON format
                report_data = {
                    "overall_valid": multi_result.overall_valid,
                    "summary": multi_result.get_summary().to_dict(),
                    "files": {
                        str(path): result.to_dict() 
                        for path, result in multi_result.file_results.items()
                    }
                }
                import json
                return json.dumps(report_data, indent=2)
            else:
                return multi_result.format_report()
                
        except Exception as e:
            self.logger.error(f"Failed to generate Digital Twin validation report: {e}")
            return None
    
    def validate_active_configuration(self) -> Optional[ValidationResult]:
        """Validate the currently loaded Digital Twin configuration.
        
        Returns:
            ValidationResult for the active configuration or None if not available
        """
        if not self.framework_config:
            return None
        
        try:
            # Validate framework Digital Twin section
            dt_config = self.framework_config.get("digital_twin", {})
            framework_result = self.validator.validate_config_dict(
                config={"digital_twin": dt_config},
                config_type="framework"
            )
            
            # Validate world model configuration if loaded
            world_model_result = None
            if self.world_model_config:
                world_model_result = self.validator.validate_config_dict(
                    config=self.world_model_config,
                    config_type="world_model"
                )
            
            # Combine results
            if world_model_result:
                framework_result.merge(world_model_result)
            
            return framework_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate active Digital Twin configuration: {e}")
            return None
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get status of validation for Digital Twin configuration.
        
        Returns:
            Dictionary with validation status information
        """
        return {
            "validation_available": True,
            "validator_initialized": hasattr(self, 'validator') and self.validator is not None,
            "framework_config_loaded": bool(self.framework_config),
            "world_model_config_loaded": bool(self.world_model_config)
        }