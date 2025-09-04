"""
Configuration management for POLARIS framework.

Provides configuration loading, validation, and management for both
the core framework and managed system plugins.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
import yaml

# Import validation capabilities
from .validation import ConfigurationValidator
from .validation_result import ValidationResult, ValidationSeverity

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent / "config.yaml",
    Path(__file__).parent / "config.json",
]



def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, str]:
    """
    Recursively flattens a nested dictionary into environment-style
    keys (uppercase, underscore separated) and string values.
    Example:
        {"swim": {"host": "localhost", "port": 4242}}
        -> {"SWIM_HOST": "localhost", "SWIM_PORT": "4242"}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key.upper()] = str(v)
    return items


def _set_env_vars(config: Dict[str, Any], overwrite: bool = False):
    """Set environment variables from dictionary."""
    for key, value in config.items():
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = str(value)
        logger.debug(f"Set environment variable: {key}={value}")


def _load_from_env_file(env_path: Path) -> bool:
    """Load variables from a .env file."""
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded environment variables from {env_path}")
        return True
    return False

def _load_from_yaml(yaml_path: Path, overwrite: bool = True) -> bool:
    """
    Load environment variables from a YAML file.
    Supports both flat and grouped YAML structures.
    Example grouped YAML:
        swim:
          host: localhost
          port: 4242
    Will produce env vars:
        SWIM_HOST=localhost
        SWIM_PORT=4242
    """
    if not yaml_path.exists():
        logger.warning(f"YAML config file not found: {yaml_path}")
        return False

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            logger.error(f"YAML file {yaml_path} is not a mapping at root level.")
            return False

        flat_data = _flatten_dict(data)
        _set_env_vars(flat_data, overwrite=overwrite)
        logger.info(f"Loaded environment variables from {yaml_path}")
        return True

    except yaml.YAMLError as e:
        logger.exception(f"Failed to parse YAML config {yaml_path}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading YAML config {yaml_path}: {e}")

    return False

def _load_from_json(json_path: Path) -> bool:
    """Load variables from a JSON config file."""
    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f) or {}
        _set_env_vars(data, overwrite=True)
        logger.info(f"Loaded environment variables from {json_path}")
        return True
    return False

def load_config(
    search_paths: Optional[list] = None,
    overwrite: bool = True,
    required_keys: Optional[list] = None
):
    """
    Load configuration variables into os.environ.

    Args:
        search_paths (list): Optional list of file paths to check.
        overwrite (bool): Whether to overwrite existing env vars.
        required_keys (list): List of keys that must be present after load.

    Raises:
        ValueError: If required keys are missing.
    """
    search_paths = search_paths or DEFAULT_CONFIG_PATHS

    loaded = False
    for path in search_paths:
        if path.suffix == ".env":
            loaded = _load_from_env_file(path) or loaded
        elif path.suffix in [".yaml", ".yml"]:
            loaded = _load_from_yaml(path) or loaded
        elif path.suffix == ".json":
            loaded = _load_from_json(path) or loaded

    if not loaded:
        logger.warning("No configuration file found, relying on system env vars only.")

    if required_keys:
        missing = [key for key in required_keys if key not in os.environ]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    return True


def get_config(key: str, default: Any = None, cast_type: type = str):
    """
    Get a configuration value from the environment.

    Args:
        key (str): Environment variable name.
        default (Any): Default value if not found.
        cast_type (type): Type to cast the value into.

    Returns:
        Any: The configuration value.
    """
    value = os.environ.get(key, default)
    try:
        return cast_type(value) if value is not None else None
    except Exception:
        logger.warning(f"Failed to cast config value for key '{key}' to {cast_type.__name__}")
        return default


class ConfigurationManager:
    """Configuration manager with comprehensive validation support.
    
    This class manages both POLARIS framework configuration and
    managed system plugin configurations with JSON schema validation
    and comprehensive error reporting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration manager.
        
        Args:
            logger: Logger instance for structured logging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.framework_config: Dict[str, Any] = {}
        self.plugin_config: Dict[str, Any] = {}
        self.schema: Optional[Dict[str, Any]] = None
        
        # Initialize validator
        try:
            self.validator = ConfigurationValidator(self.logger)
            self.logger.info("Configuration validation capabilities enabled")
        except ImportError as e:
            self.logger.error(f"Failed to initialize validator: {e}")
            raise
        
    def load_framework_config(
        self,
        config_path: Union[str, Path],
        required_keys: Optional[List[str]] = None,
        validate_config: bool = True,
        schema_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load POLARIS framework configuration with comprehensive validation.
        
        Args:
            config_path: Path to framework configuration file
            required_keys: List of required configuration keys
            validate_config: Whether to perform validation
            schema_path: Path to schema file for validation
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid or required keys missing
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        # Perform validation
        if validate_config:
            try:
                validation_result = self.validator.validate_config_file(
                    config_path=config_path,
                    schema_path=schema_path,
                    config_type="framework"
                )
                
                # Log validation results
                if validation_result.valid:
                    if validation_result.warnings or validation_result.infos:
                        self.logger.info(
                            f"Framework configuration validation passed with {len(validation_result.warnings)} warnings "
                            f"and {len(validation_result.infos)} suggestions"
                        )
                        # Log detailed validation report in debug mode
                        self.logger.debug(f"Validation report:\n{validation_result.format_report()}")
                else:
                    error_msg = f"Framework configuration validation failed with {len(validation_result.errors)} errors"
                    self.logger.error(error_msg)
                    self.logger.error(f"Validation report:\n{validation_result.format_report()}")
                    raise ValueError(f"Configuration validation failed: {error_msg}")
                    
            except Exception as e:
                if isinstance(e, ValueError):
                    raise  # Re-raise validation errors
                self.logger.warning(f"Validation failed: {e}")
        
        # Load configuration based on file type
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.framework_config = yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.framework_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path.suffix}")
        
        # Check required keys
        if required_keys:
            missing = [k for k in required_keys if k not in self.framework_config]
            if missing:
                raise ValueError(f"Missing required configuration keys: {missing}")
        
        # Flatten and set environment variables
        flat_config = _flatten_dict(self.framework_config)
        _set_env_vars(flat_config, overwrite=True)
        
        self.logger.info(
            "Framework configuration loaded successfully",
            extra={
                "config_path": str(config_path),
                "validation_enabled": validate_config
            }
        )
        
        return self.framework_config
    
    def load_schema(self, schema_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON schema for plugin validation.
        
        Args:
            schema_path: Path to JSON schema file
            
        Returns:
            Loaded schema dictionary
            
        Raises:
            ValueError: If schema file not found or invalid
        """
        schema_path = Path(schema_path)
        
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        self.logger.info(
            "Schema loaded",
            extra={"schema_path": str(schema_path)}
        )
        
        return self.schema
    
    def load_plugin_config(
        self,
        plugin_dir: Union[str, Path],
        config_filename: str = "config.yaml",
        validate: bool = True,
        schema_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load and validate managed system plugin configuration.
        
        Args:
            plugin_dir: Directory containing the plugin
            config_filename: Name of the configuration file
            validate: Whether to validate against schema
            schema_path: Path to schema file for validation
            
        Returns:
            Loaded and validated plugin configuration
            
        Raises:
            ValueError: If configuration is invalid or validation fails
        """
        plugin_dir = Path(plugin_dir)
        config_path = plugin_dir / config_filename
        
        if not config_path.exists():
            raise ValueError(f"Plugin configuration not found: {config_path}")
        
        # Perform validation
        if validate:
            try:
                validation_result = self.validator.validate_config_file(
                    config_path=config_path,
                    schema_path=schema_path or self._find_plugin_schema_path(plugin_dir),
                    config_type="plugin"
                )
                
                # Log validation results
                if validation_result.valid:
                    if validation_result.warnings or validation_result.infos:
                        self.logger.info(
                            f"Plugin configuration validation passed with {len(validation_result.warnings)} warnings "
                            f"and {len(validation_result.infos)} suggestions"
                        )
                        self.logger.debug(f"Validation report:\n{validation_result.format_report()}")
                else:
                    error_msg = f"Plugin configuration validation failed with {len(validation_result.errors)} errors"
                    self.logger.error(error_msg)
                    self.logger.error(f"Validation report:\n{validation_result.format_report()}")
                    raise ValueError(f"Configuration validation failed: {error_msg}")
                    
            except Exception as e:
                if isinstance(e, ValueError):
                    raise  # Re-raise validation errors
                self.logger.warning(f"Validation failed: {e}")
        
        # Load configuration
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.plugin_config = yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.plugin_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path.suffix}")
        
        self.logger.info(
            "Plugin configuration loaded successfully",
            extra={
                "plugin_dir": str(plugin_dir),
                "system_name": self.plugin_config.get("system_name", "unknown"),
                "validation_enabled": validate
            }
        )
        
        return self.plugin_config
    
    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            schema: JSON schema (uses self.schema if not provided)
            
        Returns:
            ValidationResult with validation information
            
        Raises:
            ValueError: If validation fails with errors
        """
        result = self.validator.validate_config_dict(
            config=config,
            schema=schema or self.schema,
            config_type="unknown"
        )
        
        if not result.valid:
            raise ValueError(f"Configuration validation failed with {len(result.errors)} errors")
        
        return result
    
    def get_plugin_connector_class(self) -> str:
        """Get the connector class path from plugin configuration.
        
        Returns:
            Connector class import path
            
        Raises:
            ValueError: If connector class not specified
        """
        if not self.plugin_config:
            raise ValueError("Plugin configuration not loaded")
        
        connector_class = self.plugin_config.get("implementation", {}).get("connector_class")
        if not connector_class:
            raise ValueError("Connector class not specified in plugin configuration")
        
        return connector_class
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration from plugin.
        
        Returns:
            Monitoring configuration dictionary
        """
        return self.plugin_config.get("monitoring", {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration from plugin.
        
        Returns:
            Execution configuration dictionary
        """
        return self.plugin_config.get("execution", {})
    
    def get_verification_config(self) -> Dict[str, Any]:
        """Get verification configuration from plugin config."""
        return self.plugin_config.get("verification", {})
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Get connection configuration from plugin.
        
        Returns:
            Connection configuration dictionary
        """
        return self.plugin_config.get("connection", {})
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configurations override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        result = {}
        for config in configs:
            result.update(config)
        return result
    
    def validate_configuration_dict(
        self,
        config: Dict[str, Any],
        config_type: str = "unknown",
        schema: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration (framework, world_model, plugin)
            schema: Optional schema for validation
            
        Returns:
            ValidationResult with validation information
        """
        result = self.validator.validate_config_dict(
            config=config,
            schema=schema,
            config_type=config_type
        )
        
        # Log validation results
        if result.valid:
            if result.warnings or result.infos:
                self.logger.info(
                    f"Configuration validation passed with {len(result.warnings)} warnings "
                    f"and {len(result.infos)} suggestions"
                )
        else:
            self.logger.warning(
                f"Configuration validation failed with {len(result.errors)} errors"
            )
        
        return result
    
    def get_validation_report(
        self,
        config_path: Union[str, Path],
        config_type: str = "unknown",
        schema_path: Optional[Union[str, Path]] = None,
        output_format: str = "text"
    ) -> str:
        """Get a detailed validation report for a configuration file.
        
        Args:
            config_path: Path to configuration file
            config_type: Type of configuration
            schema_path: Path to schema file
            output_format: Output format ('text' or 'json')
            
        Returns:
            Formatted validation report
        """
        try:
            result = self.validator.validate_config_file(
                config_path=config_path,
                schema_path=schema_path,
                config_type=config_type
            )
            
            if output_format == "json":
                return result.format_json_report()
            else:
                return result.format_report()
                
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {e}")
            return f"Failed to generate validation report: {str(e)}"
    
    def validate_multiple_configs(
        self,
        config_files: List[Dict[str, Union[str, Path]]],
        stop_on_first_error: bool = False
    ) -> str:
        """Validate multiple configuration files and return a summary report.
        
        Args:
            config_files: List of config file specifications
            stop_on_first_error: Whether to stop on first validation error
            
        Returns:
            Multi-file validation report
        """
        try:
            multi_result = self.validator.validate_multiple_files(
                file_configs=config_files,
                stop_on_first_error=stop_on_first_error
            )
            
            return multi_result.format_report()
            
        except Exception as e:
            self.logger.error(f"Failed to validate multiple configurations: {e}")
            return f"Failed to validate multiple configurations: {str(e)}"
    

    
    def _find_plugin_schema_path(self, plugin_dir: Path) -> Optional[Path]:
        """Find schema file for plugin configuration.
        
        Args:
            plugin_dir: Plugin directory path
            
        Returns:
            Path to schema file if found
        """
        # Common schema file locations and names
        schema_locations = [
            plugin_dir / "schema.json",
            plugin_dir / "config.schema.json",
            plugin_dir / "managed_system.schema.json",
            plugin_dir.parent / "schemas" / "managed_system.schema.json",
            Path(__file__).parent.parent / "config" / "managed_system.schema.json"
        ]
        
        for schema_path in schema_locations:
            if schema_path.exists():
                return schema_path
        
        return None
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get status information about validation capabilities.
        
        Returns:
            Dictionary with validation status information
        """
        return {
            "validation_available": True,
            "validator_initialized": self.validator is not None,
            "validator_type": "ConfigurationValidator"
        }