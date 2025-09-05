"""
Configuration validation utilities.
"""

import os
from typing import Dict, Any, List
from pydantic import ValidationError

from ...infrastructure.exceptions import ConfigurationError
from .models import FrameworkConfiguration, ManagedSystemConfiguration


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    
    def __init__(self, message: str, validation_errors: List[Dict[str, Any]]):
        super().__init__(message, "CONFIGURATION_VALIDATION_ERROR", {"validation_errors": validation_errors})
        self.validation_errors = validation_errors
    
    def get_detailed_message(self) -> str:
        """Get a detailed error message with all validation errors."""
        lines = [self.message]
        lines.append("Validation errors:")
        
        for error in self.validation_errors:
            location = " -> ".join(str(loc) for loc in error.get('loc', []))
            msg = error.get('msg', 'Unknown error')
            lines.append(f"- {location}: {msg}")
        
        return "\n".join(lines)


class ConfigurationValidator:
    """Validates configuration data and provides detailed error messages."""
    
    @staticmethod
    def validate_configuration(config_data: Dict[str, Any]) -> List[str]:
        """
        Validate configuration data and return list of warnings.
        
        Args:
            config_data: Raw configuration data to validate
            
        Returns:
            List of warning messages for invalid configuration
            
        Raises:
            ConfigurationValidationError: If validation fails with errors
        """
        errors = []
        warnings = []
        
        try:
            # Validate framework configuration if present
            if 'framework' in config_data:
                framework_data = config_data['framework']
                try:
                    FrameworkConfiguration(**framework_data)
                except ValidationError as e:
                    for error in e.errors():
                        errors.append({
                            'loc': ['framework'] + list(error['loc']),
                            'msg': error['msg'],
                            'type': error['type']
                        })
            
            # Validate managed systems if present
            if 'managed_systems' in config_data:
                managed_systems = config_data['managed_systems']
                if isinstance(managed_systems, dict):
                    for system_id, system_config in managed_systems.items():
                        if isinstance(system_config, dict):
                            system_config_copy = system_config.copy()
                            system_config_copy['system_id'] = system_id
                            try:
                                ManagedSystemConfiguration(**system_config_copy)
                            except ValidationError as e:
                                for error in e.errors():
                                    errors.append({
                                        'loc': ['managed_systems', system_id] + list(error['loc']),
                                        'msg': error['msg'],
                                        'type': error['type']
                                    })
            
            # Check for unknown top-level keys
            known_keys = {'framework', 'managed_systems'}
            for key in config_data.keys():
                if key not in known_keys:
                    warnings.append(f"Unknown configuration key: {key}")
            
        except Exception as e:
            errors.append({
                'loc': ['root'],
                'msg': f"Unexpected validation error: {str(e)}",
                'type': 'value_error'
            })
        
        if errors:
            raise ConfigurationValidationError(
                "Configuration validation failed",
                errors
            )
        
        return warnings
    
    @staticmethod
    def validate_environment_variables(prefix: str = "POLARIS_") -> List[str]:
        """
        Validate environment variables and return warnings for unknown variables.
        
        Args:
            prefix: Environment variable prefix to check
            
        Returns:
            List of warning messages for invalid environment variables
        """
        warnings = []
        valid_paths = {
            'framework_nats_config_servers',
            'framework_nats_config_username',
            'framework_nats_config_password',
            'framework_nats_config_token',
            'framework_nats_config_timeout',
            'framework_telemetry_config_enabled',
            'framework_telemetry_config_collection_interval',
            'framework_telemetry_config_batch_size',
            'framework_telemetry_config_retention_days',
            'framework_logging_config_level',
            'framework_logging_config_format',
            'framework_logging_config_output',
            'framework_logging_config_file_path',
            'framework_logging_config_max_file_size',
            'framework_logging_config_backup_count',
            'framework_plugin_search_paths',
            'framework_max_concurrent_adaptations',
            'framework_adaptation_timeout'
        }
        
        prefix_upper = prefix.upper()
        for key in os.environ:
            if key.startswith(prefix_upper):
                config_key = key[len(prefix_upper):].lower()
                if config_key not in valid_paths and not config_key.startswith('managed_systems_'):
                    warnings.append(f"Unknown environment variable: {key}")
        
        return warnings