"""
Comprehensive tests for the configuration management system.
"""

import os
import tempfile
import yaml
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import patch

from ..src.framework.configuration import (
    ConfigurationBuilder,
    PolarisConfiguration,
    ConfigurationValidationError,
    NATSConfiguration,
    TelemetryConfiguration,
    LoggingConfiguration,
    FrameworkConfiguration,
    ManagedSystemConfiguration,
    YAMLConfigurationSource,
    EnvironmentConfigurationSource,
    ConfigurationValidator,
    load_configuration_from_file,
    load_default_configuration,
    load_hot_reload_configuration
)


class TestConfigurationModels:
    """Test configuration data models."""
    
    def test_nats_configuration_valid(self):
        """Test valid NATS configuration."""
        config = NATSConfiguration(
            servers=["nats://localhost:4222"],
            timeout=60
        )
        assert config.servers == ["nats://localhost:4222"]
        assert config.timeout == 60
    
    def test_nats_configuration_invalid_server(self):
        """Test NATS configuration with invalid server URL."""
        with pytest.raises(ValueError, match="Invalid NATS server URL"):
            NATSConfiguration(servers=["invalid://server"])
    
    def test_nats_configuration_invalid_timeout(self):
        """Test NATS configuration with invalid timeout."""
        with pytest.raises(ValueError):
            NATSConfiguration(timeout=-1)
    
    def test_telemetry_configuration_valid(self):
        """Test valid telemetry configuration."""
        config = TelemetryConfiguration(
            enabled=True,
            collection_interval=60,
            batch_size=200
        )
        assert config.enabled is True
        assert config.collection_interval == 60
        assert config.batch_size == 200
    
    def test_telemetry_configuration_invalid_interval(self):
        """Test telemetry configuration with invalid collection interval."""
        with pytest.raises(ValueError, match="Collection interval must be at least 5 seconds"):
            TelemetryConfiguration(collection_interval=1)
    
    def test_logging_configuration_valid(self):
        """Test valid logging configuration."""
        config = LoggingConfiguration(
            level="DEBUG",
            format="json",
            output="console"
        )
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.output == "console"
    
    def test_logging_configuration_file_output_requires_path(self):
        """Test that file output requires file_path."""
        with pytest.raises(ValueError, match="file_path is required"):
            LoggingConfiguration(output="file", file_path=None)
    
    def test_framework_configuration_defaults(self):
        """Test framework configuration with defaults."""
        config = FrameworkConfiguration()
        assert isinstance(config.nats_config, NATSConfiguration)
        assert isinstance(config.telemetry_config, TelemetryConfiguration)
        assert isinstance(config.logging_config, LoggingConfiguration)
        assert config.max_concurrent_adaptations == 10
    
    def test_managed_system_configuration_valid(self):
        """Test valid managed system configuration."""
        config = ManagedSystemConfiguration(
            system_id="test_system",
            connector_type="test_connector",
            connection_params={"host": "localhost"},
            enabled=True
        )
        assert config.system_id == "test_system"
        assert config.connector_type == "test_connector"
        assert config.enabled is True
    
    def test_managed_system_configuration_invalid_id(self):
        """Test managed system configuration with invalid system ID."""
        with pytest.raises(ValueError, match="must contain only alphanumeric characters"):
            ManagedSystemConfiguration(
                system_id="invalid@system",
                connector_type="test"
            )


class TestConfigurationSources:
    """Test configuration sources."""
    
    def test_yaml_source_valid_file(self):
        """Test YAML source with valid file."""
        config_data = {
            'framework': {
                'nats_config': {
                    'servers': ['nats://test:4222'],
                    'timeout': 60
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            source = YAMLConfigurationSource(temp_file)
            loaded_data = source.load()
            
            assert loaded_data == config_data
            assert source.get_priority() == 100
        finally:
            os.unlink(temp_file)
    
    def test_yaml_source_missing_file(self):
        """Test YAML source with missing file."""
        source = YAMLConfigurationSource("/nonexistent/file.yaml")
        
        with pytest.raises(Exception, match="Configuration file not found"):
            source.load()
    
    def test_yaml_source_invalid_yaml(self):
        """Test YAML source with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_file = f.name
        
        try:
            source = YAMLConfigurationSource(temp_file)
            with pytest.raises(Exception, match="Invalid YAML"):
                source.load()
        finally:
            os.unlink(temp_file)
    
    def test_environment_source_basic(self):
        """Test environment source with basic variables."""
        with patch.dict(os.environ, {
            'TEST_FRAMEWORK_NATS_CONFIG_TIMEOUT': '120',
            'TEST_FRAMEWORK_TELEMETRY_CONFIG_ENABLED': 'false'
        }):
            source = EnvironmentConfigurationSource("TEST_")
            config = source.load()
            
            assert config['framework']['nats_config']['timeout'] == 120
            assert config['framework']['telemetry_config']['enabled'] is False
    
    def test_environment_source_list_parsing(self):
        """Test environment source list parsing."""
        with patch.dict(os.environ, {
            'TEST_FRAMEWORK_NATS_CONFIG_SERVERS': 'nats://server1:4222,nats://server2:4222'
        }):
            source = EnvironmentConfigurationSource("TEST_")
            config = source.load()
            
            expected_servers = ['nats://server1:4222', 'nats://server2:4222']
            assert config['framework']['nats_config']['servers'] == expected_servers
    
    def test_environment_source_single_server_as_list(self):
        """Test environment source converts single server to list."""
        with patch.dict(os.environ, {
            'TEST_FRAMEWORK_NATS_CONFIG_SERVERS': 'nats://single:4222'
        }):
            source = EnvironmentConfigurationSource("TEST_")
            config = source.load()
            
            assert config['framework']['nats_config']['servers'] == ['nats://single:4222']


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration."""
        config_data = {
            'framework': {
                'nats_config': {
                    'servers': ['nats://localhost:4222'],
                    'timeout': 30
                },
                'telemetry_config': {
                    'enabled': True
                }
            }
        }
        
        warnings = ConfigurationValidator.validate_configuration(config_data)
        assert isinstance(warnings, list)
    
    def test_validation_invalid_config(self):
        """Test validation with invalid configuration."""
        config_data = {
            'framework': {
                'nats_config': {
                    'servers': "not_a_list",  # Should be a list
                    'timeout': -1  # Should be positive
                }
            }
        }
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            ConfigurationValidator.validate_configuration(config_data)
        
        error = exc_info.value
        assert len(error.validation_errors) > 0
        assert "servers" in str(error.validation_errors)
    
    def test_validation_unknown_keys(self):
        """Test validation with unknown configuration keys."""
        config_data = {
            'framework': {},
            'unknown_key': 'value'
        }
        
        warnings = ConfigurationValidator.validate_configuration(config_data)
        assert any("Unknown configuration key" in warning for warning in warnings)
    
    def test_environment_variable_validation(self):
        """Test environment variable validation."""
        with patch.dict(os.environ, {
            'POLARIS_FRAMEWORK_NATS_CONFIG_TIMEOUT': '30',
            'POLARIS_UNKNOWN_VARIABLE': 'value'
        }):
            warnings = ConfigurationValidator.validate_environment_variables("POLARIS_")
            assert any("Unknown environment variable" in warning for warning in warnings)


class TestPolarisConfiguration:
    """Test main PolarisConfiguration class."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = PolarisConfiguration()
        framework_config = config.get_framework_config()
        
        assert isinstance(framework_config, FrameworkConfiguration)
        assert isinstance(framework_config.nats_config, NATSConfiguration)
    
    def test_configuration_with_yaml_source(self):
        """Test configuration with YAML source."""
        config_data = {
            'framework': {
                'nats_config': {
                    'servers': ['nats://test:4222'],
                    'timeout': 60
                }
            },
            'managed_systems': {
                'test_system': {
                    'connector_type': 'test_connector',
                    'enabled': True
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            yaml_source = YAMLConfigurationSource(temp_file)
            config = PolarisConfiguration([yaml_source])
            
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.servers == ['nats://test:4222']
            assert framework_config.nats_config.timeout == 60
            
            system_config = config.get_managed_system_config('test_system')
            assert system_config is not None
            assert system_config.connector_type == 'test_connector'
            assert system_config.enabled is True
        finally:
            os.unlink(temp_file)
    
    def test_configuration_precedence(self):
        """Test configuration source precedence."""
        # Create YAML config
        yaml_config = {
            'framework': {
                'nats_config': {
                    'timeout': 30,
                    'servers': ['nats://yaml:4222']
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_config, f)
            temp_file = f.name
        
        try:
            # Set environment variables (higher priority)
            with patch.dict(os.environ, {
                'TEST_FRAMEWORK_NATS_CONFIG_TIMEOUT': '120',
                'TEST_FRAMEWORK_NATS_CONFIG_SERVERS': 'nats://env:4222'
            }):
                yaml_source = YAMLConfigurationSource(temp_file, 100)  # Lower priority
                env_source = EnvironmentConfigurationSource("TEST_", 200)  # Higher priority
                
                config = PolarisConfiguration([yaml_source, env_source])
                framework_config = config.get_framework_config()
                
                # Environment should override YAML
                assert framework_config.nats_config.timeout == 120
                assert framework_config.nats_config.servers == ['nats://env:4222']
        finally:
            os.unlink(temp_file)
    
    def test_configuration_reload(self):
        """Test configuration reload functionality."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 30
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            yaml_source = YAMLConfigurationSource(temp_file)
            config = PolarisConfiguration([yaml_source])
            
            # Verify initial configuration
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 30
            
            # Update the file
            updated_config = {
                'framework': {
                    'nats_config': {
                        'timeout': 60
                    }
                }
            }
            
            with open(temp_file, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Reload configuration
            config.reload_configuration()
            
            # Verify updated configuration
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 60
        finally:
            os.unlink(temp_file)


class TestConfigurationBuilder:
    """Test configuration builder."""
    
    def test_builder_basic(self):
        """Test basic configuration builder usage."""
        builder = ConfigurationBuilder()
        config = builder.add_environment_source("TEST_", 100).build()
        
        assert isinstance(config, PolarisConfiguration)
    
    def test_builder_with_yaml_and_env(self):
        """Test builder with YAML and environment sources."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 30
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = (ConfigurationBuilder()
                     .add_yaml_source(temp_file, 100)
                     .add_environment_source("TEST_", 200)
                     .build())
            
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 30
        finally:
            os.unlink(temp_file)
    
    def test_builder_hot_reload(self):
        """Test builder with hot-reload enabled."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 30
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = (ConfigurationBuilder()
                     .add_yaml_source(temp_file, 100)
                     .enable_hot_reload(True)
                     .build())
            
            assert config.is_hot_reload_enabled()
            config.stop_hot_reload()
        finally:
            os.unlink(temp_file)


class TestHotReload:
    """Test hot-reload functionality."""
    
    def test_hot_reload_callback(self):
        """Test hot-reload callback functionality."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 30
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_hot_reload_configuration(temp_file)
            
            # Set up callback to track reloads
            reload_count = [0]
            def on_reload():
                reload_count[0] += 1
            
            config.add_reload_callback(on_reload)
            
            # Verify initial configuration
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 30
            
            # Wait a moment for hot-reload to start
            time.sleep(0.1)
            
            # Modify the configuration file
            updated_config = {
                'framework': {
                    'nats_config': {
                        'timeout': 60
                    }
                }
            }
            
            with open(temp_file, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Wait for hot-reload to detect the change
            max_wait = 5  # seconds
            start_time = time.time()
            
            while reload_count[0] == 0 and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
            
            # Verify reload was detected
            assert reload_count[0] > 0
            
            # Verify configuration was updated
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 60
            
            config.stop_hot_reload()
        finally:
            os.unlink(temp_file)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_load_configuration_from_file(self):
        """Test load_configuration_from_file utility."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 45
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_configuration_from_file(temp_file)
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 45
        finally:
            os.unlink(temp_file)
    
    def test_load_default_configuration(self):
        """Test load_default_configuration utility."""
        config = load_default_configuration()
        framework_config = config.get_framework_config()
        assert isinstance(framework_config, FrameworkConfiguration)
    
    def test_load_hot_reload_configuration(self):
        """Test load_hot_reload_configuration utility."""
        config_data = {
            'framework': {
                'nats_config': {
                    'timeout': 45
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = load_hot_reload_configuration(temp_file)
            assert config.is_hot_reload_enabled()
            
            framework_config = config.get_framework_config()
            assert framework_config.nats_config.timeout == 45
            
            config.stop_hot_reload()
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])