"""
Tests for Configuration Validation System.

This module tests the validation capabilities including
error reporting, suggestions, and integration with existing
configuration managers.
"""

import os
import sys
from pathlib import Path
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.validation import ConfigurationValidator
from polaris.common.validation_result import (
    ValidationResult, ValidationIssue, ValidationSeverity, ValidationCategory
)
from polaris.common.error_suggestions import ConfigurationErrorSuggestionDatabase
from polaris.common.config import ConfigurationManager
from polaris.common.digital_twin_config import DigitalTwinConfigManager


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigurationValidator()
        
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        assert self.validator is not None
        assert hasattr(self.validator, '_suggestion_db')
        assert isinstance(self.validator._suggestion_db, ConfigurationErrorSuggestionDatabase)
    
    def test_validate_valid_framework_config(self):
        """Test validation of a valid framework configuration."""
        valid_config = {
            "nats": {
                "url": "nats://localhost:4222"
            },
            "telemetry": {
                "stream_subject": "polaris.telemetry.events.stream",
                "batch_subject": "polaris.telemetry.events.batch",
                "batch_size": 100
            },
            "logger": {
                "level": "INFO",
                "format": "pretty"
            },
            "execution": {
                "action_subject": "polaris.execution.actions",
                "result_subject": "polaris.execution.results",
                "metrics_subject": "polaris.execution.metrics"
            }
        }
        
        result = self.validator.validate_config_dict(
            config=valid_config,
            config_type="framework"
        )
        
        assert result.valid is True
        # May have suggestions but no errors
        assert len(result.errors) == 0
    
    def test_validate_invalid_nats_url(self):
        """Test validation catches invalid NATS URL."""
        invalid_config = {
            "nats": {
                "url": "http://localhost:4222"  # Wrong protocol
            },
            "telemetry": {
                "stream_subject": "polaris.telemetry.events.stream",
                "batch_subject": "polaris.telemetry.events.batch"
            },
            "logger": {
                "level": "INFO",
                "format": "pretty"
            },
            "execution": {
                "action_subject": "polaris.execution.actions",
                "result_subject": "polaris.execution.results",
                "metrics_subject": "polaris.execution.metrics"
            }
        }
        
        result = self.validator.validate_config_dict(
            config=invalid_config,
            config_type="framework"
        )
        
        assert result.valid is False
        assert len(result.errors) > 0
        
        # Check that we have suggestions for the NATS URL error
        nats_errors = [e for e in result.errors if "nats.url" in e.field_path]
        assert len(nats_errors) > 0
        assert len(nats_errors[0].suggestions) > 0
        assert any("nats://" in suggestion for suggestion in nats_errors[0].suggestions)
    
    def test_validate_invalid_grpc_port(self):
        """Test validation catches invalid gRPC port."""
        invalid_config = {
            "nats": {"url": "nats://localhost:4222"},
            "telemetry": {
                "stream_subject": "polaris.telemetry.events.stream",
                "batch_subject": "polaris.telemetry.events.batch"
            },
            "logger": {"level": "INFO", "format": "pretty"},
            "execution": {
                "action_subject": "polaris.execution.actions",
                "result_subject": "polaris.execution.results",
                "metrics_subject": "polaris.execution.metrics"
            },
            "digital_twin": {
                "grpc": {
                    "port": 99999  # Invalid port
                }
            }
        }
        
        result = self.validator.validate_config_dict(
            config=invalid_config,
            config_type="framework"
        )
        
        assert result.valid is False
        port_errors = [e for e in result.errors if "grpc.port" in e.field_path]
        assert len(port_errors) > 0
        assert len(port_errors[0].suggestions) > 0
    
    def test_validate_config_file_not_found(self):
        """Test validation handles missing configuration file."""
        result = self.validator.validate_config_file(
            config_path="/nonexistent/config.yaml",
            config_type="framework"
        )
        
        assert result.valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].message.lower()
    
    def test_validate_config_file_with_temp_file(self):
        """Test validation with a temporary configuration file."""
        config_data = {
            "nats": {"url": "nats://localhost:4222"},
            "telemetry": {
                "stream_subject": "polaris.telemetry.events.stream",
                "batch_subject": "polaris.telemetry.events.batch"
            },
            "logger": {"level": "INFO", "format": "pretty"},
            "execution": {
                "action_subject": "polaris.execution.actions",
                "result_subject": "polaris.execution.results",
                "metrics_subject": "polaris.execution.metrics"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            result = self.validator.validate_config_file(
                config_path=temp_path,
                config_type="framework"
            )
            
            assert result.valid is True
            assert len(result.errors) == 0
            
        finally:
            os.unlink(temp_path)


class TestValidationResult:
    """Test cases for ValidationResult and related classes."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and basic functionality."""
        result = ValidationResult(valid=True)
        
        assert result.valid is True
        assert len(result.issues) == 0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.infos) == 0
    
    def test_validation_result_add_issue(self):
        """Test adding issues to ValidationResult."""
        result = ValidationResult(valid=True)
        
        error_issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.SCHEMA,
            message="Test error",
            field_path="test.field"
        )
        
        result.add_issue(error_issue)
        
        assert result.valid is False  # Should become false when error added
        assert len(result.errors) == 1
        assert len(result.warnings) == 0
        
        warning_issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category=ValidationCategory.PERFORMANCE,
            message="Test warning",
            field_path="test.field2"
        )
        
        result.add_issue(warning_issue)
        
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
    
    def test_validation_result_format_report(self):
        """Test ValidationResult report formatting."""
        result = ValidationResult(valid=False)
        
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.SCHEMA,
            message="Test error message",
            field_path="test.field",
            suggestions=["Fix suggestion 1", "Fix suggestion 2"]
        ))
        
        report = result.format_report()
        
        assert "❌" in report  # Error emoji
        assert "Test error message" in report
        assert "Fix suggestion 1" in report
        assert "test.field" in report


class TestErrorSuggestionDatabase:
    """Test cases for ConfigurationErrorSuggestionDatabase."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suggestion_db = ConfigurationErrorSuggestionDatabase()
    
    def test_get_suggestions_for_nats_url_error(self):
        """Test getting suggestions for NATS URL errors."""
        suggestions, examples = self.suggestion_db.get_suggestions_for_error(
            field_path="nats.url",
            error_message="NATS URL must start with 'nats://'",
            error_type="pattern_mismatch",
            config_type="framework",
            current_value="http://localhost:4222"
        )
        
        assert len(suggestions) > 0
        assert len(examples) > 0
        assert any("nats://" in suggestion for suggestion in suggestions)
        assert any("nats://localhost:4222" in example for example in examples)
    
    def test_get_suggestions_for_gemini_api_key(self):
        """Test getting suggestions for Gemini API key errors."""
        suggestions, examples = self.suggestion_db.get_suggestions_for_error(
            field_path="gemini.api_key_env",
            error_message="Environment variable not set",
            error_type="missing_env_var",
            config_type="world_model",
            current_value="GEMINI_API_KEY"
        )
        
        assert len(suggestions) > 0
        assert any("export" in suggestion for suggestion in suggestions)
        assert any("API key" in suggestion for suggestion in suggestions)


class TestConfigurationManagerIntegration:
    """Test integration with existing ConfigurationManager."""
    
    def test_configuration_manager_with_validation(self):
        """Test ConfigurationManager with validation enabled."""
        config_manager = ConfigurationManager()
        
        # Check that validation is available
        status = config_manager.get_validation_status()
        assert "validation_available" in status
        assert status["validation_available"] is True
        assert config_manager.validator is not None
    
    def test_configuration_manager_validation_dict(self):
        """Test ConfigurationManager dictionary validation."""
        config_manager = ConfigurationManager()
        
        test_config = {
            "nats": {"url": "nats://localhost:4222"},
            "logger": {"level": "INFO", "format": "pretty"}
        }
        
        result = config_manager.validate_configuration_dict(
            config=test_config,
            config_type="framework"
        )
        
        assert result is not None
        assert isinstance(result, ValidationResult)


class TestDigitalTwinConfigManagerIntegration:
    """Test integration with DigitalTwinConfigManager."""
    
    def test_digital_twin_config_manager_initialization(self):
        """Test DigitalTwinConfigManager with validation."""
        dt_config_manager = DigitalTwinConfigManager()
        
        status = dt_config_manager.get_validation_status()
        assert "validation_available" in status
        assert status["validation_available"] is True
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_world_model_config_validation(self):
        """Test world model configuration validation."""
        dt_config_manager = DigitalTwinConfigManager()
        
        # Test with valid world model config
        world_model_config = {
            "gemini": {
                "api_key_env": "GEMINI_API_KEY",
                "model": "gemini-2.5-flash",
                "temperature": 0.7
            },
            "mock": {
                "enabled": True,
                "response_delay_ms": 100
            }
        }
        
        result = dt_config_manager.validator.validate_config_dict(
            config=world_model_config,
            config_type="world_model"
        )
        
        assert result.valid is True
        assert len(result.errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])