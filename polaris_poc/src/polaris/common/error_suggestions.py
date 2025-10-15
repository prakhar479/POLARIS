"""
Configuration Error Suggestion Database.

This module provides comprehensive error suggestion mappings for common
configuration issues, context-aware suggestions based on configuration paths,
and examples of correct configuration for each error type.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .validation_result import ValidationIssue, ValidationSeverity, ValidationCategory


class ConfigurationErrorSuggestionDatabase:
    """
    Comprehensive database of error suggestions for POLARIS configuration issues.
    
    Provides context-aware suggestions based on configuration paths, error types,
    and common configuration mistakes. Includes examples of correct configuration
    and fix recommendations for interdependency violations.
    """
    
    def __init__(self):
        """Initialize the error suggestion database."""
        self._field_suggestions = self._build_field_specific_suggestions()
        self._error_type_suggestions = self._build_error_type_suggestions()
        self._interdependency_suggestions = self._build_interdependency_suggestions()
        self._security_suggestions = self._build_security_suggestions()
        self._performance_suggestions = self._build_performance_suggestions()
        self._examples_database = self._build_examples_database()
    
    def get_suggestions_for_error(
        self,
        field_path: str,
        error_message: str,
        error_type: str,
        config_type: str = "unknown",
        current_value: Any = None
    ) -> Tuple[List[str], List[str]]:
        """Get suggestions and examples for a configuration error.
        
        Args:
            field_path: Dot-separated path to the configuration field
            error_message: Original error message
            error_type: Type of validation error
            config_type: Type of configuration (framework, world_model, plugin)
            current_value: Current value that caused the error
            
        Returns:
            Tuple of (suggestions, examples)
        """
        suggestions = []
        examples = []
        
        # Get field-specific suggestions
        field_suggestions = self._get_field_specific_suggestions(field_path, error_type, current_value)
        suggestions.extend(field_suggestions)
        
        # Get error type suggestions
        type_suggestions = self._get_error_type_suggestions(error_type, error_message)
        suggestions.extend(type_suggestions)
        
        # Get context-specific suggestions
        context_suggestions = self._get_context_specific_suggestions(field_path, config_type, error_type)
        suggestions.extend(context_suggestions)
        
        # Get examples
        field_examples = self._get_field_examples(field_path, error_type)
        examples.extend(field_examples)
        
        # Remove duplicates while preserving order
        suggestions = list(dict.fromkeys(suggestions))
        examples = list(dict.fromkeys(examples))
        
        return suggestions, examples
    
    def get_interdependency_suggestions(
        self,
        source_field: str,
        target_field: str,
        dependency_type: str,
        config_type: str = "unknown"
    ) -> List[str]:
        """Get suggestions for interdependency violations.
        
        Args:
            source_field: Field that depends on another
            target_field: Field that is depended upon
            dependency_type: Type of dependency (reference, validation, etc.)
            config_type: Type of configuration
            
        Returns:
            List of suggestions for fixing the interdependency issue
        """
        key = f"{source_field}->{target_field}"
        
        if key in self._interdependency_suggestions:
            return self._interdependency_suggestions[key].copy()
        
        # Generic interdependency suggestions
        generic_suggestions = [
            f"Ensure {target_field} is properly configured before setting {source_field}",
            f"Check that {target_field} exists and has a valid value",
            f"Verify the relationship between {source_field} and {target_field}"
        ]
        
        if dependency_type == "reference":
            generic_suggestions.append(f"Update {source_field} to reference a valid {target_field}")
        elif dependency_type == "validation":
            generic_suggestions.append(f"Ensure {target_field} meets the requirements for {source_field}")
        
        return generic_suggestions
    
    def get_security_suggestions(self, field_path: str, issue_type: str) -> List[str]:
        """Get security-related suggestions for configuration fields.
        
        Args:
            field_path: Configuration field path
            issue_type: Type of security issue
            
        Returns:
            List of security suggestions
        """
        key = f"{field_path}:{issue_type}"
        
        if key in self._security_suggestions:
            return self._security_suggestions[key].copy()
        
        # Generic security suggestions based on field type
        if "password" in field_path.lower() or "key" in field_path.lower():
            return [
                "Never store sensitive values in configuration files",
                "Use environment variables for secrets",
                "Consider using a secrets management system",
                "Ensure proper file permissions (600 or 640)"
            ]
        elif "host" in field_path.lower() or "url" in field_path.lower():
            return [
                "Use specific hostnames instead of wildcards in production",
                "Enable TLS/SSL encryption for network connections",
                "Implement proper authentication and authorization",
                "Consider network segmentation and firewall rules"
            ]
        
        return ["Review security implications of this configuration"]
    
    def get_performance_suggestions(self, field_path: str, current_value: Any) -> List[str]:
        """Get performance-related suggestions for configuration fields.
        
        Args:
            field_path: Configuration field path
            current_value: Current value of the field
            
        Returns:
            List of performance suggestions
        """
        suggestions = []
        
        # Check performance suggestion database
        for pattern, perf_suggestions in self._performance_suggestions.items():
            if re.match(pattern, field_path):
                suggestions.extend(perf_suggestions)
        
        # Value-specific performance suggestions
        if isinstance(current_value, int):
            if "batch_size" in field_path and current_value > 1000:
                suggestions.append("Large batch sizes may cause memory issues - consider reducing")
            elif "timeout" in field_path and current_value < 5:
                suggestions.append("Very short timeouts may cause frequent failures")
            elif "concurrent" in field_path and current_value > 100:
                suggestions.append("High concurrency may overwhelm external services")
        
        return suggestions
    
    def _get_field_specific_suggestions(
        self,
        field_path: str,
        error_type: str,
        current_value: Any
    ) -> List[str]:
        """Get suggestions specific to a configuration field."""
        suggestions = []
        
        # Direct field match
        if field_path in self._field_suggestions:
            field_data = self._field_suggestions[field_path]
            if error_type in field_data:
                suggestions.extend(field_data[error_type])
            elif "default" in field_data:
                suggestions.extend(field_data["default"])
        
        # Pattern matching for dynamic fields
        for pattern, pattern_data in self._field_suggestions.items():
            if "*" in pattern and re.match(pattern.replace("*", ".*"), field_path):
                if error_type in pattern_data:
                    suggestions.extend(pattern_data[error_type])
                elif "default" in pattern_data:
                    suggestions.extend(pattern_data["default"])
        
        return suggestions
    
    def _get_error_type_suggestions(self, error_type: str, error_message: str) -> List[str]:
        """Get suggestions based on error type."""
        suggestions = []
        
        if error_type in self._error_type_suggestions:
            suggestions.extend(self._error_type_suggestions[error_type])
        
        # Message-based suggestions
        error_message_lower = error_message.lower()
        
        if "required" in error_message_lower:
            suggestions.append("This field is mandatory and cannot be omitted")
        elif "additional properties" in error_message_lower:
            suggestions.append("Remove unknown properties or check for typos")
        elif "not of type" in error_message_lower:
            suggestions.append("Check the data type - ensure quotes for strings, no quotes for numbers/booleans")
        
        return suggestions
    
    def _get_context_specific_suggestions(
        self,
        field_path: str,
        config_type: str,
        error_type: str
    ) -> List[str]:
        """Get suggestions based on configuration context."""
        suggestions = []
        
        # Framework-specific suggestions
        if config_type == "framework":
            if field_path.startswith("nats"):
                suggestions.append("NATS configuration affects all inter-component communication")
            elif field_path.startswith("digital_twin"):
                suggestions.append("Digital Twin settings impact AI reasoning capabilities")
            elif field_path.startswith("telemetry"):
                suggestions.append("Telemetry settings affect monitoring data flow and performance")
        
        # World Model-specific suggestions
        elif config_type == "world_model":
            if field_path.startswith("gemini"):
                suggestions.extend([
                    "Gemini configuration requires valid API key and internet access",
                    "Monitor API usage and costs for production deployments"
                ])
            elif field_path.startswith("statistical"):
                suggestions.append("Statistical models require sufficient historical data")
        
        # Plugin-specific suggestions
        elif config_type == "plugin":
            if field_path.startswith("connection"):
                suggestions.append("Connection settings must match the target system's configuration")
            elif field_path.startswith("monitoring"):
                suggestions.append("Monitoring configuration affects telemetry data collection")
        
        return suggestions
    
    def _get_field_examples(self, field_path: str, error_type: str) -> List[str]:
        """Get examples for a configuration field."""
        examples = []
        
        # Direct field examples
        if field_path in self._examples_database:
            examples.extend(self._examples_database[field_path])
        
        # Pattern-based examples
        for pattern, pattern_examples in self._examples_database.items():
            if "*" in pattern and re.match(pattern.replace("*", ".*"), field_path):
                examples.extend(pattern_examples)
        
        return examples
    
    def _build_field_specific_suggestions(self) -> Dict[str, Dict[str, List[str]]]:
        """Build field-specific error suggestions."""
        return {
            # NATS Configuration
            "nats.url": {
                "pattern_mismatch": [
                    "NATS URL must start with 'nats://' protocol",
                    "Format: nats://hostname:port",
                    "For TLS: use 'tls://hostname:port'",
                    "For authentication: nats://user:pass@hostname:port"
                ],
                "missing_required": [
                    "NATS URL is required for message bus communication",
                    "Use 'nats://localhost:4222' for local development",
                    "Use service name in Docker: 'nats://nats:4222'"
                ],
                "connection_failed": [
                    "Ensure NATS server is running and accessible",
                    "Check network connectivity and firewall rules",
                    "Verify hostname resolution and port availability"
                ]
            },
            
            # Digital Twin gRPC Configuration
            "digital_twin.grpc.port": {
                "out_of_range": [
                    "Port must be between 1 and 65535",
                    "Common gRPC ports: 50051 (default), 8080, 9090",
                    "Check if port is already in use with: netstat -an | grep :PORT"
                ],
                "type_mismatch": [
                    "Port must be an integer (number without quotes)",
                    "Remove quotes around port number in YAML"
                ],
                "port_in_use": [
                    "Choose a different port number",
                    "Stop the service using the port",
                    "Use 'lsof -i :PORT' to find which process is using the port"
                ]
            },
            
            "digital_twin.grpc.host": {
                "security_warning": [
                    "Binding to 0.0.0.0 accepts connections from any interface",
                    "Use 127.0.0.1 for local-only access",
                    "Use specific IP address for production deployments",
                    "Implement proper authentication and TLS encryption"
                ]
            },
            
            # World Model Configuration
            "digital_twin.world_model.implementation": {
                "invalid_enum": [
                    "Valid implementations: mock, gemini, bayesian, statistical, hybrid",
                    "Use 'mock' for development and testing (no external dependencies)",
                    "Use 'gemini' for AI-powered reasoning (requires API key)",
                    "Use 'bayesian' for deterministic statistical predictions (Kalman filtering)",
                    "Use 'statistical' for time series analysis (requires historical data)",
                    "Use 'hybrid' for combined approach (most robust)"
                ]
            },
            
            # Gemini Configuration
            "gemini.api_key_env": {
                "missing_env_var": [
                    "Set the environment variable: export GEMINI_API_KEY=your_key",
                    "Obtain API key from Google AI Studio (https://makersuite.google.com/app/apikey)",
                    "Never store API keys directly in configuration files",
                    "Use secrets management in production environments"
                ]
            },
            
            "gemini.temperature": {
                "out_of_range": [
                    "Temperature must be between 0.0 and 2.0",
                    "Use 0.0 for deterministic responses",
                    "Use 0.7 for balanced creativity (recommended)",
                    "Use 1.0+ for more creative/varied responses"
                ]
            },
            
            "gemini.model": {
                "invalid_model": [
                    "Valid models: gemini-2.5-flash, gemini-2.5-pro, gemini-1.0-pro",
                    "gemini-2.5-flash: Fast and cost-effective",
                    "gemini-2.5-pro: More capable but higher cost",
                    "Check Google AI documentation for latest model versions"
                ]
            },
            
            # Bayesian Configuration
            "bayesian.prediction_horizon_minutes": {
                "out_of_range": [
                    "Prediction horizon must be between 1 and 1440 minutes (24 hours)",
                    "Use 60-120 minutes for most applications",
                    "Longer horizons require more computation and may be less accurate",
                    "Consider your system's dynamics when setting horizon"
                ]
            },
            
            "bayesian.correlation_threshold": {
                "out_of_range": [
                    "Correlation threshold must be between 0.0 and 1.0",
                    "Use 0.7 for moderate correlation detection (recommended)",
                    "Use 0.8-0.9 for strong correlations only",
                    "Lower values detect more correlations but may include noise"
                ]
            },
            
            "bayesian.anomaly_threshold": {
                "out_of_range": [
                    "Anomaly threshold must be between 0.1 and 10.0 standard deviations",
                    "Use 2.0-2.5 for balanced anomaly detection (recommended)",
                    "Use 3.0+ for fewer false positives",
                    "Use 1.5-2.0 for more sensitive detection"
                ]
            },
            
            "bayesian.process_noise": {
                "out_of_range": [
                    "Process noise must be between 0.001 and 1.0",
                    "Use 0.01 for stable systems (recommended)",
                    "Use 0.05-0.1 for more dynamic systems",
                    "Higher values allow faster adaptation but reduce stability"
                ]
            },
            
            "bayesian.measurement_noise": {
                "out_of_range": [
                    "Measurement noise must be between 0.001 and 1.0",
                    "Use 0.1 for typical sensor noise (recommended)",
                    "Use 0.05 for high-quality measurements",
                    "Use 0.2-0.5 for noisy measurements"
                ]
            },
            
            # Telemetry Configuration
            "telemetry.batch_size": {
                "performance_warning": [
                    "Large batch sizes (>1000) may cause memory issues",
                    "Small batch sizes (<10) reduce efficiency",
                    "Recommended range: 50-500 for most use cases",
                    "Balance throughput vs memory consumption"
                ]
            },
            
            "telemetry.batch_max_wait": {
                "performance_tip": [
                    "Lower values = more real-time but less efficient batching",
                    "Higher values = better batching but higher latency",
                    "Recommended range: 0.5-2.0 seconds"
                ]
            },
            
            # Logger Configuration
            "logger.level": {
                "invalid_enum": [
                    "Valid log levels: DEBUG, INFO, WARNING, ERROR",
                    "Use DEBUG for development (verbose, performance impact)",
                    "Use INFO for production (balanced information)",
                    "Use WARNING for minimal logging",
                    "Use ERROR for troubleshooting only"
                ]
            },
            
            "logger.format": {
                "invalid_enum": [
                    "Valid formats: pretty, json",
                    "Use 'pretty' for development (human-readable, colored)",
                    "Use 'json' for production (structured, log aggregation)"
                ]
            },
            
            # Plugin Configuration
            "system_name": {
                "pattern_mismatch": [
                    "System name must contain only letters, numbers, hyphens, and underscores",
                    "Start with a letter or number",
                    "Keep names descriptive but concise",
                    "Examples: web_server, database-1, monitoring_system"
                ]
            },
            
            "implementation.connector_class": {
                "pattern_mismatch": [
                    "Use valid Python import path format",
                    "Format: module.submodule.ClassName",
                    "Ensure the class exists and is importable",
                    "Example: polaris.adapters.MyConnector"
                ]
            },
            
            # Connection Configuration
            "connection.host": {
                "localhost_warning": [
                    "Using localhost limits connections to local machine only",
                    "Use specific hostname or IP for remote connections",
                    "Consider DNS resolution and network accessibility"
                ]
            },
            
            "connection.port": {
                "out_of_range": [
                    "Port must be between 1 and 65535",
                    "Check if port is available on target system",
                    "Verify firewall rules allow connections"
                ]
            },
            
            # Monitoring Configuration
            "monitoring.interval": {
                "performance_tip": [
                    "Lower intervals = more real-time but higher overhead",
                    "Higher intervals = less overhead but less responsive",
                    "Consider system load and monitoring requirements"
                ]
            },
            
            # Pattern-based suggestions for dynamic fields
            "*.timeout*": {
                "performance_tip": [
                    "Balance timeout values with expected operation duration",
                    "Too short: frequent failures due to normal delays",
                    "Too long: slow failure detection and recovery"
                ]
            },
            
            "*.max_*": {
                "resource_warning": [
                    "High limits may consume excessive resources",
                    "Monitor actual usage and adjust accordingly",
                    "Consider system capacity and other processes"
                ]
            }
        }
    
    def _build_error_type_suggestions(self) -> Dict[str, List[str]]:
        """Build error type-specific suggestions."""
        return {
            "required": [
                "This field is mandatory and must be provided",
                "Check if the field name is spelled correctly",
                "Ensure proper YAML/JSON indentation and structure"
            ],
            
            "type": [
                "Check the data type of the value",
                "Strings should be quoted, numbers and booleans should not be",
                "Arrays use [] brackets, objects use {} braces"
            ],
            
            "pattern": [
                "Value must match the required format/pattern",
                "Check for typos and correct syntax",
                "Refer to documentation for valid formats"
            ],
            
            "enum": [
                "Value must be one of the allowed options",
                "Check for typos in the value",
                "Refer to documentation for valid choices"
            ],
            
            "minimum": [
                "Value is below the minimum allowed",
                "Increase the value to meet requirements",
                "Consider the impact of the minimum constraint"
            ],
            
            "maximum": [
                "Value exceeds the maximum allowed",
                "Reduce the value to meet requirements",
                "Consider the impact of the maximum constraint"
            ],
            
            "additionalProperties": [
                "Unknown or extra properties are not allowed",
                "Remove unrecognized fields",
                "Check for typos in field names"
            ],
            
            "dependencies": [
                "This field requires other fields to be present",
                "Check interdependency requirements",
                "Ensure all related fields are configured"
            ]
        }
    
    def _build_interdependency_suggestions(self) -> Dict[str, List[str]]:
        """Build interdependency-specific suggestions."""
        return {
            "digital_twin.world_model.config_path->world_model.*": [
                "Ensure world_model.yaml exists at the specified path",
                "Check that world_model.yaml is valid and readable",
                "Verify the path is relative to the framework config directory",
                "Update config_path if world_model.yaml is moved"
            ],
            
            "digital_twin.world_model.implementation->gemini.*": [
                "Gemini implementation requires gemini section in world_model.yaml",
                "Ensure GEMINI_API_KEY environment variable is set",
                "Verify internet connectivity for API access",
                "Check Google AI API quotas and billing"
            ],
            
            "digital_twin.world_model.implementation->statistical.*": [
                "Statistical implementation requires statistical section in world_model.yaml",
                "Ensure sufficient historical data is available",
                "Configure appropriate window sizes and algorithms",
                "Verify statistical libraries are installed"
            ],
            
            "digital_twin.world_model.implementation->bayesian.*": [
                "Bayesian implementation requires bayesian section in world_model.yaml",
                "Install required packages: pip install numpy scipy filterpy",
                "Configure Kalman filter parameters (process_noise, measurement_noise)",
                "Set appropriate correlation and anomaly thresholds",
                "Ensure sufficient memory for historical data storage"
            ],
            
            "telemetry.stream_subject->nats.url": [
                "Telemetry subjects require NATS server to be accessible",
                "Ensure NATS URL is valid and server is running",
                "Check NATS subject permissions and authentication",
                "Verify network connectivity to NATS server"
            ],
            
            "execution.action_subject->nats.url": [
                "Execution subjects require NATS server to be accessible",
                "Ensure NATS URL is valid and server is running",
                "Check NATS subject permissions for publishing/subscribing",
                "Verify execution adapters can connect to NATS"
            ],
            
            "plugin.monitoring.metrics->framework.telemetry": [
                "Plugin metrics must be compatible with framework telemetry configuration",
                "Ensure telemetry subjects are accessible from plugins",
                "Check metric format compatibility",
                "Verify NATS connectivity from plugin systems"
            ]
        }
    
    def _build_security_suggestions(self) -> Dict[str, List[str]]:
        """Build security-specific suggestions."""
        return {
            "nats.url:insecure_protocol": [
                "Consider using TLS: tls://hostname:port",
                "Enable NATS authentication in production",
                "Use VPN or private networks for NATS communication",
                "Implement proper network segmentation"
            ],
            
            "digital_twin.grpc.host:wildcard_binding": [
                "Avoid binding to 0.0.0.0 in production",
                "Use specific IP addresses or localhost",
                "Implement gRPC authentication (JWT, mTLS)",
                "Enable TLS encryption for gRPC"
            ],
            
            "gemini.api_key_env:missing_key": [
                "Never store API keys in configuration files",
                "Use environment variables or secrets management",
                "Rotate API keys regularly",
                "Monitor API key usage and access"
            ],
            
            "logger.level:debug_in_production": [
                "DEBUG logging may expose sensitive information",
                "Use INFO or WARNING level in production",
                "Implement log sanitization for sensitive data",
                "Secure log storage and access"
            ],
            
            "connection.auth:weak_auth": [
                "Use strong authentication methods",
                "Avoid basic authentication over unencrypted connections",
                "Implement certificate-based authentication",
                "Use secure credential storage"
            ]
        }
    
    def _build_performance_suggestions(self) -> Dict[str, List[str]]:
        """Build performance-specific suggestions."""
        return {
            r".*\.batch_size": [
                "Balance batch size with memory usage",
                "Larger batches = better throughput, higher latency",
                "Smaller batches = lower latency, more overhead",
                "Monitor memory consumption with large batches"
            ],
            
            r".*\.timeout.*": [
                "Set timeouts based on expected operation duration",
                "Consider network latency and processing time",
                "Too short = frequent failures, too long = slow recovery",
                "Monitor actual operation times and adjust"
            ],
            
            r".*\.concurrent.*": [
                "High concurrency may overwhelm target systems",
                "Consider API rate limits and system capacity",
                "Monitor resource usage and response times",
                "Implement backpressure and circuit breakers"
            ],
            
            r".*\.queue.*": [
                "Large queues consume memory but provide buffering",
                "Small queues may cause backpressure under load",
                "Monitor queue depth and processing rates",
                "Consider queue persistence for reliability"
            ]
        }
    
    def _build_examples_database(self) -> Dict[str, List[str]]:
        """Build database of configuration examples."""
        return {
            # NATS Examples
            "nats.url": [
                "nats://localhost:4222",
                "nats://nats-server:4222",
                "nats://user:pass@nats-cluster:4222",
                "tls://secure-nats:4222"
            ],
            
            # gRPC Examples
            "digital_twin.grpc.port": ["50051", "8080", "9090", "50052"],
            "digital_twin.grpc.host": ["127.0.0.1", "0.0.0.0", "10.0.1.100"],
            
            # World Model Examples
            "digital_twin.world_model.implementation": ["mock", "gemini", "bayesian", "statistical", "hybrid"],
            
            # Gemini Examples
            "gemini.model": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.0-pro"],
            "gemini.temperature": ["0.0", "0.3", "0.7", "1.0"],
            "gemini.api_key_env": ["GEMINI_API_KEY", "GOOGLE_AI_API_KEY"],
            
            # Bayesian Examples
            "bayesian.prediction_horizon_minutes": ["60", "120", "240", "480"],
            "bayesian.correlation_threshold": ["0.5", "0.7", "0.8", "0.9"],
            "bayesian.anomaly_threshold": ["2.0", "2.5", "3.0"],
            "bayesian.process_noise": ["0.01", "0.05", "0.1"],
            "bayesian.measurement_noise": ["0.1", "0.2", "0.5"],
            "bayesian.learning_rate": ["0.01", "0.05", "0.1", "0.2"],
            "bayesian.max_history_points": ["1000", "2000", "5000", "10000"],
            
            # Logger Examples
            "logger.level": ["DEBUG", "INFO", "WARNING", "ERROR"],
            "logger.format": ["pretty", "json"],
            
            # Telemetry Examples
            "telemetry.batch_size": ["10", "50", "100", "500"],
            "telemetry.batch_max_wait": ["0.5", "1.0", "2.0"],
            
            # Plugin Examples
            "system_name": ["web_server", "database-1", "monitoring_system", "api_gateway"],
            "implementation.connector_class": [
                "polaris.adapters.HttpConnector",
                "extern.swim.connector.SwimConnector",
                "plugins.database.MySQLConnector"
            ],
            
            # Connection Examples
            "connection.protocol": ["tcp", "http", "https", "grpc", "mqtt"],
            "connection.host": ["localhost", "192.168.1.100", "api.example.com"],
            "connection.port": ["80", "443", "8080", "3306", "5432"],
            
            # Monitoring Examples
            "monitoring.interval": ["1", "5", "10", "30", "60"],
            
            # Statistical Examples
            "statistical.time_series.window_size": ["50", "100", "500", "1000"],
            "statistical.anomaly_detection.method": ["isolation_forest", "one_class_svm", "local_outlier_factor"]
        }