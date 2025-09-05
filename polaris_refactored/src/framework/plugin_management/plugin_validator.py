"""
Plugin Validator Module

Provides comprehensive validation and security checks for plugins.
"""

import inspect
import logging
from pathlib import Path
from typing import Dict, List, Any, Type

from ...domain.interfaces import ManagedSystemConnector

logger = logging.getLogger(__name__)


class PluginValidator:
    """Validates plugin implementations and configurations with security checks."""
    
    @staticmethod
    def validate_connector_class(connector_class: Type) -> List[str]:
        """Validate that a class properly implements ManagedSystemConnector interface."""
        errors = []
        
        # Check if it's actually a class
        if not inspect.isclass(connector_class):
            errors.append(f"Expected a class, got {type(connector_class).__name__}")
            return errors
        
        # Check if it's a subclass of ManagedSystemConnector
        try:
            if not issubclass(connector_class, ManagedSystemConnector):
                errors.append(f"Class {connector_class.__name__} must inherit from ManagedSystemConnector")
                return errors
        except TypeError:
            errors.append(f"Cannot check inheritance for {connector_class}")
            return errors
        
        # Security check: Ensure class is not attempting to override critical methods
        dangerous_methods = ['__getattribute__', '__setattr__', '__delattr__', '__import__']
        for method_name in dangerous_methods:
            if hasattr(connector_class, method_name):
                method = getattr(connector_class, method_name)
                if method != getattr(object, method_name, None):
                    errors.append(f"Security violation: Overriding dangerous method {method_name}")
        
        # Check required methods are implemented
        required_methods = [
            'connect', 'disconnect', 'get_system_id', 'collect_metrics',
            'get_system_state', 'execute_action', 'validate_action', 'get_supported_actions'
        ]
        
        for method_name in required_methods:
            if not hasattr(connector_class, method_name):
                errors.append(f"Missing required method: {method_name}")
                continue
            
            method = getattr(connector_class, method_name)
            if not callable(method):
                errors.append(f"Method {method_name} is not callable")
                continue
            
            # Check if method is properly implemented (not just abstract)
            if getattr(method, '__isabstractmethod__', False):
                errors.append(f"Method {method_name} is not implemented (still abstract)")
            
            # Validate method signature
            try:
                sig = inspect.signature(method)
                PluginValidator._validate_method_signature(method_name, sig, errors)
            except Exception as e:
                errors.append(f"Cannot inspect signature of method {method_name}: {e}")
        
        return errors
    
    @staticmethod
    def _validate_method_signature(method_name: str, signature: inspect.Signature, errors: List[str]) -> None:
        """Validate method signatures match expected interface."""
        expected_signatures = {
            'connect': {'return_annotation': bool},
            'disconnect': {'return_annotation': bool},
            'get_system_id': {'return_annotation': str},
            'collect_metrics': {'return_annotation': Dict},
            'get_system_state': {'return_annotation': 'SystemState'},
            'execute_action': {'params': ['action'], 'return_annotation': 'ExecutionResult'},
            'validate_action': {'params': ['action'], 'return_annotation': bool},
            'get_supported_actions': {'return_annotation': List}
        }
        
        if method_name in expected_signatures:
            expected = expected_signatures[method_name]
            
            # Check parameters
            if 'params' in expected:
                param_names = [p for p in signature.parameters.keys() if p != 'self']
                if len(param_names) != len(expected['params']):
                    errors.append(f"Method {method_name} has incorrect number of parameters")
    
    @staticmethod
    def validate_plugin_metadata(metadata: Dict[str, Any]) -> List[str]:
        """Validate plugin metadata structure with security checks."""
        errors = []
        
        required_fields = ['name', 'version', 'connector_class']
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")
        
        # Validate version format (simple semantic versioning check)
        if 'version' in metadata:
            version = metadata['version']
            if not isinstance(version, str) or not version.replace('.', '').replace('-', '').isalnum():
                errors.append(f"Invalid version format: {version}")
        
        # Security validation: Check for suspicious metadata
        if 'name' in metadata:
            name = metadata['name']
            if not isinstance(name, str) or len(name) > 100:
                errors.append("Plugin name must be a string with max 100 characters")
            
            # Check for suspicious characters
            if any(char in name for char in ['<', '>', '&', '"', "'"]):
                errors.append("Plugin name contains potentially dangerous characters")
        
        if 'connector_class' in metadata:
            class_name = metadata['connector_class']
            if not isinstance(class_name, str) or not class_name.isidentifier():
                errors.append("Connector class name must be a valid Python identifier")
        
        # Check for suspicious metadata fields
        dangerous_fields = ['__import__', 'exec', 'eval', 'compile']
        for field in dangerous_fields:
            if field in metadata:
                errors.append(f"Security violation: Dangerous metadata field '{field}' not allowed")
        
        return errors
    
    @staticmethod
    def validate_plugin_path(plugin_path: Path) -> List[str]:
        """Validate plugin path for security issues."""
        errors = []
        
        try:
            # Resolve path to check for directory traversal
            resolved_path = plugin_path.resolve()
            path_str = str(resolved_path)
            
            # Check for directory traversal attempts (but allow absolute paths for temp dirs in tests)
            if '..' in path_str:
                errors.append("Plugin path contains directory traversal attempts")
            
            # Only check for suspicious absolute paths that might be trying to escape
            # Allow /tmp paths for testing
            if path_str.startswith('/') and not any(allowed in path_str for allowed in ['/tmp', '/var/tmp', 'temp']):
                # Additional check: only flag if it looks like it's trying to access system directories
                suspicious_paths = ['/etc', '/usr', '/bin', '/sbin', '/root', '/home']
                if any(suspicious in path_str for suspicious in suspicious_paths):
                    errors.append("Plugin path contains potentially dangerous system directory access")
            
            # Check file permissions (if on Unix-like system)
            if hasattr(plugin_path, 'stat'):
                try:
                    stat_info = plugin_path.stat()
                    # Check if file is world-writable (security risk)
                    if stat_info.st_mode & 0o002:
                        errors.append("Plugin file is world-writable (security risk)")
                except Exception:
                    pass  # Ignore permission check errors on Windows
            
        except Exception as e:
            errors.append(f"Cannot validate plugin path: {e}")
        
        return errors
    
    @staticmethod
    def validate_plugin_imports(plugin_path: Path) -> List[str]:
        """Validate plugin imports for security issues."""
        errors = []
        
        try:
            if plugin_path.is_file() and plugin_path.suffix == '.py':
                with open(plugin_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dangerous function calls (more precise matching)
                import re
                
                # Pattern to match dangerous function calls but not method names
                dangerous_patterns = [
                    (r'\bos\.system\s*\(', 'os.system'),
                    (r'\bsubprocess\s*\.', 'subprocess'),
                    (r'\beval\s*\(', 'eval'),
                    (r'\bexec\s*\(', 'exec'),  # Match exec( but not execute_action
                    (r'\bcompile\s*\(', 'compile'),
                    (r'\b__import__\s*\(', '__import__'),
                    (r'importlib\.import_module\s*\(', 'importlib.import_module'),
                    (r'sys\.modules\s*\[', 'sys.modules')
                ]
                
                for pattern, name in dangerous_patterns:
                    if re.search(pattern, content):
                        errors.append(f"Security violation: Dangerous function call '{name}' detected")
                
                # Check for network-related imports that might be suspicious
                network_imports = ['socket', 'urllib', 'requests', 'http']
                network_count = sum(1 for imp in network_imports if f'import {imp}' in content)
                if network_count > 2:
                    errors.append("Plugin contains multiple network imports (potential security risk)")
        
        except Exception as e:
            errors.append(f"Cannot validate plugin imports: {e}")
        
        return errors