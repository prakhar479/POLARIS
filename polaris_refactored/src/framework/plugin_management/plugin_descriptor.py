"""
Plugin Descriptor Module

Defines data structures for plugin metadata and information.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class PluginDescriptor:
    """Describes a discovered plugin with metadata and validation info."""
    plugin_id: str
    plugin_type: str
    version: str
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    connector_class_name: str = ""
    module_name: str = ""
    last_modified: float = 0.0
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)