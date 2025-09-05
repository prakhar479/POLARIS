"""
Configuration data models with validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator


class NATSConfiguration(BaseModel):
    """NATS message broker configuration with validation."""
    servers: List[str] = Field(default=["nats://localhost:4222"], min_items=1)
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    timeout: int = Field(default=30, ge=1, le=300)
    
    @validator('servers')
    def validate_servers(cls, v):
        """Validate NATS server URLs."""
        for server in v:
            if not server.startswith(('nats://', 'tls://')):
                raise ValueError(f"Invalid NATS server URL: {server}")
        return v


class TelemetryConfiguration(BaseModel):
    """Telemetry system configuration with validation."""
    enabled: bool = True
    collection_interval: int = Field(default=30, ge=1, le=3600)
    batch_size: int = Field(default=100, ge=1, le=10000)
    retention_days: int = Field(default=30, ge=1, le=365)
    
    @validator('collection_interval')
    def validate_collection_interval(cls, v):
        """Ensure collection interval is reasonable."""
        if v < 5:
            raise ValueError("Collection interval must be at least 5 seconds")
        return v


class LoggingConfiguration(BaseModel):
    """Logging system configuration with validation."""
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(default="json", pattern="^(json|text)$")
    output: str = Field(default="console", pattern="^(console|file|both)$")
    file_path: Optional[str] = None
    max_file_size: int = Field(default=10485760, ge=1048576)  # 10MB default, min 1MB
    backup_count: int = Field(default=5, ge=1, le=100)
    
    @validator('file_path')
    def validate_file_path(cls, v, values):
        """Validate file path when file output is used."""
        if values.get('output') in ['file', 'both'] and not v:
            raise ValueError("file_path is required when output is 'file' or 'both'")
        return v


class FrameworkConfiguration(BaseModel):
    """Core framework configuration with nested validation."""
    nats_config: NATSConfiguration = Field(default_factory=NATSConfiguration)
    telemetry_config: TelemetryConfiguration = Field(default_factory=TelemetryConfiguration)
    logging_config: LoggingConfiguration = Field(default_factory=LoggingConfiguration)
    plugin_search_paths: List[str] = Field(default=["./plugins"])
    max_concurrent_adaptations: int = Field(default=10, ge=1, le=100)
    adaptation_timeout: int = Field(default=300, ge=30, le=3600)


class ManagedSystemConfiguration(BaseModel):
    """Configuration for a managed system with validation."""
    system_id: str = Field(min_length=1, max_length=100)
    connector_type: str = Field(min_length=1)
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    monitoring_config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    
    @validator('system_id')
    def validate_system_id(cls, v):
        """Validate system ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("system_id must contain only alphanumeric characters, hyphens, and underscores")
        return v