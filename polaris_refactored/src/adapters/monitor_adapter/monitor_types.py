"""
This module defines dataclasses and enums that are used to configure monitoring
targets and to represent the results of metric collection.

The `MonitoringTarget` class is used to configure the monitoring of a specific
managed system. It contains information about the system, such as its ID, the
type of connector to use, the interval at which to collect metrics, and any
additional configuration options.

The `CollectionResult` class is used to represent the result of metric collection.
It contains a timestamp indicating when the collection occurred, a list of
`MetricValue` objects representing the metrics that were collected, and a reference
to the target system being monitored.

The `MetricCollectionMode` enum defines the different modes in which metrics can
be collected, including "pull" (where the monitor actively collects metrics) and
"push" (where metrics are received from the managed system).

The `MetricValue` class represents a single metric value. It contains the name of
the metric, its value, and a timestamp indicating when the value was collected.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ...domain.models import MetricValue


logger = logging.getLogger(__name__)


class MetricCollectionMode(Enum):
    """Modes for metric collection."""
    PULL = "pull"  # Actively collect metrics
    PUSH = "push"  # Receive metrics from systems
    HYBRID = "hybrid"  # Both pull and push


@dataclass
class MonitoringTarget:
    """Configuration for a monitoring target system."""
    system_id: str
    connector_type: str
    collection_interval: float = 30.0  # seconds
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate monitoring target configuration."""
        errors = []
        
        if not self.system_id or not isinstance(self.system_id, str):
            errors.append("system_id must be a non-empty string")
        
        if not self.connector_type or not isinstance(self.connector_type, str):
            errors.append("connector_type must be a non-empty string")
        
        if self.collection_interval <= 0:
            errors.append("collection_interval must be positive")
        
        if not isinstance(self.enabled, bool):
            errors.append("enabled must be a boolean")
        
        if not isinstance(self.config, dict):
            errors.append("config must be a dictionary")
        
        return errors


@dataclass
class CollectionResult:
    """Result of a metric collection operation."""
    system_id: str
    metrics: Dict[str, MetricValue]
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    collection_duration: Optional[timedelta] = None
    strategy_name: Optional[str] = None

