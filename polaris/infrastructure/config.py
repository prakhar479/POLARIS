"""
Configuration loader for Polaris.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """Configuration for a managed system."""
    id: str
    connector_type: str
    enabled: bool = True
    connection: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate system configuration after initialization."""
        if not self.id or not self.id.strip():
            raise ValueError("System ID cannot be empty")
        
        # Validate connector type
        supported_connectors = ["swim"]  # Add more as implemented
        if self.connector_type not in supported_connectors:
            raise ValueError(f"Unsupported connector type '{self.connector_type}'. Supported: {supported_connectors}")
        
        # Validate connection parameters for SWIM
        if self.connector_type == "swim" and self.connection:
            if "port" in self.connection and not isinstance(self.connection["port"], int):
                raise ValueError("SWIM connection port must be an integer")
            if "port" in self.connection and not (1 <= self.connection["port"] <= 65535):
                raise ValueError("SWIM connection port must be between 1 and 65535")


@dataclass
class StrategyConfig:
    """Configuration for adaptation strategy."""
    type: str = "threshold"
    threshold: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    hybrid: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate strategy configuration after initialization."""
        supported_strategies = ["threshold", "llm_reasoning", "hybrid"]
        if self.type not in supported_strategies:
            raise ValueError(f"Unsupported strategy type '{self.type}'. Supported: {supported_strategies}")
        
        # Validate threshold strategy parameters
        if self.type == "threshold" and self.threshold:
            thresholds = self.threshold.get("thresholds", {})
            for metric, values in thresholds.items():
                if not isinstance(values, dict):
                    raise ValueError(f"Threshold values for '{metric}' must be a dictionary")
                if "high" in values and "low" in values:
                    if values["high"] <= values["low"]:
                        raise ValueError(f"High threshold must be greater than low threshold for '{metric}'")
            
            cooldown = self.threshold.get("cooldown_seconds", 60)
            if not isinstance(cooldown, (int, float)) or cooldown < 0:
                raise ValueError("cooldown_seconds must be a non-negative number")

        # Validate hybrid strategy parameters
        if self.type == "hybrid":
            if self.hybrid is None or not isinstance(self.hybrid, dict):
                raise ValueError("Hybrid strategy requires 'hybrid' configuration block")
            # selection_mode
            sel = self.hybrid.get("selection_mode", "confidence")
            if sel not in ["first", "priority", "confidence"]:
                raise ValueError("Hybrid.selection_mode must be one of: first, priority, confidence")
            # min_confidence
            if "min_confidence" in self.hybrid:
                try:
                    mc = float(self.hybrid["min_confidence"])
                except Exception:
                    raise ValueError("Hybrid.min_confidence must be a number")
                if mc < 0.0 or mc > 1.0:
                    raise ValueError("Hybrid.min_confidence must be between 0.0 and 1.0")
            # strategies list
            strategies = self.hybrid.get("strategies", [])
            if not isinstance(strategies, list) or len(strategies) == 0:
                raise ValueError("Hybrid.strategies must be a non-empty list")
            for idx, s in enumerate(strategies):
                if not isinstance(s, dict):
                    raise ValueError(f"Hybrid.strategies[{idx}] must be a dict")
                if "type" not in s:
                    raise ValueError(f"Hybrid.strategies[{idx}] missing 'type'")
                # Priority optional but if present must be numeric
                if "priority" in s:
                    try:
                        float(s["priority"])
                    except Exception:
                        raise ValueError(f"Hybrid.strategies[{idx}].priority must be a number if provided")


@dataclass
class PolarisConfig:
    """Main Polaris configuration."""
    systems: list = field(default_factory=list)
    strategy: Optional[StrategyConfig] = None
    world_model: Optional[Dict[str, Any]] = None
    knowledge: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None  # Add monitoring configuration

    @classmethod
    def from_file(cls, path: str) -> 'PolarisConfig':
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, 'r') as f:
            content = f.read()

        # Substitute environment variables
        content = cls._substitute_env_vars(content)

        # Parse YAML
        data = yaml.safe_load(content) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolarisConfig':
        """Create config from dictionary."""
        # Parse systems
        systems = [
            SystemConfig(
                id=s['id'],
                connector_type=s.get('connector_type', 'unknown'),
                enabled=s.get('enabled', True),
                connection=s.get('connection', {}),
                monitoring=s.get('monitoring', {})
            )
            for s in data.get('systems', [])
        ]

        # Parse strategy
        strategy_data = data.get('strategy', {})
        strategy = StrategyConfig(
            type=strategy_data.get('type', 'threshold'),
            threshold=strategy_data.get('threshold'),
            llm=strategy_data.get('llm_reasoning'),
            hybrid=strategy_data.get('hybrid')
        )

        return cls(
            systems=systems,
            strategy=strategy,
            world_model=data.get('world_model'),
            knowledge=data.get('knowledge'),
            meta_learner=data.get('meta_learner'),
            observability=data.get('observability'),
            monitoring=data.get('monitoring')
        )

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Substitute ${VAR} with environment variable values."""
        def replace(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable '{var_name}' not found. Please set it or remove ${{{var_name}}} from config.")
            return value

        return re.sub(r'\$\{(\w+)\}', replace, content)


def load_config(path: str) -> PolarisConfig:
    """
    Load Polaris configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        PolarisConfig object
    """
    return PolarisConfig.from_file(path)
