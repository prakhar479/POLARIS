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


@dataclass
class StrategyConfig:
    """Configuration for adaptation strategy."""
    type: str = "threshold"
    threshold: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    hybrid: Optional[Dict[str, Any]] = None


@dataclass
class PolarisConfig:
    """Main Polaris configuration."""
    systems: list = field(default_factory=list)
    strategy: Optional[StrategyConfig] = None
    world_model: Optional[Dict[str, Any]] = None
    knowledge: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None

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
            llm=strategy_data.get('llm'),
            hybrid=strategy_data.get('hybrid')
        )

        return cls(
            systems=systems,
            strategy=strategy,
            world_model=data.get('world_model'),
            knowledge=data.get('knowledge'),
            meta_learner=data.get('meta_learner'),
            observability=data.get('observability')
        )

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Substitute ${VAR} with environment variable values."""
        def replace(match):
            var_name = match.group(1)
            return os.getenv(var_name, '')

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
