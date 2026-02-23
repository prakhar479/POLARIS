"""Configuration loader for Polaris."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from polaris.core.factories import registered_connector_types, registered_strategy_types


@dataclass
class SystemConfig:
    """Configuration for a managed system."""

    id: str
    connector_type: str
    enabled: bool = True
    connection: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate system configuration after initialization."""
        if not self.id or not self.id.strip():
            raise ValueError("System ID cannot be empty")

        # Validate connector type against registered factories
        supported_connectors = registered_connector_types()
        if self.connector_type not in supported_connectors:
            raise ValueError(
                f"Unsupported connector type '{self.connector_type}'. Supported: {supported_connectors}"
            )

        # Validate connection parameters for SWIM
        if self.connector_type == "swim" and self.connection:
            if "port" in self.connection and not isinstance(self.connection["port"], int):
                raise ValueError("SWIM connection port must be an integer")
            if "port" in self.connection and not (1 <= self.connection["port"] <= 65535):
                raise ValueError("SWIM connection port must be between 1 and 65535")

        # Validate connection parameters for Wildfire
        if self.connector_type == "wildfire" and self.connection:
            if "base_url" in self.connection and not isinstance(self.connection["base_url"], str):
                raise ValueError("Wildfire base_url must be a string")
            if "port" in self.connection and not isinstance(self.connection["port"], int):
                raise ValueError("Wildfire connection port must be an integer")
            if "port" in self.connection and not (1 <= self.connection["port"] <= 65535):
                raise ValueError("Wildfire connection port must be between 1 and 65535")


@dataclass
class StrategyConfig:
    """Configuration for adaptation strategy."""

    type: str = "threshold"
    threshold: Optional[Dict[str, Any]] = None
    llm_reasoning: Optional[Dict[str, Any]] = None
    hybrid: Optional[Dict[str, Any]] = None
    agentic_llm: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate strategy configuration after initialization."""
        supported_strategies = registered_strategy_types()
        if self.type not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy type '{self.type}'. Supported: {supported_strategies}"
            )

        # Validate threshold strategy parameters
        if self.type == "threshold" and self.threshold:
            thresholds = self.threshold.get("thresholds", {})
            for metric, values in thresholds.items():
                if not isinstance(values, dict):
                    raise ValueError(f"Threshold values for '{metric}' must be a dictionary")
                if "high" in values and "low" in values:
                    if values["high"] <= values["low"]:
                        raise ValueError(
                            f"High threshold must be greater than low threshold for '{metric}'"
                        )

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
                raise ValueError(
                    "Hybrid.selection_mode must be one of: first, priority, confidence"
                )
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
                        raise ValueError(
                            f"Hybrid.strategies[{idx}].priority must be a number if provided"
                        )

        if self.type == "llm_reasoning":
            if self.llm_reasoning is None:
                self.llm_reasoning = {}
            elif not isinstance(self.llm_reasoning, dict):
                raise ValueError(
                    "LLM reasoning strategy requires 'llm_reasoning' configuration block to be a dict"
                )

        # Validate agentic strategy parameters
        if self.type == "agentic_llm":
            if self.agentic_llm is None:
                self.agentic_llm = {}
            elif not isinstance(self.agentic_llm, dict):
                raise ValueError(
                    "Agentic LLM strategy requires 'agentic_llm' configuration block to be a dict"
                )


@dataclass
class PolarisConfig:
    """Main Polaris configuration."""

    systems: list = field(default_factory=list)
    strategy: Optional[StrategyConfig] = None
    world_model: Optional[Dict[str, Any]] = None
    knowledge_store: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    max_concurrent_connectors: int = 10

    @classmethod
    def from_file(cls, path: str) -> "PolarisConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            content = f.read()

        # Substitute environment variables
        content = cls._substitute_env_vars(content)

        # Parse YAML
        data = yaml.safe_load(content) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolarisConfig":
        """Create config from dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Config root must be a dictionary")

        if "knowledge" in data:
            raise ValueError(
                "The 'knowledge' config key was removed. Use 'knowledge_store' instead."
            )

        # Parse systems
        systems = [
            SystemConfig(
                id=s["id"],
                connector_type=s.get("connector_type", "unknown"),
                enabled=s.get("enabled", True),
                connection=s.get("connection", {}),
                monitoring=s.get("monitoring", {}),
            )
            for s in data.get("systems", [])
        ]

        # Parse strategy
        strategy_data = data.get("strategy", {})
        if strategy_data is not None and not isinstance(strategy_data, dict):
            raise ValueError("'strategy' must be a dictionary")
        if strategy_data is None:
            strategy_data = {}
        strategy = StrategyConfig(
            type=strategy_data.get("type", "threshold"),
            threshold=strategy_data.get("threshold"),
            llm_reasoning=strategy_data.get("llm_reasoning"),
            hybrid=strategy_data.get("hybrid"),
            agentic_llm=strategy_data.get("agentic_llm"),
        )
        knowledge_store_data = data.get("knowledge_store")
        if knowledge_store_data is not None and not isinstance(knowledge_store_data, dict):
            raise ValueError("'knowledge_store' must be a dictionary")

        world_model_data = data.get("world_model")
        if world_model_data is not None and not isinstance(world_model_data, dict):
            raise ValueError("'world_model' must be a dictionary")

        meta_learner_data = data.get("meta_learner")
        if meta_learner_data is not None and not isinstance(meta_learner_data, dict):
            raise ValueError("'meta_learner' must be a dictionary")

        observability_data = data.get("observability")
        if observability_data is not None and not isinstance(observability_data, dict):
            raise ValueError("'observability' must be a dictionary")

        monitoring_data = data.get("monitoring")
        if monitoring_data is not None and not isinstance(monitoring_data, dict):
            raise ValueError("'monitoring' must be a dictionary")

        max_concurrent_connectors = data.get("max_concurrent_connectors", 10)
        try:
            max_concurrent_connectors = int(max_concurrent_connectors)
        except Exception:
            max_concurrent_connectors = 10
        if max_concurrent_connectors <= 0:
            max_concurrent_connectors = 10

        return cls(
            systems=systems,
            strategy=strategy,
            world_model=world_model_data,
            knowledge_store=knowledge_store_data,
            meta_learner=meta_learner_data,
            observability=observability_data,
            monitoring=monitoring_data,
            max_concurrent_connectors=max_concurrent_connectors,
        )

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Substitute ${VAR} with environment variable values."""

        def replace(match: Any) -> str:
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' not found. Please set it or remove ${{{var_name}}} from config."
                )
            return value

        return re.sub(r"\$\{(\w+)\}", replace, content)


def load_config(path: str) -> PolarisConfig:
    """
    Load Polaris configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        PolarisConfig object
    """
    return PolarisConfig.from_file(path)
