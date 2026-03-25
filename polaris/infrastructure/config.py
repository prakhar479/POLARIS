"""Configuration loader for Polaris."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from polaris.core.factories import registered_connector_types, registered_strategy_types

from .constants import DEFAULT_COOLDOWN_SECONDS, MAX_PORT, MIN_PORT


class SystemConfig(BaseModel):
    """Configuration for a managed system."""

    id: str = Field(min_length=1)
    connector_type: str = Field(default="unknown")
    enabled: bool = True
    connection: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _validate_port(port: Any, connector_name: str) -> None:
        """Validate port number for a connector."""
        if not isinstance(port, int):
            raise ValueError(f"{connector_name} connection port must be an integer")
        if not (MIN_PORT <= port <= MAX_PORT):
            raise ValueError(
                f"{connector_name} connection port must be between {MIN_PORT} and {MAX_PORT}"
            )

    @model_validator(mode="after")
    def validate_system(self) -> "SystemConfig":
        """Validate the system configuration."""
        supported_connectors = registered_connector_types()
        if self.connector_type not in supported_connectors and self.connector_type != "unknown":
            raise ValueError(
                f"Unsupported connector type '{self.connector_type}'. Supported: {supported_connectors}"
            )

        if self.connector_type == "swim" and self.connection:
            port = self.connection.get("port")
            if port is not None:
                self._validate_port(port, "SWIM")

        if self.connector_type == "wildfire" and self.connection:
            base_url = self.connection.get("base_url")
            if base_url is not None and not isinstance(base_url, str):
                raise ValueError("Wildfire base_url must be a string")
            port = self.connection.get("port")
            if port is not None:
                self._validate_port(port, "Wildfire")

        return self


class StrategyConfig(BaseModel):
    """Configuration for adaptation strategy."""

    type: str = "threshold"
    threshold: Optional[Dict[str, Any]] = None
    llm_reasoning: Optional[Dict[str, Any]] = None
    hybrid: Optional[Dict[str, Any]] = None
    agentic_llm: Optional[Dict[str, Any]] = None
    multi_agent: Optional[Dict[str, Any]] = None

    def _validate_strategy_config_block(
        self, strategy_type: str, config_attr: str, config_block: Optional[Dict[str, Any]]
    ) -> None:
        """Validate strategy configuration block."""
        strategy_name = strategy_type.replace("_", " ").title()

        if config_block is None:
            setattr(self, config_attr, {})
        elif not isinstance(config_block, dict):
            raise ValueError(
                f"{strategy_name} strategy requires '{config_attr}' configuration block to be a dict"
            )

    @model_validator(mode="after")
    def validate_strategy(self) -> "StrategyConfig":
        """Validate the strategy configuration."""
        supported_strategies = registered_strategy_types()
        if self.type not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy type '{self.type}'. Supported: {supported_strategies}"
            )

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
            cooldown = self.threshold.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
            if not isinstance(cooldown, (int, float)) or cooldown < 0:
                raise ValueError("cooldown_seconds must be a non-negative number")

        if self.type == "hybrid":
            if not isinstance(self.hybrid, dict):
                raise ValueError("Hybrid strategy requires 'hybrid' configuration block")
            sel = self.hybrid.get("selection_mode", "confidence")
            if sel not in ["first", "priority", "confidence"]:
                raise ValueError(
                    "Hybrid.selection_mode must be one of: first, priority, confidence"
                )
            if "min_confidence" in self.hybrid:
                try:
                    mc = float(self.hybrid["min_confidence"])
                except Exception:
                    raise ValueError("Hybrid.min_confidence must be a number")
                if not (0.0 <= mc <= 1.0):
                    raise ValueError("Hybrid.min_confidence must be between 0.0 and 1.0")
            strategies = self.hybrid.get("strategies", [])
            if not isinstance(strategies, list) or len(strategies) == 0:
                raise ValueError("Hybrid.strategies must be a non-empty list")
            for idx, s in enumerate(strategies):
                if not isinstance(s, dict):
                    raise ValueError(f"Hybrid.strategies[{idx}] must be a dict")
                if "type" not in s:
                    raise ValueError(f"Hybrid.strategies[{idx}] missing 'type'")
                if "priority" in s:
                    try:
                        float(s["priority"])
                    except Exception:
                        raise ValueError(
                            f"Hybrid.strategies[{idx}].priority must be a number if provided"
                        )

        # Validate strategy-specific configuration blocks
        strategy_validations = {
            "llm_reasoning": (self.llm_reasoning, "llm_reasoning"),
            "agentic_llm": (self.agentic_llm, "agentic_llm"),
            "multi_agent": (self.multi_agent, "multi_agent"),
        }

        if self.type in strategy_validations:
            config_block, config_attr = strategy_validations[self.type]
            self._validate_strategy_config_block(self.type, config_attr, config_block)

        return self


class PolarisConfig(BaseModel):
    """Main Polaris configuration."""

    systems: List[SystemConfig] = Field(default_factory=list)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    world_model: Optional[Dict[str, Any]] = None
    knowledge_store: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    max_concurrent_connectors: int = Field(default=10, gt=0)
    # Preserve unknown top-level config keys (e.g., custom connector-specific blocks like `wildfire:`)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def preserve_unknown_top_level_keys(cls, data: Any) -> Any:
        """Preserve unknown top-level keys into `extra`.

        Some configurations include connector-specific top-level blocks (for example
        `wildfire:`) that aren't part of the strict PolarisConfig schema. We keep them
        so core code can access them for optional behaviors.
        """
        if not isinstance(data, dict):
            return data

        known = {
            "systems",
            "strategy",
            "world_model",
            "knowledge_store",
            "meta_learner",
            "observability",
            "monitoring",
            "max_concurrent_connectors",
            "extra",
        }
        extra = dict(data.get("extra") or {})
        for k, v in list(data.items()):
            if k not in known:
                extra[k] = v
        data["extra"] = extra
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_max_concurrent_connectors(cls, data: Any) -> Any:
        """Normalize max_concurrent_connectors to int and sanitize invalid values."""
        if not isinstance(data, dict):
            return data

        if "max_concurrent_connectors" in data:
            val = data["max_concurrent_connectors"]
            # Try to convert to int
            try:
                if isinstance(val, str):
                    val = int(val)
                if isinstance(val, int):
                    # Sanitize invalid values to default
                    if val <= 0:
                        data["max_concurrent_connectors"] = 10
                    else:
                        data["max_concurrent_connectors"] = val
            except (ValueError, TypeError):
                # If conversion fails, use default
                data["max_concurrent_connectors"] = 10

        return data

    @classmethod
    def from_file(cls, path: str) -> "PolarisConfig":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            content = f.read()

        content = cls._substitute_env_vars(content)
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

        # Allow pydantic to coerce
        # Default empty strategy if not given
        if "strategy" not in data or data["strategy"] is None:
            data["strategy"] = {}

        return cls.model_validate(data)

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


def load_config(path: str) -> "PolarisConfig":
    """Load Polaris configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        PolarisConfig object
    """
    return PolarisConfig.from_file(path)
