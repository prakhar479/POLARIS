"""Configuration loader for Polaris."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from polaris.core.factories import (
    discover_connector_plugins,
    get_connector_config_validator,
    registered_connector_types,
    registered_strategy_types,
)


class ActionTemplateConfig(BaseModel):
    """Template for a policy-injected adaptation action."""

    type: str = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ActionInjectionPolicyConfig(BaseModel):
    """Policy for conditionally injecting an action."""

    enabled: bool = False
    action: Optional[ActionTemplateConfig] = None

    @model_validator(mode="after")
    def validate_action_policy(self) -> "ActionInjectionPolicyConfig":
        """Ensure enabled policies define an action template."""
        if self.enabled and self.action is None:
            raise ValueError("Enabled action policy requires an action block")
        return self


class SystemActionPolicyConfig(BaseModel):
    """Per-system action injection rules used by the adaptation pipeline."""

    inject_when_no_actions: Optional[ActionInjectionPolicyConfig] = None
    append_each_cycle: Optional[ActionInjectionPolicyConfig] = None


class SystemConfig(BaseModel):
    """Configuration for a managed system."""

    id: str = Field(min_length=1)
    connector_type: str = Field(default="unknown")
    enabled: bool = True
    connection: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    action_policy: Optional[SystemActionPolicyConfig] = None

    @model_validator(mode="after")
    def validate_system(self) -> "SystemConfig":
        """Validate the system configuration."""
        supported_connectors = registered_connector_types()
        if self.connector_type not in supported_connectors and self.connector_type != "unknown":
            raise ValueError(
                f"Unsupported connector type '{self.connector_type}'. Supported: {supported_connectors}"
            )

        validator = get_connector_config_validator(self.connector_type)
        if validator is not None:
            connection = self.connection if isinstance(self.connection, dict) else {}
            validator(connection)

        return self


class StrategyConfig(BaseModel):
    """Configuration for adaptation strategy."""

    type: str = "threshold"
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def populate_params(cls, values: Any) -> Any:
        """Migrate legacy flat strategy dicts into the params payload."""
        if not isinstance(values, dict):
            return values

        stype = values.get("type", "threshold")
        if stype in values and isinstance(values[stype], dict):
            # If they provided `threshold: {...}` logic
            if "params" not in values:
                values["params"] = values.pop(stype)

        return values

    @model_validator(mode="after")
    def validate_strategy(self) -> "StrategyConfig":
        """Validate the strategy configuration."""
        supported_strategies = registered_strategy_types()
        if self.type not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy type '{self.type}'. Supported: {supported_strategies}"
            )

        if self.type == "hybrid":
            sel = self.params.get("selection_mode", "confidence")
            if sel not in ["first", "priority", "confidence"]:
                raise ValueError(
                    "hybrid selection_mode must be one of: first, priority, confidence"
                )
            if "min_confidence" in self.params:
                conf = self.params["min_confidence"]
                if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
                    raise ValueError("hybrid min_confidence must be a float between 0.0 and 1.0")

        return self


class PolarisConfig(BaseModel):
    """Main Polaris configuration."""

    model_config = ConfigDict(extra="forbid")

    systems: List[SystemConfig] = Field(default_factory=list)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    world_model: Optional[Dict[str, Any]] = None
    knowledge_store: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    observability: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    plugin_imports: List[str] = Field(default_factory=list)
    max_concurrent_connectors: int = Field(default=10, gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_max_concurrent_connectors(cls, data: Any) -> Any:
        """Normalize max_concurrent_connectors to int."""
        if not isinstance(data, dict):
            return data

        if "max_concurrent_connectors" in data:
            val = data["max_concurrent_connectors"]
            if isinstance(val, str):
                try:
                    data["max_concurrent_connectors"] = int(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError("max_concurrent_connectors must be an integer > 0") from exc

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

        data = dict(data)

        if "knowledge" in data:
            raise ValueError(
                "The 'knowledge' config key was removed. Use 'knowledge_store' instead."
            )

        plugin_imports = data.get("plugin_imports")
        if plugin_imports is None:
            normalized_plugin_imports: List[str] = []
        elif isinstance(plugin_imports, list):
            normalized_plugin_imports = []
            for path in plugin_imports:
                if not isinstance(path, str) or not path.strip():
                    raise ValueError("plugin_imports must be a list of non-empty strings")
                normalized_plugin_imports.append(path.strip())
        else:
            raise ValueError("plugin_imports must be a list of non-empty strings")

        try:
            discover_connector_plugins(normalized_plugin_imports)
        except Exception as exc:
            raise ValueError(f"Failed to load connector plugins: {exc}") from exc

        data["plugin_imports"] = normalized_plugin_imports

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
