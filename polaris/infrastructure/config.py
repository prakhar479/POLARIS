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
from polaris.infrastructure.constants import MAX_PORT, MIN_PORT
from polaris.strategies.utils import DEFAULT_ALLOWED_TOOLS
from polaris.tools import registered_tool_types

_SUPPORTED_LLM_PROVIDERS = {"google", "openai", "openrouter", "groq", "ollama"}


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

        if not isinstance(self.monitoring, dict):
            raise ValueError("systems[].monitoring must be a dictionary")

        collection_interval = self.monitoring.get("collection_interval")
        if collection_interval is not None:
            if not isinstance(collection_interval, (int, float)) or float(collection_interval) <= 0:
                raise ValueError("systems[].monitoring.collection_interval must be a number > 0")

        validator = get_connector_config_validator(self.connector_type)
        if validator is not None:
            connection = self.connection if isinstance(self.connection, dict) else {}
            validator(connection)

        if self.connector_type == "suave" and self.connection:
            host = self.connection.get("host")
            if host is not None and not isinstance(host, str):
                raise ValueError("SUAVE host must be a string")
            port = self.connection.get("port")
            if port is not None:
                self._validate_port(port, "SUAVE")

        return self


class StrategyConfig(BaseModel):
    """Configuration for adaptation strategy."""

    model_config = ConfigDict(extra="forbid")

    type: str = "threshold"
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_strategy(self) -> "StrategyConfig":
        """Validate the strategy configuration."""
        supported_strategies = registered_strategy_types()
        if self.type not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy type '{self.type}'. Supported: {supported_strategies}"
            )

        if not isinstance(self.params, dict):
            raise ValueError("strategy.params must be a dictionary")

        if self.type == "threshold":
            self._validate_threshold_params()
        elif self.type == "llm_reasoning":
            self._validate_llm_params(self.params, label="llm_reasoning")
        elif self.type == "agentic_llm":
            self._validate_llm_params(self.params, label="agentic_llm")
            self._validate_int_min("steps_limit", self.params.get("steps_limit"), minimum=1)
            self._validate_float_min(
                "decision_cooldown_seconds",
                self.params.get("decision_cooldown_seconds"),
                minimum=0.0,
            )
            self._validate_int_min(
                "max_tool_result_chars",
                self.params.get("max_tool_result_chars"),
                minimum=1,
            )
            self._validate_choice(
                "native_tools_unsupported_policy",
                self.params.get("native_tools_unsupported_policy"),
                {"skip_cycle", "json_fallback", "strict_fail"},
            )
            self._validate_tools_block(self.params.get("tools"), label="agentic_llm.tools")
            enabled_tools = self._effective_enabled_tools(
                self.params.get("tools"),
                default_tools=DEFAULT_ALLOWED_TOOLS,
            )
            self._validate_native_tools_block(
                self.params.get("native_tools"),
                label="agentic_llm.native_tools",
                enabled_tools=enabled_tools,
            )
        elif self.type == "thread_agentic":
            self._validate_llm_params(self.params, label="thread_agentic")
            self._validate_int_min("steps_limit", self.params.get("steps_limit"), minimum=1)
            self._validate_int_min(
                "max_thread_depth", self.params.get("max_thread_depth"), minimum=0
            )
            self._validate_int_min(
                "max_total_threads", self.params.get("max_total_threads"), minimum=1
            )
            self._validate_float_min(
                "child_timeout_seconds",
                self.params.get("child_timeout_seconds"),
                minimum=0.000001,
            )
            self._validate_int_min(
                "max_repeated_spawns", self.params.get("max_repeated_spawns"), minimum=1
            )
            self._validate_float_min(
                "assessment_cooldown_seconds",
                self.params.get("assessment_cooldown_seconds"),
                minimum=0.0,
            )
            self._validate_int_min(
                "max_tool_result_chars", self.params.get("max_tool_result_chars"), minimum=1
            )
            self._validate_int_min(
                "max_child_payload_chars", self.params.get("max_child_payload_chars"), minimum=1
            )
            self._validate_int_min("phi_max_lines", self.params.get("phi_max_lines"), minimum=1)
            self._validate_tools_block(self.params.get("tools"), label="thread_agentic.tools")

            phi_mode = self.params.get("phi_mode")
            if phi_mode is not None and phi_mode not in {"last_line", "recent_lines"}:
                raise ValueError("thread_agentic phi_mode must be 'last_line' or 'recent_lines'")

            for token_name in ("listen_token", "return_token"):
                token_value = self.params.get(token_name)
                if token_value is not None and (
                    not isinstance(token_value, str) or not token_value.strip()
                ):
                    raise ValueError(f"thread_agentic {token_name} must be a non-empty string")
        elif self.type == "multi_agent":
            self._validate_llm_params(self.params, label="multi_agent")
            self._validate_int_min("steps_limit", self.params.get("steps_limit"), minimum=1)
            self._validate_int_min(
                "max_tool_result_chars",
                self.params.get("max_tool_result_chars"),
                minimum=1,
            )
            self._validate_tools_block(self.params.get("tools"), label="multi_agent.tools")

            for role in ("diagnostician", "planner", "validator"):
                role_cfg = self.params.get(role)
                if role_cfg is None:
                    continue
                if not isinstance(role_cfg, dict):
                    raise ValueError(f"multi_agent {role} config must be a dictionary")
                self._validate_llm_params(role_cfg, label=f"multi_agent.{role}")
                self._validate_int_min(
                    f"multi_agent.{role}.steps_limit",
                    role_cfg.get("steps_limit"),
                    minimum=1,
                )
                self._validate_int_min(
                    f"multi_agent.{role}.max_tokens",
                    role_cfg.get("max_tokens"),
                    minimum=1,
                )
                self._validate_tools_block(
                    role_cfg.get("tools"),
                    label=f"multi_agent.{role}.tools",
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

            strategies = self.params.get("strategies")
            if strategies is not None:
                if not isinstance(strategies, list) or not strategies:
                    raise ValueError("hybrid strategies must be a non-empty list")
                for idx, strategy in enumerate(strategies):
                    if not isinstance(strategy, dict):
                        raise ValueError(f"hybrid strategies[{idx}] must be a dictionary")
                    if "type" not in strategy:
                        raise ValueError(f"hybrid strategies[{idx}] requires a 'type' field")
                    strategy_type = strategy["type"]
                    if not isinstance(strategy_type, str) or not strategy_type.strip():
                        raise ValueError(
                            f"hybrid strategies[{idx}].type must be a non-empty string"
                        )

                    unknown_keys = set(strategy.keys()) - {"type", "priority", "params"}
                    if unknown_keys:
                        unknown = sorted(unknown_keys)
                        raise ValueError(
                            f"hybrid strategies[{idx}] has unsupported keys: {unknown}. "
                            "Use a 'params' block for sub-strategy configuration"
                        )

                    priority = strategy.get("priority")
                    if priority is not None and not isinstance(priority, (int, float)):
                        raise ValueError(f"hybrid strategies[{idx}].priority must be numeric")

                    sub_params = strategy.get("params", {})
                    if sub_params is None:
                        sub_params = {}
                    if not isinstance(sub_params, dict):
                        raise ValueError(f"hybrid strategies[{idx}].params must be a dictionary")

                    try:
                        StrategyConfig(type=strategy_type, params=sub_params)
                    except ValueError as exc:
                        raise ValueError(f"hybrid strategies[{idx}] invalid params: {exc}") from exc

        return self

    def _validate_threshold_params(self) -> None:
        """Validate threshold strategy params structure."""
        thresholds = self.params.get("thresholds")
        if thresholds is None:
            return
        if not isinstance(thresholds, dict):
            raise ValueError("threshold thresholds must be a dictionary")

        for metric_name, bounds in thresholds.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError("threshold metric names must be non-empty strings")
            if not isinstance(bounds, dict):
                raise ValueError(
                    f"threshold bounds for metric '{metric_name}' must be a dictionary"
                )

            high = bounds.get("high")
            low = bounds.get("low")
            if high is not None and not isinstance(high, (int, float)):
                raise ValueError(f"threshold high bound for metric '{metric_name}' must be numeric")
            if low is not None and not isinstance(low, (int, float)):
                raise ValueError(f"threshold low bound for metric '{metric_name}' must be numeric")
            if isinstance(high, (int, float)) and isinstance(low, (int, float)) and high <= low:
                raise ValueError(
                    f"threshold high bound for metric '{metric_name}' must be greater than low bound"
                )

        action_templates = self.params.get("action_templates")
        if action_templates is not None and not isinstance(action_templates, dict):
            raise ValueError("threshold action_templates must be a dictionary")

        self._validate_int_min("cooldown_seconds", self.params.get("cooldown_seconds"), minimum=0)

    def _validate_llm_params(self, params: Dict[str, Any], label: str) -> None:
        """Validate common provider and temperature fields for LLM-backed strategies."""
        provider = params.get("provider")
        if provider is not None:
            if not isinstance(provider, str) or provider not in _SUPPORTED_LLM_PROVIDERS:
                supported = sorted(_SUPPORTED_LLM_PROVIDERS)
                raise ValueError(f"{label} provider must be one of: {supported}")

        temperature = params.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or not 0.0 <= float(temperature) <= 2.0:
                raise ValueError(f"{label} temperature must be a float between 0.0 and 2.0")

        per_system_prompts = params.get("per_system_prompts")
        if per_system_prompts is not None:
            if not isinstance(per_system_prompts, dict):
                raise ValueError(f"{label} per_system_prompts must be a dictionary")
            for system_id, prompt in per_system_prompts.items():
                if not isinstance(system_id, str) or not system_id.strip():
                    raise ValueError(f"{label} per_system_prompts keys must be non-empty strings")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{label} per_system_prompts values must be non-empty strings")

    def _validate_tools_block(self, tools: Any, label: str) -> None:
        """Validate tools config shape used by LLM-backed strategies."""
        if tools is None:
            return

        known_tools = self._known_tool_names()

        if isinstance(tools, list):
            normalized: List[str] = []
            for tool in tools:
                if not isinstance(tool, str) or not tool.strip():
                    raise ValueError(f"{label} list entries must be non-empty strings")
                normalized.append(tool.strip())
            self._validate_known_tools(label, normalized, known_tools)
            return
        if isinstance(tools, dict):
            enabled = tools.get("enabled")
            if enabled is None:
                return
            if not isinstance(enabled, list):
                raise ValueError(f"{label}.enabled must be a list of strings")
            normalized = []
            for tool in enabled:
                if not isinstance(tool, str) or not tool.strip():
                    raise ValueError(f"{label}.enabled entries must be non-empty strings")
                normalized.append(tool.strip())
            self._validate_known_tools(f"{label}.enabled", normalized, known_tools)
            return
        raise ValueError(f"{label} must be either a list of strings or a dictionary")

    def _validate_int_min(self, field_name: str, value: Any, minimum: int) -> None:
        """Validate optional integer field with minimum bound."""
        if value is None:
            return
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field_name} must be an integer >= {minimum}")

    def _validate_float_min(self, field_name: str, value: Any, minimum: float) -> None:
        """Validate optional float field with minimum bound."""
        if value is None:
            return
        if not isinstance(value, (int, float)) or float(value) < minimum:
            raise ValueError(f"{field_name} must be a number >= {minimum}")

    def _validate_native_tools_block(
        self,
        native_tools: Any,
        label: str,
        enabled_tools: Optional[set[str]] = None,
    ) -> None:
        """Validate an OpenAI-format native tool definitions list.

        Each entry must follow the OpenAI function-calling schema::

            {"type": "function", "function": {"name": str, "description": str,
                "parameters": {"type": "object", "properties": {...}, "required": [...]}}}

        The ``description`` and ``parameters`` fields are optional here because some
        simple use-cases (e.g. a sentinel ``no_adaptation`` function) may omit them.
        """
        if native_tools is None:
            return
        if not isinstance(native_tools, list):
            raise ValueError(f"{label} must be a list of function definition dicts")

        known_tools = self._known_tool_names()
        for i, tool in enumerate(native_tools):
            if not isinstance(tool, dict):
                raise ValueError(f"{label}[{i}] must be a dict")
            tool_type = tool.get("type")
            if tool_type != "function":
                raise ValueError(f"{label}[{i}].type must be 'function' (got {tool_type!r})")
            fn = tool.get("function")
            if not isinstance(fn, dict):
                raise ValueError(f"{label}[{i}].function must be a dict")
            fn_name = fn.get("name")
            if not isinstance(fn_name, str) or not fn_name.strip():
                raise ValueError(f"{label}[{i}].function.name must be a non-empty string")
            normalized_name = fn_name.strip()
            params = fn.get("parameters")
            if params is not None and not isinstance(params, dict):
                raise ValueError(
                    f"{label}[{i}].function.parameters must be a dict (JSON Schema object)"
                )

            if (
                enabled_tools is not None
                and normalized_name in known_tools
                and normalized_name not in enabled_tools
            ):
                raise ValueError(
                    f"{label}[{i}].function.name references Polaris tool '{normalized_name}' "
                    "that is not enabled under strategy.params.tools"
                )

    def _validate_choice(
        self,
        field_name: str,
        value: Any,
        supported_values: set[str],
    ) -> None:
        """Validate optional string field against supported values."""
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be one of: {sorted(supported_values)}")
        normalized = value.strip().lower()
        if normalized not in supported_values:
            raise ValueError(f"{field_name} must be one of: {sorted(supported_values)}")

    def _known_tool_names(self) -> set[str]:
        """Return currently registered tool names."""
        try:
            return {name for name in registered_tool_types() if isinstance(name, str)}
        except Exception:
            return set()

    def _validate_known_tools(self, label: str, tools: List[str], known_tools: set[str]) -> None:
        """Ensure configured tool names are known to the registry."""
        if not known_tools:
            return
        unknown = sorted({tool for tool in tools if tool not in known_tools})
        if unknown:
            known = sorted(known_tools)
            raise ValueError(
                f"{label} contains unknown tool name(s): {unknown}. " f"Known tools: {known}"
            )

    def _effective_enabled_tools(self, tools: Any, default_tools: List[str]) -> set[str]:
        """Resolve effective enabled tool names from tools block plus defaults."""
        if isinstance(tools, list):
            return {tool.strip() for tool in tools if isinstance(tool, str) and tool.strip()}
        if isinstance(tools, dict):
            enabled = tools.get("enabled")
            if isinstance(enabled, list):
                return {tool.strip() for tool in enabled if isinstance(tool, str) and tool.strip()}
        return {tool.strip() for tool in default_tools if isinstance(tool, str) and tool.strip()}


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
