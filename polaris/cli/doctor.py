"""Diagnostics helpers for Polaris CLI."""

import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml

from polaris.infrastructure.config import PolarisConfig
from polaris.infrastructure.llm.contracts import (
    get_provider_multi_key_env_var,
    get_provider_required_modules,
    get_provider_single_key_env_vars,
)

_LEGACY_STRATEGY_BLOCK_KEYS = (
    "threshold",
    "llm_reasoning",
    "agentic_llm",
    "thread_agentic",
    "multi_agent",
    "hybrid",
)


@dataclass
class Diagnostic:
    """Single doctor diagnostic finding."""

    status: str
    category: str
    message: str


def _is_set(env_var: str) -> bool:
    value = os.getenv(env_var)
    return bool(value and value.strip())


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _normalize_provider(provider: Any) -> str:
    return str(provider or "google").lower()


def _extract_llm_requirements(raw_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []

    def add_llm(path: str, block: Any) -> None:
        llm_cfg = block if isinstance(block, dict) else {}
        requirements.append(
            {
                "path": path,
                "provider": _normalize_provider(llm_cfg.get("provider", "google")),
                "resilience": llm_cfg.get("resilience"),
            }
        )

    strategy = raw_config.get("strategy")
    if isinstance(strategy, dict):
        strategy_type = str(strategy.get("type", "threshold")).lower()
        strategy_params = strategy.get("params", {})
        if not isinstance(strategy_params, dict):
            strategy_params = {}

        def add_multi_agent_role_requirements(path_prefix: str, cfg: Dict[str, Any]) -> None:
            shared_provider = _normalize_provider(cfg.get("provider", "google"))
            for role in ("diagnostician", "planner", "validator"):
                role_cfg = cfg.get(role)
                if not isinstance(role_cfg, dict):
                    continue
                merged = dict(role_cfg)
                merged.setdefault("provider", shared_provider)
                add_llm(f"{path_prefix}.{role}", merged)

        if strategy_type == "llm_reasoning":
            add_llm("strategy.params", strategy_params)
        elif strategy_type == "agentic_llm":
            add_llm("strategy.params", strategy_params)
        elif strategy_type == "thread_agentic":
            add_llm("strategy.params", strategy_params)
        elif strategy_type == "multi_agent":
            add_llm("strategy.params", strategy_params)
            add_multi_agent_role_requirements("strategy.params", strategy_params)
        elif strategy_type == "hybrid":
            sub_defs = strategy_params.get("strategies", [])
            if isinstance(sub_defs, list):
                for index, sub in enumerate(sub_defs):
                    if not isinstance(sub, dict):
                        continue
                    sub_type = str(sub.get("type", "")).lower()
                    sub_params = sub.get("params", {})
                    if not isinstance(sub_params, dict):
                        continue

                    if sub_type == "llm_reasoning":
                        add_llm(
                            f"strategy.params.strategies[{index}].params",
                            sub_params,
                        )
                    elif sub_type == "agentic_llm":
                        add_llm(
                            f"strategy.params.strategies[{index}].params",
                            sub_params,
                        )
                    elif sub_type == "thread_agentic":
                        add_llm(
                            f"strategy.params.strategies[{index}].params",
                            sub_params,
                        )
                    elif sub_type == "multi_agent":
                        base_path = f"strategy.params.strategies[{index}].params"
                        add_llm(base_path, sub_params)
                        add_multi_agent_role_requirements(base_path, sub_params)

    meta_learner = raw_config.get("meta_learner")
    if isinstance(meta_learner, dict) and bool(meta_learner.get("enabled", False)):
        meta_type = str(meta_learner.get("type", "statistical")).lower()
        if meta_type == "llm":
            add_llm("meta_learner.llm", meta_learner.get("llm"))

    return requirements


def _extract_connectors(raw_config: Dict[str, Any]) -> Set[str]:
    connector_types: Set[str] = set()
    systems = raw_config.get("systems", [])
    if not isinstance(systems, list):
        return connector_types

    for system in systems:
        if not isinstance(system, dict):
            continue
        connector_type = system.get("connector_type")
        if connector_type:
            connector_types.add(str(connector_type).lower())
    return connector_types


def _normalize_tools_block(raw_tools: Any) -> List[str]:
    """Normalize tools block into a list of tool names."""
    if isinstance(raw_tools, list):
        return [tool for tool in raw_tools if isinstance(tool, str)]
    if isinstance(raw_tools, dict):
        enabled = raw_tools.get("enabled")
        if isinstance(enabled, list):
            return [tool for tool in enabled if isinstance(tool, str)]
    return []


def _extract_tooling_requirements(raw_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract strategy tool/native-tool requirements for diagnostics."""
    requirements: List[Dict[str, Any]] = []

    def walk_strategy(path: str, strategy_type: str, params: Dict[str, Any]) -> None:
        if strategy_type in {"agentic_llm", "thread_agentic"}:
            requirements.append(
                {
                    "path": path,
                    "strategy_type": strategy_type,
                    "provider": _normalize_provider(params.get("provider", "google")),
                    "tools": _normalize_tools_block(params.get("tools")),
                    "native_tools": params.get("native_tools"),
                    "generate_mode": params.get("generate_mode"),
                }
            )
            return

        if strategy_type == "multi_agent":
            requirements.append(
                {
                    "path": path,
                    "strategy_type": strategy_type,
                    "provider": _normalize_provider(params.get("provider", "google")),
                    "tools": _normalize_tools_block(params.get("tools")),
                    "native_tools": None,
                    "generate_mode": params.get("generate_mode"),
                }
            )
            shared_provider = _normalize_provider(params.get("provider", "google"))
            for role in ("diagnostician", "planner", "validator"):
                role_cfg = params.get(role)
                if not isinstance(role_cfg, dict):
                    continue
                requirements.append(
                    {
                        "path": f"{path}.{role}",
                        "strategy_type": f"multi_agent.{role}",
                        "provider": _normalize_provider(role_cfg.get("provider", shared_provider)),
                        "tools": _normalize_tools_block(role_cfg.get("tools")),
                        "native_tools": None,
                        "generate_mode": role_cfg.get("generate_mode", params.get("generate_mode")),
                    }
                )
            return

        if strategy_type == "hybrid":
            sub_defs = params.get("strategies", [])
            if not isinstance(sub_defs, list):
                return
            for index, sub in enumerate(sub_defs):
                if not isinstance(sub, dict):
                    continue
                sub_type = str(sub.get("type", "")).lower()
                sub_params = sub.get("params")
                if not isinstance(sub_params, dict):
                    continue
                walk_strategy(
                    path=f"{path}.strategies[{index}].params",
                    strategy_type=sub_type,
                    params=sub_params,
                )

    strategy = raw_config.get("strategy")
    if not isinstance(strategy, dict):
        return requirements

    strategy_type = str(strategy.get("type", "threshold")).lower()
    params = strategy.get("params")
    if not isinstance(params, dict):
        params = {}

    walk_strategy("strategy.params", strategy_type, params)
    return requirements


def _find_legacy_strategy_schema_paths(raw_config: Dict[str, Any]) -> List[str]:
    """Return paths that still use deprecated type-keyed strategy blocks."""
    paths: List[str] = []

    strategy = raw_config.get("strategy")
    if not isinstance(strategy, dict):
        return paths

    for key in _LEGACY_STRATEGY_BLOCK_KEYS:
        if key in strategy:
            paths.append(f"strategy.{key}")

    params = strategy.get("params")
    if not isinstance(params, dict):
        return paths

    sub_defs = params.get("strategies")
    if not isinstance(sub_defs, list):
        return paths

    for index, sub in enumerate(sub_defs):
        if not isinstance(sub, dict):
            continue
        for key in _LEGACY_STRATEGY_BLOCK_KEYS:
            if key in sub:
                paths.append(f"strategy.params.strategies[{index}].{key}")

    return paths


def _diagnose_config(config_path: str, diagnostics: List[Diagnostic]) -> Optional[Dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "config",
                f"Config file not found: {config_path}",
            )
        )
        return None

    diagnostics.append(
        Diagnostic(
            "OK",
            "config",
            f"Config file exists: {config_path}",
        )
    )

    try:
        raw_content = path.read_text(encoding="utf-8")
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "config",
                f"Unable to read config file: {exc}",
            )
        )
        return None

    placeholders = sorted(set(re.findall(r"\$\{(\w+)\}", raw_content)))
    if placeholders:
        missing = [name for name in placeholders if not _is_set(name)]
        if missing:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "env",
                    f"Missing environment variables referenced in config: {', '.join(missing)}",
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    "OK",
                    "env",
                    f"All config placeholders resolved: {', '.join(placeholders)}",
                )
            )
    else:
        diagnostics.append(
            Diagnostic(
                "OK",
                "env",
                "No ${VAR} placeholders found in config",
            )
        )

    try:
        raw_data = yaml.safe_load(raw_content) or {}
        if not isinstance(raw_data, dict):
            raise ValueError("Config root must be a mapping")

        legacy_paths = _find_legacy_strategy_schema_paths(raw_data)
        if legacy_paths:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "config",
                    "Legacy strategy schema detected at "
                    f"{', '.join(legacy_paths)}. "
                    "Use strategy.params and, for hybrid sub-strategies, "
                    "strategy.params.strategies[].params.",
                )
            )
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "config",
                f"Raw YAML parsing failed: {exc}",
            )
        )
        raw_data = None

    try:
        PolarisConfig.from_file(config_path)
        diagnostics.append(
            Diagnostic(
                "OK",
                "config",
                "Config schema and semantic validation passed",
            )
        )
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "config",
                f"Config validation failed: {exc}",
            )
        )

    return raw_data


def _diagnose_dependencies(
    raw_config: Dict[str, Any],
    diagnostics: List[Diagnostic],
) -> None:
    llm_requirements = _extract_llm_requirements(raw_config)
    connectors = _extract_connectors(raw_config)

    if _module_available("rich"):
        diagnostics.append(
            Diagnostic(
                "OK",
                "dependency",
                "Optional dependency available: rich (dashboard/interactive UI)",
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                "WARN",
                "dependency",
                "Optional dependency missing: rich (install with `pip install rich`)",
            )
        )

    if "wildfire" in connectors:
        if _module_available("httpx"):
            diagnostics.append(
                Diagnostic(
                    "OK",
                    "dependency",
                    "Dependency available for wildfire connector: httpx",
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "dependency",
                    "Missing dependency for wildfire connector: httpx",
                )
            )

    if not llm_requirements:
        diagnostics.append(
            Diagnostic(
                "OK",
                "dependency",
                "No LLM-backed strategy/meta-learner enabled in config",
            )
        )
        return

    for requirement in llm_requirements:
        provider = requirement["provider"]
        path = requirement["path"]
        resilience = requirement.get("resilience")
        required_modules = get_provider_required_modules(provider)

        if not required_modules:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "dependency",
                    f"{path}: unsupported provider '{provider}'",
                )
            )
            continue

        missing_modules = [mod for mod in required_modules if not _module_available(mod)]
        if missing_modules:
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "dependency",
                    f"{path}: missing module(s) for provider '{provider}': {', '.join(missing_modules)}",
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    "OK",
                    "dependency",
                    f"{path}: provider '{provider}' dependencies available",
                )
            )

        keys_env_var = None
        if isinstance(resilience, dict):
            candidate = resilience.get("keys_env_var")
            if isinstance(candidate, str) and candidate.strip():
                keys_env_var = candidate.strip()

        # Ollama is commonly local and doesn't require credentials by default.
        if provider == "ollama":
            diagnostics.append(
                Diagnostic(
                    "OK",
                    "env",
                    f"{path}: provider 'ollama' typically requires no API key (local endpoint)",
                )
            )
        else:
            env_candidates = list(get_provider_single_key_env_vars(provider))
            env_candidates.append(keys_env_var or get_provider_multi_key_env_var(provider) or "")
            env_candidates = [name for name in env_candidates if name]

            if any(_is_set(name) for name in env_candidates):
                diagnostics.append(
                    Diagnostic(
                        "OK",
                        "env",
                        f"{path}: credentials detected ({', '.join(env_candidates)})",
                    )
                )
            else:
                diagnostics.append(
                    Diagnostic(
                        "FAIL",
                        "env",
                        f"{path}: missing credentials. Set one of: {', '.join(env_candidates)}",
                    )
                )


def _diagnose_tooling(raw_config: Dict[str, Any], diagnostics: List[Diagnostic]) -> None:
    """Diagnose tool-name validity and native-tool/provider compatibility."""
    requirements = _extract_tooling_requirements(raw_config)
    if not requirements:
        diagnostics.append(
            Diagnostic(
                "OK",
                "tooling",
                "No tool-using strategy configuration detected",
            )
        )
        return

    try:
        from polaris.tools import registered_tool_types

        known_tools = {name for name in registered_tool_types() if isinstance(name, str)}
    except Exception:
        known_tools = set()

    for requirement in requirements:
        path = requirement["path"]
        tools = requirement.get("tools") or []

        if tools and known_tools:
            unknown = sorted({tool for tool in tools if tool not in known_tools})
            if unknown:
                diagnostics.append(
                    Diagnostic(
                        "WARN",
                        "tooling",
                        f"{path}.tools references unknown tool names: {', '.join(unknown)}",
                    )
                )
            else:
                diagnostics.append(
                    Diagnostic(
                        "OK",
                        "tooling",
                        f"{path}.tools resolved against builtin tool registry",
                    )
                )

        native_tools = requirement.get("native_tools")
        if native_tools is None:
            continue

        provider = str(requirement.get("provider", "google")).lower()
        generate_mode = str(requirement.get("generate_mode", "openai_compat")).lower()

        if provider == "ollama" and generate_mode == "native":
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "tooling",
                    f"{path}: provider=ollama with generate_mode=native does not support native tool calling",
                )
            )

        if not isinstance(native_tools, list):
            diagnostics.append(
                Diagnostic(
                    "FAIL",
                    "tooling",
                    f"{path}.native_tools must be a list",
                )
            )
            continue

        function_names: List[str] = []
        malformed = 0
        for item in native_tools:
            if not isinstance(item, dict):
                malformed += 1
                continue
            fn = item.get("function")
            if not isinstance(fn, dict):
                malformed += 1
                continue
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                function_names.append(name.strip())
            else:
                malformed += 1

        if malformed:
            diagnostics.append(
                Diagnostic(
                    "WARN",
                    "tooling",
                    f"{path}.native_tools contains {malformed} malformed function entries",
                )
            )

        if function_names and tools and known_tools:
            native_polaris_tools = sorted(
                {name for name in function_names if name in known_tools and name not in set(tools)}
            )
            if native_polaris_tools:
                diagnostics.append(
                    Diagnostic(
                        "WARN",
                        "tooling",
                        f"{path}.native_tools includes Polaris tool(s) not enabled under tools: "
                        f"{', '.join(native_polaris_tools)}",
                    )
                )

        diagnostics.append(
            Diagnostic(
                "OK",
                "tooling",
                f"{path}.native_tools parsed ({len(function_names)} function definition(s))",
            )
        )


def run_doctor(config_path: str) -> List[Diagnostic]:
    """Run all doctor diagnostics and return findings."""
    diagnostics: List[Diagnostic] = []

    if sys.version_info < (3, 10):
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "runtime",
                f"Unsupported Python version: {sys.version.split()[0]} (requires >= 3.10)",
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                "OK",
                "runtime",
                f"Python version: {sys.version.split()[0]}",
            )
        )

    raw_config = _diagnose_config(config_path, diagnostics)
    if isinstance(raw_config, dict):
        _diagnose_dependencies(raw_config, diagnostics)
        _diagnose_tooling(raw_config, diagnostics)

    return diagnostics


def run_doctor_cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for `polaris doctor`."""
    parser = argparse.ArgumentParser(
        prog="polaris doctor",
        description="Run Polaris diagnostics for config, environment, and optional dependencies",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/default.yaml",
        help="Path to configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    diagnostics = run_doctor(args.config)

    print("Polaris Doctor")
    print(f"Config: {args.config}")
    print()

    for item in diagnostics:
        print(f"[{item.status}] ({item.category}) {item.message}")

    failures = sum(1 for item in diagnostics if item.status == "FAIL")
    warnings = sum(1 for item in diagnostics if item.status == "WARN")

    print()
    print(f"Summary: {len(diagnostics)} checks, {warnings} warning(s), {failures} failure(s)")

    if failures > 0 or (args.strict and warnings > 0):
        return 1
    return 0
