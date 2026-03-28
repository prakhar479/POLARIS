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
    provider_name = str(provider or "google").lower()
    if provider_name == "gemini":
        return "google"
    return provider_name


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
        if strategy_type == "llm_reasoning":
            add_llm("strategy.llm_reasoning", strategy.get("llm_reasoning"))
        elif strategy_type == "agentic_llm":
            add_llm("strategy.agentic_llm", strategy.get("agentic_llm"))
        elif strategy_type == "thread_agentic":
            add_llm("strategy.thread_agentic", strategy.get("thread_agentic"))
        elif strategy_type == "hybrid":
            hybrid_cfg = strategy.get("hybrid")
            if isinstance(hybrid_cfg, dict):
                for index, sub in enumerate(hybrid_cfg.get("strategies", [])):
                    if not isinstance(sub, dict):
                        continue
                    sub_type = str(sub.get("type", "")).lower()
                    if sub_type == "llm_reasoning":
                        add_llm(
                            f"strategy.hybrid.strategies[{index}].llm_reasoning",
                            sub.get("llm_reasoning"),
                        )
                    elif sub_type == "agentic_llm":
                        add_llm(
                            f"strategy.hybrid.strategies[{index}].agentic_llm",
                            sub.get("agentic_llm"),
                        )
                    elif sub_type == "thread_agentic":
                        add_llm(
                            f"strategy.hybrid.strategies[{index}].thread_agentic",
                            sub.get("thread_agentic"),
                        )

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

    module_requirements = {
        "google": ["google.generativeai", "google.ai.generativelanguage_v1beta"],
        "openai": ["openai"],
        "openrouter": ["openai"],
        "groq": ["groq"],
        "ollama": ["openai"],
    }
    provider_single_key_env = {
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "groq": ["GROQ_API_KEY"],
    }
    provider_multi_key_env = {
        "google": "GEMINI_API_KEYS",
        "openai": "OPENAI_API_KEYS",
        "openrouter": "OPENROUTER_API_KEYS",
        "groq": "GROQ_API_KEYS",
    }

    for requirement in llm_requirements:
        provider = requirement["provider"]
        path = requirement["path"]
        resilience = requirement.get("resilience")
        required_modules = module_requirements.get(provider, [])

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
            env_candidates = list(provider_single_key_env.get(provider, []))
            env_candidates.append(keys_env_var or provider_multi_key_env.get(provider, ""))
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


def run_doctor(config_path: str) -> List[Diagnostic]:
    """Run all doctor diagnostics and return findings."""
    diagnostics: List[Diagnostic] = []

    if sys.version_info < (3, 8):
        diagnostics.append(
            Diagnostic(
                "FAIL",
                "runtime",
                f"Unsupported Python version: {sys.version.split()[0]} (requires >= 3.8)",
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
