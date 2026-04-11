"""CLI command to initialize a new Polaris configuration."""

import argparse
from pathlib import Path

from polaris.core.factories import registered_connector_types, registered_strategy_types
from polaris.infrastructure.llm.contracts import ordered_canonical_llm_providers


def _prompt_choice(prompt: str, allowed: list[str], default: str, attempts: int = 3) -> str:
    """Prompt for a constrained value with strict validation."""
    for _ in range(attempts):
        value = input(prompt).strip() or default
        if value in allowed:
            return value
        print(f"Invalid value '{value}'. Allowed values: {', '.join(allowed)}")
    raise ValueError(f"Failed to provide a valid value after {attempts} attempts")


def _build_strategy_params_block(strategy: str, llm_connector: str, llm_model: str) -> str:
    """Build a strategy.params block for generated starter configs."""
    llm_backed_strategies = {"llm_reasoning", "agentic_llm", "thread_agentic", "multi_agent"}

    if strategy == "threshold":
        return (
            "  params:\n"
            "    thresholds:\n"
            "      cpu_usage:\n"
            "        high: 80.0\n"
            "        low: 20.0\n"
            "    cooldown_seconds: 60\n"
        )

    if strategy == "hybrid":
        return (
            "  params:\n"
            '    selection_mode: "first"\n'
            "    min_confidence: 0.7\n"
            "    strategies:\n"
            '      - type: "threshold"\n'
            "        priority: 0.9\n"
            "        params:\n"
            "          thresholds:\n"
            "            cpu_usage:\n"
            "              high: 80.0\n"
            "              low: 20.0\n"
            "          cooldown_seconds: 60\n"
            '      - type: "agentic_llm"\n'
            "        priority: 0.6\n"
            "        params:\n"
            f'          provider: "{llm_connector}"\n'
            f'          model: "{llm_model}"\n'
            "          temperature: 0.1\n"
        )

    if strategy in llm_backed_strategies:
        return (
            "  params:\n"
            f'    provider: "{llm_connector}"\n'
            f'    model: "{llm_model}"\n'
            "    temperature: 0.1\n"
        )

    return "  params: {}\n"


def _build_connection_block(system_type: str) -> str:
    """Build connector-specific connection settings for generated configs."""
    if system_type == "swim":
        return '      host: "localhost"\n' "      port: 4242\n"

    if system_type == "wildfire":
        return '      base_url: "http://localhost:5000"\n' "      timeout: 5.0\n"

    if system_type == "kubernetes":
        return (
            '      namespace: "default"\n'
            "      in_cluster: false\n"
            '      kubeconfig_path: "~/.kube/config"\n'
        )

    return '      host: "localhost"\n' "      port: 4242\n"


def run_init_cli(args: list[str]) -> int:
    """Run the interactive init wizard."""
    parser = argparse.ArgumentParser(description="Initialize a new Polaris configuration file")
    parser.add_argument(
        "--output", "-o", default="config/custom.yaml", help="Output path for the generated config"
    )
    parsed_args = parser.parse_args(args)

    output_path = Path(parsed_args.output)
    if output_path.exists():
        print(f"Warning: {output_path} already exists.")
        choice = input("Overwrite? (y/N): ").strip().lower()
        if choice != "y":
            print("Aborted.")
            return 1

    print("\n" + "=" * 50)
    print("Welcome to the Polaris Configuration Wizard!")
    print("=" * 50 + "\n")
    print("This wizard will help you generate a starting configuration.\n")

    connector_types = [name for name in registered_connector_types() if name != "unknown"]
    if not connector_types:
        print("No connector types are registered. Cannot initialize config.")
        return 1
    default_connector = "swim" if "swim" in connector_types else connector_types[0]
    connector_options = "/".join(connector_types)

    try:
        system_type = _prompt_choice(
            f"Which managed system do you want to use? ({connector_options}) [{default_connector}]: ",
            connector_types,
            default_connector,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    interval = input("Monitoring interval in seconds [5]: ").strip() or "5"

    enable_metrics = (
        input("Enable metrics collection and background export? (Y/n): ").strip().lower() != "n"
    )

    strategy_types = registered_strategy_types()
    if not strategy_types:
        print("No strategy types are registered. Cannot initialize config.")
        return 1
    default_strategy = "hybrid" if "hybrid" in strategy_types else strategy_types[0]
    strategy_options = "/".join(strategy_types)

    try:
        strategy = _prompt_choice(
            f"Which strategy to use? ({strategy_options}) [{default_strategy}]: ",
            strategy_types,
            default_strategy,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    llm_connector = "openai"
    llm_model = "gpt-4o"
    llm_needed = strategy in {
        "llm_reasoning",
        "agentic_llm",
        "thread_agentic",
        "multi_agent",
        "hybrid",
    }
    if llm_needed:
        provider_types = list(ordered_canonical_llm_providers())
        default_provider = "openai"
        try:
            llm_connector = _prompt_choice(
                "Which LLM provider to use? (google/openai/openrouter/groq/ollama) [openai]: ",
                provider_types,
                default_provider,
            )
        except ValueError as exc:
            print(str(exc))
            return 1
        llm_model = input("Which model version? [gpt-4o]: ").strip() or "gpt-4o"

    strategy_params_block = _build_strategy_params_block(strategy, llm_connector, llm_model)
    connection_block = _build_connection_block(system_type)

    config_content = (
        "################################################################################\n"
        f"# Polaris Generated Configuration ({output_path.name})\n"
        "################################################################################\n\n"
        "systems:\n"
        f'  - id: "{system_type}"\n'
        f'    connector_type: "{system_type}"\n'
        "    enabled: true\n"
        "    connection:\n"
        f"{connection_block}"
        "    monitoring:\n"
        f"      collection_interval: {interval}\n"
        "      # Effective cadence = max(global interval_seconds, collection_interval)\n\n"
        "monitoring:\n"
        "  interval_seconds: 30\n\n"
        "observability:\n"
        "  logging:\n"
        '    type: "human"\n'
        '    level: "INFO"\n'
        "  metrics:\n"
        f"    enabled: {str(enable_metrics).lower()}\n"
        "    export:\n"
        f"      enabled: {str(enable_metrics).lower()}\n"
        '      formats: ["json", "csv"]\n'
        '      output_dir: "metrics"\n'
        "      auto_export_interval_minutes: 5\n\n"
        "strategy:\n"
        f'  type: "{strategy}"\n'
        f"{strategy_params_block}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_content)

    print("\n" + "=" * 50)
    print(f"Success! Configuration generated at: {output_path}")
    print(f"Validate config with: python -m polaris.cli doctor --config {output_path}")
    print(f"Run Polaris with: python -m polaris.cli --config {output_path}")
    print("=" * 50 + "\n")

    return 0
