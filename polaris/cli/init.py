"""CLI command to initialize a new Polaris configuration."""

import argparse
from pathlib import Path

from polaris.core.factories import registered_connector_types, registered_strategy_types


def _prompt_choice(prompt: str, allowed: list[str], default: str, attempts: int = 3) -> str:
    """Prompt for a constrained value with strict validation."""
    for _ in range(attempts):
        value = input(prompt).strip() or default
        if value in allowed:
            return value
        print(f"Invalid value '{value}'. Allowed values: {', '.join(allowed)}")
    raise ValueError(f"Failed to provide a valid value after {attempts} attempts")


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

    provider_types = ["google", "openai", "openrouter", "groq", "ollama"]
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

    config_content = f"""################################################################################
# Polaris Generated Configuration ({output_path.name})
################################################################################

systems:
  - id: "{system_type}"
    connector_type: "{system_type}"
    enabled: true
    connection:
      host: "localhost"
      port: 4242
    monitoring:
      collection_interval: {interval}

observability:
  metrics:
    enabled: {str(enable_metrics).lower()}
    export:
      enabled: {str(enable_metrics).lower()}
      format: "both"
      directory: "metrics"
      auto_export_interval_minutes: 5

llm_reasoning:
  provider: "{llm_connector}"
  model: "{llm_model}"

strategy:
  type: "{strategy}"
  {strategy}:
    temperature: 0.1
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_content)

    print("\n" + "=" * 50)
    print(f"Success! Configuration generated at: {output_path}")
    print(f"Run Polaris with: python -m polaris.cli --config {output_path}")
    print("=" * 50 + "\n")

    return 0
