"""CLI command to initialize a new Polaris configuration."""

import argparse
from pathlib import Path


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

    system_type = (
        input(
            "Which managed system do you want to use? (swim/wildfire/kubernetes) [swim]: "
        ).strip()
        or "swim"
    )
    interval = input("Monitoring interval in seconds [5]: ").strip() or "5"

    enable_metrics = (
        input("Enable metrics collection and background export? (Y/n): ").strip().lower() != "n"
    )

    llm_connector = (
        input("Which LLM provider to use? (openai/anthropic/ollama) [openai]: ").strip() or "openai"
    )
    llm_model = input("Which model version? [gpt-4o]: ").strip() or "gpt-4o"

    strategy = (
        input("Which strategy to use? (threshold/hybrid/agentic_llm) [hybrid]: ").strip()
        or "hybrid"
    )

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
