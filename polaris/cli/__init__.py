"""CLI package.

Keep imports lazy so optional UI dependencies (for example ``rich``) are not
required for non-interactive CLI commands such as ``polaris --version``.
"""

from typing import Any

__all__ = ["main", "Dashboard", "PolarisInteractiveCLI", "run_interactive_cli"]


def __getattr__(name: str) -> Any:
    """Lazily resolve CLI exports to avoid eager optional dependency imports."""
    if name == "main":
        from polaris.cli.main import main

        return main
    if name == "Dashboard":
        from polaris.cli.dashboard import Dashboard

        return Dashboard
    if name in {"PolarisInteractiveCLI", "run_interactive_cli"}:
        from polaris.cli.interactive import PolarisInteractiveCLI, run_interactive_cli

        return {
            "PolarisInteractiveCLI": PolarisInteractiveCLI,
            "run_interactive_cli": run_interactive_cli,
        }[name]

    raise AttributeError(f"module 'polaris.cli' has no attribute {name!r}")
