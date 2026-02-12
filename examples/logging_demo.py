#!/usr/bin/env python3
"""
Polaris Logging Demo.

This example demonstrates the different logging formats and CLI export options.
"""

import asyncio
import sys
from pathlib import Path

from polaris.infrastructure.observability.logger import create_logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


async def demo_structured_logging():
    """Demonstrate structured JSON logging."""
    print("=== Structured (JSON) Logging Demo ===")

    logger = create_logger(logger_type="structured", name="demo", level="INFO", console=True)

    logger.info("Starting structured logging demo")
    logger.info("Processing user request", user_id=12345, action="login")
    logger.warning("High memory usage detected", memory_percent=85.2, threshold=80.0)
    logger.error(
        "Database connection failed", error="Connection timeout", retry_count=3, database="user_db"
    )

    print()


async def demo_human_readable_logging():
    """Demonstrate human-readable logging."""
    print("=== Human-Readable Logging Demo ===")

    logger = create_logger(
        logger_type="human", name="demo", level="INFO", console=True, use_colors=True
    )

    logger.info("Starting human-readable logging demo")
    logger.info("Processing user request", user_id=12345, action="login")
    logger.warning("High memory usage detected", memory_percent=85.2, threshold=80.0)
    logger.error(
        "Database connection failed", error="Connection timeout", retry_count=3, database="user_db"
    )

    print()


async def demo_file_export():
    """Demonstrate logging to file."""
    print("=== File Export Demo ===")

    # Create logs directory
    Path("./demo_logs").mkdir(exist_ok=True)

    # Structured logging to file
    structured_logger = create_logger(
        logger_type="structured",
        name="structured_demo",
        level="DEBUG",
        log_file="./demo_logs/structured.log",
        console=False,
    )

    # Human-readable logging to file
    human_logger = create_logger(
        logger_type="human",
        name="human_demo",
        level="DEBUG",
        log_file="./demo_logs/human.log",
        console=False,
    )

    # Log some messages
    for i in range(5):
        structured_logger.info(f"Structured log entry {i}", iteration=i, status="active")
        human_logger.info(f"Human-readable log entry {i}", iteration=i, status="active")

    print("Logs written to:")
    print("  - ./demo_logs/structured.log (JSON format)")
    print("  - ./demo_logs/human.log (Human-readable format)")
    print()


async def demo_polaris_with_different_loggers():
    """Demonstrate Polaris with different logger configurations."""
    print("=== Polaris Integration Demo ===")

    # This would normally use a real config file
    # For demo purposes, we'll show how CLI overrides work

    print("Example CLI commands:")
    print()
    print("1. Use human-readable logging:")
    print(
        "   python -m polaris.cli.main --config config/default.yaml "
        "--log-format structured --export-logs ./structured-export.log"
    )
    print("2. Export logs to custom file:")
    print("   python -m polaris.cli.main --config config/default.yaml --export-logs ./my-logs.log")
    print()
    print("3. Set log level and format:")
    print(
        "   python -m polaris.cli.main --config config/default.yaml --log-level DEBUG --log-format human"
    )
    print()
    print("4. Use structured logging with file export:")
    print(
        "   python -m polaris.cli.main --config config/default.yaml "
        "--log-format structured --export-logs ./structured-export.log"
    )


async def main():
    """Run all logging demos."""
    print("Polaris Logging System Demo")
    print("=" * 50)
    print()

    await demo_structured_logging()
    await demo_human_readable_logging()
    await demo_file_export()
    await demo_polaris_with_different_loggers()

    print("Demo completed!")
    print()
    print("Configuration options:")
    print("- Set 'type: structured' for JSON logging")
    print("- Set 'type: human' for human-readable logging")
    print("- Use CLI flags to override config settings")
    print("- Export logs to any file with --export-logs")


if __name__ == "__main__":
    asyncio.run(main())
