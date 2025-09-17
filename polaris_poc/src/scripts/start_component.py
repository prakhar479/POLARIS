#!/usr/bin/env python3
"""
Entry point script for starting POLARIS components.

This script can launch monitor, execution adapters with a specified
managed system plugin, or the Digital Twin component.

Examples:
    # Start monitor with SWIM plugin
    python start_component.py monitor --plugin-dir extern

    # Start execution adapter with custom config
    python start_component.py execution --plugin-dir extern --config custom_config.yaml

    # Start Digital Twin component
    python start_component.py digital-twin

    # Start Digital Twin with specific World Model
    python start_component.py digital-twin --world-model gemini

    # Start with debug logging
    python start_component.py monitor --plugin-dir extern --log-level DEBUG
"""

import argparse
import asyncio
import logging
import os
import signal
import dotenv
import subprocess
import sys
from pathlib import Path
from typing import Optional


dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from polaris.adapters.monitor import MonitorAdapter
from polaris.adapters.execution import ExecutionAdapter
from polaris.adapters.verification import VerificationAdapter
from polaris.agents.digital_twin_agent import DigitalTwinAgent
from polaris.services.knowledge_base_service import KnowledgeBaseService
from polaris.common.logging_setup import setup_logging
from polaris.common.digital_twin_config import DigitalTwinConfigManager, DigitalTwinConfigError
from polaris.common.digital_twin_logging import setup_digital_twin_logging
from polaris.agents.reasoner_agent import (
    create_reasoner_agent,
    SkeletonReasoningImplementation,
    ReasoningType,
)
from polaris.agents.llm_reasoner import create_llm_reasoner_agent


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start POLARIS components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s monitor --plugin-dir extern          # Start monitor adapter
  %(prog)s execution --plugin-dir extern        # Start execution adapter
  %(prog)s digital-twin                         # Start Digital Twin
  %(prog)s digital-twin --world-model gemini    # Start Digital Twin with Gemini
  %(prog)s knowledge-base                       # Start Knowledge Base Service
  %(prog)s kernel                               # Start Kernel Service 
        """,
    )

    parser.add_argument(
        "component",
        choices=[
            "monitor",
            "execution",
            "verification",
            "digital-twin",
            "knowledge-base",
            "kernel",
            "reasoner",
        ],
        help="Component to start",
    )

    parser.add_argument(
        "--plugin-dir",
        help="Directory containing the managed system plugin (required for monitor/execution)",
    )

    parser.add_argument(
        "--config",
        default="src/config/polaris_config.yaml",
        help="Path to POLARIS framework configuration",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration and exit"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Initialize component but don't start processing"
    )

    # Digital Twin specific arguments
    parser.add_argument(
        "--world-model",
        choices=["mock", "gemini", "statistical", "hybrid"],
        help="Override World Model implementation (Digital Twin only)",
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit (Digital Twin only)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.component in ["monitor", "execution"] and not args.plugin_dir:
        parser.error(f"{args.component} component requires --plugin-dir")

    # Verification can work with or without plugin-dir
    if args.component == "verification" and not args.plugin_dir:
        print(
            "⚠️  Warning: Starting verification without plugin directory - using framework defaults only"
        )

    if args.component not in ["digital-twin", "knowledge-base"] and (
        args.world_model or args.health_check
    ):
        parser.error("--world-model and --health-check are only valid for digital-twin component")

    # Resolve configuration path
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Handle Digital Twin component
    if args.component == "digital-twin":
        await start_digital_twin(args, config_path)
        return

    # Handle Knowledge Base Service
    if args.component == "knowledge-base":
        await start_knowledge_base(args, config_path)
        return

    if args.component == "kernel":
        await start_kernel(args, config_path)
        return

    if args.component == "reasoner":
        await start_reasoner(args, config_path)
        return

    # Handle adapter components (monitor/execution/verification)
    await start_adapter(args, config_path)


async def start_adapter(args, config_path: Path):
    """Start monitor, execution, or verification adapter."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Handle plugin directory requirements
    plugin_dir = None
    if args.plugin_dir:
        plugin_dir = Path(args.plugin_dir).resolve()
        if not plugin_dir.exists():
            logger.error(f"Plugin directory not found: {plugin_dir}")
            sys.exit(1)

    # Create adapter
    try:
        logger.info(f"Creating {args.component} adapter...")

        if args.component == "monitor":
            if not plugin_dir:
                logger.error("Monitor adapter requires --plugin-dir")
                sys.exit(1)
            adapter = MonitorAdapter(
                polaris_config_path=str(config_path), plugin_dir=str(plugin_dir), logger=logger
            )
        elif args.component == "execution":
            if not plugin_dir:
                logger.error("Execution adapter requires --plugin-dir")
                sys.exit(1)
            adapter = ExecutionAdapter(
                polaris_config_path=str(config_path), plugin_dir=str(plugin_dir), logger=logger
            )
        else:  # verification
            adapter = VerificationAdapter(
                polaris_config_path=str(config_path),
                plugin_dir=str(plugin_dir) if plugin_dir else None,
                logger=logger,
            )

        logger.info(f"✅ {args.component.capitalize()} adapter created successfully")

        # Show different info based on adapter type
        if args.component == "verification":
            system_name = getattr(adapter, "system_name", "polaris_framework")
            constraints_count = len(getattr(adapter, "constraints", []))
            policies_count = len(getattr(adapter, "policies", []))
            logger.info(f"   System: {system_name}")
            logger.info(f"   Constraints: {constraints_count}")
            logger.info(f"   Policies: {policies_count}")
        else:
            logger.info(f"{args.component}")
            logger.info(f"   System: {adapter.plugin_config.get('system_name')}")
            logger.info(f"   Connector: {adapter.connector.__class__.__name__}")

        # Validation-only mode
        if args.validate_only:
            logger.info("✅ Configuration validation passed")
            logger.info("🏁 Validation complete - exiting")
            return

    except Exception as e:
        logger.error(f"❌ Failed to create adapter: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Setup signal handling
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda *_: signal_handler())

    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - adapter initialized but not started")
        logger.info("🏁 Dry run complete - exiting")
        return

    # Start adapter
    logger.info(f"🚀 Starting {args.component} adapter...")

    try:
        async with adapter:
            logger.info(f"✅ {args.component.capitalize()} adapter started successfully")
            logger.info("📡 Adapter is running - press Ctrl+C to stop")
            await stop_event.wait()
    except Exception as e:
        logger.error(f"❌ Adapter error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)

    logger.info(f"🛑 {args.component.capitalize()} adapter stopped")


async def start_kernel(args, config_path: Path):
    """Start POLARIS kernel."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Validation-only mode
    if args.validate_only:
        logger.info("✅ Kernel validation passed (configuration file exists)")
        logger.info("🏁 Validation complete - exiting")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - kernel would be started but not executing")
        logger.info("🏁 Dry run complete - exiting")
        return

    # Find the src directory (should be one level up from the script location)
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent

    if not src_dir.exists():
        logger.error(f"❌ Source directory not found: {src_dir}")
        sys.exit(1)

    # Prepare the kernel command
    kernel_cmd = [sys.executable, "-m", "polaris.kernel.kernel"]

    # Add config argument if not default
    if args.config != "src/config/polaris_config.yaml":
        kernel_cmd.extend(["--config", str(config_path)])

    logger.info("🚀 Starting POLARIS kernel...")
    logger.info(f"   Working directory: {src_dir}")
    logger.info(f"   Command: {' '.join(kernel_cmd)}")

    # Setup signal handling for graceful shutdown
    kernel_process = None

    def signal_handler(signum, frame):
        logger.info(f"🔔 Received signal {signum}")
        if kernel_process:
            logger.info("🛑 Terminating kernel process...")
            kernel_process.terminate()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the kernel process
        kernel_process = subprocess.Popen(
            kernel_cmd, cwd=src_dir, env=dict(os.environ, PYTHONPATH=str(src_dir))
        )

        logger.info("✅ POLARIS kernel started successfully")
        logger.info("📡 Kernel is running - press Ctrl+C to stop")

        # Wait for the process to complete
        return_code = kernel_process.wait()

        if return_code == 0:
            logger.info("✅ POLARIS kernel stopped cleanly")
        else:
            logger.error(f"❌ POLARIS kernel exited with code: {return_code}")
            sys.exit(return_code)

    except FileNotFoundError:
        logger.error("❌ Could not find Python interpreter or kernel module")
        logger.error("   Make sure you're running from the correct directory")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Kernel process failed: {e}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("🔔 Received interrupt signal")
        if kernel_process:
            logger.info("🛑 Terminating kernel process...")
            kernel_process.terminate()
            try:
                # Wait a bit for graceful shutdown
                kernel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Kernel didn't stop gracefully, killing...")
                kernel_process.kill()
                kernel_process.wait()
        logger.info("🛑 POLARIS kernel stopped")
    except Exception as e:
        logger.error(f"❌ Unexpected error starting kernel: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def start_knowledge_base(args, config_path: Path):
    """Start Knowledge Base Service."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Create Knowledge Base Service
    service = KnowledgeBaseService(
        nats_url="nats://localhost:4222", telemetry_buffer_size=50, logger=logger
    )

    # Setup graceful shutdown
    stop_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info(f"🔔 Received signal {signum}")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start service in the background
        service_task = asyncio.create_task(service.start())

        # Wait for shutdown signal
        await stop_event.wait()

        # Graceful shutdown
        await service.shutdown()
        service_task.cancel()

        try:
            await service_task
        except asyncio.CancelledError:
            pass

    except Exception as e:
        logger.error(f"❌ Knowledge Base Service error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        raise

        logger.info("🛑 Knowledge Base Service stopped")


async def start_digital_twin(args, config_path: Path):
    """Start Digital Twin component."""
    # Setup basic logging for startup
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load Digital Twin configuration
        logger.info(f"Loading Digital Twin configuration from {config_path}")
        config_manager = DigitalTwinConfigManager(logger)
        config = config_manager.load_configuration(str(config_path))

        # Apply World Model override if specified
        if args.world_model:
            dt_config = config["digital_twin"]
            dt_config["world_model"]["implementation"] = args.world_model
            logger.info(f"World Model implementation overridden to: {args.world_model}")

        # Log configuration summary
        summary = config_manager.create_model_config_summary()
        logger.info("Configuration loaded successfully:")
        logger.info(f"  Implementation: {summary['implementation']}")
        logger.info(f"  Available models: {summary['available_models']}")
        logger.info(f"  gRPC port: {summary['grpc_port']}")
        logger.info(f"  NATS subjects: {summary['nats_subjects']}")

        # Validate environment
        logger.info("Validating environment...")
        issues = []

        # Check World Model implementation requirements
        dt_config = config["digital_twin"]
        implementation = dt_config["world_model"]["implementation"]

        if implementation == "gemini":
            # Check for Gemini API key
            active_config = config_manager.get_active_model_config()
            cfg = active_config.get("config", {}) if isinstance(active_config, dict) else {}
            api_key_env = cfg.get("api_key_env", "GEMINI_API_KEY")

            if api_key_env not in os.environ:
                issues.append(f"Gemini API key not found in environment variable: {api_key_env}")

            # Check for required packages
            try:
                import google.genai

                logger.info("✅ Google Generative AI package available")
            except ImportError:
                issues.append(
                    "Google Generative AI package not installed (pip install google-genai)"
                )

            try:
                import langchain

                logger.info("✅ LangChain package available")
            except ImportError:
                issues.append("LangChain package not installed (pip install langchain)")

        # Check gRPC port availability
        grpc_config = config_manager.get_grpc_config()
        grpc_port = grpc_config.get("port", 50051)

        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(("localhost", grpc_port))
                if result == 0:
                    issues.append(f"gRPC port {grpc_port} is already in use")
                else:
                    logger.info(f"✅ gRPC port {grpc_port} is available")
        except Exception as e:
            logger.warning(f"Could not check gRPC port availability: {e}")

        # Report validation results
        if issues:
            logger.error("Environment validation failed:")
            for issue in issues:
                logger.error(f"  ❌ {issue}")
            sys.exit(1)
        else:
            logger.info("✅ Environment validation passed")

        # Validation-only mode
        if args.validate_only:
            logger.info("✅ Configuration and environment validation passed")
            logger.info("🏁 Validation complete - exiting")
            return

        # Setup Digital Twin specific logging
        dt_logger = setup_digital_twin_logging(config["digital_twin"])

        # Create Digital Twin agent
        logger.info("Creating Digital Twin agent...")
        agent = DigitalTwinAgent(config_path=str(config_path), logger=dt_logger.get_logger())
        logger.info("✅ Digital Twin agent created successfully")

        # Health check mode
        if args.health_check:
            logger.info("Performing health check...")
            try:
                await agent.start()  # Start agent for health check

                # Check World Model health
                world_model = agent.world_model
                if world_model:
                    health_status = await world_model.get_health_status()
                    logger.info(f"World Model health: {health_status}")

                    if health_status.get("status") != "healthy":
                        logger.warning("World Model is not healthy")
                        sys.exit(1)

                # Check NATS connection
                if hasattr(agent, "nats_client") and agent.nats_client:
                    if agent.nats_client.is_connected:
                        logger.info("✅ NATS connection is healthy")
                    else:
                        logger.warning("❌ NATS connection is not healthy")
                        sys.exit(1)

                logger.info("✅ Health check passed")
                return

            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                sys.exit(1)
            finally:
                try:
                    await agent.stop()
                except Exception as e:
                    logger.warning(f"Error during health check cleanup: {e}")

        # Dry run mode
        if args.dry_run:
            logger.info("🧪 Dry run mode - agent initialized but not started")
            logger.info("🏁 Dry run complete - exiting")
            return

        # Setup signal handling
        stop_event = asyncio.Event()

        def signal_handler():
            logger.info("Received shutdown signal")
            stop_event.set()

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda *_: signal_handler())

        # Start Digital Twin agent
        logger.info("🚀 Starting Digital Twin agent...")

        try:
            await agent.start()
            logger.info("✅ Digital Twin agent started successfully")
            logger.info("📡 Digital Twin is running - press Ctrl+C to stop")

            # Log service endpoints
            grpc_host = grpc_config.get("host", "0.0.0.0")
            grpc_port = grpc_config.get("port", 50051)
            logger.info(f"🌐 gRPC service available at {grpc_host}:{grpc_port}")

            nats_config = config_manager.get_nats_config()
            calibrate_subject = nats_config.get("calibrate_subject")
            logger.info(
                f"📨 NATS listening on telemetry/execution streams and calibrate subject: {calibrate_subject}"
            )

            # Wait for shutdown signal
            await stop_event.wait()

        except Exception as e:
            logger.error(f"❌ Digital Twin agent error: {e}")
            raise
        finally:
            # Shutdown agent
            logger.info("🛑 Shutting down Digital Twin agent...")
            try:
                await agent.stop()
                logger.info("✅ Digital Twin agent stopped cleanly")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

    except DigitalTwinConfigError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)


async def start_reasoner(args, config_path: Path):
    """Start Reasoner agent."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Validation-only mode
    if args.validate_only:
        logger.info("✅ Reasoner validation passed (configuration file exists)")
        logger.info("🏁 Validation complete - exiting")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - reasoner would be started but not executing")
        logger.info("🏁 Dry run complete - exiting")
        return

    # Build Reasoner
    agent = create_llm_reasoner_agent(
        "polaris_reasoner_001", config_path, API_KEY, nats_url=None, logger=logger, mode="llm"
    )

    # Setup shutdown handling
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: signal_handler())

    # Start Reasoner
    logger.info("🚀 Starting Reasoner agent...")
    try:
        await agent.connect()
        logger.info("✅ Reasoner agent started successfully")
        logger.info("📡 Reasoner is running - press Ctrl+C to stop")

        # Wait until shutdown
        await stop_event.wait()
    except Exception as e:
        logger.error(f"❌ Reasoner agent error: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Reasoner agent...")
        await agent.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
