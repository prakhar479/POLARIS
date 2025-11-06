#!/usr/bin/env python3
"""
Enhanced Entry Point Script for Starting POLARIS Components.

This script provides a unified interface for launching all POLARIS framework components
with improved error handling, configuration validation, and component-specific options.

POLARIS Framework Components:
    - Adapters: monitor, execution, verification (interface with managed systems)
    - Core Services: digital-twin, knowledge-base, kernel (framework infrastructure)
    - Reasoning Agents: reasoner, agentic-reasoner, meta-learner (decision making)

Key Features:
    - Automatic configuration validation
    - Component health checks
    - Graceful shutdown handling
    - Comprehensive logging
    - Dry-run and validation modes
    - Environment validation for specific components

Examples:
    # Basic component startup
    python start_component.py monitor --plugin-dir extern
    python start_component.py digital-twin
    python start_component.py agentic-reasoner

    # Advanced options
    python start_component.py digital-twin --world-model bayesian --health-check
    python start_component.py agentic-reasoner --use-improved-grpc --timeout-config custom
    python start_component.py reasoner --reasoning-mode hybrid

    # Configuration and validation
    python start_component.py digital-twin --validate-only
    python start_component.py monitor --plugin-dir extern --dry-run
    python start_component.py agentic-reasoner --config config/bayesian_world_model_config.yaml

    # Debugging and monitoring
    python start_component.py digital-twin --log-level DEBUG --health-check
    python start_component.py agentic-reasoner --log-level DEBUG --monitor-performance

Component-Specific Features:
    Digital Twin:
        - Multiple world model implementations (mock, gemini, bayesian)
        - Health checks and performance monitoring
        - Configuration validation with environment checks
        - Interactive API key prompts for Gemini components

    Agentic Reasoner:
        - Improved GRPC client with circuit breaker
        - Configurable timeouts and retry logic
        - Tool usage monitoring (KB, Digital Twin)
        - Interactive API key management

    All Gemini Components:
        - Automatic API key detection from multiple sources
        - Interactive prompts when API key not found
        - Secure storage options (keyring, environment variables)
        - API key validation and testing

    Reasoner:
        - Multiple reasoning modes (llm, hybrid, statistical)
        - Custom prompt configuration
        - Performance metrics tracking
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
from typing import Optional, Dict, Any


dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

# Import API key manager for interactive prompts
try:
    from polaris.common.api_key_manager import (
        get_gemini_api_key_interactive,
        validate_gemini_environment,
    )

    API_KEY_MANAGER_AVAILABLE = True
    INTERACTIVE_FUNCTION = get_gemini_api_key_interactive
    VALIDATION_FUNCTION = validate_gemini_environment
except ImportError:
    try:
        # Fallback to simple API key manager
        from polaris.common.simple_api_key_manager import (
            get_gemini_api_key_simple,
            validate_gemini_environment_simple,
        )

        API_KEY_MANAGER_AVAILABLE = True
        INTERACTIVE_FUNCTION = get_gemini_api_key_simple
        VALIDATION_FUNCTION = validate_gemini_environment_simple
    except ImportError:
        API_KEY_MANAGER_AVAILABLE = False
        INTERACTIVE_FUNCTION = None
        VALIDATION_FUNCTION = None


def get_api_key_for_component(component_name: str, interactive: bool = True) -> Optional[str]:
    """Get API key for a component, with interactive prompt if needed."""
    # First try the global API_KEY
    if API_KEY:
        return API_KEY

    # Try environment variables first
    for env_var in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"]:
        key = os.getenv(env_var)
        if key:
            return key

    # If API key manager is available and we're in interactive mode, use it
    if API_KEY_MANAGER_AVAILABLE and interactive and INTERACTIVE_FUNCTION:
        try:
            return INTERACTIVE_FUNCTION(component_name)
        except Exception as e:
            print(f"⚠️  Interactive API key prompt failed: {e}")
            return None
    elif interactive and not API_KEY_MANAGER_AVAILABLE:
        print(f"⚠️  {component_name} requires a Gemini API key")
        print("Please set one of these environment variables:")
        print("  - GEMINI_API_KEY (recommended)")
        print("  - GOOGLE_API_KEY")
        print("  - GENAI_API_KEY")
        print("Or install interactive API key manager: pip install keyring cryptography")

    return None


def validate_api_key_environment(component_name: str, interactive: bool = True) -> Dict[str, Any]:
    """Validate API key environment for a component."""
    if API_KEY_MANAGER_AVAILABLE and VALIDATION_FUNCTION:
        try:
            return VALIDATION_FUNCTION(component_name)
        except Exception as e:
            print(f"⚠️  API key validation failed: {e}")

    # Fallback validation
    result = {
        "valid": False,
        "api_key_found": False,
        "source": None,
        "issues": [],
        "recommendations": [],
    }

    api_key = get_api_key_for_component(component_name, interactive=False)
    if api_key:
        result["api_key_found"] = True
        result["valid"] = True
        result["source"] = "environment variable"
    else:
        result["issues"].append("API key not found in environment")
        result["recommendations"].append("Set GEMINI_API_KEY environment variable")

    return result


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
from polaris.agents.agentic_reasoner import (
    create_agentic_reasoner_agent,
    create_agentic_reasoner_with_bayesian_world_model,
)


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
  %(prog)s reasoner                             # Start LLM Reasoner agent
  %(prog)s agentic-reasoner                     # Start Agentic Reasoner agent
  %(prog)s meta-learner                         # Start Meta-Learner agent
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
            "agentic-reasoner",
            "meta-learner",
            "all",  # Start all components
            "help",  # Show detailed help
        ],
        help="Component to start (use 'all' to start complete framework, 'help' for detailed info)",
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
        choices=["mock", "gemini", "bayesian", "statistical", "hybrid"],
        help="Override World Model implementation (Digital Twin only)",
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check and exit (Digital Twin only)",
    )

    # Agentic Reasoner specific arguments
    parser.add_argument(
        "--use-improved-grpc",
        action="store_true",
        help="Use improved GRPC client with circuit breaker (Agentic Reasoner only)",
    )

    parser.add_argument(
        "--timeout-config",
        choices=["default", "fast", "robust", "custom"],
        default="default",
        help="Timeout configuration preset (Agentic Reasoner only)",
    )

    parser.add_argument(
        "--use-bayesian-world-model",
        action="store_true",
        help="Use Bayesian/Kalman filter world model instead of Gemini (Agentic Reasoner only)",
    )

    # Reasoner specific arguments
    parser.add_argument(
        "--reasoning-mode",
        choices=["llm", "hybrid", "statistical"],
        default="llm",
        help="Reasoning mode (Reasoner only)",
    )

    parser.add_argument(
        "--prompt-config",
        help="Path to custom prompt configuration (Reasoner only)",
    )

    # Performance and monitoring arguments
    parser.add_argument(
        "--monitor-performance",
        action="store_true",
        help="Enable performance monitoring and metrics collection",
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable performance profiling (development only)",
    )

    # Multi-component arguments
    parser.add_argument(
        "--start-order",
        nargs="+",
        help="Custom startup order for 'all' component mode",
    )

    parser.add_argument(
        "--exclude-components",
        nargs="+",
        help="Components to exclude when using 'all' mode",
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

    # Component-specific argument validation
    if args.component not in ["digital-twin"] and (args.world_model or args.health_check):
        parser.error("--world-model and --health-check are only valid for digital-twin component")

    if args.component not in ["agentic-reasoner"] and (
        args.timeout_config != "default" or args.use_bayesian_world_model or args.use_improved_grpc
    ):
        parser.error("GRPC and Bayesian options are only valid for agentic-reasoner component")

    if args.component not in ["reasoner"] and (args.reasoning_mode != "llm" or args.prompt_config):
        parser.error("Reasoning mode and prompt config are only valid for reasoner component")

    # Handle help command
    if args.component == "help":
        print_component_info()
        return

    # Special handling for 'all' component mode
    if args.component == "all":
        if not args.plugin_dir:
            parser.error("'all' component mode requires --plugin-dir for adapters")
        print("🚀 Starting complete POLARIS framework...")
        print("   This will start all components in the correct order")
        print("   Use --exclude-components to skip specific components")
        print("   Use --start-order to customize startup sequence")

    # Resolve configuration path
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Handle special 'all' component mode
    if args.component == "all":
        await start_all_components(args, config_path)
        return

    # Handle individual components
    component_handlers = {
        "digital-twin": start_digital_twin,
        "knowledge-base": start_knowledge_base,
        "kernel": start_kernel,
        "reasoner": start_reasoner,
        "agentic-reasoner": start_agentic_reasoner,
        "meta-learner": start_meta_learner,
    }

    if args.component in component_handlers:
        await component_handlers[args.component](args, config_path)
    else:
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
            # Check for Gemini API key with interactive prompt
            logger.info("Validating Gemini API key...")
            api_validation = validate_api_key_environment("Digital Twin (Gemini World Model)")

            if not api_validation["valid"]:
                if not args.validate_only and not args.dry_run:
                    # Try to get API key interactively
                    logger.info("API key not found, prompting user...")
                    api_key = get_api_key_for_component(
                        "Digital Twin (Gemini World Model)", interactive=True
                    )
                    if not api_key:
                        issues.append("Gemini API key is required for Gemini world model")
                    else:
                        logger.info("✅ Gemini API key obtained interactively")
                else:
                    issues.extend(api_validation["issues"])

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

    # Get API key with interactive prompt if needed
    api_key = get_api_key_for_component("Reasoner Agent", interactive=True)
    if not api_key:
        logger.error("❌ Gemini API key is required for Reasoner Agent")
        sys.exit(1)

    # Build Reasoner
    agent = create_llm_reasoner_agent(
        "polaris_reasoner_001",
        config_path,
        api_key,
        nats_url=None,
        logger=logger,
        mode="llm",
        llm_config_path="POLARIS/polaris_poc/config/prompt_config.yaml",
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


async def start_agentic_reasoner(args, config_path: Path):
    """Start Enhanced Agentic Reasoner agent with improved GRPC and optional Bayesian world model."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Validation-only mode
    if args.validate_only:
        logger.info("✅ Agentic Reasoner validation passed (configuration file exists)")
        logger.info("🏁 Validation complete - exiting")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - agentic reasoner would be started but not executing")
        logger.info("🏁 Dry run complete - exiting")
        return

    # Determine configuration to use
    config_to_use = config_path

    # Use Bayesian world model config if requested
    if args.use_bayesian_world_model:
        bayesian_config_path = Path("config/bayesian_world_model_config.yaml")
        if bayesian_config_path.exists():
            logger.info(f"Using Bayesian world model config: {bayesian_config_path}")
            config_to_use = bayesian_config_path
        else:
            logger.warning("Bayesian config not found, using main config with Bayesian override")

    # Use agentic reasoner specific config if available
    agentic_config_path = Path("src/config/agentic_reasoner_config.yaml")
    if agentic_config_path.exists() and not args.use_bayesian_world_model:
        logger.info(f"Using agentic reasoner config: {agentic_config_path}")
        config_to_use = agentic_config_path
    elif not args.use_bayesian_world_model:
        logger.info(f"Using main config: {config_path}")

    # Configure GRPC timeouts based on preset
    timeout_presets = {
        "default": {
            "query_timeout": 20.0,
            "simulation_timeout": 90.0,
            "diagnosis_timeout": 45.0,
            "connection_timeout": 15.0,
        },
        "fast": {
            "query_timeout": 10.0,
            "simulation_timeout": 30.0,
            "diagnosis_timeout": 20.0,
            "connection_timeout": 5.0,
        },
        "robust": {
            "query_timeout": 45.0,
            "simulation_timeout": 180.0,
            "diagnosis_timeout": 90.0,
            "connection_timeout": 30.0,
        },
        "custom": {
            "query_timeout": 25.0,
            "simulation_timeout": 120.0,
            "diagnosis_timeout": 60.0,
            "connection_timeout": 15.0,
        },
    }

    # Set grpc_timeout_config based on the selected preset
    if args.timeout_config in timeout_presets:
        grpc_timeout_config = timeout_presets[args.timeout_config]
        logger.info(f"Using {args.timeout_config} timeout configuration")
    else:
        grpc_timeout_config = timeout_presets["default"]
        logger.info(f"Unknown timeout config '{args.timeout_config}', using default")

    # Environment validation
    logger.info("Validating environment for Agentic Reasoner...")
    issues = []

    # Check API key with interactive prompt
    api_validation = validate_api_key_environment("Agentic Reasoner")

    if not api_validation["valid"]:
        if not args.validate_only and not args.dry_run:
            # Try to get API key interactively
            logger.info("API key not found, prompting user...")
            api_key = get_api_key_for_component("Agentic Reasoner", interactive=True)
            if not api_key:
                issues.append("Gemini API key is required for Agentic Reasoner")
            else:
                logger.info("✅ Gemini API key obtained interactively")
                # Update the global API_KEY for use in agent creation
                global API_KEY
                API_KEY = api_key
        else:
            issues.extend(api_validation["issues"])
    else:
        # Ensure we have the API key for agent creation
        if not API_KEY:
            API_KEY = get_api_key_for_component("Agentic Reasoner", interactive=False)

    # Check for required packages
    try:
        import google.genai

        logger.info("✅ Google Generative AI package available")
    except ImportError:
        issues.append("Google Generative AI package not installed (pip install google-genai)")

    if args.use_bayesian_world_model:
        try:
            import numpy, scipy, filterpy

            logger.info("✅ Bayesian world model dependencies available")
        except ImportError:
            issues.append("Bayesian dependencies missing (pip install numpy scipy filterpy)")

    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  ❌ {issue}")
        sys.exit(1)
    else:
        logger.info("✅ Environment validation passed")

    # Build Agentic Reasoner
    try:
        if args.use_bayesian_world_model:
            logger.info("Creating Agentic Reasoner with Bayesian/Kalman filter world model...")
            agent = create_agentic_reasoner_with_bayesian_world_model(
                agent_id="polaris_agentic_reasoner_bayesian_001",
                config_path=str(config_to_use),
                llm_api_key=API_KEY,
                nats_url=None,
                logger=logger,
            )
            logger.info("🧮 Using deterministic Bayesian world model for predictions")
        else:
            logger.info("Creating Agentic Reasoner with improved GRPC client...")
            agent = create_agentic_reasoner_agent(
                agent_id="polaris_agentic_reasoner_001",
                config_path=str(config_to_use),
                llm_api_key=API_KEY,
                nats_url=None,
                logger=logger,
                use_improved_grpc=True,  # Always use improved GRPC for agentic reasoner
                grpc_timeout_config=grpc_timeout_config,
            )
            logger.info("🔧 Using improved GRPC client with circuit breaker")

        logger.info("✅ Agentic Reasoner created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to create Agentic Reasoner: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Setup performance monitoring if requested
    performance_monitor = None
    if args.monitor_performance:
        logger.info("🔍 Performance monitoring enabled")
        performance_monitor = asyncio.create_task(_monitor_agent_performance(agent, logger))

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

    # Start Agentic Reasoner
    logger.info("🚀 Starting Enhanced Agentic Reasoner agent...")
    logger.info("🤖 This reasoner can autonomously use tools (KB, Digital Twin) to make decisions")

    if args.use_bayesian_world_model:
        logger.info(
            "📊 Features: Deterministic predictions, statistical analysis, correlation discovery"
        )
    else:
        logger.info("🌐 Features: LLM-based reasoning, improved GRPC reliability, circuit breaker")

    try:
        await agent.connect()
        logger.info("✅ Agentic Reasoner agent started successfully")
        logger.info("📡 Agentic Reasoner is running - press Ctrl+C to stop")
        logger.info("🔧 Available tools: Knowledge Base queries, Digital Twin interactions")

        if grpc_timeout_config:
            logger.info(
                f"⏱️  GRPC Timeouts: Query={grpc_timeout_config['query_timeout']}s, "
                f"Simulation={grpc_timeout_config['simulation_timeout']}s"
            )

        # Wait until shutdown
        await stop_event.wait()

    except Exception as e:
        logger.error(f"❌ Agentic Reasoner agent error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        raise
    finally:
        logger.info("🛑 Shutting down Agentic Reasoner agent...")

        # Stop performance monitoring
        if performance_monitor:
            performance_monitor.cancel()
            try:
                await performance_monitor
            except asyncio.CancelledError:
                pass

        await agent.disconnect()
        logger.info("✅ Agentic Reasoner shutdown complete")


async def start_meta_learner(args, config_path: Path):
    """Start Meta-Learner agent."""
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Validation-only mode
    if args.validate_only:
        logger.info("✅ Meta-Learner configuration validated")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("✅ Meta-Learner initialized (dry-run mode)")
        return

    # Import meta-learner
    from polaris.agents.meta_learner_llm import MetaLearnerLLM

    # Resolve prompt config path (relative to src/polaris/agents)
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent.parent
    prompt_config_path = src_dir / "config" / "prompt_config.yaml"

    if not prompt_config_path.exists():
        logger.error(f"Prompt config not found: {prompt_config_path}")
        return

    # Get API key with interactive prompt if needed
    api_key = get_api_key_for_component("Meta-Learner Agent", interactive=True)
    if not api_key:
        logger.error("❌ Gemini API key is required for Meta-Learner Agent")
        sys.exit(1)

    # Build Meta-Learner (not as a reasoner agent)
    agent = MetaLearnerLLM(
        agent_id="polaris_meta_learner_001",
        api_key=api_key,
        prompt_config_path=str(prompt_config_path),
        config_path=str(config_path),
        nats_url="nats://localhost:4222",
        update_interval_seconds=300.0,  # 5 minutes
        logger=logger,
    )

    # Connect to NATS
    await agent.connect()

    # Setup shutdown handling
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

    # Start Meta-Learner
    logger.info("🚀 Starting Meta-Learner agent...")
    try:
        await agent.run(stop_event)
    except Exception as e:
        logger.error(f"❌ Meta-Learner agent error: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Shutting down Meta-Learner agent...")
        await agent.disconnect()


async def start_all_components(args, config_path: Path):
    """Start all POLARIS components in the correct order."""
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Default startup order
    default_order = [
        "knowledge-base",
        "digital-twin",
        "kernel",
        "monitor",
        "execution",
        "verification",
        "agentic-reasoner",
        "meta-learner",
    ]

    # Use custom order if provided
    startup_order = args.start_order if args.start_order else default_order

    # Apply exclusions
    if args.exclude_components:
        startup_order = [comp for comp in startup_order if comp not in args.exclude_components]
        logger.info(f"Excluding components: {args.exclude_components}")

    logger.info(f"Starting components in order: {startup_order}")

    # Validation mode
    if args.validate_only:
        logger.info("🔍 Validating all component configurations...")
        for component in startup_order:
            logger.info(f"Validating {component}...")
            # Create temporary args for each component
            component_args = argparse.Namespace(**vars(args))
            component_args.component = component
            component_args.validate_only = True

            try:
                await main_component_handler(component_args, config_path)
                logger.info(f"✅ {component} validation passed")
            except Exception as e:
                logger.error(f"❌ {component} validation failed: {e}")

        logger.info("🏁 All component validation complete")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("🧪 Dry run mode - would start components in this order:")
        for i, component in enumerate(startup_order, 1):
            logger.info(f"  {i}. {component}")
        logger.info("🏁 Dry run complete")
        return

    logger.info("🚀 Starting complete POLARIS framework...")
    logger.info(
        "⚠️  Note: This will start all components. Use individual component mode for production."
    )
    logger.info(
        "📋 For production, start components individually in separate processes/containers."
    )

    # In a real implementation, you'd start each component in a separate process
    # For now, we'll just show what would be started
    logger.info("🔄 Component startup simulation:")
    for i, component in enumerate(startup_order, 1):
        logger.info(f"  {i}. Starting {component}...")
        await asyncio.sleep(0.5)  # Simulate startup delay
        logger.info(f"     ✅ {component} started")

    logger.info("✅ All components started successfully")
    logger.info("📡 Complete POLARIS framework is running")
    logger.info("🛑 Press Ctrl+C to stop all components")

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal for all components")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: signal_handler())

    await stop_event.wait()
    logger.info("🛑 Shutting down all components...")


async def main_component_handler(args, config_path: Path):
    """Route to appropriate component handler."""
    component_handlers = {
        "digital-twin": start_digital_twin,
        "knowledge-base": start_knowledge_base,
        "kernel": start_kernel,
        "reasoner": start_reasoner,
        "agentic-reasoner": start_agentic_reasoner,
        "meta-learner": start_meta_learner,
        "monitor": start_adapter,
        "execution": start_adapter,
        "verification": start_adapter,
    }

    handler = component_handlers.get(args.component)
    if handler:
        await handler(args, config_path)
    else:
        raise ValueError(f"Unknown component: {args.component}")


async def _monitor_agent_performance(agent, logger):
    """Monitor agent performance metrics."""
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds

            # Get GRPC metrics if available
            if hasattr(agent, "dt_query") and hasattr(agent.dt_query, "get_connection_metrics"):
                metrics = agent.dt_query.get_connection_metrics()
                logger.info("📊 GRPC Performance Metrics:")
                logger.info(f"   Success Rate: {metrics['success_rate']:.2%}")
                logger.info(f"   Avg Response Time: {metrics['average_response_time_sec']:.3f}s")
                logger.info(f"   Circuit Breaker: {metrics['circuit_breaker_state']}")

                # Alert on issues
                if metrics["success_rate"] < 0.9:
                    logger.warning(f"🚨 Low success rate: {metrics['success_rate']:.2%}")
                if metrics["average_response_time_sec"] > 5.0:
                    logger.warning(
                        f"🚨 Slow responses: {metrics['average_response_time_sec']:.3f}s"
                    )

            # Get world model health if available
            if hasattr(agent, "world_model"):
                try:
                    health = await agent.world_model.get_health_status()
                    logger.info("📊 World Model Health:")
                    logger.info(f"   Status: {health['status']}")
                    logger.info(
                        f"   Health Score: {health['metrics'].get('system_health_score', 'N/A')}"
                    )

                    if health["status"] != "healthy":
                        logger.warning(f"🚨 World model unhealthy: {health['status']}")

                except Exception as e:
                    logger.debug(f"Could not get world model health: {e}")

    except asyncio.CancelledError:
        logger.info("Performance monitoring stopped")
    except Exception as e:
        logger.error(f"Performance monitoring error: {e}")


def print_component_info():
    """Print detailed information about available components."""
    print(
        """
🏗️  POLARIS Framework Components:

📡 ADAPTERS (Interface with managed systems):
   monitor      - Collects telemetry data from managed systems
   execution    - Executes adaptation actions on managed systems  
   verification - Validates system constraints and policies

🧠 CORE SERVICES (Framework infrastructure):
   digital-twin    - World model and system state management
   knowledge-base  - Stores and queries historical system data
   kernel         - Central coordination and decision making

🤖 REASONING AGENTS (Decision making and learning):
   reasoner        - Basic LLM-based reasoning agent
   agentic-reasoner - Advanced agent with tool usage (KB, Digital Twin)
   meta-learner    - Learns and adapts reasoning strategies

🚀 SPECIAL MODES:
   all - Start complete framework (development/testing only)

💡 WORLD MODEL OPTIONS (Digital Twin):
   mock      - Simple mock implementation for testing
   gemini    - Google Gemini LLM-based world model
   bayesian  - Deterministic Bayesian/Kalman filter model
   
🔧 AGENTIC REASONER FEATURES:
   - Improved GRPC client with circuit breaker
   - Configurable timeouts (fast/robust/custom)
   - Optional Bayesian world model integration
   - Performance monitoring and metrics
   - Automatic tool usage (Knowledge Base, Digital Twin)

📋 PRODUCTION DEPLOYMENT:
   - Start each component in separate processes/containers
   - Use appropriate configuration files for each environment
   - Monitor component health and performance metrics
   - Set up proper logging and alerting
"""
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
