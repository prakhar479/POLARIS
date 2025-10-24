#!/usr/bin/env python3
"""
Enhanced demo script for the Switch System Connector.

This script provides a comprehensive demonstration of the Switch system connector's
capabilities, broken down into various scenarios. It shows how to interact
with the YOLO model switching system through the POLARIS framework.

Version 2: Includes robust health polling after server restart.
"""

import asyncio
import logging
import sys
from pathlib import Path
import argparse
import time

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import from the switch_plugin directory
from switch_plugin.connector import SwitchSystemConnector
from polaris.common.config import ConfigurationManager


# --- New Helper Function ---

async def await_system_healthy(connector: SwitchSystemConnector, logger: logging.Logger, timeout: int = 30, interval: int = 2):
    """Poll the system's health check until it becomes healthy or times out."""
    logger.info(f"Waiting for system to become healthy (timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await connector.health_check():
            logger.info("System is healthy and responsive.")
            return True
        logger.info("  - System not ready yet, retrying...")
        await asyncio.sleep(interval)
    
    logger.error(f"System did not become healthy within {timeout} seconds.")
    return False

# --- Scenario Functions ---

async def scenario_get_system_info(connector: SwitchSystemConnector, logger: logging.Logger):
    """Scenario 1: Get basic system information and state."""
    logger.info("--- SCENARIO 1: GETTING SYSTEM INFORMATION ---")
    
    # 1. Health Check
    logger.info("1.1 Performing health check...")
    is_healthy = await connector.health_check()
    logger.info(f"System is {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
    
    # 2. Get System State
    logger.info("\n1.2 Getting system state...")
    state = await connector.get_system_state()
    logger.info(f"  - Current model: {state.get('current_model', 'N/A')}")
    logger.info(f"  - Available models: {state.get('available_models', [])}")
    logger.info(f"  - Timestamp: {state.get('timestamp', 0)}")


async def scenario_model_switching(connector: SwitchSystemConnector, logger: logging.Logger):
    """Scenario 2: Demonstrate manual and optimal model switching."""
    logger.info("\n--- SCENARIO 2: MODEL SWITCHING ---")
    
    # 1. Manual Model Switching
    logger.info("2.1 Demonstrating manual model switching...")
    models_to_test = ["yolov5s", "yolov5m", "yolov5n"]
    for model in models_to_test:
        logger.info(f"  - Switching to {model}...")
        result = await connector.execute_command("switch_model", {"model": model})
        logger.info(f"    - Result: {result}")
        current_model = await connector.execute_command("get_current_model")
        logger.info(f"    - Verified current model: {current_model}")
        await asyncio.sleep(2)
        
    # 2. Optimal Model Selection
    logger.info("\n2.2 Demonstrating optimal model selection based on input rate...")
    test_rates = [1.0, 4.0, 8.0, 15.0, 30.0]
    for rate in test_rates:
        logger.info(f"  - Finding optimal model for input rate: {rate} images/sec")
        optimal_model = await connector.switch_to_optimal_model(rate)
        logger.info(f"    - Optimal model selected: {optimal_model}")
        await asyncio.sleep(1)


async def scenario_knowledge_update(connector: SwitchSystemConnector, logger: logging.Logger):
    """Scenario 3: Update the knowledge base (model switching thresholds)."""
    logger.info("\n--- SCENARIO 3: KNOWLEDGE BASE UPDATE ---")
    
    logger.info("3.1 Updating model switching thresholds...")
    new_thresholds = {
        "yolov5nLower": "0.0", "yolov5nUpper": "1.5",
        "yolov5sLower": "1.5", "yolov5sUpper": "4.0",
        "yolov5mLower": "4.0", "yolov5mUpper": "8.0",
        "yolov5lLower": "8.0", "yolov5lUpper": "15.0",
        "yolov5xLower": "15.0", "yolov5xUpper": "50.0"
    }
    result = await connector.execute_command("update_knowledge", new_thresholds)
    logger.info(f"  - Knowledge update result: {result}")


async def scenario_processing_control(connector: SwitchSystemConnector, logger: logging.Logger):
    """Scenario 4: Demonstrate starting, stopping, and restarting the processing pipeline."""
    logger.info("\n--- SCENARIO 4: PROCESSING PIPELINE CONTROL ---")
    
    # 1. Start processing
    logger.info("4.1 Starting processing...")
    result = await connector.execute_command("start_processing")
    logger.info(f"  - Result: {result}")
    await asyncio.sleep(5)
    
    # 2. Stop processing
    logger.info("\n4.2 Stopping processing...")
    result = await connector.execute_command("stop_processing")
    logger.info(f"  - Result: {result}")
    await asyncio.sleep(2)
    
    # 3. Restart processing
    logger.info("\n4.3 Restarting processing...")
    result = await connector.execute_command("restart_processing")
    logger.info(f"  - Result: {result}")

    # 4. **MODIFIED**: Wait for the system to be healthy again
    logger.info("\n4.4 Waiting for system to stabilize after restart...")
    is_healthy = await await_system_healthy(connector, logger)
    if not is_healthy:
        logger.error("Could not confirm system health after restart. Aborting.")
        raise RuntimeError("System failed to restart correctly.")
    
    # 5. Final stop
    logger.info("\n4.5 Stopping processing again now that system is stable...")
    result = await connector.execute_command("stop_processing")
    logger.info(f"  - Result: {result}")
    
    
async def scenario_metrics_and_logs(connector: SwitchSystemConnector, logger: logging.Logger):
    """Scenario 5: Retrieve and display various metrics and logs."""
    logger.info("\n--- SCENARIO 5: METRICS AND LOGS ---")
    
    # 1. Get raw metrics
    logger.info("5.1 Getting raw metrics from the system...")
    raw_metrics = await connector.execute_command("get_metrics", {})
    logger.info(f"  - Raw metrics: {raw_metrics}")

    # 2. Get latest logs
    logger.info("\n5.2 Getting latest logs from the system...")
    latest_logs = await connector.execute_command("get_latest_logs")
    logger.info(f"  - Latest logs: {latest_logs}")
    
    # 3. Get formatted performance metrics
    logger.info("\n5.3 Getting key performance metrics...")
    perf_metrics = await connector.get_performance_metrics()
    for key, value in perf_metrics.items():
        logger.info(f"  - {key.replace('_', ' ').title()}: {value}")


async def run_full_demo():
    """Run a comprehensive demo of the Switch system connector."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("SwitchDemo")

    config_manager = ConfigurationManager(logger)
    connector = None

    try:
        # Load config from switch_plugin directory
        config_path = Path(__file__).parent / "switch_plugin"
        system_config = config_manager.load_plugin_config(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"System name: {system_config.get('system_name', 'N/A')}")

        connector = SwitchSystemConnector(system_config, logger)

        logger.info("Connecting to Switch system...")
        await connector.connect()

        # Run all scenarios
        await scenario_get_system_info(connector, logger)
        await scenario_model_switching(connector, logger)
        await scenario_knowledge_update(connector, logger)
        await scenario_processing_control(connector, logger)
        await scenario_metrics_and_logs(connector, logger)
        
        logger.info("\n=== All demo scenarios completed successfully! ===")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=False) # Set to True for full traceback
    finally:
        if connector:
            logger.info("Disconnecting from Switch system.")
            await connector.disconnect()


async def run_health_monitoring():
    """Demonstrate continuous health monitoring."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("HealthDemo")

    config_manager = ConfigurationManager(logger)
    connector = None

    try:
        # Load config from switch_plugin directory
        config_path = Path(__file__).parent / "switch_plugin"
        system_config = config_manager.load_plugin_config(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
        
        connector = SwitchSystemConnector(system_config, logger)
        await connector.connect()

        logger.info("=== Starting Health Monitoring Demo (run for 60 seconds) ===")
        for i in range(12):
            is_healthy = await connector.health_check()
            state = await connector.get_system_state()
            metrics = await connector.get_performance_metrics()
            
            logger.info(
                f"[{i+1}/12] Health: {'OK' if is_healthy else 'FAIL'} | "
                f"Model: {state['current_model']} | "
                f"Processing Time: {metrics['image_processing_time']:.3f}s | "
                f"Confidence: {metrics['confidence']:.3f}"
            )
            await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Health monitoring failed: {e}", exc_info=False)
    finally:
        if connector:
            await connector.disconnect()
            logger.info("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced demo for the Switch System Connector.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["full", "health"],
        default="full",
        help=(
            "Select the demo mode to run:\n"
            "  - full:   Run all demonstration scenarios sequentially.\n"
            "  - health: Run a continuous health monitoring loop."
        )
    )

    args = parser.parse_args()

    if args.mode == "full":
        asyncio.run(run_full_demo())
    elif args.mode == "health":
        asyncio.run(run_health_monitoring())