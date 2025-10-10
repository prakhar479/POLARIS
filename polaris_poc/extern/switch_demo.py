#!/usr/bin/env python3
"""
Demo script for Switch System Connector.

This script demonstrates how to use the Switch system connector
to interact with the YOLO model switching system through POLARIS.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from switch_connector import SwitchSystemConnector
from polaris.common.config import ConfigurationManager


async def demo_switch_connector():
    """Demonstrate the Switch system connector functionality."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("switch_demo")
    
    # Load configuration
    config_manager = ConfigurationManager(logger)
    
    try:
        # Load the switch system configuration
        config_path = Path(__file__).parent 
        system_config = config_manager.load_plugin_config(config_path)
        
        logger.info("Configuration loaded successfully")
        
        # Create connector instance
        connector = SwitchSystemConnector(system_config, logger)
        
        # Connect to the system
        logger.info("Connecting to Switch system...")
        await connector.connect()
        
        # Demonstrate basic operations
        logger.info("=== Switch System Demo ===")
        
        # 1. Get current system state
        logger.info("1. Getting system state...")
        state = await connector.get_system_state()
        logger.info(f"Current model: {state['current_model']}")
        logger.info(f"Available models: {state['available_models']}")
        
        # 2. Get performance metrics
        logger.info("2. Getting performance metrics...")
        metrics = await connector.get_performance_metrics()
        logger.info(f"Processing time: {metrics['image_processing_time']:.3f}s")
        logger.info(f"Confidence: {metrics['confidence']:.3f}")
        logger.info(f"Utility: {metrics['utility']:.3f}")
        
        # 3. Switch models
        logger.info("3. Demonstrating model switching...")
        
        # Switch to different models
        for model in ["yolov5s", "yolov5m", "yolov5n"]:
            logger.info(f"Switching to {model}...")
            result = await connector.execute_command("switch_model", {"model": model})
            logger.info(f"Result: {result}")
            
            # Verify the switch
            current = await connector.execute_command("get_current_model")
            logger.info(f"Current model after switch: {current}")
            
            # Wait a bit between switches
            await asyncio.sleep(2)
        
        # 4. Demonstrate optimal model selection
        logger.info("4. Demonstrating optimal model selection...")
        
        # Test different input rates
        test_rates = [1.0, 3.0, 7.0, 15.0, 25.0]
        
        for rate in test_rates:
            logger.info(f"Finding optimal model for input rate: {rate}")
            optimal_model = await connector.switch_to_optimal_model(rate)
            logger.info(f"Optimal model for rate {rate}: {optimal_model}")
            await asyncio.sleep(1)
        
        # 5. Update knowledge base (thresholds)
        logger.info("5. Demonstrating knowledge base update...")
        
        new_thresholds = {
            "yolov5nLower": "0.0",
            "yolov5nUpper": "1.5",
            "yolov5sLower": "1.5", 
            "yolov5sUpper": "4.0",
            "yolov5mLower": "4.0",
            "yolov5mUpper": "8.0",
            "yolov5lLower": "8.0",
            "yolov5lUpper": "15.0",
            "yolov5xLower": "15.0",
            "yolov5xUpper": "50.0"
        }
        
        result = await connector.execute_command("update_knowledge", new_thresholds)
        logger.info(f"Knowledge update result: {result}")
        
        # 6. Test processing control
        logger.info("6. Testing processing control...")
        
        # Note: These commands control the actual processing pipeline
        # Uncomment only if you want to test with a running system
        
        logger.info("Starting processing...")
        result = await connector.execute_command("start_processing")
        logger.info(f"Start result: {result}")
        
        await asyncio.sleep(5)
        
        logger.info("Stopping processing...")
        result = await connector.execute_command("stop_processing")
        logger.info(f"Stop result: {result}")
        
        logger.info("=== Demo completed successfully ===")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Always disconnect
        try:
            await connector.disconnect()
            logger.info("Disconnected from Switch system")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


async def demo_health_monitoring():
    """Demonstrate health monitoring capabilities."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("health_demo")
    
    # Load configuration
    config_manager = ConfigurationManager(logger)
    config_path = Path(__file__).parent / "switch_system_config.yaml"
    system_config = config_manager._load_from_yaml(config_path)
    
    # Create connector
    connector = SwitchSystemConnector(system_config, logger)
    
    try:
        await connector.connect()
        
        logger.info("=== Health Monitoring Demo ===")
        
        # Monitor system health for a period
        for i in range(10):
            is_healthy = await connector.health_check()
            state = await connector.get_system_state()
            metrics = await connector.get_performance_metrics()
            
            logger.info(
                f"Health check {i+1}: {'HEALTHY' if is_healthy else 'UNHEALTHY'} | "
                f"Model: {state['current_model']} | "
                f"Processing time: {metrics['image_processing_time']:.3f}s | "
                f"Confidence: {metrics['confidence']:.3f}"
            )
            
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
    
    finally:
        await connector.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Switch System Connector Demo")
    parser.add_argument(
        "--mode",
        choices=["basic", "health"],
        default="basic",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        asyncio.run(demo_switch_connector())
    elif args.mode == "health":
        asyncio.run(demo_health_monitoring())