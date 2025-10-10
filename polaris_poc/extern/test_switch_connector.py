#!/usr/bin/env python3
"""
Integration tests for Switch System Connector.

These tests verify that the connector can properly interface with
the Switch system and handle various scenarios.
"""

import asyncio
import logging
import sys
import tempfile
import csv
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from switch_connector import SwitchSystemConnector


class TestSwitchConnector:
    """Test suite for Switch System Connector."""
    
    def __init__(self):
        self.logger = logging.getLogger("test_switch_connector")
        self.test_config = {
            "system_name": "test_switch_system",
            "connection": {
                "host": "localhost",
                "port": 3001,
                "protocol": "http"
            },
            "implementation": {
                "connector_class": "SwitchSystemConnector",
                "timeout": 10.0,
                "max_retries": 2,
                "retry_base_delay": 0.5,
                "retry_max_delay": 2.0
            },
            "switch_system": {
                "model_file_path": "test_model.csv",
                "monitor_file_path": "test_monitor.csv",
                "knowledge_file_path": "test_knowledge.csv",
                "metrics_file_path": "test_metrics.csv"
            }
        }
    
    async def test_connector_initialization(self):
        """Test connector initialization."""
        print("Testing connector initialization...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        assert connector.host == "localhost"
        assert connector.port == 3001
        assert connector.base_url == "http://localhost:3001"
        assert connector.timeout == 10.0
        assert connector.max_retries == 2
        assert len(connector.available_models) == 5
        
        print("✓ Connector initialization test passed")
    
    async def test_model_file_operations(self):
        """Test model file read/write operations."""
        print("Testing model file operations...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Test writing model file
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = os.path.join(temp_dir, "test_model.csv")
            connector.model_file_path = model_file
            
            # Write model
            await connector._write_model_file("yolov5s")
            
            # Read model
            current_model = await connector._get_current_model()
            assert current_model == "yolov5s"
        
        print("✓ Model file operations test passed")
    
    async def test_knowledge_file_operations(self):
        """Test knowledge file operations."""
        print("Testing knowledge file operations...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Create test knowledge file
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge_file = os.path.join(temp_dir, "test_knowledge.csv")
            connector.knowledge_file_path = knowledge_file
            
            # Write test knowledge data
            test_data = [
                ["0", "0.0", "2.0"],
                ["1", "2.0", "5.0"],
                ["2", "5.0", "10.0"],
                ["3", "10.0", "20.0"],
                ["4", "20.0", "100.0"]
            ]
            
            with open(knowledge_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(test_data)
            
            # Read knowledge
            knowledge = await connector._read_knowledge_file()
            
            assert len(knowledge) == 5
            assert knowledge[0] == ("yolov5n", 0.0, 2.0)
            assert knowledge[1] == ("yolov5s", 2.0, 5.0)
            assert knowledge[4] == ("yolov5x", 20.0, 100.0)
        
        print("✓ Knowledge file operations test passed")
    
    async def test_optimal_model_selection(self):
        """Test optimal model selection logic."""
        print("Testing optimal model selection...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Mock the knowledge file and model switching
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge_file = os.path.join(temp_dir, "test_knowledge.csv")
            model_file = os.path.join(temp_dir, "test_model.csv")
            
            connector.knowledge_file_path = knowledge_file
            connector.model_file_path = model_file
            
            # Create test knowledge data
            test_data = [
                ["0", "0.0", "2.0"],
                ["1", "2.0", "5.0"],
                ["2", "5.0", "10.0"],
                ["3", "10.0", "20.0"],
                ["4", "20.0", "100.0"]
            ]
            
            with open(knowledge_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(test_data)
            
            # Test different input rates
            test_cases = [
                (1.0, "yolov5n"),
                (3.0, "yolov5s"),
                (7.0, "yolov5m"),
                (15.0, "yolov5l"),
                (25.0, "yolov5x")
            ]
            
            for input_rate, expected_model in test_cases:
                selected_model = await connector.switch_to_optimal_model(input_rate)
                assert selected_model == expected_model, f"Rate {input_rate}: expected {expected_model}, got {selected_model}"
        
        print("✓ Optimal model selection test passed")
    
    async def test_command_routing(self):
        """Test command routing to appropriate handlers."""
        print("Testing command routing...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Mock the session and connection
        connector._connected = True
        connector._session = AsyncMock()
        
        # Mock file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = os.path.join(temp_dir, "test_model.csv")
            connector.model_file_path = model_file
            
            # Test get_current_model command
            await connector._write_model_file("yolov5m")
            result = await connector.execute_command("get_current_model")
            assert result == "yolov5m"
            
            # Test switch_model command
            result = await connector.execute_command("switch_model", {"model": "yolov5l"})
            assert "Successfully switched" in result
            
            current = await connector.execute_command("get_current_model")
            assert current == "yolov5l"
        
        print("✓ Command routing test passed")
    
    async def test_error_handling(self):
        """Test error handling scenarios."""
        print("Testing error handling...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Test invalid model
        try:
            await connector.execute_command("switch_model", {"model": "invalid_model"})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid model" in str(e)
        
        # Test missing parameters
        try:
            await connector.execute_command("switch_model", {})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Model parameter required" in str(e)
        
        # Test unknown command
        try:
            await connector.execute_command("unknown_command")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown command" in str(e)
        
        # Test not connected
        connector._connected = False
        try:
            await connector.execute_command("get_current_model")
            assert False, "Should have raised ConnectionError"
        except ConnectionError as e:
            assert "Not connected" in str(e)
        
        print("✓ Error handling test passed")
    
    async def test_performance_metrics_parsing(self):
        """Test performance metrics parsing."""
        print("Testing performance metrics parsing...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Mock metrics data
        mock_metrics = {
            "image_processing_time": "1.234",
            "confidence": "0.85",
            "utility": "0.75",
            "cpu_usage": "45.6",
            "detection_boxes": "3",
            "total_processed": "150"
        }
        
        # Mock the _get_latest_metrics method
        connector._get_latest_metrics = AsyncMock(return_value=mock_metrics)
        
        # Test metrics parsing
        performance = await connector.get_performance_metrics()
        
        assert performance["image_processing_time"] == 1.234
        assert performance["confidence"] == 0.85
        assert performance["utility"] == 0.75
        assert performance["cpu_usage"] == 45.6
        assert performance["detection_boxes"] == 3
        assert performance["total_processed"] == 150
        
        print("✓ Performance metrics parsing test passed")
    
    async def test_system_state_aggregation(self):
        """Test system state aggregation."""
        print("Testing system state aggregation...")
        
        connector = SwitchSystemConnector(self.test_config, self.logger)
        
        # Mock individual methods
        connector._get_current_model = AsyncMock(return_value="yolov5s")
        connector._get_latest_metrics = AsyncMock(return_value={"confidence": "0.8"})
        connector._get_latest_logs = AsyncMock(return_value={"last_action": "switch_model"})
        
        # Test system state aggregation
        state = await connector.get_system_state()
        
        assert state["current_model"] == "yolov5s"
        assert state["metrics"]["confidence"] == "0.8"
        assert state["logs"]["last_action"] == "switch_model"
        assert state["available_models"] == connector.available_models
        assert "timestamp" in state
        
        print("✓ System state aggregation test passed")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("=" * 50)
        print("Running Switch Connector Tests")
        print("=" * 50)
        
        tests = [
            self.test_connector_initialization,
            self.test_model_file_operations,
            self.test_knowledge_file_operations,
            self.test_optimal_model_selection,
            self.test_command_routing,
            self.test_error_handling,
            self.test_performance_metrics_parsing,
            self.test_system_state_aggregation
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                await test()
                passed += 1
            except Exception as e:
                print(f"✗ {test.__name__} FAILED: {e}")
                failed += 1
        
        print("=" * 50)
        print(f"Test Results: {passed} passed, {failed} failed")
        print("=" * 50)
        
        return failed == 0


async def main():
    """Run the test suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = TestSwitchConnector()
    success = await test_suite.run_all_tests()
    
    if success:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed! ✗")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())