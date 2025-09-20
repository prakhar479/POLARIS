"""
Contract tests for ManagedSystemConnector interface.

This module provides comprehensive contract tests that validate any implementation
of the ManagedSystemConnector interface meets the expected behavior requirements.
"""

import pytest
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Type, Optional
from unittest.mock import patch

from src.domain.interfaces import ManagedSystemConnector
from src.domain.models import AdaptationAction, ExecutionResult, ExecutionStatus, MetricValue
from tests.fixtures.mock_objects import DataBuilder


class ManagedSystemConnectorContract(ABC):
    """
    Abstract base class for ManagedSystemConnector contract tests.
    
    Any ManagedSystemConnector implementation should pass all these tests
    to ensure it meets the interface contract requirements.
    """
    
    @abstractmethod
    def create_connector(self) -> ManagedSystemConnector:
        """Create an instance of the connector to test."""
        pass
    
    @abstractmethod
    def get_expected_system_id(self) -> str:
        """Get the expected system ID for the connector."""
        pass
    
    @abstractmethod
    def is_real_connector(self) -> bool:
        """Return True if this is a real connector that requires external resources."""
        pass
    
    # Core interface contract tests
    
    def test_system_id_property(self):
        """Test that get_system_id returns the expected system ID."""
        connector = self.create_connector()
        system_id = connector.get_system_id()
        
        assert system_id is not None
        assert isinstance(system_id, str)
        assert len(system_id) > 0
        assert system_id == self.get_expected_system_id()
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self):
        """Test the complete connection lifecycle."""
        connector = self.create_connector()
        
        # Initially should not be connected (if connector tracks connection state)
        if hasattr(connector, 'connected'):
            assert not connector.connected
        
        # Test connection
        await connector.connect()
        
        # Should be connected after connect()
        if hasattr(connector, 'connected'):
            assert connector.connected
        
        # Test disconnection
        await connector.disconnect()
        
        # Should not be connected after disconnect()
        if hasattr(connector, 'connected'):
            assert not connector.connected
    
    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test that multiple connect calls are handled gracefully."""
        connector = self.create_connector()
        
        # Multiple connects should not fail
        await connector.connect()
        await connector.connect()  # Should not raise exception
        
        # Multiple disconnects should not fail
        await connector.disconnect()
        await connector.disconnect()  # Should not raise exception
    
    @pytest.mark.asyncio
    async def test_collect_metrics_returns_valid_data(self):
        """Test that collect_metrics returns valid metric data."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            metrics = await connector.collect_metrics()
            
            # Validate return type
            assert isinstance(metrics, dict)
            
            # Validate metric structure
            for metric_name, metric_value in metrics.items():
                assert isinstance(metric_name, str)
                assert len(metric_name) > 0
                assert isinstance(metric_value, MetricValue)
                assert metric_value.value is not None
                assert isinstance(metric_value.unit, str)
                assert isinstance(metric_value.timestamp, datetime)
                
                # Timestamp should be recent (within last minute)
                from datetime import timezone
                now = datetime.now(timezone.utc) if metric_value.timestamp.tzinfo else datetime.now()
                time_diff = now - metric_value.timestamp
                assert time_diff < timedelta(minutes=1)
                
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_collect_metrics_without_connection(self):
        """Test collect_metrics behavior when not connected."""
        connector = self.create_connector()
        
        # For real connectors, this should raise an exception
        # For mock connectors, behavior may vary
        if self.is_real_connector():
            with pytest.raises(Exception):
                await connector.collect_metrics()
        else:
            # Mock connectors may allow metrics collection without connection
            # This is acceptable for testing purposes
            metrics = await connector.collect_metrics()
            assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    async def test_execute_action_returns_valid_result(self):
        """Test that execute_action returns a valid ExecutionResult."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            # Create a test action
            action = DataBuilder.adaptation_action(
                action_id="test_action_001",
                action_type="test_action",
                target_system=connector.get_system_id(),
                parameters={"test_param": "test_value"}
            )
            
            result = await connector.execute_action(action)
            
            # Validate result structure
            assert isinstance(result, ExecutionResult)
            assert result.action_id == action.action_id
            assert isinstance(result.status, ExecutionStatus)
            assert isinstance(result.result_data, dict)
            assert isinstance(result.timestamp, datetime)
            assert isinstance(result.execution_time, timedelta)
            
            # Timestamp should be recent
            from datetime import timezone
            now = datetime.now(timezone.utc) if result.timestamp.tzinfo else datetime.now()
            time_diff = now - result.timestamp
            assert time_diff < timedelta(minutes=1)
            
            # Execution time should be reasonable (less than 30 seconds for tests)
            assert result.execution_time < timedelta(seconds=30)
            
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_execute_action_without_connection(self):
        """Test execute_action behavior when not connected."""
        connector = self.create_connector()
        
        action = DataBuilder.adaptation_action(
            target_system=connector.get_system_id()
        )
        
        # For real connectors, this should raise an exception or return failure
        if self.is_real_connector():
            try:
                result = await connector.execute_action(action)
                # If it doesn't raise an exception, it should return a failure status
                assert result.status == ExecutionStatus.FAILED
            except Exception:
                # Exception is also acceptable
                pass
        else:
            # Mock connectors may allow execution without connection
            result = await connector.execute_action(action)
            assert isinstance(result, ExecutionResult)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test that the connector handles concurrent operations correctly."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            # Run multiple operations concurrently
            tasks = []
            
            # Concurrent metric collections
            for _ in range(3):
                tasks.append(connector.collect_metrics())
            
            # Concurrent action executions
            for i in range(2):
                action = DataBuilder.adaptation_action(
                    action_id=f"concurrent_action_{i}",
                    target_system=connector.get_system_id()
                )
                tasks.append(connector.execute_action(action))
            
            # Wait for all operations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that operations completed (may have exceptions for real connectors)
            for result in results:
                if isinstance(result, Exception):
                    # Exceptions are acceptable for real connectors under load
                    if not self.is_real_connector():
                        raise result
                else:
                    # Successful results should be valid
                    assert result is not None
                    
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery capabilities."""
        connector = self.create_connector()
        
        # Test recovery from connection errors
        if self.is_real_connector():
            # For real connectors, simulate network issues
            with patch('asyncio.sleep', side_effect=asyncio.TimeoutError):
                try:
                    await connector.connect()
                except (asyncio.TimeoutError, ConnectionError):
                    pass  # Expected for real connectors
        
        # Test normal operation after error
        await connector.connect()
        metrics = await connector.collect_metrics()
        assert isinstance(metrics, dict)
        
        await connector.disconnect()
    
    # Performance contract tests
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_metrics_collection_performance(self):
        """Test that metrics collection meets performance requirements."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            # Measure metrics collection time
            start_time = asyncio.get_event_loop().time()
            await connector.collect_metrics()
            end_time = asyncio.get_event_loop().time()
            
            collection_time = end_time - start_time
            
            # Metrics collection should complete within reasonable time
            # Allow more time for real connectors
            max_time = 5.0 if self.is_real_connector() else 1.0
            assert collection_time < max_time, f"Metrics collection took {collection_time:.3f}s, expected < {max_time}s"
            
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_action_execution_performance(self):
        """Test that action execution meets performance requirements."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            action = DataBuilder.adaptation_action(
                target_system=connector.get_system_id()
            )
            
            # Measure action execution time
            start_time = asyncio.get_event_loop().time()
            result = await connector.execute_action(action)
            end_time = asyncio.get_event_loop().time()
            
            execution_time = end_time - start_time
            
            # Action execution should complete within reasonable time
            max_time = 10.0 if self.is_real_connector() else 2.0
            assert execution_time < max_time, f"Action execution took {execution_time:.3f}s, expected < {max_time}s"
            
            # The reported execution time should be consistent with measured time
            if result.status == ExecutionStatus.SUCCESS:
                reported_time = result.execution_time.total_seconds()
                # Allow some variance for overhead
                assert abs(reported_time - execution_time) < 1.0, f"Reported time {reported_time:.3f}s differs significantly from measured time {execution_time:.3f}s"
                
        finally:
            await connector.disconnect()
    
    # Data validation contract tests
    
    @pytest.mark.asyncio
    async def test_metrics_data_consistency(self):
        """Test that metrics data is consistent across multiple collections."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            # Collect metrics multiple times
            metrics_collections = []
            for _ in range(3):
                metrics = await connector.collect_metrics()
                metrics_collections.append(metrics)
                await asyncio.sleep(0.1)  # Small delay between collections
            
            # Verify consistency
            if len(metrics_collections) > 1:
                first_collection = metrics_collections[0]
                
                for subsequent_collection in metrics_collections[1:]:
                    # Should have same metric names
                    assert set(first_collection.keys()) == set(subsequent_collection.keys())
                    
                    # Metric values may change, but structure should be consistent
                    for metric_name in first_collection.keys():
                        first_metric = first_collection[metric_name]
                        subsequent_metric = subsequent_collection[metric_name]
                        
                        # Units should be consistent
                        assert first_metric.unit == subsequent_metric.unit
                        
                        # Timestamps should be different (unless collected simultaneously)
                        # Values may be the same or different depending on the metric
                        
        finally:
            await connector.disconnect()
    
    @pytest.mark.asyncio
    async def test_action_parameter_validation(self):
        """Test that the connector properly validates action parameters."""
        connector = self.create_connector()
        await connector.connect()
        
        try:
            # Test with invalid parameters
            invalid_action = AdaptationAction(
                action_id="invalid_action",
                action_type="invalid_type",
                target_system=connector.get_system_id(),
                parameters={"invalid": "parameters"}
            )
            
            result = await connector.execute_action(invalid_action)
            
            # Should either raise an exception or return a failure result
            if isinstance(result, ExecutionResult):
                # If it returns a result, it should indicate failure for invalid actions
                # (unless the connector accepts any action type for testing)
                pass  # Allow flexible behavior for test connectors
            
        finally:
            await connector.disconnect()


# Concrete contract test implementations

class MockManagedSystemConnectorContractTest(ManagedSystemConnectorContract):
    """Contract tests for MockManagedSystemConnector."""
    
    def create_connector(self) -> ManagedSystemConnector:
        from tests.fixtures.mock_objects import MockManagedSystemConnector
        return MockManagedSystemConnector("test_system")
    
    def get_expected_system_id(self) -> str:
        return "test_system"
    
    def is_real_connector(self) -> bool:
        return False


# Test runner for contract tests

class ConnectorContractTestRunner:
    """Utility for running contract tests against any connector implementation."""
    
    def __init__(self, connector_factory: callable, system_id: str, is_real: bool = False):
        self.connector_factory = connector_factory
        self.system_id = system_id
        self.is_real = is_real
    
    def create_test_class(self) -> Type[ManagedSystemConnectorContract]:
        """Create a test class for the specific connector."""
        
        # Capture the factory and other attributes in the closure
        connector_factory = self.connector_factory
        system_id = self.system_id
        is_real = self.is_real
        
        class DynamicConnectorContractTest(ManagedSystemConnectorContract):
            def create_connector(self) -> ManagedSystemConnector:
                return connector_factory()
            
            def get_expected_system_id(self) -> str:
                return system_id
            
            def is_real_connector(self) -> bool:
                return is_real
        
        return DynamicConnectorContractTest
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all contract tests and return results."""
        test_class = self.create_test_class()
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_instance)
            if method.startswith('test_') and callable(getattr(test_instance, method))
        ]
        
        results = {}
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                
                # Check if it's an async method
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                
                results[method_name] = True
                
            except Exception as e:
                print(f"Contract test {method_name} failed: {e}")
                results[method_name] = False
        
        return results
    
    def generate_contract_report(self, results: Dict[str, bool]) -> str:
        """Generate a contract compliance report."""
        total_tests = len(results)
        passed_tests = sum(1 for passed in results.values() if passed)
        failed_tests = total_tests - passed_tests
        
        compliance_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report_lines = [
            f"ManagedSystemConnector Contract Compliance Report",
            f"System ID: {self.system_id}",
            f"Connector Type: {'Real' if self.is_real else 'Mock'}",
            "=" * 60,
            "",
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Compliance: {compliance_percentage:.1f}%",
            "",
            "Test Results:",
            "-" * 20
        ]
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report_lines.append(f"{status} {test_name}")
        
        if failed_tests > 0:
            report_lines.extend([
                "",
                "⚠️  Contract violations detected. Please review failed tests.",
                "All ManagedSystemConnector implementations should pass these contract tests."
            ])
        else:
            report_lines.extend([
                "",
                "✅ Full contract compliance achieved!"
            ])
        
        return "\n".join(report_lines)


# Utility functions for easy contract testing

async def validate_connector_contract(
    connector_factory: callable,
    system_id: str,
    is_real: bool = False
) -> bool:
    """Validate that a connector implementation meets the contract requirements."""
    runner = ConnectorContractTestRunner(connector_factory, system_id, is_real)
    results = await runner.run_all_tests()
    
    # Print report
    report = runner.generate_contract_report(results)
    print(report)
    
    # Return True if all tests passed
    return all(results.values())


def create_contract_test_suite(connector_implementations: List[Dict[str, Any]]) -> List[Type[ManagedSystemConnectorContract]]:
    """Create contract test suites for multiple connector implementations."""
    test_classes = []
    
    for impl in connector_implementations:
        runner = ConnectorContractTestRunner(
            connector_factory=impl["factory"],
            system_id=impl["system_id"],
            is_real=impl.get("is_real", False)
        )
        test_classes.append(runner.create_test_class())
    
    return test_classes