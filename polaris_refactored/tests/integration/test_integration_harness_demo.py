"""
Integration test demonstration using the POLARIS Integration Test Harness.

This module demonstrates how to use the integration testing harness and
validates that the harness itself works correctly.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from tests.integration.harness.polaris_integration_test_harness import (
    PolarisIntegrationTestHarness, IntegrationTestConfig,
    create_simple_harness, create_performance_harness, create_failure_testing_harness
)
from tests.integration.contracts.managed_system_connector_contract import (
    MockManagedSystemConnectorContractTest, validate_connector_contract
)
from tests.fixtures.mock_objects import MockManagedSystemConnector, DataBuilder
from src.domain.models import MetricValue, ExecutionStatus


class TestIntegrationHarness:
    """Test the integration test harness functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_harness_basic_setup_and_teardown(self):
        """Test basic harness setup and teardown."""
        config = IntegrationTestConfig(
            test_name="basic_setup_test",
            systems=["test_system_1", "test_system_2"],
            enable_observability=True
        )
        
        harness = PolarisIntegrationTestHarness(config)
        
        # Test setup
        await harness.setup()
        
        # Verify components are initialized
        assert harness.framework is not None
        assert harness.di_container is not None
        assert len(harness.connectors) == 2
        assert "test_system_1" in harness.connectors
        assert "test_system_2" in harness.connectors
        
        # Test teardown
        await harness.teardown()
        
        # Verify cleanup
        assert harness.test_end_time is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_harness_context_manager(self):
        """Test harness as async context manager."""
        async with create_simple_harness("context_manager_test", ["system_1"]) as harness:
            # Harness should be fully set up
            assert harness.framework is not None
            assert len(harness.connectors) == 1
            
            # Test basic functionality
            await harness.inject_telemetry("system_1", {
                "cpu_usage": MetricValue(name="cpu_usage", value=50.0, unit="percent", timestamp=datetime.now())
            })
            
            events = await harness.wait_for_events("telemetry", 1, timeout=5.0)
            assert len(events) == 1
        
        # Harness should be cleaned up after context exit
        assert harness.test_end_time is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_telemetry_injection_and_monitoring(self):
        """Test telemetry injection and event monitoring."""
        async with create_simple_harness("telemetry_test", ["web_server", "database"]) as harness:
            # Inject telemetry for web server
            web_metrics = {
                "cpu_usage": MetricValue(name="cpu_usage", value=75.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(name="memory_usage", value=1024.0, unit="MB", timestamp=datetime.now()),
                "active_connections": MetricValue(name="active_connections", value=150.0, unit="connections", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("web_server", web_metrics)
            
            # Inject telemetry for database
            db_metrics = {
                "cpu_usage": MetricValue(name="cpu_usage", value=60.0, unit="percent", timestamp=datetime.now()),
                "query_latency": MetricValue(name="query_latency", value=25.0, unit="ms", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("database", db_metrics)
            
            # Wait for events and validate
            events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            
            assert len(events) == 2
            
            # Verify event structure
            for event in events:
                assert event["type"] == "telemetry"
                assert "system_id" in event["data"]
                assert "metrics" in event["data"]
                assert "timestamp" in event["data"]
                assert "correlation_id" in event["data"]
            
            # Verify system IDs
            system_ids = {event["data"]["system_id"] for event in events}
            assert system_ids == {"web_server", "database"}
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_adaptation_triggering_and_execution(self):
        """Test adaptation triggering and execution through harness."""
        async with create_simple_harness("adaptation_test", ["app_server"]) as harness:
            # Create adaptation action
            scale_action = DataBuilder.adaptation_action(
                action_id="test_scale_001",
                action_type="horizontal_scale",
                target_system="app_server",
                parameters={"replicas": 3, "cpu_limit": "500m"}
            )
            
            # Trigger adaptation
            await harness.trigger_adaptation("app_server", [scale_action])
            
            # Wait for adaptation event
            adaptation_events = await harness.wait_for_events("adaptation", 1, timeout=5.0)
            
            assert len(adaptation_events) == 1
            event_data = adaptation_events[0]["data"]

            assert event_data["system_id"] == "app_server"
            assert len(event_data["suggested_actions"]) == 1
            assert event_data["suggested_actions"][0]["action_id"] == "test_scale_001"
            
            # Execute the action
            result = await harness.execute_action(scale_action)
            
            # Validate execution result
            assert result.action_id == "test_scale_001"
            assert result.status == ExecutionStatus.SUCCESS
            
            # Verify through harness assertions
            harness.assert_adaptation_executed("test_scale_001")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_failure_scenario_configuration(self):
        """Test configuration and handling of failure scenarios."""
        failure_modes = {
            "failing_system": ["connection", "metrics"],
            "healthy_system": []
        }
        
        async with create_failure_testing_harness("failure_test", ["failing_system", "healthy_system"], failure_modes) as harness:
            # Verify failure configuration
            failing_connector = harness.connectors["failing_system"]
            healthy_connector = harness.connectors["healthy_system"]
            
            assert failing_connector.should_fail_connection is True
            assert failing_connector.should_fail_metrics is True
            assert healthy_connector.should_fail_connection is False
            
            # Test that healthy system works
            await harness.inject_telemetry("healthy_system", {
                "cpu_usage": MetricValue(name="cpu_usage", value=30.0, unit="percent", timestamp=datetime.now())
            })
            
            events = await harness.wait_for_events("telemetry", 1, timeout=5.0)
            assert len(events) == 1
            assert events[0]["data"]["system_id"] == "healthy_system"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_harness_metrics_and_reporting(self):
        """Test harness metrics collection and reporting."""
        async with create_simple_harness("metrics_test", ["system_1", "system_2"]) as harness:
            # Perform various operations
            await harness.inject_telemetry("system_1", {
                "cpu_usage": MetricValue(name="cpu_usage", value=40.0, unit="percent", timestamp=datetime.now())
            })
            
            action = DataBuilder.adaptation_action(
                action_id="metrics_test_action",
                target_system="system_2"
            )
            
            await harness.trigger_adaptation("system_2", [action])
            await harness.execute_action(action)
            
            # Get test metrics
            metrics = harness.get_test_metrics()
            
            assert metrics["test_name"] == "metrics_test"
            assert metrics["systems_tested"] == 2
            assert metrics["events_published"] >= 2  # 1 telemetry + 1 adaptation
            assert metrics["adaptations_executed"] >= 1
            assert metrics["successful_adaptations"] >= 1
            assert metrics["failed_adaptations"] == 0
            
            # Generate report
            report = harness.generate_test_report()
            
            assert "POLARIS Integration Test Report" in report
            assert "metrics_test" in report
            assert "system_1" in report
            assert "system_2" in report


class TestContractValidation:
    """Test contract validation functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mock_connector_contract_compliance(self):
        """Test that MockManagedSystemConnector meets contract requirements."""
        # Create connector factory
        def create_mock_connector():
            return MockManagedSystemConnector("contract_test_system")
        
        # Validate contract compliance
        is_compliant = await validate_connector_contract(
            connector_factory=create_mock_connector,
            system_id="contract_test_system",
            is_real=False
        )
        
        assert is_compliant, "MockManagedSystemConnector should be fully contract compliant"
    
    def test_contract_test_class_creation(self):
        """Test creation of contract test classes."""
        contract_test = MockManagedSystemConnectorContractTest()
        
        # Test that contract test methods exist
        assert hasattr(contract_test, 'test_system_id_property')
        assert hasattr(contract_test, 'test_connection_lifecycle')
        assert hasattr(contract_test, 'test_collect_metrics_returns_valid_data')
        assert hasattr(contract_test, 'test_execute_action_returns_valid_result')
        
        # Test connector creation
        connector = contract_test.create_connector()
        assert isinstance(connector, MockManagedSystemConnector)
        assert contract_test.get_expected_system_id() == "test_system"
        assert not contract_test.is_real_connector()


class TestPerformanceHarness:
    """Test performance testing capabilities of the harness."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_performance_harness_throughput(self):
        """Test performance harness with high throughput scenario."""
        systems = [f"perf_system_{i}" for i in range(5)]
        
        async with create_performance_harness("throughput_test", systems) as harness:
            # Generate multiple telemetry events concurrently
            tasks = []
            
            for i in range(25):  # 25 events across 5 systems
                system_id = systems[i % len(systems)]
                metrics = {
                    "cpu_usage": MetricValue(name="cpu_usage", value=float(30 + i % 70), unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(name="memory_usage", value=float(500 + i * 20), unit="MB", timestamp=datetime.now())
                }
                
                tasks.append(harness.inject_telemetry(system_id, metrics))
            
            # Measure execution time
            start_time = datetime.now()
            await asyncio.gather(*tasks)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Should handle high throughput efficiently
            assert execution_time < 10.0, f"High throughput test took {execution_time:.2f}s"
            
            # Verify all events were processed
            events = await harness.wait_for_events("telemetry", 25, timeout=15.0)
            assert len(events) == 25
            
            # Verify events from all systems
            system_ids = {event["data"]["system_id"] for event in events}
            assert len(system_ids) == 5
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_adaptations_performance(self):
        """Test performance with concurrent adaptations."""
        systems = ["concurrent_1", "concurrent_2", "concurrent_3"]
        
        async with create_performance_harness("concurrent_adaptations", systems) as harness:
            # Create multiple adaptation actions
            actions = []
            for i, system_id in enumerate(systems):
                action = DataBuilder.adaptation_action(
                    action_id=f"concurrent_action_{i}",
                    action_type="scale_operation",
                    target_system=system_id,
                    parameters={"scale_factor": i + 1}
                )
                actions.append(action)
            
            # Execute adaptations concurrently
            start_time = datetime.now()
            results = await asyncio.gather(*[harness.execute_action(action) for action in actions])
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Should execute concurrently efficiently
            assert execution_time < 5.0, f"Concurrent adaptations took {execution_time:.2f}s"
            
            # Verify all adaptations succeeded
            for result in results:
                assert result.status == ExecutionStatus.SUCCESS
            
            # Verify through harness
            for action in actions:
                harness.assert_adaptation_executed(action.action_id)


class TestComplexIntegrationScenarios:
    """Test complex integration scenarios using the harness."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_tier_application_scenario(self):
        """Test a complete multi-tier application scenario."""
        systems = ["load_balancer", "web_tier", "app_tier", "database_tier", "cache_tier"]
        
        async with create_simple_harness("multi_tier_app", systems) as harness:
            # Phase 1: Inject load at the load balancer
            await harness.inject_telemetry("load_balancer", {
                "requests_per_second": MetricValue(name="requests_per_second", value=500.0, unit="req/s", timestamp=datetime.now()),
                "cpu_usage": MetricValue(name="cpu_usage", value=70.0, unit="percent", timestamp=datetime.now())
            })
            
            # Phase 2: Web tier responds to increased load
            await harness.inject_telemetry("web_tier", {
                "cpu_usage": MetricValue(name="cpu_usage", value=85.0, unit="percent", timestamp=datetime.now()),
                "response_time": MetricValue(name="response_time", value=200.0, unit="ms", timestamp=datetime.now())
            })
            
            # Phase 3: Trigger web tier scaling
            web_scale_action = DataBuilder.adaptation_action(
                action_id="web_tier_scale",
                action_type="horizontal_scale",
                target_system="web_tier",
                parameters={"target_instances": 4}
            )
            
            await harness.trigger_adaptation("web_tier", [web_scale_action])
            await harness.execute_action(web_scale_action)
            
            # Phase 4: App tier experiences increased load
            await harness.inject_telemetry("app_tier", {
                "cpu_usage": MetricValue(name="cpu_usage", value=80.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(name="memory_usage", value=2048.0, unit="MB", timestamp=datetime.now())
            })
            
            # Phase 5: Database tier shows impact
            await harness.inject_telemetry("database_tier", {
                "active_connections": MetricValue(name="active_connections", value=200.0, unit="connections", timestamp=datetime.now()),
                "query_latency": MetricValue(name="query_latency", value=150.0, unit="ms", timestamp=datetime.now())
            })
            
            # Phase 6: Enable caching to reduce database load
            cache_action = DataBuilder.adaptation_action(
                action_id="enable_caching",
                action_type="enable_cache",
                target_system="cache_tier",
                parameters={"cache_size": "1GB", "ttl": 300}
            )
            
            await harness.trigger_adaptation("cache_tier", [cache_action])
            await harness.execute_action(cache_action)
            
            # Verify the complete scenario
            telemetry_events = await harness.wait_for_events("telemetry", 4, timeout=15.0)
            adaptation_events = await harness.wait_for_events("adaptation", 2, timeout=10.0)
            
            assert len(telemetry_events) >= 4
            assert len(adaptation_events) >= 2
            
            harness.assert_adaptation_executed("web_tier_scale")
            harness.assert_adaptation_executed("enable_caching")
            harness.assert_no_errors_logged()
            
            # Generate comprehensive report
            report = harness.generate_test_report()
            assert "multi_tier_app" in report
            
            # Verify all systems were involved
            for system in systems:
                assert system in report
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_event_driven_workflow_validation(self):
        """Test validation of event-driven workflows."""
        async with create_simple_harness("event_workflow", ["event_producer", "event_consumer"]) as harness:
            # Add custom event validator
            received_high_cpu_event = {"value": False}
            
            def high_cpu_validator(event_data):
                if "metrics" in event_data:
                    cpu_metric = event_data["metrics"].get("cpu_usage")
                    if cpu_metric and float(cpu_metric.get("value", 0)) > 80:
                        received_high_cpu_event["value"] = True
                return True
            
            harness.add_event_validator(high_cpu_validator)
            
            # Inject high CPU telemetry
            await harness.inject_telemetry("event_producer", {
                "cpu_usage": MetricValue(name="cpu_usage", value=90.0, unit="percent", timestamp=datetime.now())
            })
            
            # Wait for event processing
            await harness.wait_for_events("telemetry", 1, timeout=5.0)
            
            # Verify validator was triggered
            assert received_high_cpu_event["value"], "High CPU event validator should have been triggered"
            
            # Test adaptation based on the event
            cpu_action = DataBuilder.adaptation_action(
                action_id="cpu_response",
                action_type="cpu_optimization",
                target_system="event_consumer",
                parameters={"optimization_level": "aggressive"}
            )
            
            await harness.trigger_adaptation("event_consumer", [cpu_action])
            await harness.execute_action(cpu_action)
            
            harness.assert_adaptation_executed("cpu_response")