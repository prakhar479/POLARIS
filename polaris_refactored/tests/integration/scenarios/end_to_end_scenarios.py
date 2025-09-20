"""
End-to-end testing scenarios for POLARIS framework.

This module provides comprehensive end-to-end test scenarios that validate
complete workflows from telemetry ingestion to adaptation execution.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from tests.integration.harness.polaris_integration_test_harness import (
    PolarisIntegrationTestHarness, IntegrationTestConfig,
    create_simple_harness, create_performance_harness, create_failure_testing_harness
)
from tests.fixtures.mock_objects import TestDataBuilder
from src.domain.models import MetricValue, HealthStatus, ExecutionStatus


class TestBasicWorkflows:
    """Test basic POLARIS workflows end-to-end."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_telemetry_collection_and_processing(self):
        """Test complete telemetry collection and processing workflow."""
        async with create_simple_harness("telemetry_workflow", ["web_server", "database"]) as harness:
            # Inject telemetry data for web server
            web_metrics = {
                "cpu_usage": MetricValue(value=85.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(value=2048.0, unit="MB", timestamp=datetime.now()),
                "response_time": MetricValue(value=250.0, unit="ms", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("web_server", web_metrics)
            
            # Inject telemetry data for database
            db_metrics = {
                "cpu_usage": MetricValue(value=45.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(value=4096.0, unit="MB", timestamp=datetime.now()),
                "query_time": MetricValue(value=50.0, unit="ms", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("database", db_metrics)
            
            # Wait for telemetry events to be processed
            telemetry_events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            
            # Validate events
            assert len(telemetry_events) == 2
            
            system_ids = {event["data"]["system_id"] for event in telemetry_events}
            assert system_ids == {"web_server", "database"}
            
            # Verify no errors occurred
            harness.assert_no_errors_logged()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_adaptation_trigger_and_execution(self):
        """Test adaptation triggering and execution workflow."""
        async with create_simple_harness("adaptation_workflow", ["web_server"]) as harness:
            # Create adaptation actions
            scale_up_action = TestDataBuilder.adaptation_action(
                action_id="scale_up_001",
                action_type="scale_up",
                target_system="web_server",
                parameters={"replicas": 5, "cpu_limit": "1000m"}
            )
            
            # Trigger adaptation
            await harness.trigger_adaptation("web_server", [scale_up_action])
            
            # Wait for adaptation event
            adaptation_events = await harness.wait_for_events("adaptation", 1, timeout=10.0)
            
            # Validate adaptation event
            assert len(adaptation_events) == 1
            event_data = adaptation_events[0]["data"]
            assert event_data["system_id"] == "web_server"
            assert event_data["trigger_reason"] == "test_triggered"
            assert len(event_data["actions"]) == 1
            
            # Execute the action
            result = await harness.execute_action(scale_up_action)
            
            # Validate execution result
            assert result.action_id == "scale_up_001"
            assert result.status == ExecutionStatus.SUCCESS
            assert "message" in result.result_data
            
            # Verify adaptation was recorded
            harness.assert_adaptation_executed("scale_up_001")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_mape_k_loop(self):
        """Test complete MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) loop."""
        async with create_simple_harness("mape_k_loop", ["application_server"]) as harness:
            # Monitor: Inject high CPU usage telemetry
            high_cpu_metrics = {
                "cpu_usage": MetricValue(value=95.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(value=1024.0, unit="MB", timestamp=datetime.now()),
                "request_rate": MetricValue(value=1000.0, unit="req/s", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("application_server", high_cpu_metrics)
            
            # Wait for telemetry processing
            await harness.wait_for_events("telemetry", 1, timeout=5.0)
            
            # Analyze & Plan: Trigger adaptation based on high CPU
            cpu_scaling_action = TestDataBuilder.adaptation_action(
                action_id="cpu_scale_001",
                action_type="horizontal_scale",
                target_system="application_server",
                parameters={"scale_factor": 2, "reason": "high_cpu_usage"}
            )
            
            await harness.trigger_adaptation("application_server", [cpu_scaling_action])
            
            # Execute: Execute the scaling action
            execution_result = await harness.execute_action(cpu_scaling_action)
            
            # Validate execution
            assert execution_result.status == ExecutionStatus.SUCCESS
            
            # Knowledge: Verify system learned from the adaptation
            # (In a real system, this would update the knowledge base)
            
            # Monitor again: Inject improved metrics after scaling
            improved_metrics = {
                "cpu_usage": MetricValue(value=60.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(value=1024.0, unit="MB", timestamp=datetime.now()),
                "request_rate": MetricValue(value=1000.0, unit="req/s", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("application_server", improved_metrics)
            
            # Verify the complete loop executed successfully
            telemetry_events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            adaptation_events = await harness.wait_for_events("adaptation", 1, timeout=5.0)
            
            assert len(telemetry_events) == 2
            assert len(adaptation_events) == 1
            
            harness.assert_adaptation_executed("cpu_scale_001")
            harness.assert_no_errors_logged()


class TestMultiSystemScenarios:
    """Test scenarios involving multiple interconnected systems."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascading_adaptations(self):
        """Test cascading adaptations across multiple systems."""
        systems = ["load_balancer", "web_server", "database", "cache"]
        
        async with create_simple_harness("cascading_adaptations", systems) as harness:
            # Simulate load increase at load balancer
            load_balancer_metrics = {
                "incoming_requests": MetricValue(value=5000.0, unit="req/s", timestamp=datetime.now()),
                "cpu_usage": MetricValue(value=80.0, unit="percent", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("load_balancer", load_balancer_metrics)
            
            # This should trigger scaling of web servers
            web_scale_action = TestDataBuilder.adaptation_action(
                action_id="web_scale_001",
                action_type="scale_out",
                target_system="web_server",
                parameters={"instances": 3}
            )
            
            await harness.trigger_adaptation("web_server", [web_scale_action])
            await harness.execute_action(web_scale_action)
            
            # Increased web server load should affect database
            db_metrics = {
                "connection_count": MetricValue(value=150.0, unit="connections", timestamp=datetime.now()),
                "query_latency": MetricValue(value=200.0, unit="ms", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("database", db_metrics)
            
            # This should trigger database optimization
            db_optimize_action = TestDataBuilder.adaptation_action(
                action_id="db_optimize_001",
                action_type="optimize_queries",
                target_system="database",
                parameters={"enable_caching": True, "connection_pool_size": 200}
            )
            
            await harness.trigger_adaptation("database", [db_optimize_action])
            await harness.execute_action(db_optimize_action)
            
            # Verify all systems were adapted
            harness.assert_adaptation_executed("web_scale_001")
            harness.assert_adaptation_executed("db_optimize_001")
            
            # Verify telemetry was collected from all systems
            telemetry_events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            assert len(telemetry_events) >= 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_system_dependency_management(self):
        """Test management of system dependencies during adaptations."""
        systems = ["frontend", "api_gateway", "microservice_a", "microservice_b", "shared_database"]
        
        async with create_simple_harness("dependency_management", systems) as harness:
            # Simulate microservice_a needing to scale
            service_a_metrics = {
                "cpu_usage": MetricValue(value=90.0, unit="percent", timestamp=datetime.now()),
                "memory_usage": MetricValue(value=3072.0, unit="MB", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("microservice_a", service_a_metrics)
            
            # Scale microservice_a
            scale_action = TestDataBuilder.adaptation_action(
                action_id="service_a_scale",
                action_type="horizontal_scale",
                target_system="microservice_a",
                parameters={"replicas": 4}
            )
            
            await harness.trigger_adaptation("microservice_a", [scale_action])
            await harness.execute_action(scale_action)
            
            # This should affect the shared database
            db_metrics = {
                "active_connections": MetricValue(value=180.0, unit="connections", timestamp=datetime.now()),
                "cpu_usage": MetricValue(value=75.0, unit="percent", timestamp=datetime.now())
            }
            
            await harness.inject_telemetry("shared_database", db_metrics)
            
            # Verify dependency impact was handled
            harness.assert_adaptation_executed("service_a_scale")
            
            # Check that all dependent systems received telemetry updates
            telemetry_events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            system_ids = {event["data"]["system_id"] for event in telemetry_events}
            assert "microservice_a" in system_ids
            assert "shared_database" in system_ids


class TestFailureScenarios:
    """Test failure handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_connector_failure_recovery(self):
        """Test recovery from connector failures."""
        failure_modes = {
            "failing_system": ["connection"],
            "healthy_system": []
        }
        
        async with create_failure_testing_harness("connector_failure", ["failing_system", "healthy_system"], failure_modes) as harness:
            # Try to collect metrics from failing system
            try:
                await harness.inject_telemetry("failing_system", {
                    "cpu_usage": MetricValue(value=50.0, unit="percent", timestamp=datetime.now())
                })
            except Exception:
                pass  # Expected to fail
            
            # Healthy system should still work
            await harness.inject_telemetry("healthy_system", {
                "cpu_usage": MetricValue(value=30.0, unit="percent", timestamp=datetime.now())
            })
            
            # Verify healthy system events were processed
            telemetry_events = await harness.wait_for_events("telemetry", 1, timeout=5.0)
            assert len(telemetry_events) >= 1
            
            # Verify the healthy system event
            healthy_events = [e for e in telemetry_events if e["data"]["system_id"] == "healthy_system"]
            assert len(healthy_events) >= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_adaptation_failure_handling(self):
        """Test handling of adaptation execution failures."""
        failure_modes = {
            "execution_failing_system": ["execution"]
        }
        
        async with create_failure_testing_harness("adaptation_failure", ["execution_failing_system"], failure_modes) as harness:
            # Try to execute action on failing system
            failing_action = TestDataBuilder.adaptation_action(
                action_id="failing_action_001",
                action_type="scale_up",
                target_system="execution_failing_system",
                parameters={"replicas": 3}
            )
            
            result = await harness.execute_action(failing_action)
            
            # Should return failure result
            assert result.status == ExecutionStatus.FAILED
            assert "error" in result.result_data
            
            # System should continue operating despite failure
            # (In real system, this might trigger alternative adaptations)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_infrastructure_failure_resilience(self):
        """Test resilience to infrastructure component failures."""
        async with create_simple_harness("infrastructure_failure", ["test_system"]) as harness:
            # Simulate data store failure
            data_store = harness.di_container.get("data_store")
            if hasattr(data_store, 'should_fail_operations'):
                data_store.should_fail_operations = True
            
            # System should continue operating despite data store issues
            await harness.inject_telemetry("test_system", {
                "cpu_usage": MetricValue(value=60.0, unit="percent", timestamp=datetime.now())
            })
            
            # Events should still be processed (may not be persisted)
            telemetry_events = await harness.wait_for_events("telemetry", 1, timeout=5.0)
            assert len(telemetry_events) >= 1


class TestPerformanceScenarios:
    """Test performance characteristics under various conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_high_throughput_telemetry(self):
        """Test system performance under high telemetry throughput."""
        systems = [f"system_{i}" for i in range(10)]
        
        async with create_performance_harness("high_throughput", systems) as harness:
            # Generate high volume of telemetry data
            tasks = []
            
            for i in range(50):  # 50 telemetry injections
                system_id = systems[i % len(systems)]
                metrics = {
                    "cpu_usage": MetricValue(value=float(50 + i % 50), unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(value=float(1000 + i * 10), unit="MB", timestamp=datetime.now())
                }
                
                tasks.append(harness.inject_telemetry(system_id, metrics))
            
            # Execute all telemetry injections concurrently
            start_time = datetime.now()
            await asyncio.gather(*tasks)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Should process all telemetry within reasonable time
            assert processing_time < 30.0, f"High throughput processing took {processing_time:.2f}s"
            
            # Verify all events were processed
            telemetry_events = await harness.wait_for_events("telemetry", 50, timeout=30.0)
            assert len(telemetry_events) == 50
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_adaptations(self):
        """Test performance with concurrent adaptations."""
        systems = ["system_1", "system_2", "system_3"]
        
        async with create_performance_harness("concurrent_adaptations", systems) as harness:
            # Create multiple adaptation actions
            actions = []
            for i, system_id in enumerate(systems):
                action = TestDataBuilder.adaptation_action(
                    action_id=f"concurrent_action_{i}",
                    action_type="scale_up",
                    target_system=system_id,
                    parameters={"replicas": 2 + i}
                )
                actions.append(action)
            
            # Execute all adaptations concurrently
            start_time = datetime.now()
            results = await asyncio.gather(*[harness.execute_action(action) for action in actions])
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Should execute all adaptations within reasonable time
            assert execution_time < 15.0, f"Concurrent adaptations took {execution_time:.2f}s"
            
            # Verify all adaptations succeeded
            for result in results:
                assert result.status == ExecutionStatus.SUCCESS
            
            # Verify all adaptations were recorded
            for action in actions:
                harness.assert_adaptation_executed(action.action_id)


class TestComplexWorkflows:
    """Test complex, realistic workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_auto_scaling_workflow(self):
        """Test complete auto-scaling workflow based on load patterns."""
        systems = ["load_balancer", "web_tier", "app_tier", "database"]
        
        async with create_simple_harness("auto_scaling", systems) as harness:
            # Phase 1: Normal load
            normal_metrics = {
                "load_balancer": {
                    "requests_per_second": MetricValue(value=100.0, unit="req/s", timestamp=datetime.now()),
                    "cpu_usage": MetricValue(value=30.0, unit="percent", timestamp=datetime.now())
                },
                "web_tier": {
                    "cpu_usage": MetricValue(value=40.0, unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(value=512.0, unit="MB", timestamp=datetime.now())
                }
            }
            
            for system_id, metrics in normal_metrics.items():
                await harness.inject_telemetry(system_id, metrics)
            
            # Phase 2: Load spike
            spike_metrics = {
                "load_balancer": {
                    "requests_per_second": MetricValue(value=1000.0, unit="req/s", timestamp=datetime.now()),
                    "cpu_usage": MetricValue(value=85.0, unit="percent", timestamp=datetime.now())
                },
                "web_tier": {
                    "cpu_usage": MetricValue(value=95.0, unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(value=1536.0, unit="MB", timestamp=datetime.now())
                }
            }
            
            for system_id, metrics in spike_metrics.items():
                await harness.inject_telemetry(system_id, metrics)
            
            # Trigger scaling actions
            web_scale_action = TestDataBuilder.adaptation_action(
                action_id="web_scale_spike",
                action_type="horizontal_scale",
                target_system="web_tier",
                parameters={"target_instances": 5, "trigger": "load_spike"}
            )
            
            await harness.trigger_adaptation("web_tier", [web_scale_action])
            await harness.execute_action(web_scale_action)
            
            # Phase 3: Load normalization
            normalized_metrics = {
                "web_tier": {
                    "cpu_usage": MetricValue(value=50.0, unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(value=768.0, unit="MB", timestamp=datetime.now())
                }
            }
            
            await harness.inject_telemetry("web_tier", normalized_metrics["web_tier"])
            
            # Verify the complete workflow
            telemetry_events = await harness.wait_for_events("telemetry", 5, timeout=15.0)
            adaptation_events = await harness.wait_for_events("adaptation", 1, timeout=10.0)
            
            assert len(telemetry_events) >= 5
            assert len(adaptation_events) >= 1
            
            harness.assert_adaptation_executed("web_scale_spike")
            harness.assert_no_errors_logged()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_disaster_recovery_workflow(self):
        """Test disaster recovery and failover workflow."""
        systems = ["primary_db", "secondary_db", "application", "monitoring"]
        
        async with create_simple_harness("disaster_recovery", systems) as harness:
            # Normal operation
            await harness.inject_telemetry("primary_db", {
                "availability": MetricValue(value=100.0, unit="percent", timestamp=datetime.now()),
                "response_time": MetricValue(value=10.0, unit="ms", timestamp=datetime.now())
            })
            
            # Simulate primary database failure
            await harness.inject_telemetry("primary_db", {
                "availability": MetricValue(value=0.0, unit="percent", timestamp=datetime.now()),
                "response_time": MetricValue(value=5000.0, unit="ms", timestamp=datetime.now())
            })
            
            # Trigger failover to secondary
            failover_action = TestDataBuilder.adaptation_action(
                action_id="db_failover_001",
                action_type="failover",
                target_system="secondary_db",
                parameters={"promote_to_primary": True, "update_dns": True}
            )
            
            await harness.trigger_adaptation("secondary_db", [failover_action])
            await harness.execute_action(failover_action)
            
            # Update application configuration
            app_config_action = TestDataBuilder.adaptation_action(
                action_id="app_config_001",
                action_type="update_config",
                target_system="application",
                parameters={"database_endpoint": "secondary_db", "connection_pool_size": 50}
            )
            
            await harness.trigger_adaptation("application", [app_config_action])
            await harness.execute_action(app_config_action)
            
            # Verify recovery workflow
            harness.assert_adaptation_executed("db_failover_001")
            harness.assert_adaptation_executed("app_config_001")
            
            # Verify monitoring detected the issue
            telemetry_events = await harness.wait_for_events("telemetry", 2, timeout=10.0)
            assert len(telemetry_events) >= 2