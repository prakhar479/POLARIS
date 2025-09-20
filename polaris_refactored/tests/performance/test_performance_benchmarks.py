"""
Performance benchmarks and tests for POLARIS framework.

This module contains comprehensive performance tests that validate the framework's
performance characteristics under various load conditions.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from tests.performance.polaris_performance_test_suite import (
    PolarisPerformanceTestSuite, LoadTestConfig, PerformanceThresholds,
    run_basic_performance_suite, run_comprehensive_performance_suite
)
from tests.integration.harness.polaris_integration_test_harness import create_performance_harness
from tests.fixtures.mock_objects import DataBuilder
from src.domain.models import MetricValue


class TestTelemetryPerformance:
    """Performance tests for telemetry processing."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_telemetry_throughput_benchmark(self):
        """Benchmark telemetry processing throughput."""
        systems = ["telemetry_system_1", "telemetry_system_2", "telemetry_system_3"]
        suite = PolarisPerformanceTestSuite("telemetry_benchmarks")
        
        # Define performance requirements
        thresholds = PerformanceThresholds(
            min_throughput=50.0,  # Minimum 50 telemetry events per second
            max_avg_latency=0.1,  # Maximum 100ms average latency
            max_p95_latency=0.2,  # Maximum 200ms P95 latency
            max_error_rate=1.0,   # Maximum 1% error rate
            max_cpu_usage=80.0    # Maximum 80% CPU usage
        )
        
        config = LoadTestConfig(
            test_name="telemetry_throughput",
            duration=60.0,
            concurrent_users=20,
            ramp_up_time=10.0,
            target_throughput=100.0,  # Target 100 ops/sec
            warmup_duration=5.0
        )
        
        metrics, passed = await suite.run_throughput_test(systems, config, thresholds)
        
        # Validate performance requirements
        assert passed, f"Telemetry throughput test failed: {metrics.error_rate}% error rate"
        assert metrics.throughput >= thresholds.min_throughput, f"Throughput {metrics.throughput:.2f} below threshold {thresholds.min_throughput}"
        assert metrics.avg_latency <= thresholds.max_avg_latency, f"Average latency {metrics.avg_latency:.3f}s above threshold {thresholds.max_avg_latency}s"
        
        print(f"✅ Telemetry Throughput: {metrics.throughput:.2f} ops/sec")
        print(f"✅ Average Latency: {metrics.avg_latency*1000:.2f} ms")
        print(f"✅ P95 Latency: {metrics.p95_latency*1000:.2f} ms")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_high_frequency_telemetry(self):
        """Test performance with high-frequency telemetry data."""
        systems = [f"hf_system_{i}" for i in range(10)]
        suite = PolarisPerformanceTestSuite("high_frequency_telemetry")
        
        config = LoadTestConfig(
            test_name="high_frequency_telemetry",
            duration=30.0,
            concurrent_users=50,  # High concurrency
            ramp_up_time=5.0,
            target_throughput=500.0,  # Very high target throughput
            max_errors=50
        )
        
        thresholds = PerformanceThresholds(
            min_throughput=200.0,  # Lower threshold for high-frequency test
            max_avg_latency=0.5,   # Allow higher latency under extreme load
            max_error_rate=10.0,   # Allow higher error rate under stress
            max_cpu_usage=95.0     # Allow higher CPU usage
        )
        
        metrics, passed = await suite.run_throughput_test(systems, config, thresholds)
        
        # Even if it doesn't pass all thresholds, it should handle the load gracefully
        assert metrics.error_rate < 50.0, f"Error rate {metrics.error_rate}% too high for high-frequency test"
        assert metrics.throughput > 50.0, f"Throughput {metrics.throughput:.2f} too low for high-frequency test"
        
        print(f"📊 High-Frequency Throughput: {metrics.throughput:.2f} ops/sec")
        print(f"📊 Error Rate: {metrics.error_rate:.2f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_telemetry_burst_handling(self):
        """Test handling of telemetry bursts."""
        systems = ["burst_system_1", "burst_system_2"]
        
        async with create_performance_harness("telemetry_burst", systems) as harness:
            # Simulate burst of telemetry data
            burst_tasks = []
            
            # Create a burst of 100 telemetry events in quick succession
            for i in range(100):
                system_id = systems[i % len(systems)]
                metrics = {
                    "cpu_usage": MetricValue(name="cpu_usage", value=float(50 + i % 50), unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(name="memory_usage", value=float(1000 + i * 10), unit="MB", timestamp=datetime.now()),
                    "burst_id": MetricValue(name="burst_id", value=float(i), unit="count", timestamp=datetime.now())
                }
                
                burst_tasks.append(harness.inject_telemetry(system_id, metrics))
            
            # Execute burst
            start_time = datetime.now()
            await asyncio.gather(*burst_tasks)
            end_time = datetime.now()
            
            burst_duration = (end_time - start_time).total_seconds()
            
            # Wait for all events to be processed
            events = await harness.wait_for_events("telemetry", 100, timeout=30.0)
            
            # Validate burst handling
            assert len(events) == 100, f"Expected 100 events, got {len(events)}"
            assert burst_duration < 10.0, f"Burst processing took {burst_duration:.2f}s, expected < 10s"
            
            print(f"💥 Burst Duration: {burst_duration:.2f}s")
            print(f"💥 Burst Throughput: {100/burst_duration:.2f} ops/sec")


class TestAdaptationPerformance:
    """Performance tests for adaptation execution."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_adaptation_execution_latency(self):
        """Test adaptation execution latency."""
        systems = ["adaptation_system_1", "adaptation_system_2"]
        suite = PolarisPerformanceTestSuite("adaptation_latency")
        
        thresholds = PerformanceThresholds(
            max_avg_latency=0.5,   # Maximum 500ms for adaptation execution
            max_p95_latency=1.0,   # Maximum 1s P95 latency
            min_throughput=5.0,    # Minimum 5 adaptations per second
            max_error_rate=2.0     # Maximum 2% error rate
        )
        
        config = LoadTestConfig(
            test_name="adaptation_latency",
            duration=30.0,
            concurrent_users=10,
            ramp_up_time=5.0,
            warmup_duration=3.0
        )
        
        metrics, passed = await suite.run_latency_test(systems, config, thresholds)
        
        assert passed, f"Adaptation latency test failed"
        assert metrics.avg_latency <= thresholds.max_avg_latency, f"Average latency {metrics.avg_latency:.3f}s above threshold"
        
        print(f"⚡ Adaptation Latency: {metrics.avg_latency*1000:.2f} ms")
        print(f"⚡ P95 Latency: {metrics.p95_latency*1000:.2f} ms")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_adaptations(self):
        """Test performance with concurrent adaptations."""
        systems = [f"concurrent_system_{i}" for i in range(5)]
        
        async with create_performance_harness("concurrent_adaptations", systems) as harness:
            # Create multiple adaptation actions
            actions = []
            for i in range(20):
                action = DataBuilder.adaptation_action(
                    action_id=f"concurrent_action_{i}",
                    action_type="performance_test",
                    target_system=systems[i % len(systems)],
                    parameters={"test_id": i, "concurrency_test": True}
                )
                actions.append(action)
            
            # Execute all adaptations concurrently
            start_time = datetime.now()
            results = await asyncio.gather(*[harness.execute_action(action) for action in actions])
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Validate concurrent execution
            successful_results = [r for r in results if r.status.name == "SUCCESS"]
            success_rate = len(successful_results) / len(results) * 100
            
            assert success_rate >= 90.0, f"Success rate {success_rate:.1f}% below 90%"
            assert execution_time < 15.0, f"Concurrent execution took {execution_time:.2f}s, expected < 15s"
            
            print(f"🔄 Concurrent Adaptations: {len(actions)} actions in {execution_time:.2f}s")
            print(f"🔄 Success Rate: {success_rate:.1f}%")
            print(f"🔄 Throughput: {len(actions)/execution_time:.2f} adaptations/sec")


class TestScalabilityBenchmarks:
    """Scalability benchmarks for POLARIS framework."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_system_count_scalability(self):
        """Test scalability with increasing number of systems."""
        base_systems = ["scalability_base_1", "scalability_base_2"]
        suite = PolarisPerformanceTestSuite("scalability_benchmarks")
        
        base_config = LoadTestConfig(
            test_name="system_scalability",
            duration=30.0,
            concurrent_users=5,
            ramp_up_time=5.0
        )
        
        thresholds = PerformanceThresholds(
            min_throughput=5.0,
            max_avg_latency=2.0,
            max_error_rate=10.0
        )
        
        # Test with 1x, 2x, and 4x system counts
        scale_factors = [1, 2, 4]
        results = await suite.run_scalability_test(base_systems, scale_factors, base_config, thresholds)
        
        # Analyze scalability characteristics
        throughputs = [metrics.throughput for _, metrics, _ in results]
        latencies = [metrics.avg_latency for _, metrics, _ in results]
        
        # Throughput should scale reasonably with system count
        throughput_scaling = throughputs[-1] / throughputs[0]  # 4x vs 1x
        assert throughput_scaling >= 2.0, f"Throughput scaling {throughput_scaling:.2f}x insufficient"
        
        # Latency should not degrade significantly
        latency_degradation = latencies[-1] / latencies[0]
        assert latency_degradation <= 3.0, f"Latency degradation {latency_degradation:.2f}x too high"
        
        print(f"📈 Scalability Results:")
        for scale_factor, metrics, passed in results:
            print(f"  {scale_factor}x: {metrics.throughput:.2f} ops/sec, {metrics.avg_latency*1000:.2f} ms")
        
        print(f"📈 Throughput Scaling: {throughput_scaling:.2f}x")
        print(f"📈 Latency Degradation: {latency_degradation:.2f}x")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_load_scalability(self):
        """Test scalability with increasing load."""
        systems = ["load_system_1", "load_system_2", "load_system_3"]
        suite = PolarisPerformanceTestSuite("load_scalability")
        
        # Test with different load levels
        load_configs = [
            LoadTestConfig(test_name="light_load", duration=20.0, concurrent_users=5, ramp_up_time=3.0),
            LoadTestConfig(test_name="medium_load", duration=20.0, concurrent_users=15, ramp_up_time=3.0),
            LoadTestConfig(test_name="heavy_load", duration=20.0, concurrent_users=30, ramp_up_time=5.0)
        ]
        
        results = []
        for config in load_configs:
            metrics, passed = await suite.run_throughput_test(systems, config)
            results.append((config.concurrent_users, metrics, passed))
        
        # Analyze load handling
        for users, metrics, passed in results:
            print(f"👥 {users} users: {metrics.throughput:.2f} ops/sec, {metrics.error_rate:.2f}% errors")
        
        # System should handle increasing load gracefully
        error_rates = [metrics.error_rate for _, metrics, _ in results]
        assert all(rate < 20.0 for rate in error_rates), f"Error rates too high: {error_rates}"


class TestStressBenchmarks:
    """Stress testing benchmarks."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load_stress(self):
        """Test performance under sustained load."""
        systems = ["stress_system_1", "stress_system_2"]
        suite = PolarisPerformanceTestSuite("stress_benchmarks")
        
        config = LoadTestConfig(
            test_name="sustained_stress",
            duration=120.0,  # 2 minutes of sustained load
            concurrent_users=25,
            ramp_up_time=15.0,
            ramp_down_time=10.0,
            max_errors=200
        )
        
        thresholds = PerformanceThresholds(
            min_throughput=10.0,
            max_avg_latency=3.0,
            max_error_rate=15.0,  # More lenient for stress test
            max_cpu_usage=95.0
        )
        
        metrics, passed = await suite.run_stress_test(systems, config, thresholds)
        
        # System should survive stress test
        assert metrics.error_rate < 25.0, f"Error rate {metrics.error_rate:.2f}% too high for stress test"
        assert metrics.throughput > 5.0, f"Throughput {metrics.throughput:.2f} too low for stress test"
        
        print(f"💪 Stress Test Results:")
        print(f"  Duration: {metrics.duration:.1f}s")
        print(f"  Throughput: {metrics.throughput:.2f} ops/sec")
        print(f"  Error Rate: {metrics.error_rate:.2f}%")
        print(f"  CPU Usage: {metrics.cpu_usage.get('cpu_avg', 0):.1f}%")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_memory_stress(self):
        """Test memory usage under stress."""
        systems = [f"memory_system_{i}" for i in range(8)]
        
        async with create_performance_harness("memory_stress", systems) as harness:
            # Generate large amounts of telemetry data to stress memory
            large_data_tasks = []
            
            for i in range(200):  # Large number of telemetry events
                system_id = systems[i % len(systems)]
                
                # Create metrics with larger data payloads
                metrics = {
                    "cpu_usage": MetricValue(name="cpu_usage", value=float(30 + i % 70), unit="percent", timestamp=datetime.now()),
                    "memory_usage": MetricValue(name="memory_usage", value=float(500 + i * 5), unit="MB", timestamp=datetime.now()),
                    "large_data": MetricValue(name="large_data", value=float(i), unit="bytes", timestamp=datetime.now()),
                    # Simulate larger metric payloads
                    f"metric_{i % 10}": MetricValue(name="metric", value=float(i * 1.5), unit="custom", timestamp=datetime.now())
                }
                
                large_data_tasks.append(harness.inject_telemetry(system_id, metrics))
            
            # Execute memory stress test
            start_time = datetime.now()
            await asyncio.gather(*large_data_tasks)
            end_time = datetime.now()
            
            # Wait for processing
            events = await harness.wait_for_events("telemetry", 200, timeout=60.0)
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Validate memory stress handling
            assert len(events) == 200, f"Expected 200 events, got {len(events)}"
            assert processing_time < 45.0, f"Memory stress test took {processing_time:.2f}s"
            
            print(f"🧠 Memory Stress: 200 events in {processing_time:.2f}s")
            print(f"🧠 Throughput: {200/processing_time:.2f} ops/sec")


class TestRegressionBenchmarks:
    """Performance regression testing."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_regression_detection(self):
        """Test performance regression detection."""
        systems = ["regression_system"]
        suite = PolarisPerformanceTestSuite("regression_benchmarks")
        
        # Run baseline test
        baseline_config = LoadTestConfig(
            test_name="regression_baseline",
            duration=20.0,
            concurrent_users=10,
            ramp_up_time=3.0
        )
        
        baseline_metrics, _ = await suite.run_throughput_test(systems, baseline_config)
        
        # Store as baseline
        suite.baseline_results["regression_test"] = baseline_metrics
        
        # Run current test (should be similar to baseline)
        current_config = LoadTestConfig(
            test_name="regression_current",
            duration=20.0,
            concurrent_users=10,
            ramp_up_time=3.0
        )
        
        current_metrics, _ = await suite.run_throughput_test(systems, current_config)
        
        # Check for regression
        passed, regressions = suite.run_regression_test("regression_test", current_metrics, regression_threshold=0.2)
        
        # Should not detect significant regression in identical test
        assert passed or len(regressions) <= 1, f"False regression detected: {regressions}"
        
        print(f"🔍 Regression Test:")
        print(f"  Baseline Throughput: {baseline_metrics.throughput:.2f} ops/sec")
        print(f"  Current Throughput: {current_metrics.throughput:.2f} ops/sec")
        print(f"  Regression Detected: {not passed}")
        if regressions:
            print(f"  Regressions: {regressions}")


class TestComprehensivePerformanceSuite:
    """Comprehensive performance test suite."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_full_performance_suite(self):
        """Run the complete performance test suite."""
        systems = ["perf_system_1", "perf_system_2", "perf_system_3"]
        
        # Run comprehensive suite
        results = await run_comprehensive_performance_suite(systems)
        
        # Validate overall results
        assert results["throughput"]["passed"], "Throughput test failed"
        assert results["stress"]["passed"], "Stress test failed"
        
        # Check scalability results
        scalability_results = results["scalability"]
        assert len(scalability_results) == 3, "Expected 3 scalability test results"
        
        # At least 2 out of 3 scalability tests should pass
        scalability_passes = sum(1 for _, _, passed in scalability_results if passed)
        assert scalability_passes >= 2, f"Only {scalability_passes}/3 scalability tests passed"
        
        # Print comprehensive report
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE SUITE RESULTS")
        print("="*60)
        print(results["report"])
        print("="*60)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_basic_performance_suite(self):
        """Run basic performance test suite for quick validation."""
        systems = ["basic_system_1", "basic_system_2"]
        
        results = await run_basic_performance_suite(systems)
        
        # Validate basic performance requirements
        throughput_metrics = results["throughput"]
        latency_metrics = results["latency"]
        
        assert throughput_metrics.throughput > 5.0, f"Basic throughput {throughput_metrics.throughput:.2f} too low"
        assert latency_metrics.avg_latency < 2.0, f"Basic latency {latency_metrics.avg_latency:.3f}s too high"
        assert throughput_metrics.error_rate < 10.0, f"Basic error rate {throughput_metrics.error_rate:.2f}% too high"
        
        print(f"✅ Basic Performance Suite Passed")
        print(f"  Throughput: {throughput_metrics.throughput:.2f} ops/sec")
        print(f"  Latency: {latency_metrics.avg_latency*1000:.2f} ms")
        print(f"  Error Rate: {throughput_metrics.error_rate:.2f}%")