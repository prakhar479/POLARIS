"""
POLARIS Performance Test Suite

This module provides comprehensive performance testing capabilities for the POLARIS framework,
including throughput testing, latency measurement, load generation, and performance regression testing.
"""

import asyncio
import time
import statistics
import psutil
import gc
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tests.integration.harness.polaris_integration_test_harness import (
    PolarisIntegrationTestHarness, IntegrationTestConfig, create_performance_harness
)
from tests.fixtures.mock_objects import DataBuilder
from src.domain.models import MetricValue, AdaptationAction


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    throughput: float  # operations per second
    avg_latency: float  # average response time in seconds
    p50_latency: float  # 50th percentile latency
    p95_latency: float  # 95th percentile latency
    p99_latency: float  # 99th percentile latency
    max_latency: float  # maximum latency
    min_latency: float  # minimum latency
    error_rate: float  # percentage of failed operations
    cpu_usage: Dict[str, float] = field(default_factory=dict)  # CPU usage statistics
    memory_usage: Dict[str, float] = field(default_factory=dict)  # Memory usage statistics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)  # Custom test-specific metrics


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    test_name: str
    duration: float = 60.0  # Test duration in seconds
    concurrent_users: int = 10  # Number of concurrent operations
    ramp_up_time: float = 10.0  # Time to ramp up to full load
    ramp_down_time: float = 5.0  # Time to ramp down
    target_throughput: Optional[float] = None  # Target operations per second
    max_errors: int = 100  # Maximum allowed errors before stopping
    warmup_duration: float = 5.0  # Warmup period before measurements
    cooldown_duration: float = 2.0  # Cooldown period after test


@dataclass
class PerformanceThresholds:
    """Performance thresholds for pass/fail criteria."""
    max_avg_latency: Optional[float] = None  # Maximum acceptable average latency
    max_p95_latency: Optional[float] = None  # Maximum acceptable 95th percentile latency
    min_throughput: Optional[float] = None  # Minimum acceptable throughput
    max_error_rate: float = 5.0  # Maximum acceptable error rate percentage
    max_cpu_usage: float = 90.0  # Maximum acceptable CPU usage percentage
    max_memory_usage: Optional[float] = None  # Maximum acceptable memory usage in MB


class PerformanceMonitor:
    """Monitors system performance during tests."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_task:
            await self.monitor_task
        
        return self._calculate_stats()
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Sample CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)
                
                # Sample memory usage
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                await asyncio.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                break
    
    def _calculate_stats(self) -> Dict[str, float]:
        """Calculate performance statistics."""
        stats = {}
        
        if self.cpu_samples:
            stats.update({
                "cpu_avg": statistics.mean(self.cpu_samples),
                "cpu_max": max(self.cpu_samples),
                "cpu_min": min(self.cpu_samples),
                "cpu_p95": np.percentile(self.cpu_samples, 95) if len(self.cpu_samples) > 1 else self.cpu_samples[0]
            })
        
        if self.memory_samples:
            stats.update({
                "memory_avg": statistics.mean(self.memory_samples),
                "memory_max": max(self.memory_samples),
                "memory_min": min(self.memory_samples),
                "memory_p95": np.percentile(self.memory_samples, 95) if len(self.memory_samples) > 1 else self.memory_samples[0]
            })
        
        return stats


class LoadGenerator:
    """Generates load for performance testing."""
    
    def __init__(self, harness: PolarisIntegrationTestHarness):
        self.harness = harness
        self.operation_times: List[float] = []
        self.errors: List[Exception] = []
        self.successful_operations = 0
        self.failed_operations = 0
    
    async def generate_telemetry_load(
        self,
        systems: List[str],
        config: LoadTestConfig
    ) -> PerformanceMetrics:
        """Generate telemetry load across multiple systems."""
        return await self._run_load_test(
            operation=self._telemetry_operation,
            operation_args={"systems": systems},
            config=config
        )
    
    async def generate_adaptation_load(
        self,
        systems: List[str],
        config: LoadTestConfig
    ) -> PerformanceMetrics:
        """Generate adaptation execution load."""
        return await self._run_load_test(
            operation=self._adaptation_operation,
            operation_args={"systems": systems},
            config=config
        )
    
    async def generate_mixed_load(
        self,
        systems: List[str],
        telemetry_ratio: float,
        config: LoadTestConfig
    ) -> PerformanceMetrics:
        """Generate mixed telemetry and adaptation load."""
        return await self._run_load_test(
            operation=self._mixed_operation,
            operation_args={"systems": systems, "telemetry_ratio": telemetry_ratio},
            config=config
        )
    
    async def _run_load_test(
        self,
        operation: Callable,
        operation_args: Dict[str, Any],
        config: LoadTestConfig
    ) -> PerformanceMetrics:
        """Run a load test with the specified operation."""
        # Reset counters
        self.operation_times.clear()
        self.errors.clear()
        self.successful_operations = 0
        self.failed_operations = 0
        
        # Start performance monitoring
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()
        
        start_time = datetime.now()
        
        try:
            # Warmup phase
            if config.warmup_duration > 0:
                await self._warmup_phase(operation, operation_args, config.warmup_duration)
            
            # Main load test phase
            await self._load_test_phase(operation, operation_args, config)
            
            # Cooldown phase
            if config.cooldown_duration > 0:
                await asyncio.sleep(config.cooldown_duration)
        
        finally:
            end_time = datetime.now()
            system_stats = await monitor.stop_monitoring()
        
        # Calculate metrics
        return self._calculate_performance_metrics(
            config.test_name, start_time, end_time, system_stats
        )
    
    async def _warmup_phase(
        self,
        operation: Callable,
        operation_args: Dict[str, Any],
        duration: float
    ) -> None:
        """Run warmup operations."""
        warmup_end = time.time() + duration
        
        while time.time() < warmup_end:
            try:
                await operation(**operation_args)
                await asyncio.sleep(0.1)  # Small delay between warmup operations
            except Exception:
                pass  # Ignore warmup errors
    
    async def _load_test_phase(
        self,
        operation: Callable,
        operation_args: Dict[str, Any],
        config: LoadTestConfig
    ) -> None:
        """Run the main load test phase."""
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        # Calculate operation timing
        if config.target_throughput:
            operation_interval = 1.0 / config.target_throughput
        else:
            operation_interval = 0.01  # Default to high frequency
        
        # Run load test
        test_end = time.time() + config.duration
        tasks = []
        
        while time.time() < test_end and len(self.errors) < config.max_errors:
            # Create operation task
            task = asyncio.create_task(
                self._execute_operation_with_timing(semaphore, operation, operation_args)
            )
            tasks.append(task)
            
            # Control operation rate
            await asyncio.sleep(operation_interval)
        
        # Wait for all operations to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_operation_with_timing(
        self,
        semaphore: asyncio.Semaphore,
        operation: Callable,
        operation_args: Dict[str, Any]
    ) -> None:
        """Execute an operation with timing measurement."""
        async with semaphore:
            start_time = time.time()
            
            try:
                await operation(**operation_args)
                self.successful_operations += 1
                
            except Exception as e:
                self.errors.append(e)
                self.failed_operations += 1
            
            finally:
                end_time = time.time()
                self.operation_times.append(end_time - start_time)
    
    async def _telemetry_operation(self, systems: List[str]) -> None:
        """Single telemetry operation."""
        system_id = systems[len(self.operation_times) % len(systems)]
        
        metrics = {
            "cpu_usage": MetricValue(
                name="cpu_usage",
                value=float(30 + (len(self.operation_times) % 70)),
                unit="percent",
                timestamp=datetime.now()
            ),
            "memory_usage": MetricValue(
                name="memory_usage",
                value=float(500 + (len(self.operation_times) % 1000)),
                unit="MB",
                timestamp=datetime.now()
            )
        }
        
        await self.harness.inject_telemetry(system_id, metrics)
    
    async def _adaptation_operation(self, systems: List[str]) -> None:
        """Single adaptation operation."""
        system_id = systems[len(self.operation_times) % len(systems)]
        
        action = DataBuilder.adaptation_action(
            action_id=f"perf_action_{len(self.operation_times)}",
            action_type="performance_test",
            target_system=system_id,
            parameters={"test_param": len(self.operation_times)}
        )
        
        await self.harness.execute_action(action)
    
    async def _mixed_operation(self, systems: List[str], telemetry_ratio: float) -> None:
        """Mixed telemetry and adaptation operation."""
        if len(self.operation_times) % 100 < (telemetry_ratio * 100):
            await self._telemetry_operation(systems)
        else:
            await self._adaptation_operation(systems)
    
    def _calculate_performance_metrics(
        self,
        test_name: str,
        start_time: datetime,
        end_time: datetime,
        system_stats: Dict[str, float]
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        duration = (end_time - start_time).total_seconds()
        total_operations = self.successful_operations + self.failed_operations
        
        # Calculate latency statistics
        if self.operation_times:
            avg_latency = statistics.mean(self.operation_times)
            p50_latency = np.percentile(self.operation_times, 50)
            p95_latency = np.percentile(self.operation_times, 95)
            p99_latency = np.percentile(self.operation_times, 99)
            max_latency = max(self.operation_times)
            min_latency = min(self.operation_times)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = max_latency = min_latency = 0.0
        
        # Calculate throughput and error rate
        throughput = total_operations / duration if duration > 0 else 0.0
        error_rate = (self.failed_operations / total_operations * 100) if total_operations > 0 else 0.0
        
        return PerformanceMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            throughput=throughput,
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            max_latency=max_latency,
            min_latency=min_latency,
            error_rate=error_rate,
            cpu_usage={k: v for k, v in system_stats.items() if k.startswith("cpu_")},
            memory_usage={k: v for k, v in system_stats.items() if k.startswith("memory_")}
        )


class PolarisPerformanceTestSuite:
    """
    Main performance test suite for POLARIS framework.
    
    Provides comprehensive performance testing capabilities including:
    - Throughput testing
    - Latency measurement
    - Load generation
    - Performance regression testing
    - Benchmarking
    """
    
    def __init__(self, results_dir: str = "performance_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.test_results: List[PerformanceMetrics] = []
        self.baseline_results: Dict[str, PerformanceMetrics] = {}
    
    async def run_throughput_test(
        self,
        systems: List[str],
        config: LoadTestConfig,
        thresholds: Optional[PerformanceThresholds] = None
    ) -> Tuple[PerformanceMetrics, bool]:
        """Run throughput performance test."""
        async with create_performance_harness(f"throughput_{config.test_name}", systems) as harness:
            load_generator = LoadGenerator(harness)
            
            # Run telemetry throughput test
            metrics = await load_generator.generate_telemetry_load(systems, config)
            
            # Evaluate against thresholds
            passed = self._evaluate_thresholds(metrics, thresholds) if thresholds else True
            
            # Store results
            self.test_results.append(metrics)
            self._save_results(metrics)
            
            return metrics, passed
    
    async def run_latency_test(
        self,
        systems: List[str],
        config: LoadTestConfig,
        thresholds: Optional[PerformanceThresholds] = None
    ) -> Tuple[PerformanceMetrics, bool]:
        """Run latency performance test."""
        # Configure for latency testing (lower concurrency, focus on response times)
        latency_config = LoadTestConfig(
            test_name=f"latency_{config.test_name}",
            duration=config.duration,
            concurrent_users=min(config.concurrent_users, 5),  # Lower concurrency for latency
            ramp_up_time=config.ramp_up_time,
            ramp_down_time=config.ramp_down_time,
            warmup_duration=config.warmup_duration,
            cooldown_duration=config.cooldown_duration
        )
        
        async with create_performance_harness(latency_config.test_name, systems) as harness:
            load_generator = LoadGenerator(harness)
            
            # Run adaptation latency test
            metrics = await load_generator.generate_adaptation_load(systems, latency_config)
            
            # Evaluate against thresholds
            passed = self._evaluate_thresholds(metrics, thresholds) if thresholds else True
            
            # Store results
            self.test_results.append(metrics)
            self._save_results(metrics)
            
            return metrics, passed
    
    async def run_stress_test(
        self,
        systems: List[str],
        config: LoadTestConfig,
        thresholds: Optional[PerformanceThresholds] = None
    ) -> Tuple[PerformanceMetrics, bool]:
        """Run stress test with high load."""
        # Configure for stress testing (high concurrency and duration)
        stress_config = LoadTestConfig(
            test_name=f"stress_{config.test_name}",
            duration=max(config.duration, 120.0),  # Minimum 2 minutes for stress test
            concurrent_users=config.concurrent_users * 2,  # Double the concurrency
            ramp_up_time=config.ramp_up_time * 2,
            ramp_down_time=config.ramp_down_time,
            max_errors=config.max_errors * 2,  # Allow more errors in stress test
            warmup_duration=config.warmup_duration,
            cooldown_duration=config.cooldown_duration
        )
        
        async with create_performance_harness(stress_config.test_name, systems) as harness:
            load_generator = LoadGenerator(harness)
            
            # Run mixed load stress test
            metrics = await load_generator.generate_mixed_load(systems, 0.7, stress_config)  # 70% telemetry, 30% adaptations
            
            # Evaluate against thresholds (may be more lenient for stress tests)
            passed = self._evaluate_thresholds(metrics, thresholds) if thresholds else True
            
            # Store results
            self.test_results.append(metrics)
            self._save_results(metrics)
            
            return metrics, passed
    
    async def run_scalability_test(
        self,
        base_systems: List[str],
        scale_factors: List[int],
        base_config: LoadTestConfig,
        thresholds: Optional[PerformanceThresholds] = None
    ) -> List[Tuple[int, PerformanceMetrics, bool]]:
        """Run scalability test with increasing system counts."""
        results = []
        
        for scale_factor in scale_factors:
            # Scale up systems
            scaled_systems = []
            for i in range(scale_factor):
                scaled_systems.extend([f"{system}_{i}" for system in base_systems])
            
            # Scale up load proportionally
            scaled_config = LoadTestConfig(
                test_name=f"scalability_{base_config.test_name}_x{scale_factor}",
                duration=base_config.duration,
                concurrent_users=base_config.concurrent_users * scale_factor,
                ramp_up_time=base_config.ramp_up_time,
                ramp_down_time=base_config.ramp_down_time,
                target_throughput=base_config.target_throughput * scale_factor if base_config.target_throughput else None,
                max_errors=base_config.max_errors * scale_factor,
                warmup_duration=base_config.warmup_duration,
                cooldown_duration=base_config.cooldown_duration
            )
            
            # Run test
            metrics, passed = await self.run_throughput_test(scaled_systems, scaled_config, thresholds)
            results.append((scale_factor, metrics, passed))
        
        return results
    
    async def run_endurance_test(
        self,
        systems: List[str],
        duration_hours: float,
        base_config: LoadTestConfig,
        thresholds: Optional[PerformanceThresholds] = None
    ) -> Tuple[PerformanceMetrics, bool]:
        """Run endurance test for extended duration."""
        endurance_config = LoadTestConfig(
            test_name=f"endurance_{base_config.test_name}_{duration_hours}h",
            duration=duration_hours * 3600,  # Convert hours to seconds
            concurrent_users=base_config.concurrent_users,
            ramp_up_time=base_config.ramp_up_time * 2,  # Longer ramp up for endurance
            ramp_down_time=base_config.ramp_down_time,
            target_throughput=base_config.target_throughput,
            max_errors=base_config.max_errors * 10,  # Allow more errors over long duration
            warmup_duration=base_config.warmup_duration * 2,
            cooldown_duration=base_config.cooldown_duration
        )
        
        return await self.run_throughput_test(systems, endurance_config, thresholds)
    
    def run_regression_test(
        self,
        test_name: str,
        current_metrics: PerformanceMetrics,
        regression_threshold: float = 0.1  # 10% regression threshold
    ) -> Tuple[bool, Dict[str, float]]:
        """Run performance regression test against baseline."""
        if test_name not in self.baseline_results:
            # No baseline available, store current as baseline
            self.baseline_results[test_name] = current_metrics
            return True, {}
        
        baseline = self.baseline_results[test_name]
        regressions = {}
        
        # Check throughput regression
        throughput_change = (baseline.throughput - current_metrics.throughput) / baseline.throughput
        if throughput_change > regression_threshold:
            regressions["throughput"] = throughput_change
        
        # Check latency regression
        latency_change = (current_metrics.avg_latency - baseline.avg_latency) / baseline.avg_latency
        if latency_change > regression_threshold:
            regressions["avg_latency"] = latency_change
        
        p95_change = (current_metrics.p95_latency - baseline.p95_latency) / baseline.p95_latency
        if p95_change > regression_threshold:
            regressions["p95_latency"] = p95_change
        
        # Check error rate regression
        error_rate_change = current_metrics.error_rate - baseline.error_rate
        if error_rate_change > regression_threshold * 100:  # Error rate is in percentage
            regressions["error_rate"] = error_rate_change
        
        passed = len(regressions) == 0
        return passed, regressions
    
    def _evaluate_thresholds(
        self,
        metrics: PerformanceMetrics,
        thresholds: PerformanceThresholds
    ) -> bool:
        """Evaluate performance metrics against thresholds."""
        if thresholds.max_avg_latency and metrics.avg_latency > thresholds.max_avg_latency:
            return False
        
        if thresholds.max_p95_latency and metrics.p95_latency > thresholds.max_p95_latency:
            return False
        
        if thresholds.min_throughput and metrics.throughput < thresholds.min_throughput:
            return False
        
        if metrics.error_rate > thresholds.max_error_rate:
            return False
        
        if metrics.cpu_usage.get("cpu_avg", 0) > thresholds.max_cpu_usage:
            return False
        
        if thresholds.max_memory_usage and metrics.memory_usage.get("memory_avg", 0) > thresholds.max_memory_usage:
            return False
        
        return True
    
    def _save_results(self, metrics: PerformanceMetrics) -> None:
        """Save performance results to file."""
        results_file = self.results_dir / f"{metrics.test_name}_{metrics.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert metrics to JSON-serializable format
        results_data = {
            "test_name": metrics.test_name,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat(),
            "duration": metrics.duration,
            "throughput": metrics.throughput,
            "avg_latency": metrics.avg_latency,
            "p50_latency": metrics.p50_latency,
            "p95_latency": metrics.p95_latency,
            "p99_latency": metrics.p99_latency,
            "max_latency": metrics.max_latency,
            "min_latency": metrics.min_latency,
            "error_rate": metrics.error_rate,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "custom_metrics": metrics.custom_metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def generate_performance_report(self, include_charts: bool = True) -> str:
        """Generate comprehensive performance report."""
        if not self.test_results:
            return "No performance test results available."
        
        report_lines = [
            "POLARIS Performance Test Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {len(self.test_results)}",
            ""
        ]
        
        # Summary statistics
        throughputs = [r.throughput for r in self.test_results]
        latencies = [r.avg_latency for r in self.test_results]
        error_rates = [r.error_rate for r in self.test_results]
        
        report_lines.extend([
            "Summary Statistics:",
            "-" * 20,
            f"Average Throughput: {statistics.mean(throughputs):.2f} ops/sec",
            f"Average Latency: {statistics.mean(latencies)*1000:.2f} ms",
            f"Average Error Rate: {statistics.mean(error_rates):.2f}%",
            ""
        ])
        
        # Individual test results
        report_lines.extend([
            "Individual Test Results:",
            "-" * 25
        ])
        
        for metrics in self.test_results:
            status = "✅ PASS" if metrics.error_rate < 5.0 else "❌ FAIL"
            report_lines.extend([
                f"{status} {metrics.test_name}",
                f"  Duration: {metrics.duration:.1f}s",
                f"  Throughput: {metrics.throughput:.2f} ops/sec",
                f"  Avg Latency: {metrics.avg_latency*1000:.2f} ms",
                f"  P95 Latency: {metrics.p95_latency*1000:.2f} ms",
                f"  Error Rate: {metrics.error_rate:.2f}%",
                f"  CPU Usage: {metrics.cpu_usage.get('cpu_avg', 0):.1f}%",
                ""
            ])
        
        # Generate charts if requested
        if include_charts:
            self._generate_performance_charts()
            report_lines.extend([
                "Performance charts generated in:", 
                f"  {self.results_dir}/performance_charts.png"
            ])
        
        return "\n".join(report_lines)
    
    def _generate_performance_charts(self) -> None:
        """Generate performance visualization charts."""
        if not self.test_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        test_names = [r.test_name for r in self.test_results]
        throughputs = [r.throughput for r in self.test_results]
        avg_latencies = [r.avg_latency * 1000 for r in self.test_results]  # Convert to ms
        p95_latencies = [r.p95_latency * 1000 for r in self.test_results]
        error_rates = [r.error_rate for r in self.test_results]
        
        # Throughput chart
        ax1.bar(range(len(test_names)), throughputs)
        ax1.set_title('Throughput (ops/sec)')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        
        # Latency chart
        x = range(len(test_names))
        ax2.bar(x, avg_latencies, alpha=0.7, label='Average')
        ax2.bar(x, p95_latencies, alpha=0.7, label='P95')
        ax2.set_title('Latency (ms)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.legend()
        
        # Error rate chart
        ax3.bar(range(len(test_names)), error_rates)
        ax3.set_title('Error Rate (%)')
        ax3.set_xticks(range(len(test_names)))
        ax3.set_xticklabels(test_names, rotation=45, ha='right')
        
        # CPU usage chart
        cpu_usages = [r.cpu_usage.get('cpu_avg', 0) for r in self.test_results]
        ax4.bar(range(len(test_names)), cpu_usages)
        ax4.set_title('CPU Usage (%)')
        ax4.set_xticks(range(len(test_names)))
        ax4.set_xticklabels(test_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_charts.png', dpi=300, bbox_inches='tight')
        plt.close()


# Utility functions for common performance testing scenarios

async def run_basic_performance_suite(systems: List[str]) -> Dict[str, PerformanceMetrics]:
    """Run a basic performance test suite with common scenarios."""
    suite = PolarisPerformanceTestSuite()
    results = {}
    
    # Basic throughput test
    throughput_config = LoadTestConfig(
        test_name="basic_throughput",
        duration=30.0,
        concurrent_users=10,
        ramp_up_time=5.0
    )
    
    metrics, _ = await suite.run_throughput_test(systems, throughput_config)
    results["throughput"] = metrics
    
    # Basic latency test
    latency_config = LoadTestConfig(
        test_name="basic_latency",
        duration=20.0,
        concurrent_users=5,
        ramp_up_time=3.0
    )
    
    metrics, _ = await suite.run_latency_test(systems, latency_config)
    results["latency"] = metrics
    
    return results


async def run_comprehensive_performance_suite(systems: List[str]) -> Dict[str, Any]:
    """Run a comprehensive performance test suite."""
    suite = PolarisPerformanceTestSuite()
    results = {}
    
    # Define performance thresholds
    thresholds = PerformanceThresholds(
        max_avg_latency=1.0,  # 1 second
        max_p95_latency=2.0,  # 2 seconds
        min_throughput=10.0,  # 10 ops/sec
        max_error_rate=5.0,   # 5%
        max_cpu_usage=85.0    # 85%
    )
    
    # Throughput test
    throughput_config = LoadTestConfig(
        test_name="comprehensive_throughput",
        duration=60.0,
        concurrent_users=20,
        ramp_up_time=10.0
    )
    
    throughput_metrics, throughput_passed = await suite.run_throughput_test(systems, throughput_config, thresholds)
    results["throughput"] = {"metrics": throughput_metrics, "passed": throughput_passed}
    
    # Stress test
    stress_config = LoadTestConfig(
        test_name="comprehensive_stress",
        duration=120.0,
        concurrent_users=15,
        ramp_up_time=15.0
    )
    
    stress_metrics, stress_passed = await suite.run_stress_test(systems, stress_config, thresholds)
    results["stress"] = {"metrics": stress_metrics, "passed": stress_passed}
    
    # Scalability test
    scalability_results = await suite.run_scalability_test(
        base_systems=systems[:2],  # Use first 2 systems as base
        scale_factors=[1, 2, 4],
        base_config=LoadTestConfig(
            test_name="comprehensive_scalability",
            duration=30.0,
            concurrent_users=5,
            ramp_up_time=5.0
        ),
        thresholds=thresholds
    )
    
    results["scalability"] = scalability_results
    
    # Generate report
    report = suite.generate_performance_report(include_charts=True)
    results["report"] = report
    
    return results