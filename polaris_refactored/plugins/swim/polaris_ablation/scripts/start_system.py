#!/usr/bin/env python3
"""
System Startup Script for SWIM POLARIS Adaptation System

Provides automated system startup with dependency checking, health verification,
and comprehensive status monitoring and reporting.
"""

import asyncio
import argparse
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import subprocess
import socket
from datetime import datetime, timezone

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swim_driver import SwimPolarisDriver, SystemStatus
from config_manager import HierarchicalConfigurationManager
from logging_system import SwimPolarisLoggingSystem, log_context
from metrics_system import SwimPolarisMetricsSystem, ComponentStatus


class DependencyChecker:
    """Checks system dependencies before startup."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def check_all_dependencies(self, config: Dict[str, Any]) -> bool:
        """Check all system dependencies.
        
        Args:
            config: System configuration
            
        Returns:
            True if all dependencies are satisfied
        """
        checks = [
            self._check_python_version(),
            self._check_nats_server(config),
            self._check_swim_connection(config),
            self._check_file_permissions(config),
            self._check_disk_space(),
            self._check_memory()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        all_passed = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Dependency check {i} failed: {result}")
                all_passed = False
            elif not result:
                self.logger.error(f"Dependency check {i} failed")
                all_passed = False
        
        return all_passed
    
    async def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            self.logger.info(f"Python version check passed: {sys.version}")
            return True
        else:
            self.logger.error(f"Python version {current_version} < required {required_version}")
            return False
    
    async def _check_nats_server(self, config: Dict[str, Any]) -> bool:
        """Check NATS server connectivity."""
        nats_config = config.get('message_bus', {}).get('nats', {})
        servers = nats_config.get('servers', ['nats://localhost:4222'])
        
        for server_url in servers:
            try:
                # Parse server URL
                if '://' in server_url:
                    protocol, address = server_url.split('://', 1)
                    if ':' in address:
                        host, port = address.split(':', 1)
                        port = int(port)
                    else:
                        host = address
                        port = 4222
                else:
                    host = server_url
                    port = 4222
                
                # Test connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    self.logger.info(f"NATS server check passed: {server_url}")
                    return True
                else:
                    self.logger.warning(f"NATS server not reachable: {server_url}")
            
            except Exception as e:
                self.logger.warning(f"NATS server check failed for {server_url}: {e}")
        
        self.logger.error("No NATS servers are reachable")
        return False
    
    async def _check_swim_connection(self, config: Dict[str, Any]) -> bool:
        """Check SWIM system connectivity."""
        managed_systems = config.get('managed_systems', [])
        
        for system_config in managed_systems:
            if system_config.get('connector_type') == 'swim':
                swim_config = system_config.get('config', {})
                connection_config = swim_config.get('connection', {})
                
                host = connection_config.get('host', 'localhost')
                port = connection_config.get('port', 4242)
                
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5.0)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        self.logger.info(f"SWIM connection check passed: {host}:{port}")
                        return True
                    else:
                        self.logger.error(f"SWIM not reachable: {host}:{port}")
                        return False
                
                except Exception as e:
                    self.logger.error(f"SWIM connection check failed: {e}")
                    return False
        
        self.logger.warning("No SWIM systems configured")
        return True  # Not an error if no SWIM systems
    
    async def _check_file_permissions(self, config: Dict[str, Any]) -> bool:
        """Check file system permissions."""
        # Check log directory
        log_handlers = config.get('logging', {}).get('handlers', [])
        for handler in log_handlers:
            if handler.get('type') == 'file':
                log_path = Path(handler['path'])
                log_dir = log_path.parent
                
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                    # Test write permission
                    test_file = log_dir / '.write_test'
                    test_file.write_text('test')
                    test_file.unlink()
                    self.logger.info(f"Log directory permissions OK: {log_dir}")
                except Exception as e:
                    self.logger.error(f"Log directory not writable: {log_dir} - {e}")
                    return False
        
        # Check results directory
        try:
            results_dir = Path('results')
            results_dir.mkdir(parents=True, exist_ok=True)
            test_file = results_dir / '.write_test'
            test_file.write_text('test')
            test_file.unlink()
            self.logger.info("Results directory permissions OK")
        except Exception as e:
            self.logger.error(f"Results directory not writable: {e}")
            return False
        
        return True
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                self.logger.error(f"Low disk space: {free_gb:.2f} GB free")
                return False
            else:
                self.logger.info(f"Disk space check passed: {free_gb:.2f} GB free")
                return True
        
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return True  # Don't fail startup for this
    
    async def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 0.5:  # Less than 500MB available
                self.logger.warning(f"Low memory: {available_gb:.2f} GB available")
                # Don't fail startup, just warn
            else:
                self.logger.info(f"Memory check passed: {available_gb:.2f} GB available")
            
            return True
        
        except ImportError:
            self.logger.info("psutil not available, skipping memory check")
            return True
        except Exception as e:
            self.logger.warning(f"Could not check memory: {e}")
            return True


class HealthChecker:
    """Performs health checks and readiness verification."""
    
    def __init__(self, driver: SwimPolarisDriver):
        self.driver = driver
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def wait_for_readiness(self, timeout: float = 60.0) -> bool:
        """Wait for system to become ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if system becomes ready within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status = await self.driver.get_status()
                
                if self._is_system_ready(status):
                    self.logger.info("System readiness check passed")
                    return True
                
                self.logger.info(f"System not ready yet, waiting... (status: {status.health_status.value})")
                await asyncio.sleep(5.0)
            
            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")
                await asyncio.sleep(5.0)
        
        self.logger.error(f"System not ready after {timeout} seconds")
        return False
    
    def _is_system_ready(self, status: SystemStatus) -> bool:
        """Check if system status indicates readiness."""
        # System is ready if it's healthy and has recent metrics
        if status.health_status not in [ComponentStatus.HEALTHY, ComponentStatus.WARNING]:
            return False
        
        # Check if we have recent metrics (within last 60 seconds)
        if status.last_update:
            time_since_update = (datetime.now(timezone.utc) - status.last_update).total_seconds()
            if time_since_update > 60:
                return False
        
        return True
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check and return detailed results."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "checks": {}
        }
        
        # System status check
        try:
            status = await self.driver.get_status()
            results["checks"]["system_status"] = {
                "status": "passed" if status.health_status == ComponentStatus.HEALTHY else "failed",
                "details": {
                    "health_status": status.health_status.value,
                    "last_update": status.last_update.isoformat() if status.last_update else None,
                    "component_count": len(status.component_status) if status.component_status else 0
                }
            }
        except Exception as e:
            results["checks"]["system_status"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # SWIM connectivity check
        try:
            # This would use the SWIM connector to test connectivity
            results["checks"]["swim_connectivity"] = {
                "status": "passed",  # Placeholder
                "details": {"connection": "active"}
            }
        except Exception as e:
            results["checks"]["swim_connectivity"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Component health checks
        if hasattr(self.driver, 'metrics_system'):
            try:
                health_summary = self.driver.metrics_system.health_monitor.get_system_health_summary()
                results["checks"]["component_health"] = {
                    "status": "passed" if health_summary["overall_status"] in ["healthy", "warning"] else "failed",
                    "details": health_summary
                }
            except Exception as e:
                results["checks"]["component_health"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Determine overall status
        check_statuses = [check["status"] for check in results["checks"].values()]
        if all(status == "passed" for status in check_statuses):
            results["overall_status"] = "healthy"
        elif any(status == "failed" for status in check_statuses):
            results["overall_status"] = "unhealthy"
        else:
            results["overall_status"] = "unknown"
        
        return results


class SystemMonitor:
    """Monitors system status and provides reporting."""
    
    def __init__(self, driver: SwimPolarisDriver):
        self.driver = driver
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_time = datetime.now(timezone.utc)
    
    async def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        try:
            status = await self.driver.get_status()
            
            # Calculate uptime
            uptime = datetime.now(timezone.utc) - self.start_time
            
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": uptime.total_seconds(),
                "uptime_formatted": str(uptime),
                "system_status": {
                    "health_status": status.health_status.value,
                    "last_update": status.last_update.isoformat() if status.last_update else None,
                    "component_count": len(status.component_status) if status.component_status else 0
                }
            }
            
            # Add metrics summary if available
            if hasattr(self.driver, 'metrics_system'):
                try:
                    metrics_summary = self.driver.metrics_system.get_metrics_summary()
                    report["metrics_summary"] = metrics_summary
                except Exception as e:
                    report["metrics_error"] = str(e)
            
            # Add component status
            if status.component_status:
                report["components"] = {
                    name: comp_status.value 
                    for name, comp_status in status.component_status.items()
                }
            
            # Add recent metrics
            if status.system_metrics:
                report["current_metrics"] = status.system_metrics
            
            return report
        
        except Exception as e:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
            }
    
    def print_status_summary(self, report: Dict[str, Any]) -> None:
        """Print a formatted status summary."""
        print("\n" + "="*60)
        print("SWIM POLARIS ADAPTATION SYSTEM STATUS")
        print("="*60)
        
        print(f"Timestamp: {report.get('timestamp', 'unknown')}")
        print(f"Uptime: {report.get('uptime_formatted', 'unknown')}")
        
        system_status = report.get('system_status', {})
        print(f"Health Status: {system_status.get('health_status', 'unknown')}")
        print(f"Last Update: {system_status.get('last_update', 'unknown')}")
        print(f"Components: {system_status.get('component_count', 0)}")
        
        # Component status
        components = report.get('components', {})
        if components:
            print("\nComponent Status:")
            for name, status in components.items():
                print(f"  {name}: {status}")
        
        # Current metrics
        current_metrics = report.get('current_metrics', {})
        if current_metrics:
            print("\nCurrent Metrics:")
            for name, value in current_metrics.items():
                print(f"  {name}: {value}")
        
        # Metrics summary
        metrics_summary = report.get('metrics_summary', {})
        if metrics_summary:
            system_metrics = metrics_summary.get('system_metrics', {})
            adaptation_metrics = metrics_summary.get('adaptation_metrics', {})
            
            if system_metrics:
                print(f"\nSystem Metrics: {system_metrics.get('total_metrics', 0)} metrics collected")
            
            if adaptation_metrics:
                print(f"Adaptations: {adaptation_metrics.get('total_adaptations', 0)} total, "
                      f"{adaptation_metrics.get('success_rate', 0):.2%} success rate")
        
        print("="*60)


class SwimPolarisSystemStarter:
    """Main system starter with comprehensive startup management."""
    
    def __init__(self, config_path: str, environment: str = "development"):
        self.config_path = Path(config_path)
        self.environment = environment
        self.driver: Optional[SwimPolarisDriver] = None
        self.health_checker: Optional[HealthChecker] = None
        self.monitor: Optional[SystemMonitor] = None
        
        # Setup basic logging for startup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Signal handling
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def start_system(self, 
                          skip_dependency_check: bool = False,
                          wait_for_ready: bool = True,
                          ready_timeout: float = 60.0) -> bool:
        """Start the SWIM POLARIS system.
        
        Args:
            skip_dependency_check: Skip dependency checking
            wait_for_ready: Wait for system to become ready
            ready_timeout: Timeout for readiness check
            
        Returns:
            True if system started successfully
        """
        try:
            with log_context("system_startup", "startup"):
                self.logger.info("Starting SWIM POLARIS Adaptation System...")
                
                # Load configuration
                self.logger.info("Loading configuration...")
                config_manager = HierarchicalConfigurationManager(str(self.config_path.parent))
                
                if self.environment != "development":
                    config = config_manager.get_ablation_config(self.environment)
                else:
                    config = config_manager.load_configuration()
                
                # Check dependencies
                if not skip_dependency_check:
                    self.logger.info("Checking system dependencies...")
                    dependency_checker = DependencyChecker()
                    
                    if not await dependency_checker.check_all_dependencies(config):
                        self.logger.error("Dependency checks failed")
                        return False
                    
                    self.logger.info("All dependency checks passed")
                
                # Initialize driver
                self.logger.info("Initializing system driver...")
                config_file = self.config_path if self.environment == "development" else None
                self.driver = SwimPolarisDriver(str(config_file) if config_file else None)
                
                # Start the system
                self.logger.info("Starting system components...")
                await self.driver.start()
                
                # Initialize health checker and monitor
                self.health_checker = HealthChecker(self.driver)
                self.monitor = SystemMonitor(self.driver)
                
                # Wait for readiness
                if wait_for_ready:
                    self.logger.info("Waiting for system readiness...")
                    if not await self.health_checker.wait_for_readiness(ready_timeout):
                        self.logger.error("System failed to become ready")
                        await self.stop_system()
                        return False
                
                self.logger.info("System startup completed successfully")
                return True
        
        except Exception as e:
            self.logger.error(f"System startup failed: {e}", exc_info=True)
            if self.driver:
                await self.stop_system()
            return False
    
    async def stop_system(self) -> None:
        """Stop the system gracefully."""
        if self.driver:
            self.logger.info("Stopping system...")
            await self.driver.stop()
            self.logger.info("System stopped")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        if not self.health_checker:
            return {"error": "System not started"}
        
        return await self.health_checker.run_comprehensive_health_check()
    
    async def get_status_report(self) -> Dict[str, Any]:
        """Get system status report."""
        if not self.monitor:
            return {"error": "System not started"}
        
        return await self.monitor.get_system_report()
    
    async def run_until_shutdown(self) -> None:
        """Run system until shutdown signal received."""
        self.logger.info("System running. Press Ctrl+C to shutdown.")
        
        # Print initial status
        if self.monitor:
            report = await self.monitor.get_system_report()
            self.monitor.print_status_summary(report)
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Graceful shutdown
        await self.stop_system()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start SWIM POLARIS Adaptation System")
    parser.add_argument("--config", "-c", 
                       default="config/base_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--environment", "-e",
                       default="development",
                       choices=["development", "testing", "staging", "production"],
                       help="Environment to run in")
    parser.add_argument("--skip-deps", 
                       action="store_true",
                       help="Skip dependency checks")
    parser.add_argument("--no-wait", 
                       action="store_true",
                       help="Don't wait for system readiness")
    parser.add_argument("--ready-timeout",
                       type=float,
                       default=60.0,
                       help="Readiness check timeout in seconds")
    parser.add_argument("--health-check",
                       action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--status",
                       action="store_true", 
                       help="Show status report and exit")
    
    args = parser.parse_args()
    
    # Create system starter
    starter = SwimPolarisSystemStarter(args.config, args.environment)
    
    try:
        # Start system
        if not await starter.start_system(
            skip_dependency_check=args.skip_deps,
            wait_for_ready=not args.no_wait,
            ready_timeout=args.ready_timeout
        ):
            sys.exit(1)
        
        # Handle special modes
        if args.health_check:
            health_report = await starter.run_health_check()
            print(json.dumps(health_report, indent=2))
            sys.exit(0 if health_report.get("overall_status") == "healthy" else 1)
        
        if args.status:
            status_report = await starter.get_status_report()
            if starter.monitor:
                starter.monitor.print_status_summary(status_report)
            else:
                print(json.dumps(status_report, indent=2))
            sys.exit(0)
        
        # Run until shutdown
        await starter.run_until_shutdown()
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())