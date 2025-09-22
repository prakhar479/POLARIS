#!/usr/bin/env python3
"""
System Status Monitor for SWIM POLARIS Adaptation System

Provides real-time monitoring, status reporting, and system health checking
for the running SWIM POLARIS adaptation system.
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swim_driver import SwimPolarisDriver
from metrics_system import ComponentStatus


class SystemStatusMonitor:
    """Real-time system status monitor."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.driver: Optional[SwimPolarisDriver] = None
    
    async def connect(self) -> bool:
        """Connect to running system."""
        try:
            self.driver = SwimPolarisDriver(self.config_path)
            # Don't start the driver, just use it to query status
            return True
        except Exception as e:
            print(f"Failed to connect to system: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.driver:
            return {"error": "Not connected to system"}
        
        try:
            status = await self.driver.get_status()
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "health_status": status.health_status.value,
                "last_update": status.last_update.isoformat() if status.last_update else None,
                "components": {
                    name: comp_status.value 
                    for name, comp_status in (status.component_status or {}).items()
                },
                "metrics": status.system_metrics or {},
                "recent_adaptations": getattr(status, 'recent_adaptations', [])
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def monitor_continuous(self, interval: float = 10.0, duration: Optional[float] = None):
        """Monitor system status continuously."""
        start_time = time.time()
        
        print("Starting continuous monitoring...")
        print(f"Update interval: {interval} seconds")
        if duration:
            print(f"Duration: {duration} seconds")
        print("-" * 60)
        
        try:
            while True:
                status = await self.get_status()
                self._print_status_line(status)
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
    
    def _print_status_line(self, status: Dict[str, Any]):
        """Print a single status line."""
        timestamp = status.get('timestamp', 'unknown')
        health = status.get('health_status', 'unknown')
        
        # Component summary
        components = status.get('components', {})
        healthy_count = sum(1 for s in components.values() if s == 'healthy')
        total_count = len(components)
        
        # Metrics summary
        metrics = status.get('metrics', {})
        metric_count = len(metrics)
        
        # Recent adaptations
        adaptations = status.get('recent_adaptations', [])
        adaptation_count = len(adaptations)
        
        print(f"{timestamp[:19]} | {health:8} | "
              f"Components: {healthy_count}/{total_count} | "
              f"Metrics: {metric_count:3} | "
              f"Adaptations: {adaptation_count:3}")
    
    def print_detailed_status(self, status: Dict[str, Any]):
        """Print detailed status information."""
        print("\n" + "="*80)
        print("SWIM POLARIS ADAPTATION SYSTEM - DETAILED STATUS")
        print("="*80)
        
        print(f"Timestamp: {status.get('timestamp', 'unknown')}")
        print(f"Health Status: {status.get('health_status', 'unknown')}")
        print(f"Last Update: {status.get('last_update', 'unknown')}")
        
        # Component status
        components = status.get('components', {})
        if components:
            print(f"\nComponents ({len(components)}):")
            for name, comp_status in components.items():
                status_icon = "✓" if comp_status == "healthy" else "⚠" if comp_status == "warning" else "✗"
                print(f"  {status_icon} {name}: {comp_status}")
        
        # Current metrics
        metrics = status.get('metrics', {})
        if metrics:
            print(f"\nCurrent Metrics ({len(metrics)}):")
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.3f}")
                else:
                    print(f"  {name}: {value}")
        
        # Recent adaptations
        adaptations = status.get('recent_adaptations', [])
        if adaptations:
            print(f"\nRecent Adaptations ({len(adaptations)}):")
            for adaptation in adaptations[-5:]:  # Show last 5
                action = adaptation.get('action_type', 'unknown')
                success = adaptation.get('status', 'unknown')
                timestamp = adaptation.get('timestamp', 'unknown')
                print(f"  {timestamp[:19]} | {action:15} | {success}")
        
        # Error information
        if 'error' in status:
            print(f"\nError: {status['error']}")
        
        print("="*80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor SWIM POLARIS System Status")
    parser.add_argument("--config", "-c",
                       help="Configuration file path")
    parser.add_argument("--continuous", "-m",
                       action="store_true",
                       help="Monitor continuously")
    parser.add_argument("--interval", "-i",
                       type=float,
                       default=10.0,
                       help="Update interval for continuous monitoring (seconds)")
    parser.add_argument("--duration", "-d",
                       type=float,
                       help="Duration for continuous monitoring (seconds)")
    parser.add_argument("--json", "-j",
                       action="store_true",
                       help="Output in JSON format")
    parser.add_argument("--detailed", "-v",
                       action="store_true",
                       help="Show detailed status information")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SystemStatusMonitor(args.config)
    
    # Connect to system
    if not await monitor.connect():
        sys.exit(1)
    
    try:
        if args.continuous:
            # Continuous monitoring
            await monitor.monitor_continuous(args.interval, args.duration)
        else:
            # Single status check
            status = await monitor.get_status()
            
            if args.json:
                print(json.dumps(status, indent=2))
            elif args.detailed:
                monitor.print_detailed_status(status)
            else:
                # Simple status
                health = status.get('health_status', 'unknown')
                components = status.get('components', {})
                metrics = status.get('metrics', {})
                
                print(f"System Status: {health}")
                print(f"Components: {len(components)} registered")
                print(f"Metrics: {len(metrics)} current values")
                
                if 'error' in status:
                    print(f"Error: {status['error']}")
                    sys.exit(1)
    
    except Exception as e:
        print(f"Monitoring error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())