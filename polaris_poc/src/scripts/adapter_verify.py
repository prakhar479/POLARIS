"""
Verification scripts for testing SWIM adapters independently

This module provides utilities to test the monitor and execution adapters
without requiring the full POLARIS system to be running.
"""

import asyncio
import json
import logging
from datetime import datetime
from nats.aio.client import Client as NATS


class AdapterTester:
    """Test utilities for SWIM adapters"""
    
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc: NATS = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def connect(self):
        """Connect to NATS"""
        self.nc = NATS()
        await self.nc.connect(self.nats_url)
        self.logger.info(f"Connected to NATS at {self.nats_url}")
    
    async def close(self):
        """Close NATS connection"""
        if self.nc:
            await self.nc.close()


class MonitorAdapterTester(AdapterTester):
    """Test the monitor adapter"""
    
    async def listen_to_telemetry(self, duration: int = 30):
        """Listen to telemetry events for specified duration"""
        await self.connect()
        
        received_events = []
        
        async def telemetry_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                received_events.append(data)
                
                payload = data.get('payload', {})
                metric_name = payload.get('name', 'unknown')
                metric_value = payload.get('value', 'unknown')
                timestamp = data.get('timestamp', 'unknown')
                
                self.logger.info(f"Received: {metric_name} = {metric_value} at {timestamp}")
                
            except Exception as e:
                self.logger.error(f"Error processing telemetry: {e}")
        
        # Subscribe to telemetry events
        await self.nc.subscribe("polaris.telemetry.events", cb=telemetry_handler)
        self.logger.info(f"Listening for telemetry events for {duration} seconds...")
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        await self.close()
        
        self.logger.info(f"Test completed. Received {len(received_events)} events")
        return received_events
    
    async def verify_telemetry_structure(self, events):
        """Verify the structure of received telemetry events"""
        if not events:
            self.logger.warning("No telemetry events received!")
            return False
        
        required_fields = ['event_type', 'timestamp', 'source', 'payload']
        required_payload_fields = ['name', 'value', 'unit']
        
        for i, event in enumerate(events):
            # Check top-level fields
            for field in required_fields:
                if field not in event:
                    self.logger.error(f"Event {i}: Missing field '{field}'")
                    return False
            
            # Check payload fields
            payload = event.get('payload', {})
            for field in required_payload_fields:
                if field not in payload:
                    self.logger.error(f"Event {i}: Missing payload field '{field}'")
                    return False
            
            # Check value is numeric
            if not isinstance(payload.get('value'), (int, float)):
                self.logger.error(f"Event {i}: Value is not numeric: {payload.get('value')}")
                return False
        
        self.logger.info("All telemetry events have correct structure")
        return True


class ExecutionAdapterTester(AdapterTester):
    """Test the execution adapter"""
    
    async def test_action_execution(self, actions):
        """Test sending various control actions"""
        await self.connect()
        
        results = []
        
        # Subscribe to execution results
        async def result_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                results.append(data)
                
                action_type = data.get('action_type', 'unknown')
                success = data.get('success', False)
                message = data.get('message', 'no message')
                
                status = "SUCCESS" if success else "FAILED"
                self.logger.info(f"Result: {action_type} - {status}: {message}")
                
            except Exception as e:
                self.logger.error(f"Error processing result: {e}")
        
        await self.nc.subscribe("polaris.execution.results", cb=result_handler)
        
        # Send test actions
        for action in actions:
            self.logger.info(f"Sending action: {action['action_type']}")
            await self.nc.publish("polaris.actions.swim_adapter", json.dumps(action).encode())
            await asyncio.sleep(2)  # Wait for execution
        
        # Wait a bit more for final results
        await asyncio.sleep(5)
        
        await self.close()
        
        self.logger.info(f"Test completed. Received {len(results)} execution results")
        return results
    
    def create_test_actions(self):
        """Create a set of test actions"""
        timestamp = datetime.now().isoformat()
        
        return [
            {
                "action_type": "ADD_SERVER",
                "timestamp": timestamp,
                "source": "test_client",
                "params": {}
            },
            {
                "action_type": "SET_DIMMER", 
                "timestamp": timestamp,
                "source": "test_client",
                "params": {"value": 0.8}
            },
            {
                "action_type": "ADJUST_QOS",
                "timestamp": timestamp, 
                "source": "test_client",
                "params": {"value": 0.5}
            },
            {
                "action_type": "REMOVE_SERVER",
                "timestamp": timestamp,
                "source": "test_client", 
                "params": {}
            }
        ]


async def test_monitor_adapter():
    """Full test of monitor adapter"""
    print("=" * 60)
    print("TESTING MONITOR ADAPTER")
    print("=" * 60)
    
    tester = MonitorAdapterTester()
    
    print("1. Listening to telemetry events...")
    events = await tester.listen_to_telemetry(duration=20)
    
    print("\n2. Verifying telemetry structure...")
    structure_ok = await tester.verify_telemetry_structure(events)
    
    print(f"\nMONITOR ADAPTER TEST RESULTS:")
    print(f"- Events received: {len(events)}")
    print(f"- Structure valid: {structure_ok}")
    
    if events:
        metrics = set()
        for event in events:
            metrics.add(event.get('payload', {}).get('name', 'unknown'))
        print(f"- Unique metrics: {len(metrics)}")
        print(f"- Metric types: {', '.join(sorted(metrics))}")


async def test_execution_adapter():
    """Full test of execution adapter"""
    print("=" * 60)
    print("TESTING EXECUTION ADAPTER")
    print("=" * 60)
    
    tester = ExecutionAdapterTester()
    
    print("1. Creating test actions...")
    actions = tester.create_test_actions()
    
    print("2. Sending actions and monitoring results...")
    results = await tester.test_action_execution(actions)
    
    print(f"\nEXECUTION ADAPTER TEST RESULTS:")
    print(f"- Actions sent: {len(actions)}")
    print(f"- Results received: {len(results)}")
    
    if results:
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"- Successful actions: {success_count}/{len(results)}")
        
        print("\nAction Results:")
        for result in results:
            action = result.get('action_type', 'unknown')
            success = 'SUCCESS' if result.get('success', False) else 'FAILED'
            message = result.get('message', 'no message')
            print(f"  {action}: {success} - {message}")


async def test_both_adapters():
    """Test both adapters in sequence"""
    print("POLARIS SWIM Adapter Testing Suite")
    print("=" * 60)
    
    try:
        await test_monitor_adapter()
        print("\n")
        await test_execution_adapter()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            asyncio.run(test_monitor_adapter())
        elif sys.argv[1] == "execution":
            asyncio.run(test_execution_adapter())
        else:
            print("Usage: python verify_adapters.py [monitor|execution]")
    else:
        asyncio.run(test_both_adapters())