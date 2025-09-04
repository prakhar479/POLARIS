#!/usr/bin/env python3
"""
Verification Agent Demo Script

This script demonstrates the POLARIS Verification Agent capabilities
by sending various control actions for verification and showing the results.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from polaris.common.nats_client import NATSClient
from polaris.models.actions import ControlAction


class VerificationDemo:
    """Demo class for testing verification agent functionality."""
    
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.logger = logging.getLogger(__name__)
        self.nats_client = None
        self.results = {}
        
    async def start(self):
        """Start the demo by connecting to NATS."""
        self.nats_client = NATSClient(
            nats_url=self.nats_url,
            logger=self.logger,
            name="verification-demo"
        )
        
        await self.nats_client.connect()
        
        # Subscribe to verification results
        await self.nats_client.subscribe(
            "polaris.verification.results",
            self._handle_verification_result
        )
        
        self.logger.info("Verification demo started")
    
    async def stop(self):
        """Stop the demo and close connections."""
        if self.nats_client:
            await self.nats_client.close()
        self.logger.info("Verification demo stopped")
    
    async def _handle_verification_result(self, msg):
        """Handle verification results from the verification adapter."""
        try:
            result_data = json.loads(msg.data.decode())
            request_id = result_data["request_id"]
            
            self.results[request_id] = result_data
            
            self.logger.info(
                f"Received verification result for {request_id}: "
                f"{'APPROVED' if result_data['approved'] else 'REJECTED'}"
            )
            
            if not result_data["approved"]:
                violations = result_data.get("violations", [])
                self.logger.warning(f"Violations: {len(violations)}")
                for violation in violations:
                    self.logger.warning(f"  - {violation['description']}")
            
        except Exception as e:
            self.logger.error(f"Error handling verification result: {e}")
    
    async def send_verification_request(
        self, 
        action: ControlAction, 
        context: dict = None,
        verification_level: str = "basic",
        timeout_sec: float = 30.0
    ) -> str:
        """Send a verification request and return the request ID."""
        request_id = str(uuid.uuid4())
        
        verification_request = {
            "request_id": request_id,
            "action": action.to_dict(),
            "context": context or {},
            "verification_level": verification_level,
            "timeout_sec": timeout_sec,
            "requester": "verification_demo"
        }
        
        await self.nats_client.publish_json(
            "polaris.verification.requests",
            verification_request
        )
        
        self.logger.info(f"Sent verification request {request_id} for {action.action_type}")
        return request_id
    
    async def wait_for_result(self, request_id: str, timeout: float = 35.0) -> dict:
        """Wait for a verification result."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if request_id in self.results:
                return self.results[request_id]
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"No result received for request {request_id}")
    
    async def demo_valid_action(self):
        """Demo a valid action that should be approved."""
        print("\n=== Demo 1: Valid Action (ADD_SERVER) ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="ADD_SERVER",
            source="demo",
            params={"count": 1, "server_type": "compute"},
            priority="normal"
        )
        
        context = {
            "active_servers": 3,
            "max_servers": 10,
            "utilization": 0.7,
            "response_time": 0.4
        }
        
        request_id = await self.send_verification_request(action, context)
        result = await self.wait_for_result(request_id)
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Verification time: {result['verification_time_ms']:.1f}ms")
        
        return result
    
    async def demo_constraint_violation(self):
        """Demo an action that violates constraints."""
        print("\n=== Demo 2: Constraint Violation (Exceed Max Servers) ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="ADD_SERVER",
            source="demo",
            params={"count": 5},  # This will exceed max_servers
            priority="normal"
        )
        
        context = {
            "active_servers": 8,
            "max_servers": 10,
            "utilization": 0.9,
            "response_time": 0.6
        }
        
        request_id = await self.send_verification_request(action, context)
        result = await self.wait_for_result(request_id)
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Violations: {len(result['violations'])}")
        
        for violation in result['violations']:
            print(f"  - {violation['severity'].upper()}: {violation['description']}")
            if violation.get('suggested_fix'):
                print(f"    Fix: {violation['suggested_fix']}")
        
        return result
    
    async def demo_policy_violation(self):
        """Demo an action that violates policies."""
        print("\n=== Demo 3: Policy Violation (Server Removal Without Approval) ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="REMOVE_SERVER",
            source="demo",
            params={"count": 1},
            priority="normal"
            # Note: No 'approved_by' field - violates policy
        )
        
        context = {
            "active_servers": 5,
            "max_servers": 10,
            "utilization": 0.4,
            "response_time": 0.2
        }
        
        request_id = await self.send_verification_request(action, context, "policy")
        result = await self.wait_for_result(request_id)
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Violations: {len(result['violations'])}")
        
        for violation in result['violations']:
            print(f"  - {violation['severity'].upper()}: {violation['description']}")
        
        print(f"Recommendations: {len(result['recommendations'])}")
        for rec in result['recommendations']:
            print(f"  - {rec}")
        
        return result
    
    async def demo_valid_dimmer_action(self):
        """Demo a valid dimmer adjustment."""
        print("\n=== Demo 4: Valid Dimmer Adjustment ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="SET_DIMMER",
            source="demo",
            params={"value": 0.8},
            priority="normal"
        )
        
        context = {
            "active_servers": 4,
            "max_servers": 10,
            "utilization": 0.6,
            "response_time": 0.5,
            "dimmer": 1.0
        }
        
        request_id = await self.send_verification_request(action, context)
        result = await self.wait_for_result(request_id)
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Verification time: {result['verification_time_ms']:.1f}ms")
        
        return result
    
    async def demo_invalid_dimmer_action(self):
        """Demo an invalid dimmer value."""
        print("\n=== Demo 5: Invalid Dimmer Value ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="SET_DIMMER",
            source="demo",
            params={"value": 1.5},  # Invalid: > 1.0
            priority="normal"
        )
        
        context = {
            "active_servers": 4,
            "max_servers": 10,
            "utilization": 0.6,
            "response_time": 0.5,
            "dimmer": 1.0
        }
        
        request_id = await self.send_verification_request(action, context)
        result = await self.wait_for_result(request_id)
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Violations: {len(result['violations'])}")
        
        for violation in result['violations']:
            print(f"  - {violation['severity'].upper()}: {violation['description']}")
        
        return result
    
    async def demo_comprehensive_verification(self):
        """Demo comprehensive verification with all checks."""
        print("\n=== Demo 6: Comprehensive Verification ===")
        
        action = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="ADD_SERVER",
            source="demo",
            params={"count": 2},
            priority="normal"
        )
        
        context = {
            "active_servers": 6,
            "max_servers": 10,
            "utilization": 0.8,
            "response_time": 0.7
        }
        
        request_id = await self.send_verification_request(action, context, "comprehensive")
        result = await self.wait_for_result(request_id, timeout=70.0)  # Longer timeout
        
        print(f"Result: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Verification time: {result['verification_time_ms']:.1f}ms")
        print(f"Metadata: {result.get('metadata', {})}")
        
        return result
    
    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("🚀 Starting POLARIS Verification Agent Demo")
        print("=" * 50)
        
        try:
            # Run all demo scenarios
            await self.demo_valid_action()
            await asyncio.sleep(1)
            
            await self.demo_constraint_violation()
            await asyncio.sleep(1)
            
            await self.demo_policy_violation()
            await asyncio.sleep(1)
            
            await self.demo_valid_dimmer_action()
            await asyncio.sleep(1)
            
            await self.demo_invalid_dimmer_action()
            await asyncio.sleep(1)
            
            await self.demo_comprehensive_verification()
            
            print("\n" + "=" * 50)
            print("✅ All demos completed successfully!")
            
            # Summary
            approved = sum(1 for r in self.results.values() if r['approved'])
            total = len(self.results)
            print(f"📊 Summary: {approved}/{total} actions approved")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise


async def main():
    """Main demo function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo = VerificationDemo()
    
    try:
        await demo.start()
        await demo.run_all_demos()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        logging.exception("Demo failed")
    finally:
        await demo.stop()


if __name__ == "__main__":
    print("POLARIS Verification Agent Demo")
    print("Make sure the verification adapter is running:")
    print("  python src/scripts/start_component.py verification --plugin-dir extern")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped.")