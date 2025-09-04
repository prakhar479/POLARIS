#!/usr/bin/env python3
"""
Standalone Verification Agent Demo

This script demonstrates running the verification agent without
any external system plugin, using only framework-level defaults.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from polaris.common.nats_client import NATSClient
from polaris.models.actions import ControlAction


async def demo_standalone_verification():
    """Demo verification agent running in standalone mode."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🚀 POLARIS Standalone Verification Demo")
    print("=" * 50)
    print("This demo shows verification working without external system plugins")
    print()
    
    # Connect to NATS
    nats_client = NATSClient(
        nats_url="nats://localhost:4222",
        logger=logger,
        name="standalone-verification-demo"
    )
    
    results = {}
    
    async def handle_verification_result(msg):
        """Handle verification results."""
        try:
            result_data = json.loads(msg.data.decode())
            request_id = result_data["request_id"]
            results[request_id] = result_data
            
            status = "✅ APPROVED" if result_data["approved"] else "❌ REJECTED"
            print(f"Result: {status} (confidence: {result_data['confidence']:.2f})")
            
            if not result_data["approved"]:
                for violation in result_data.get("violations", []):
                    print(f"  - {violation['severity'].upper()}: {violation['description']}")
            
        except Exception as e:
            logger.error(f"Error handling result: {e}")
    
    try:
        await nats_client.connect()
        await nats_client.subscribe("polaris.verification.results", handle_verification_result)
        
        print("Connected to NATS, starting verification tests...")
        print()
        
        # Test 1: Valid action with proper metadata
        print("Test 1: Valid action with proper metadata")
        action1 = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="ADD_SERVER",
            source="demo_controller",
            params={"count": 1},
            priority="normal"
        )
        
        request1 = {
            "request_id": str(uuid.uuid4()),
            "action": action1.to_dict(),
            "context": {},
            "verification_level": "basic",
            "timeout_sec": 30.0,
            "requester": "standalone_demo"
        }
        
        await nats_client.publish_json("polaris.verification.requests", request1)
        await asyncio.sleep(2)
        
        # Test 2: Invalid action type
        print("\nTest 2: Invalid action type")
        action2 = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="INVALID_ACTION",  # This should fail
            source="demo_controller",
            params={"count": 1},
            priority="normal"
        )
        
        request2 = {
            "request_id": str(uuid.uuid4()),
            "action": action2.to_dict(),
            "context": {},
            "verification_level": "basic",
            "timeout_sec": 30.0,
            "requester": "standalone_demo"
        }
        
        await nats_client.publish_json("polaris.verification.requests", request2)
        await asyncio.sleep(2)
        
        # Test 3: Action without source (policy violation)
        print("\nTest 3: Action without source (policy violation)")
        action3 = ControlAction(
            action_id=str(uuid.uuid4()),
            action_type="SCALE_UP",
            source="",  # Empty source should trigger policy violation
            params={"count": 2},
            priority="normal"
        )
        
        request3 = {
            "request_id": str(uuid.uuid4()),
            "action": action3.to_dict(),
            "context": {},
            "verification_level": "policy",
            "timeout_sec": 30.0,
            "requester": "standalone_demo"
        }
        
        await nats_client.publish_json("polaris.verification.requests", request3)
        await asyncio.sleep(2)
        
        print("\n" + "=" * 50)
        print("✅ Standalone verification demo completed!")
        
        # Summary
        approved = sum(1 for r in results.values() if r['approved'])
        total = len(results)
        print(f"📊 Summary: {approved}/{total} actions approved")
        print()
        print("This demonstrates that verification works even without")
        print("external system plugins, using framework-level defaults.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Demo error")
    finally:
        await nats_client.close()


if __name__ == "__main__":
    print("POLARIS Standalone Verification Demo")
    print("Make sure the verification adapter is running:")
    print("  python src/scripts/start_component.py verification")
    print("  (Note: No --plugin-dir needed for standalone mode)")
    print()
    
    try:
        asyncio.run(demo_standalone_verification())
    except KeyboardInterrupt:
        print("\nDemo stopped.")