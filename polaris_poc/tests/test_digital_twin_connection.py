#!/usr/bin/env python3
"""
Digital Twin Connection Test Script

This script tests the connection between the agentic reasoner and the digital twin
to help diagnose connection issues.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.agents.reasoner_agent import GRPCDigitalTwinClient, DTQuery, DTSimulation, DTDiagnosis


async def test_digital_twin_connection():
    """Test digital twin gRPC connection."""
    print("🔍 Testing Digital Twin Connection")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt_test")
    
    # Test different addresses
    test_addresses = [
        "localhost:50051",
        "0.0.0.0:50051",
        "127.0.0.1:50051"
    ]
    
    for address in test_addresses:
        print(f"\n🧪 Testing connection to {address}")
        
        client = GRPCDigitalTwinClient(address, logger)
        
        try:
            # Test connection
            await client.connect()
            print(f"✅ Connection to {address} successful")
            
            # Test query
            print("🔍 Testing query operation...")
            query = DTQuery(
                query_type="current_state",
                query_content="Test query from agentic reasoner"
            )
            
            response = await client.query(query)
            if response:
                print(f"✅ Query successful: {response.success}")
                print(f"   Confidence: {response.confidence}")
                print(f"   Explanation: {response.explanation}")
            else:
                print("❌ Query returned no response")
            
            # Test simulation
            print("🎯 Testing simulation operation...")
            simulation = DTSimulation(
                simulation_type="forecast",
                actions=[{"action_type": "ADD_SERVER", "params": {"count": 1}}],
                horizon_minutes=30
            )
            
            sim_response = await client.simulate(simulation)
            if sim_response:
                print(f"✅ Simulation successful: {sim_response.success}")
                print(f"   Confidence: {sim_response.confidence}")
                if hasattr(sim_response, 'future_states') and sim_response.future_states:
                    print(f"   Future states: {len(sim_response.future_states)}")
            else:
                print("❌ Simulation returned no response")
            
            # Test diagnosis
            print("🩺 Testing diagnosis operation...")
            diagnosis = DTDiagnosis(
                anomaly_description="Test anomaly for diagnosis",
                context={"test": "true"}
            )
            
            diag_response = await client.diagnose(diagnosis)
            if diag_response:
                print(f"✅ Diagnosis successful: {diag_response.success}")
                print(f"   Confidence: {diag_response.confidence}")
                if hasattr(diag_response, 'hypotheses') and diag_response.hypotheses:
                    print(f"   Hypotheses: {len(diag_response.hypotheses)}")
            else:
                print("❌ Diagnosis returned no response")
            
            await client.disconnect()
            print(f"✅ Successfully tested {address}")
            return True
            
        except Exception as e:
            print(f"❌ Connection to {address} failed: {e}")
            try:
                await client.disconnect()
            except:
                pass
    
    print("\n❌ All connection attempts failed")
    return False


async def test_agentic_reasoner_dt_integration():
    """Test the agentic reasoner's digital twin integration."""
    print("\n🤖 Testing Agentic Reasoner Digital Twin Integration")
    print("=" * 60)
    
    from polaris.agents.agentic_reasoner import DigitalTwinTool
    
    logger = logging.getLogger("agentic_test")
    
    # Create digital twin client
    dt_client = GRPCDigitalTwinClient("localhost:50051", logger)
    
    try:
        await dt_client.connect()
        print("✅ Digital Twin client connected")
        
        # Create digital twin tool
        dt_tool = DigitalTwinTool(dt_client, logger)
        print("✅ Digital Twin tool created")
        
        # Test query operation
        print("\n🔍 Testing tool query operation...")
        result = await dt_tool.execute(
            operation="query",
            query_type="current_state",
            query_content="Get system overview for agentic reasoner"
        )
        
        print(f"Query result: {result}")
        
        if result.get("success"):
            print("✅ Tool query operation successful")
        else:
            print(f"❌ Tool query operation failed: {result.get('error')}")
        
        # Test simulation operation
        print("\n🎯 Testing tool simulation operation...")
        sim_result = await dt_tool.execute(
            operation="simulate",
            simulation_type="forecast",
            actions=[{"action_type": "ADD_SERVER", "params": {"count": 1}}],
            horizon_minutes=15
        )
        
        print(f"Simulation result: {sim_result}")
        
        if sim_result.get("success"):
            print("✅ Tool simulation operation successful")
        else:
            print(f"❌ Tool simulation operation failed: {sim_result.get('error')}")
        
        await dt_client.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Agentic reasoner DT integration test failed: {e}")
        try:
            await dt_client.disconnect()
        except:
            pass
        return False


async def main():
    """Main test function."""
    print("🚀 Digital Twin Connection Diagnostic")
    print("=" * 50)
    print("This script tests the connection between the agentic reasoner")
    print("and the digital twin to help diagnose issues.")
    print("")
    print("Prerequisites:")
    print("1. Digital Twin should be running: python src/scripts/start_component.py digital-twin")
    print("2. NATS server should be running: nats-server")
    print("")
    
    # Test basic connection
    connection_success = await test_digital_twin_connection()
    
    if connection_success:
        # Test agentic reasoner integration
        integration_success = await test_agentic_reasoner_dt_integration()
        
        if integration_success:
            print("\n🎉 All tests passed! Digital Twin integration is working correctly.")
            return True
        else:
            print("\n💥 Integration tests failed. Check the agentic reasoner implementation.")
            return False
    else:
        print("\n💥 Connection tests failed. Check if Digital Twin is running and accessible.")
        print("\nTroubleshooting steps:")
        print("1. Ensure Digital Twin is running: python src/scripts/start_component.py digital-twin")
        print("2. Check if port 50051 is available and not blocked by firewall")
        print("3. Verify Digital Twin configuration in polaris_config.yaml")
        print("4. Check Digital Twin logs for errors")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)