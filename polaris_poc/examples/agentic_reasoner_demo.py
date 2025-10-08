#!/usr/bin/env python3
"""
Agentic Reasoner Demo Script

This script demonstrates the POLARIS Agentic Reasoner capabilities
by sending various telemetry scenarios and showing how the agent
autonomously decides which tools to use and what actions to take.
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.nats_client import NATSClient
from polaris.agents.reasoner_core import ReasoningContext, ReasoningType


class AgenticReasonerDemo:
    """Demo class for testing agentic reasoner functionality."""
    
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
            name="agentic-reasoner-demo"
        )
        
        await self.nats_client.connect()
        self.logger.info("Agentic reasoner demo started")
    
    async def stop(self):
        """Stop the demo and close connections."""
        if self.nats_client:
            await self.nats_client.close()
        self.logger.info("Agentic reasoner demo stopped")
    
    async def send_reasoning_request(
        self, 
        scenario_name: str,
        telemetry_data: dict,
        reasoning_type: str = "decision",
        timeout_sec: float = 60.0
    ) -> dict:
        """Send a reasoning request to the agentic reasoner."""
        request_id = str(uuid.uuid4())
        
        request = {
            "session_id": request_id,
            "reasoning_type": reasoning_type,
            "timestamp": time.time(),
            "scenario": scenario_name,
            **telemetry_data
        }
        
        self.logger.info(f"🧠 Sending reasoning request for scenario: {scenario_name}")
        self.logger.info(f"📊 Telemetry data: {json.dumps(telemetry_data, indent=2)}")
        
        try:
            response = await self.nats_client.request_json(
                "polaris.reasoner.kernel.requests",
                request,
                timeout=timeout_sec
            )
            
            self.results[request_id] = response
            self.logger.info(f"✅ Received response for {scenario_name}")
            self.logger.info(f"🎯 Action: {response.get('result', {}).get('result', {}).get('action_type', 'Unknown')}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Request failed for {scenario_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def demo_high_response_time_scenario(self):
        """Demo scenario: High response time requiring immediate action."""
        self.logger.info("\n" + "="*60)
        self.logger.info("🚨 SCENARIO 1: High Response Time Crisis")
        self.logger.info("="*60)
        
        telemetry_data = {
            "events": [
                {"name": "average_response_time", "value": 2.5},  # 2.5 seconds - very high!
                {"name": "utilization", "value": 0.95},
                {"name": "active_servers", "value": 2},
                {"name": "max_servers", "value": 10},
                {"name": "dimmer", "value": 1.0},
                {"name": "arrival_rate", "value": 150},
                {"name": "basic_throughput", "value": 45},
                {"name": "optional_throughput", "value": 30}
            ],
            "metadata": {
                "description": "System is experiencing severe performance degradation",
                "severity": "critical"
            }
        }
        
        response = await self.send_reasoning_request(
            "high_response_time_crisis",
            telemetry_data
        )
        
        self.logger.info("🤔 Expected behavior: Agent should query KB for trends, possibly check DT for predictions, then decide to ADD_SERVER")
        return response
    
    async def demo_underutilized_system_scenario(self):
        """Demo scenario: System is underutilized, should consider scaling down."""
        self.logger.info("\n" + "="*60)
        self.logger.info("💰 SCENARIO 2: Underutilized System - Cost Optimization")
        self.logger.info("="*60)
        
        telemetry_data = {
            "events": [
                {"name": "average_response_time", "value": 0.3},  # Very good response time
                {"name": "utilization", "value": 0.25},  # Very low utilization
                {"name": "active_servers", "value": 4},
                {"name": "max_servers", "value": 10},
                {"name": "dimmer", "value": 0.8},
                {"name": "arrival_rate", "value": 50},
                {"name": "basic_throughput", "value": 40},
                {"name": "optional_throughput", "value": 35}
            ],
            "metadata": {
                "description": "System is running efficiently but may be over-provisioned",
                "severity": "low"
            }
        }
        
        response = await self.send_reasoning_request(
            "underutilized_system",
            telemetry_data
        )
        
        self.logger.info("🤔 Expected behavior: Agent should analyze trends, check if it's safe to scale down, then decide to REMOVE_SERVER")
        return response
    
    async def demo_optimal_system_scenario(self):
        """Demo scenario: System is running optimally."""
        self.logger.info("\n" + "="*60)
        self.logger.info("✅ SCENARIO 3: Optimal System Performance")
        self.logger.info("="*60)
        
        telemetry_data = {
            "events": [
                {"name": "average_response_time", "value": 0.8},  # Good response time
                {"name": "utilization", "value": 0.65},  # Optimal utilization
                {"name": "active_servers", "value": 3},
                {"name": "max_servers", "value": 10},
                {"name": "dimmer", "value": 0.9},
                {"name": "arrival_rate", "value": 100},
                {"name": "basic_throughput", "value": 85},
                {"name": "optional_throughput", "value": 75}
            ],
            "metadata": {
                "description": "System is performing within optimal parameters",
                "severity": "normal"
            }
        }
        
        response = await self.send_reasoning_request(
            "optimal_system",
            telemetry_data
        )
        
        self.logger.info("🤔 Expected behavior: Agent should verify system health, possibly check trends, then decide NO_ACTION")
        return response
    
    async def demo_borderline_scenario(self):
        """Demo scenario: System is on the borderline, requiring careful analysis."""
        self.logger.info("\n" + "="*60)
        self.logger.info("⚖️  SCENARIO 4: Borderline Performance - Requires Deep Analysis")
        self.logger.info("="*60)
        
        telemetry_data = {
            "events": [
                {"name": "average_response_time", "value": 0.95},  # Close to threshold
                {"name": "utilization", "value": 0.82},  # High but not critical
                {"name": "active_servers", "value": 3},
                {"name": "max_servers", "value": 10},
                {"name": "dimmer", "value": 1.0},
                {"name": "arrival_rate", "value": 120},
                {"name": "basic_throughput", "value": 70},
                {"name": "optional_throughput", "value": 65}
            ],
            "metadata": {
                "description": "System metrics are approaching thresholds - requires careful analysis",
                "severity": "warning"
            }
        }
        
        response = await self.send_reasoning_request(
            "borderline_performance",
            telemetry_data
        )
        
        self.logger.info("🤔 Expected behavior: Agent should use multiple tools to analyze trends, check DT predictions, then make informed decision")
        return response
    
    async def demo_anomaly_scenario(self):
        """Demo scenario: Unusual system behavior requiring diagnosis."""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔍 SCENARIO 5: System Anomaly - Diagnostic Required")
        self.logger.info("="*60)
        
        telemetry_data = {
            "events": [
                {"name": "average_response_time", "value": 1.2},  # High response time
                {"name": "utilization", "value": 0.45},  # But low utilization - anomaly!
                {"name": "active_servers", "value": 3},
                {"name": "max_servers", "value": 10},
                {"name": "dimmer", "value": 0.9},
                {"name": "arrival_rate", "value": 80},
                {"name": "basic_throughput", "value": 25},  # Very low throughput
                {"name": "optional_throughput", "value": 20}
            ],
            "metadata": {
                "description": "High response time with low utilization - possible system issue",
                "severity": "warning",
                "anomaly_detected": True
            }
        }
        
        response = await self.send_reasoning_request(
            "system_anomaly",
            telemetry_data
        )
        
        self.logger.info("🤔 Expected behavior: Agent should use DT diagnostic tools, analyze historical patterns, then decide on corrective action")
        return response
    
    async def run_all_demos(self):
        """Run all demo scenarios."""
        self.logger.info("🚀 Starting Agentic Reasoner Demo Suite")
        self.logger.info("This demo shows how the agentic reasoner autonomously uses tools to make decisions")
        
        scenarios = [
            self.demo_high_response_time_scenario,
            self.demo_underutilized_system_scenario,
            self.demo_optimal_system_scenario,
            self.demo_borderline_scenario,
            self.demo_anomaly_scenario,
        ]
        
        results = []
        for i, scenario in enumerate(scenarios, 1):
            self.logger.info(f"\n🎬 Running scenario {i}/{len(scenarios)}")
            try:
                result = await scenario()
                results.append(result)
                
                # Wait between scenarios
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"❌ Scenario {i} failed: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 DEMO SUMMARY")
        self.logger.info("="*60)
        
        successful = sum(1 for r in results if r.get("success", False))
        self.logger.info(f"✅ Successful scenarios: {successful}/{len(results)}")
        
        for i, result in enumerate(results, 1):
            if result.get("success"):
                action = result.get("result", {}).get("result", {}).get("action_type", "Unknown")
                self.logger.info(f"   Scenario {i}: {action}")
            else:
                self.logger.info(f"   Scenario {i}: FAILED - {result.get('error', 'Unknown error')}")
        
        self.logger.info("\n🎉 Demo completed! Check the agentic reasoner logs to see tool usage patterns.")
        
        return results


async def main():
    """Main demo function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 POLARIS Agentic Reasoner Demo")
    logger.info("=" * 50)
    logger.info("This demo showcases the autonomous decision-making capabilities")
    logger.info("of the agentic reasoner, including dynamic tool usage.")
    logger.info("")
    logger.info("Prerequisites:")
    logger.info("1. Start NATS server: nats-server")
    logger.info("2. Start Knowledge Base: python start_component.py knowledge-base")
    logger.info("3. Start Digital Twin: python start_component.py digital-twin")
    logger.info("4. Start Agentic Reasoner: python start_component.py agentic-reasoner")
    logger.info("")
    
    demo = AgenticReasonerDemo()
    
    try:
        await demo.start()
        await demo.run_all_demos()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        await demo.stop()


if __name__ == "__main__":
    asyncio.run(main())