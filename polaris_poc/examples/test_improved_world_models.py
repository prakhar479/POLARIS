#!/usr/bin/env python3
"""
Test script for improved world models and GRPC client.

This script demonstrates the usage of both the improved GRPC client
and the new Bayesian/Kalman filter world model.
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime, timezone

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polaris.agents.improved_grpc_client import ImprovedGRPCDigitalTwinClient
from polaris.agents.reasoner_agent import DTQuery, DTSimulation, DTDiagnosis
from polaris.models.bayesian_world_model import BayesianWorldModel
from polaris.models.digital_twin_events import KnowledgeEvent
from polaris.models.world_model import QueryRequest, SimulationRequest, DiagnosisRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_improved_grpc_client():
    """Test the improved GRPC client with timeout handling."""
    logger.info("Testing Improved GRPC Client...")
    
    client = ImprovedGRPCDigitalTwinClient(
        grpc_address="localhost:50051",
        logger=logger,
        query_timeout=10.0,
        simulation_timeout=30.0,
        diagnosis_timeout=20.0,
        max_retries=2,
        circuit_breaker_enabled=True
    )
    
    try:
        # Test connection
        logger.info("Attempting to connect to Digital Twin...")
        await client.connect()
        logger.info("✓ Connection successful")
        
        # Test query
        query = DTQuery(
            query_type="current_state",
            query_content="Get system overview"
        )
        
        logger.info("Testing query operation...")
        start_time = time.time()
        response = await client.query(query)
        query_time = time.time() - start_time
        
        if response:
            logger.info(f"✓ Query successful in {query_time:.2f}s")
            logger.info(f"  Confidence: {response.confidence}")
            logger.info(f"  Success: {response.success}")
        else:
            logger.warning("✗ Query failed or timed out")
        
        # Test simulation
        simulation = DTSimulation(
            simulation_type="forecast",
            actions=[
                {"action_type": "ADD_SERVER", "params": {"count": 1}},
                {"action_type": "SET_DIMMER", "params": {"value": 0.8}}
            ],
            horizon_minutes=30
        )
        
        logger.info("Testing simulation operation...")
        start_time = time.time()
        sim_response = await client.simulate(simulation)
        sim_time = time.time() - start_time
        
        if sim_response:
            logger.info(f"✓ Simulation successful in {sim_time:.2f}s")
            logger.info(f"  Future states: {len(sim_response.future_states or [])}")
        else:
            logger.warning("✗ Simulation failed or timed out")
        
        # Get connection metrics
        metrics = client.get_connection_metrics()
        logger.info("Connection Metrics:")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"  Total Requests: {metrics['total_requests']}")
        logger.info(f"  Failed Requests: {metrics['failed_requests']}")
        logger.info(f"  Average Response Time: {metrics['average_response_time_sec']:.3f}s")
        logger.info(f"  Circuit Breaker State: {metrics['circuit_breaker_state']}")
        
    except Exception as e:
        logger.error(f"GRPC client test failed: {e}")
    
    finally:
        await client.disconnect()
        logger.info("Disconnected from Digital Twin")


async def test_bayesian_world_model():
    """Test the Bayesian/Kalman filter world model."""
    logger.info("Testing Bayesian World Model...")
    
    config = {
        'prediction_horizon_minutes': 60,
        'max_history_points': 1000,
        'correlation_threshold': 0.7,
        'anomaly_threshold': 2.0,
        'process_noise': 0.01,
        'measurement_noise': 0.1,
        'learning_rate': 0.1
    }
    
    world_model = BayesianWorldModel(config, logger)
    
    try:
        # Initialize the model
        logger.info("Initializing Bayesian World Model...")
        await world_model.initialize()
        logger.info("✓ Model initialized successfully")
        
        # Simulate some telemetry data
        logger.info("Feeding sample telemetry data...")
        
        # Create mock telemetry events
        metrics = ['cpu_utilization', 'memory_usage', 'response_time', 'throughput']
        base_values = [65.0, 70.0, 250.0, 1000.0]
        
        for i in range(20):  # 20 data points
            for j, metric_name in enumerate(metrics):
                # Add some noise and trend
                noise = (i % 3 - 1) * 5  # Simple noise pattern
                trend = i * 0.5  # Slight upward trend
                value = base_values[j] + noise + trend
                
                # Create mock telemetry data as a simple dict
                telemetry_data = {
                    "name": metric_name,
                    "value": value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "test_monitor"
                }
                
                event = KnowledgeEvent(
                    source="test_monitor",
                    event_type="telemetry",
                    data=telemetry_data,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
                await world_model.update_state(event)
            
            # Small delay to simulate real-time data
            await asyncio.sleep(0.1)
        
        logger.info("✓ Sample data ingested")
        
        # Test current state query
        logger.info("Testing current state query...")
        query_request = QueryRequest(
            query_id="test-query-001",
            query_type="current_state",
            query_content="Get current system state with confidence intervals"
        )
        
        response = await world_model.query_state(query_request)
        
        if response.success:
            logger.info("✓ Current state query successful")
            logger.info(f"  Confidence: {response.confidence:.2f}")
            logger.info(f"  Metrics analyzed: {response.metadata.get('metrics_analyzed', 0)}")
            
            # Show some results
            if isinstance(response.result, dict):
                for metric, data in list(response.result.items())[:2]:  # Show first 2 metrics
                    logger.info(f"  {metric}:")
                    logger.info(f"    Value: {data.get('value', 'N/A'):.2f}")
                    logger.info(f"    Confidence: {data.get('confidence', 'N/A'):.2f}")
                    logger.info(f"    Trend: {data.get('trend', 'N/A'):.3f}")
        else:
            logger.warning("✗ Current state query failed")
        
        # Test prediction simulation
        logger.info("Testing prediction simulation...")
        sim_request = SimulationRequest(
            simulation_id="test-sim-001",
            simulation_type="forecast",
            actions=[],  # No actions, just forecast
            horizon_minutes=30
        )
        
        sim_response = await world_model.simulate(sim_request)
        
        if sim_response.success:
            logger.info("✓ Prediction simulation successful")
            logger.info(f"  Confidence: {sim_response.confidence:.2f}")
            logger.info(f"  Future states: {len(sim_response.future_states)}")
            logger.info(f"  Uncertainty range: [{sim_response.uncertainty_lower:.2f}, {sim_response.uncertainty_upper:.2f}]")
        else:
            logger.warning("✗ Prediction simulation failed")
        
        # Test correlation query
        logger.info("Testing correlation analysis...")
        corr_request = QueryRequest(
            query_id="test-corr-001",
            query_type="correlation",
            query_content="Get metric correlations"
        )
        
        corr_response = await world_model.query_state(corr_request)
        
        if corr_response.success and isinstance(corr_response.result, dict):
            logger.info("✓ Correlation analysis successful")
            logger.info(f"  Correlations found: {len(corr_response.result)}")
            
            # Show strongest correlations
            correlations = [(k, v) for k, v in corr_response.result.items()]
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for pair, corr in correlations[:3]:  # Show top 3
                logger.info(f"    {pair}: {corr:.3f}")
        else:
            logger.info("No significant correlations found yet (need more data)")
        
        # Get health status
        health = await world_model.get_health_status()
        logger.info("Model Health Status:")
        logger.info(f"  Status: {health['status']}")
        logger.info(f"  Metrics Tracked: {health['metrics']['metrics_tracked']}")
        logger.info(f"  System Health Score: {health['metrics']['system_health_score']:.2f}")
        logger.info(f"  Background Tasks: {health['metrics']['background_tasks_running']}")
        
    except Exception as e:
        logger.error(f"Bayesian world model test failed: {e}")
    
    finally:
        await world_model.shutdown()
        logger.info("Bayesian World Model shutdown complete")


async def main():
    """Run all tests."""
    logger.info("Starting World Model and GRPC Client Tests")
    logger.info("=" * 60)
    
    # Test Bayesian World Model (always works)
    await test_bayesian_world_model()
    
    logger.info("=" * 60)
    
    # Test GRPC Client (may fail if Digital Twin not running)
    logger.info("Note: GRPC client test requires Digital Twin agent to be running")
    logger.info("If it fails, that's expected when Digital Twin is not available")
    
    await test_improved_grpc_client()
    
    logger.info("=" * 60)
    logger.info("Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())