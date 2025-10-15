#!/usr/bin/env python3
"""
Production Usage Example for Improved POLARIS Components

This example shows how to use the improved GRPC client and Bayesian world model
in a production environment with proper error handling and monitoring.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polaris.agents.agentic_reasoner import (
    create_agentic_reasoner_agent,
    create_agentic_reasoner_with_bayesian_world_model
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def production_example_with_improved_grpc():
    """Example using improved GRPC client for better reliability."""
    logger.info("=== Production Example: Improved GRPC Client ===")
    
    # Custom timeout configuration for production
    grpc_timeout_config = {
        'query_timeout': 30.0,        # Longer timeout for complex queries
        'simulation_timeout': 180.0,   # 3 minutes for complex simulations
        'diagnosis_timeout': 60.0,     # 1 minute for diagnosis
        'connection_timeout': 20.0,    # 20 seconds to establish connection
        'default_timeout': 45.0        # Default for other operations
    }
    
    try:
        # Create reasoner with improved GRPC client
        agent = create_agentic_reasoner_agent(
            agent_id="production-reasoner-001",
            config_path="config/polaris_config.yaml",  # Your production config
            llm_api_key=os.getenv("GEMINI_API_KEY", "your-api-key"),
            use_improved_grpc=True,
            grpc_timeout_config=grpc_timeout_config
        )
        
        logger.info("✓ Agentic reasoner created with improved GRPC client")
        
        # Start the agent
        await agent.start()
        logger.info("✓ Agent started successfully")
        
        # Example reasoning context
        from polaris.agents.reasoner_core import ReasoningContext, ReasoningType
        
        context = ReasoningContext(
            session_id="prod-session-001",
            timestamp=datetime.now(timezone.utc).timestamp(),
            input_data={
                "system_metrics": {
                    "cpu_utilization": 85.0,
                    "memory_usage": 78.0,
                    "response_time": 950.0,
                    "error_rate": 0.02
                },
                "alert": "High CPU utilization detected",
                "severity": "medium"
            },
            reasoning_type=ReasoningType.DECISION
        )
        
        # Perform reasoning with automatic tool usage
        logger.info("Performing reasoning with tool usage...")
        result = await agent.reason(context)
        
        if result:
            logger.info("✓ Reasoning completed successfully")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Execution time: {result.execution_time:.2f}s")
            logger.info(f"  KB queries made: {result.kb_queries_made}")
            logger.info(f"  DT queries made: {result.dt_queries_made}")
            logger.info(f"  Action: {result.result.get('action_type', 'N/A')}")
        else:
            logger.warning("✗ Reasoning failed")
        
        # Get GRPC client metrics if available
        if hasattr(agent, 'dt_query') and hasattr(agent.dt_query, 'get_connection_metrics'):
            metrics = agent.dt_query.get_connection_metrics()
            logger.info("GRPC Client Metrics:")
            logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
            logger.info(f"  Average Response Time: {metrics['average_response_time_sec']:.3f}s")
            logger.info(f"  Circuit Breaker State: {metrics['circuit_breaker_state']}")
        
    except Exception as e:
        logger.error(f"Production example failed: {e}")
    
    finally:
        try:
            await agent.stop()
            logger.info("Agent stopped gracefully")
        except:
            pass


async def production_example_with_bayesian_world_model():
    """Example using Bayesian world model for deterministic predictions."""
    logger.info("=== Production Example: Bayesian World Model ===")
    
    try:
        # Create reasoner with Bayesian world model
        agent = create_agentic_reasoner_with_bayesian_world_model(
            agent_id="bayesian-reasoner-001",
            config_path="config/bayesian_world_model_config.yaml",
            llm_api_key=os.getenv("GEMINI_API_KEY", "your-api-key")
        )
        
        logger.info("✓ Agentic reasoner created with Bayesian world model")
        
        # Start the agent
        await agent.start()
        logger.info("✓ Agent started successfully")
        
        # Simulate some system monitoring data
        logger.info("Simulating system monitoring data...")
        
        # In production, this would come from your monitoring system
        monitoring_data = [
            {"metric": "cpu_utilization", "value": 65.0, "timestamp": datetime.now(timezone.utc)},
            {"metric": "memory_usage", "value": 70.0, "timestamp": datetime.now(timezone.utc)},
            {"metric": "response_time", "value": 250.0, "timestamp": datetime.now(timezone.utc)},
            {"metric": "throughput", "value": 1000.0, "timestamp": datetime.now(timezone.utc)},
        ]
        
        # Feed data to the world model (this would be automatic in production)
        from polaris.models.digital_twin_events import KnowledgeEvent
        
        for data in monitoring_data:
            event = KnowledgeEvent(
                source="production_monitor",
                event_type="telemetry",
                data={
                    "name": data["metric"],
                    "value": data["value"],
                    "timestamp": data["timestamp"].isoformat()
                },
                timestamp=data["timestamp"].isoformat()
            )
            
            # In production, this would be sent via NATS
            # Here we simulate direct world model update
            if hasattr(agent, 'world_model'):
                await agent.world_model.update_state(event)
        
        logger.info("✓ Monitoring data processed")
        
        # Example reasoning for capacity planning
        from polaris.agents.reasoner_core import ReasoningContext, ReasoningType
        
        context = ReasoningContext(
            session_id="capacity-planning-001",
            timestamp=datetime.now(timezone.utc).timestamp(),
            input_data={
                "scenario": "capacity_planning",
                "current_load": 75.0,
                "projected_growth": 0.15,  # 15% growth expected
                "time_horizon": "next_hour",
                "business_constraints": {
                    "max_response_time": 1000.0,
                    "cost_optimization": True
                }
            },
            reasoning_type=ReasoningType.PLANNING
        )
        
        # Perform reasoning
        logger.info("Performing capacity planning reasoning...")
        result = await agent.reason(context)
        
        if result:
            logger.info("✓ Capacity planning completed")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Recommendation: {result.result.get('action_type', 'N/A')}")
            logger.info(f"  Reasoning: {result.result.get('reasoning', 'N/A')}")
        else:
            logger.warning("✗ Capacity planning failed")
        
        # Get world model health status
        if hasattr(agent, 'world_model'):
            health = await agent.world_model.get_health_status()
            logger.info("Bayesian World Model Health:")
            logger.info(f"  Status: {health['status']}")
            logger.info(f"  Metrics Tracked: {health['metrics']['metrics_tracked']}")
            logger.info(f"  System Health Score: {health['metrics']['system_health_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Bayesian world model example failed: {e}")
    
    finally:
        try:
            await agent.stop()
            logger.info("Agent stopped gracefully")
        except:
            pass


async def monitoring_and_alerting_example():
    """Example of monitoring both GRPC client and world model health."""
    logger.info("=== Monitoring and Alerting Example ===")
    
    # This would typically run as a separate monitoring service
    async def monitor_agent_health(agent, check_interval=30):
        """Monitor agent health and alert on issues."""
        while True:
            try:
                # Check GRPC client health
                if hasattr(agent, 'dt_query') and hasattr(agent.dt_query, 'get_connection_metrics'):
                    grpc_metrics = agent.dt_query.get_connection_metrics()
                    
                    # Alert on high failure rate
                    if grpc_metrics['failure_rate'] > 0.1:  # 10% failure rate
                        logger.warning(f"🚨 High GRPC failure rate: {grpc_metrics['failure_rate']:.2%}")
                    
                    # Alert on circuit breaker open
                    if grpc_metrics['circuit_breaker_state'] == 'open':
                        logger.error("🚨 GRPC Circuit breaker is OPEN - service degraded")
                    
                    # Alert on slow responses
                    if grpc_metrics['average_response_time_sec'] > 5.0:
                        logger.warning(f"🚨 Slow GRPC responses: {grpc_metrics['average_response_time_sec']:.2f}s")
                
                # Check world model health
                if hasattr(agent, 'world_model'):
                    health = await agent.world_model.get_health_status()
                    
                    # Alert on low system health
                    if health['metrics']['system_health_score'] < 0.7:
                        logger.warning(f"🚨 Low system health score: {health['metrics']['system_health_score']:.2f}")
                    
                    # Alert on no recent data
                    if health['metrics']['metrics_tracked'] == 0:
                        logger.error("🚨 No metrics being tracked - data pipeline issue")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    # Example: Start monitoring (in production, this would be a separate service)
    logger.info("Health monitoring would run continuously in production")
    logger.info("Key metrics to monitor:")
    logger.info("  - GRPC failure rate < 10%")
    logger.info("  - Circuit breaker state = closed")
    logger.info("  - Response times < 5s")
    logger.info("  - System health score > 0.7")
    logger.info("  - Active metric tracking")


async def main():
    """Run production examples."""
    logger.info("POLARIS Production Usage Examples")
    logger.info("=" * 60)
    
    # Example 1: Improved GRPC Client
    await production_example_with_improved_grpc()
    
    logger.info("\n" + "=" * 60)
    
    # Example 2: Bayesian World Model
    await production_example_with_bayesian_world_model()
    
    logger.info("\n" + "=" * 60)
    
    # Example 3: Monitoring
    await monitoring_and_alerting_example()
    
    logger.info("\n" + "=" * 60)
    logger.info("Production examples completed!")
    
    logger.info("\n📋 Production Deployment Checklist:")
    logger.info("  ✓ Configure appropriate timeouts for your environment")
    logger.info("  ✓ Set up monitoring for GRPC metrics and world model health")
    logger.info("  ✓ Configure circuit breaker thresholds based on SLA requirements")
    logger.info("  ✓ Set up alerting for high failure rates and degraded performance")
    logger.info("  ✓ Choose world model implementation based on requirements:")
    logger.info("    - Gemini LLM: Creative, flexible reasoning")
    logger.info("    - Bayesian: Deterministic, fast, mathematically rigorous")
    logger.info("  ✓ Test failover scenarios and recovery procedures")


if __name__ == "__main__":
    asyncio.run(main())