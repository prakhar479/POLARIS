#!/usr/bin/env python3
"""
Test suite for enhanced state tracking functionality in Digital Twin.
"""

import asyncio
import json
import dotenv
import pytest
import pytest_asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.models.gemini_world_model import GeminiWorldModel
from polaris.models.digital_twin_events import KnowledgeEvent
from polaris.models.world_model import QueryRequest
from polaris.models.telemetry import TelemetryEvent

dotenv.load_dotenv(
    Path("/home/vyakhya/Desktop/serc/self_adapt/POLARIS/polaris_poc/.env")
)


class TestEnhancedStateTracking:
    """Test suite for enhanced state tracking functionality."""

    @pytest_asyncio.fixture
    async def world_model(self):
        """Create and initialize a test world model."""
        config = {
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemini-1.5-pro",
            "temperature": 0.1,
            "max_tokens": 2000,
            "concurrent_requests": 5,
            "max_history_events": 1000,
            "retention_days": 30,
            "compression_threshold": 1000,
            "enable_causal_tracking": True,
            "min_confidence_threshold": 0.3,
            "uncertainty_estimation": True,
            "calibration_window_hours": 24,
        }

        model = GeminiWorldModel(config)
        await model.initialize()
        yield model
        await model.shutdown()

    def create_test_telemetry_events(self) -> List[Tuple[str, float]]:
        """Create a series of test telemetry events with various patterns."""
        return [
            ("cpu_usage", 45.2),  # Normal baseline
            ("memory_usage", 67.8),  # Normal baseline
            ("response_time", 120.5),  # Normal baseline
            ("cpu_usage", 48.1),  # Slight increase
            ("memory_usage", 69.2),  # Slight increase
            ("response_time", 115.3),  # Improvement
            ("cpu_usage", 85.7),  # Significant jump (potential anomaly)
            ("memory_usage", 71.1),  # Continued increase
            ("response_time", 95.2),  # Further improvement
            ("cpu_usage", 87.3),  # High but stable
            ("memory_usage", 72.5),  # Gradual increase
            ("response_time", 92.1),  # Stable improvement
        ]

    async def process_test_events(
        self, world_model: GeminiWorldModel, events: List[Tuple[str, float]]
    ) -> None:
        """Process a series of test events."""
        for i, (metric_name, value) in enumerate(events):
            event = KnowledgeEvent(
                event_id=f"test_event_{i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="test_monitor",
                event_type="telemetry",
                data=TelemetryEvent(name=metric_name, value=value),
            )
            await world_model.update_state(event)

    @pytest.mark.asyncio
    async def test_enhanced_telemetry_processing(self, world_model):
        """Test enhanced telemetry event processing with LLM reasoning."""
        # Process test events
        events = self.create_test_telemetry_events()
        await self.process_test_events(world_model, events)

        # Verify state tracking structures are populated
        assert (
            len(world_model._current_system_state) > 0
        ), "Current state should be populated"
        assert (
            len(world_model._system_state_history) > 0
        ), "State history should be populated"
        assert (
            len(world_model._state_embeddings) > 0
        ), "State embeddings should be generated"
        assert (
            len(world_model._temporal_state_index) > 0
        ), "Temporal index should be populated"

        # Verify metadata is present in current state
        for metric_name, state_entry in world_model._current_system_state.items():
            assert "metadata" in state_entry, f"Metadata missing for {metric_name}"
            metadata = state_entry["metadata"]
            assert "confidence" in metadata, f"Confidence missing for {metric_name}"
            assert (
                "anomaly_score" in metadata
            ), f"Anomaly score missing for {metric_name}"
            assert "trend" in metadata, f"Trend missing for {metric_name}"

            # Verify confidence is in valid range
            assert (
                0.0 <= metadata["confidence"] <= 1.0
            ), f"Invalid confidence for {metric_name}"
            assert (
                0.0 <= metadata["anomaly_score"] <= 1.0
            ), f"Invalid anomaly score for {metric_name}"

    @pytest.mark.asyncio
    async def test_state_consistency_validation(self, world_model):
        """Test state consistency validation functionality."""
        # Process events with some inconsistencies
        events = [
            ("cpu_usage", 50.0),
            ("cpu_usage", 150.0),  # Invalid value (>100%)
            ("cpu_usage", 25.0),  # Large drop
            ("memory_usage", -10.0),  # Invalid negative value
            ("memory_usage", 75.0),
        ]

        await self.process_test_events(world_model, events)

        # Verify consistency checks were performed
        assert (
            len(world_model._state_consistency_log) > 0
        ), "Consistency checks should be logged"

        # Check for detected issues
        consistency_issues = [
            check
            for check in world_model._state_consistency_log
            if check.get("consistency_score", 1.0) < 0.8
        ]

        assert len(consistency_issues) > 0, "Should detect consistency issues"

        # Verify issue types are recorded
        for issue in consistency_issues:
            assert "issues" in issue, "Issues should be recorded"
            assert len(issue["issues"]) > 0, "Specific issues should be identified"

    @pytest.mark.asyncio
    async def test_vector_embedding_generation(self, world_model):
        """Test vector embedding generation and storage."""
        # Process some events
        events = self.create_test_telemetry_events()[:6]  # First 6 events
        await self.process_test_events(world_model, events)

        # Verify embeddings are generated
        assert len(world_model._state_embeddings) > 0, "Embeddings should be generated"

        # Verify embedding structure
        for embedding_key, embedding_data in world_model._state_embeddings.items():
            assert (
                "embedding" in embedding_data
            ), f"Embedding vector missing for {embedding_key}"
            assert (
                "metric_name" in embedding_data
            ), f"Metric name missing for {embedding_key}"
            assert (
                "timestamp" in embedding_data
            ), f"Timestamp missing for {embedding_key}"
            assert "metadata" in embedding_data, f"Metadata missing for {embedding_key}"

            # Verify embedding is a list of floats
            embedding = embedding_data["embedding"]
            assert isinstance(
                embedding, list
            ), f"Embedding should be a list for {embedding_key}"
            assert (
                len(embedding) == 8
            ), f"Embedding should have fixed size for {embedding_key}"
            assert all(
                isinstance(x, float) for x in embedding
            ), f"Embedding should contain floats for {embedding_key}"

    @pytest.mark.asyncio
    async def test_temporal_indexing(self, world_model):
        """Test temporal indexing for historical queries."""
        # Process events with specific timestamps
        base_time = datetime.now(timezone.utc)
        events = [("cpu_usage", 50.0), ("cpu_usage", 55.0), ("memory_usage", 60.0)]

        for i, (metric_name, value) in enumerate(events):
            # Create events with different timestamps
            timestamp = (base_time - timedelta(hours=i)).isoformat()
            event = KnowledgeEvent(
                event_id=f"temporal_test_{i}",
                timestamp=timestamp,
                source="test_monitor",
                event_type="telemetry",
                data=TelemetryEvent(name=metric_name, value=value),
            )
            await world_model.update_state(event)

        # Verify temporal index is populated
        assert (
            len(world_model._temporal_state_index) > 0
        ), "Temporal index should be populated"

        # Verify index structure
        for index_key, states in world_model._temporal_state_index.items():
            assert isinstance(
                states, list
            ), f"Index should contain list for {index_key}"
            for state in states:
                assert "value" in state, f"State should have value in index {index_key}"
                assert (
                    "timestamp" in state
                ), f"State should have timestamp in index {index_key}"

    @pytest.mark.asyncio
    async def test_enhanced_current_state_query(self, world_model):
        """Test enhanced current state queries with confidence scoring."""
        # Process some events
        events = self.create_test_telemetry_events()[:6]
        await self.process_test_events(world_model, events)

        # Test specific metric query
        query_request = QueryRequest(
            query_id="test_specific_metric",
            query_type="current_state",
            query_content="Get CPU usage with metadata",
            parameters={"metric": "cpu_usage"},
        )

        response = await world_model.query_state(query_request)

        assert response.success, "Specific metric query should succeed"
        assert response.confidence > 0.0, "Query should have confidence score"
        assert "cpu_usage" in response.result, "Result should contain CPU usage data"

        # Verify JSON structure for specific metric
        try:
            result_data = json.loads(response.result)
            assert "requested_metric" in result_data, "Result should have metric field"
            assert "value" in result_data, "Result should have value field"
            assert "confidence" in result_data, "Result should have confidence field"
            assert "anomaly_score" in result_data, "Result should have anomaly score"
            assert "trend" in result_data, "Result should have trend information"
        except json.JSONDecodeError:
            pytest.fail("Specific metric result should be valid JSON")

        # Test overall state query
        query_request = QueryRequest(
            query_id="test_overall_state",
            query_type="current_state",
            query_content="Get overall system state",
            parameters={},
        )

        response = await world_model.query_state(query_request)

        assert response.success, "Overall state query should succeed"
        assert response.confidence > 0.0, "Query should have confidence score"
        assert "System" in response.result, "Result should contain system state summary"

    @pytest.mark.asyncio
    async def test_enhanced_historical_query(self, world_model):
        """Test enhanced historical queries with vector embeddings."""
        # Process events to build history
        events = self.create_test_telemetry_events()
        await self.process_test_events(world_model, events)

        # Test recent historical query
        query_request = QueryRequest(
            query_id="test_recent_history",
            query_type="historical",
            query_content="Get recent CPU usage history",
            parameters={"timestamp": "recent", "metric": "cpu_usage"},
        )

        response = await world_model.query_state(query_request)

        assert response.success, "Historical query should succeed"
        assert response.confidence >= 0.0, "Query should have confidence score"
        assert response.explanation, "Query should have explanation"

        # Test specific timestamp query
        target_time = datetime.now(timezone.utc).isoformat()
        query_request = QueryRequest(
            query_id="test_timestamp_history",
            query_type="historical",
            query_content="Get historical state at specific time",
            parameters={"timestamp": target_time, "metric": "cpu_usage"},
        )

        response = await world_model.query_state(query_request)

        assert response.success, "Timestamp-based historical query should succeed"
        assert response.explanation, "Query should have explanation"

    @pytest.mark.asyncio
    async def test_llm_state_evolution_analysis(self, world_model):
        """Test LLM-based state evolution analysis."""
        # Process events with clear patterns
        events = [
            ("cpu_usage", 30.0),  # Low baseline
            ("cpu_usage", 35.0),  # Slight increase
            ("cpu_usage", 85.0),  # Sudden spike (should trigger analysis)
            ("cpu_usage", 90.0),  # Continued high usage
        ]

        await self.process_test_events(world_model, events)

        # Verify evolution patterns are tracked
        assert (
            len(world_model._state_evolution_patterns) > 0
        ), "Evolution patterns should be tracked"

        # Check for CPU usage evolution pattern
        cpu_pattern_key = "cpu.usage_evolution"
        assert (
            cpu_pattern_key in world_model._state_evolution_patterns
        ), "CPU evolution pattern should exist"

        cpu_patterns = world_model._state_evolution_patterns[cpu_pattern_key]
        assert len(cpu_patterns) > 0, "CPU patterns should be recorded"

        # Verify pattern structure
        for pattern in cpu_patterns:
            assert "timestamp" in pattern, "Pattern should have timestamp"
            assert "analysis" in pattern, "Pattern should have analysis"
            assert "context" in pattern, "Pattern should have context"

            analysis = pattern["analysis"]
            assert analysis, "Analysis should not be empty"

    @pytest.mark.asyncio
    async def test_enhanced_health_status(self, world_model):
        """Test enhanced health status reporting."""
        # Process some events to populate metrics
        events = self.create_test_telemetry_events()
        await self.process_test_events(world_model, events)

        # Get health status
        health_status = await world_model.get_health_status()

        # Verify basic structure
        assert "status" in health_status, "Health status should have status field"
        assert "metrics" in health_status, "Health status should have metrics"
        assert (
            "configuration" in health_status
        ), "Health status should have configuration"
        assert (
            "state_tracking_status" in health_status
        ), "Health status should have state tracking status"

        # Verify enhanced metrics
        metrics = health_status["metrics"]
        enhanced_metrics = [
            "current_state_metrics",
            "state_embeddings_stored",
            "temporal_index_keys",
            "evolution_patterns_tracked",
            "anomalies_detected",
            "low_confidence_metrics",
            "consistency_issues",
        ]

        for metric in enhanced_metrics:
            assert metric in metrics, f"Enhanced metric {metric} should be present"
            assert isinstance(
                metrics[metric], int
            ), f"Enhanced metric {metric} should be integer"

        # Verify configuration flags
        config = health_status["configuration"]
        enhanced_config = [
            "vector_embeddings_enabled",
            "state_consistency_validation",
            "temporal_indexing_enabled",
            "llm_state_analysis_enabled",
        ]

        for config_flag in enhanced_config:
            assert (
                config_flag in config
            ), f"Enhanced config {config_flag} should be present"
            assert isinstance(
                config[config_flag], bool
            ), f"Enhanced config {config_flag} should be boolean"

        # Verify state tracking status
        tracking_status = health_status["state_tracking_status"]
        status_fields = ["embeddings_health", "consistency_health", "anomaly_detection"]

        for field in status_fields:
            assert (
                field in tracking_status
            ), f"State tracking status {field} should be present"
            assert isinstance(
                tracking_status[field], str
            ), f"State tracking status {field} should be string"

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, world_model):
        """Test anomaly detection in telemetry processing."""
        # Process normal baseline events
        baseline_events = [
            ("cpu_usage", 45.0),
            ("cpu_usage", 47.0),
            ("cpu_usage", 46.0),
            ("cpu_usage", 48.0),
            ("cpu_usage", 44.0),
        ]

        await self.process_test_events(world_model, baseline_events)

        # Process anomalous event
        anomaly_event = KnowledgeEvent(
            event_id="anomaly_test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="test_monitor",
            event_type="telemetry",
            data=TelemetryEvent(name="cpu_usage", value=95.0),  # Significant deviation
        )

        await world_model.update_state(anomaly_event)

        # Check if anomaly was detected
        cpu_state = world_model._current_system_state.get("cpu.usage")
        assert cpu_state is not None, "CPU state should exist"

        anomaly_score = cpu_state["metadata"]["anomaly_score"]
        assert anomaly_score > 0.5, f"High anomaly score expected, got {anomaly_score}"

    @pytest.mark.asyncio
    async def test_confidence_scoring(self, world_model):
        """Test confidence scoring for telemetry data."""
        # Test with complete, valid data
        complete_event = KnowledgeEvent(
            event_id="complete_test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="reliable_monitor",
            event_type="telemetry",
            data=TelemetryEvent(name="cpu_usage", value=75.0),
        )

        await world_model.update_state(complete_event)

        cpu_state = world_model._current_system_state.get("cpu.usage")
        confidence = cpu_state["metadata"]["confidence"]

        assert (
            confidence > 0.8
        ), f"High confidence expected for complete data, got {confidence}"


# Integration test function for manual execution
async def run_integration_test():
    """Run comprehensive integration test for enhanced state tracking."""
    print("🧪 Running Enhanced State Tracking Integration Test...")

    # Create test instance
    test_instance = TestEnhancedStateTracking()

    # Create world model
    config = {
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 2000,
        "concurrent_requests": 5,
        "max_history_events": 1000,
        "retention_days": 30,
        "compression_threshold": 1000,
        "enable_causal_tracking": True,
        "min_confidence_threshold": 0.3,
        "uncertainty_estimation": True,
        "calibration_window_hours": 24,
    }

    world_model = GeminiWorldModel(config)
    await world_model.initialize()

    try:
        print("✅ Testing enhanced telemetry processing...")
        await test_instance.test_enhanced_telemetry_processing(world_model)

        print("✅ Testing state consistency validation...")
        await test_instance.test_state_consistency_validation(world_model)

        print("✅ Testing vector embedding generation...")
        await test_instance.test_vector_embedding_generation(world_model)

        print("✅ Testing temporal indexing...")
        await test_instance.test_temporal_indexing(world_model)

        print("✅ Testing enhanced current state queries...")
        await test_instance.test_enhanced_current_state_query(world_model)

        print("✅ Testing enhanced historical queries...")
        await test_instance.test_enhanced_historical_query(world_model)

        print("✅ Testing LLM state evolution analysis...")
        await test_instance.test_llm_state_evolution_analysis(world_model)

        print("✅ Testing enhanced health status...")
        await test_instance.test_enhanced_health_status(world_model)

        print("✅ Testing anomaly detection...")
        await test_instance.test_anomaly_detection(world_model)

        print("✅ Testing confidence scoring...")
        await test_instance.test_confidence_scoring(world_model)

        print("\n🎉 All enhanced state tracking tests passed!")

        # Display summary metrics
        health_status = await world_model.get_health_status()
        print(f"\n📊 Final Test Metrics:")
        print(
            f"   • State metrics tracked: {health_status['metrics']['current_state_metrics']}"
        )
        print(
            f"   • Embeddings generated: {health_status['metrics']['state_embeddings_stored']}"
        )
        print(
            f"   • Temporal index keys: {health_status['metrics']['temporal_index_keys']}"
        )
        print(
            f"   • Evolution patterns: {health_status['metrics']['evolution_patterns_tracked']}"
        )
        print(f"   • Consistency checks: {len(world_model._state_consistency_log)}")
        print(
            f"   • Anomalies detected: {health_status['metrics']['anomalies_detected']}"
        )

    finally:
        await world_model.shutdown()


if __name__ == "__main__":
    # Run integration test when executed directly
    asyncio.run(run_integration_test())
