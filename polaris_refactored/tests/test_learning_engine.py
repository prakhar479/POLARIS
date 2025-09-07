import pytest
from datetime import datetime
import uuid

from polaris_refactored.src.digital_twin.learning_engine import (
    PolarisLearningEngine,
    LearningContext,
    ReinforcementLearningStrategy,
    PatternRecognitionStrategy,
)
from polaris_refactored.src.digital_twin.knowledge_base import PolarisKnowledgeBase
from polaris_refactored.src.infrastructure.data_storage import InMemoryGraphStorageBackend, PolarisDataStore
from polaris_refactored.src.domain.models import ExecutionResult, ExecutionStatus, LearnedPattern


@pytest.mark.asyncio
async def test_reinforcement_strategy_basic_reward():
    strat = ReinforcementLearningStrategy()
    res = ExecutionResult(
        action_id=str(uuid.uuid4()),
        status=ExecutionStatus.SUCCESS,
        result_data={}
    )
    ctx = LearningContext("sys-1", res, {}, {})
    knowledge = await strat.learn(ctx)
    assert knowledge.knowledge_type == "reinforcement"
    assert knowledge.data.get("estimated_reward") == 1.0
    assert knowledge.confidence >= 0.5


@pytest.mark.asyncio
async def test_pattern_recognition_strategy_diffs():
    strat = PatternRecognitionStrategy()
    res = ExecutionResult(
        action_id=str(uuid.uuid4()),
        status=ExecutionStatus.SUCCESS,
        result_data={}
    )
    ctx = LearningContext("sys-1", res, {"cpu": 0.5}, {"cpu": 0.8})
    knowledge = await strat.learn(ctx)
    assert knowledge.knowledge_type == "pattern"
    assert "diffs" in knowledge.data
    assert knowledge.data["diffs"].get("delta_cpu") == pytest.approx(0.3, rel=1e-6, abs=1e-6)


@pytest.mark.asyncio
async def test_learning_engine_persists_patterns():
    backend = InMemoryGraphStorageBackend()
    ds = PolarisDataStore({
        "graph": backend,
        "document": backend,
        "time_series": backend,
    })
    await ds.start()
    kb = PolarisKnowledgeBase(ds)
    engine = PolarisLearningEngine(knowledge_base=kb)

    res = ExecutionResult(
        action_id=str(uuid.uuid4()),
        status=ExecutionStatus.SUCCESS,
        result_data={"system_id": "S1"}
    )
    await engine.learn_from_adaptation(res)

    insights = await engine.get_learning_insights("S1")
    assert len(insights) >= 1

    recs = await engine.recommend_adaptations("S1", current_state={})
    assert len(recs) >= 1

    await ds.stop()
