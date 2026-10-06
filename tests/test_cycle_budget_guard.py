"""Tests for cycle execution budget guard and token accounting across strategies."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext
from polaris.abstractions.system_contract import SystemContract
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import HealthStatus, MetricValue, SystemState
from polaris.infrastructure.llm.base import LLMResponse
from polaris.strategies.agentic_llm import AgenticLLMStrategy
from polaris.strategies.thread_agentic import ThreadAgenticStrategy


@pytest.fixture
def base_context():
    contract = SystemContract(
        system_id="test-sys",
        connector_type="test",
        supported_action_types=("scale_up", "scale_down"),
    )
    return AdaptationContext(
        system_id="test-sys",
        system_contract=contract,
        historical_states=[],
    )


@pytest.fixture
def base_state():
    return SystemState(
        system_id="test-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 85.0)},
        health_status=HealthStatus.HEALTHY,
    )


@pytest.mark.asyncio
async def test_agentic_llm_cycle_budget_timeout_json(base_context, base_state):
    """AgenticLLMStrategy respects max_cycle_time_seconds when JSON call hangs."""
    llm = Mock()

    async def hanging_generate(*args, **kwargs):
        await asyncio.sleep(0.5)
        return LLMResponse(content="{}", model="mock-model")

    llm.generate = AsyncMock(side_effect=hanging_generate)
    metrics = Mock(spec=MetricsCollector)
    logger = Mock(spec=Logger)

    strategy = AgenticLLMStrategy(
        llm_client=llm,
        knowledge_store=Mock(spec=KnowledgeStore),
        world_model=Mock(spec=WorldModel),
        max_cycle_time_seconds=0.05,  # 50ms timeout
        metrics=metrics,
        logger=logger,
    )

    actions = await strategy.assess(base_state, base_context)
    assert actions == []
    metrics.increment.assert_any_call(
        "polaris.strategy.agentic.cycle_timeout",
        tags={"system_id": "test-sys"},
    )


@pytest.mark.asyncio
async def test_agentic_llm_cycle_budget_timeout_native_tools(base_context, base_state):
    """AgenticLLMStrategy respects max_cycle_time_seconds when native tools call hangs."""
    llm = Mock()

    async def hanging_generate(*args, **kwargs):
        await asyncio.sleep(0.5)
        return LLMResponse(content="", model="mock-model", tool_calls=[])

    llm.generate_with_tools = AsyncMock(side_effect=hanging_generate)
    metrics = Mock(spec=MetricsCollector)

    strategy = AgenticLLMStrategy(
        llm_client=llm,
        knowledge_store=Mock(spec=KnowledgeStore),
        world_model=Mock(spec=WorldModel),
        native_tools=[{"type": "function", "function": {"name": "test"}}],
        max_cycle_time_seconds=0.05,
        metrics=metrics,
    )

    actions = await strategy.assess(base_state, base_context)
    assert actions == []
    metrics.increment.assert_any_call(
        "polaris.strategy.agentic.cycle_timeout",
        tags={"system_id": "test-sys"},
    )


@pytest.mark.asyncio
async def test_agentic_llm_token_accounting(base_context, base_state):
    """AgenticLLMStrategy records total, prompt, and completion tokens."""
    llm = Mock()
    resp_payload = '{"final": {"needs_adaptation": false, "reasoning": "stable", "actions": []}}'
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content=resp_payload,
            model="gpt-4",
            tokens_used=120,
            prompt_tokens=100,
            completion_tokens=20,
        )
    )
    metrics = Mock(spec=MetricsCollector)

    strategy = AgenticLLMStrategy(
        llm_client=llm,
        knowledge_store=Mock(spec=KnowledgeStore),
        world_model=Mock(spec=WorldModel),
        metrics=metrics,
    )

    actions = await strategy.assess(base_state, base_context)
    assert actions == []

    metrics.increment.assert_any_call(
        "polaris.llm.tokens.total",
        value=120,
        tags={"system_id": "test-sys", "strategy": "agentic"},
    )
    metrics.increment(
        "polaris.llm.tokens.prompt",
        value=100,
        tags={"system_id": "test-sys", "strategy": "agentic"},
    )
    metrics.increment(
        "polaris.llm.tokens.completion",
        value=20,
        tags={"system_id": "test-sys", "strategy": "agentic"},
    )


@pytest.mark.asyncio
async def test_thread_agentic_cycle_budget_timeout(base_context, base_state):
    """ThreadAgenticStrategy aborts when reasoning tree exceeds max_cycle_time_seconds."""
    llm = Mock()

    async def hanging_generate(*args, **kwargs):
        await asyncio.sleep(0.5)
        return LLMResponse(content="{}", model="mock")

    llm.generate = AsyncMock(side_effect=hanging_generate)
    metrics = Mock(spec=MetricsCollector)

    strategy = ThreadAgenticStrategy(
        llm_client=llm,
        knowledge_store=Mock(spec=KnowledgeStore),
        world_model=Mock(spec=WorldModel),
        max_cycle_time_seconds=0.05,
        metrics=metrics,
    )

    actions = await strategy.assess(base_state, base_context)
    assert actions == []
    metrics.increment.assert_any_call(
        "polaris.strategy.thread_agentic.cycle_timeout",
        tags={"system_id": "test-sys"},
    )


@pytest.mark.asyncio
async def test_thread_agentic_token_accounting(base_context, base_state):
    """ThreadAgenticStrategy records tokens used during thread reasoning."""
    llm = Mock()
    resp_payload = '{"final": {"needs_adaptation": false, "reasoning": "stable", "actions": []}}'
    llm.generate = AsyncMock(
        return_value=LLMResponse(
            content=resp_payload,
            model="gpt-4",
            tokens_used=150,
            prompt_tokens=110,
            completion_tokens=40,
        )
    )
    metrics = Mock(spec=MetricsCollector)

    strategy = ThreadAgenticStrategy(
        llm_client=llm,
        knowledge_store=Mock(spec=KnowledgeStore),
        world_model=Mock(spec=WorldModel),
        metrics=metrics,
    )

    actions = await strategy.assess(base_state, base_context)
    assert actions == []

    metrics.increment.assert_any_call(
        "polaris.llm.tokens.total",
        value=150,
        tags={"system_id": "test-sys", "strategy": "thread_agentic"},
    )
    metrics.increment.assert_any_call(
        "polaris.llm.tokens.prompt",
        value=110,
        tags={"system_id": "test-sys", "strategy": "thread_agentic"},
    )
    metrics.increment.assert_any_call(
        "polaris.llm.tokens.completion",
        value=40,
        tags={"system_id": "test-sys", "strategy": "thread_agentic"},
    )
