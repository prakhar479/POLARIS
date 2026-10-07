"""Tests for Cluster Safety Guardrails, Rate Limiting, and Blast Radius Isolation."""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.core.polaris import Polaris, PolarisConfig
from polaris.core.safety import SafetyConfig, SafetyPolicyEngine
from polaris.core.topology import SystemTopology


def test_safety_config_defaults_and_custom():
    """Verify SafetyConfig parsing with defaults and custom overrides."""
    default_cfg = SafetyConfig.from_dict(None)
    assert default_cfg.enabled is True
    assert default_cfg.concurrency_mode == "blast_radius_isolated"
    assert "scale_up" in default_cfg.opposing_actions

    custom_cfg = SafetyConfig.from_dict(
        {
            "enabled": True,
            "max_cluster_adaptations_per_window": 5,
            "cluster_window_seconds": 30.0,
            "concurrency_mode": "serialized",
            "opposing_actions": {"warm_cache": "purge_cache"},
            "mutual_exclusions": [["backup", "restore"]],
        }
    )
    assert custom_cfg.max_cluster_adaptations_per_window == 5
    assert custom_cfg.cluster_window_seconds == 30.0
    assert custom_cfg.concurrency_mode == "serialized"
    assert custom_cfg.opposing_actions.get("warm_cache") == "purge_cache"
    assert custom_cfg.opposing_actions.get("purge_cache") == "warm_cache"
    assert custom_cfg.mutual_exclusions == [["backup", "restore"]]


def test_cluster_rate_limiting():
    """Verify cluster-wide adaptation rate limits across multiple systems."""
    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            max_cluster_adaptations_per_window=2,
            cluster_window_seconds=60.0,
        )
    )

    act1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="sys_a")
    act2 = AdaptationAction(action_id="2", action_type="scale_up", target_system="sys_b")
    act3 = AdaptationAction(action_id="3", action_type="scale_up", target_system="sys_c")

    # Action 1 allowed
    ok1, reason1 = engine.check_action_safety(act1)
    assert ok1 is True
    assert reason1 is None
    engine.record_action_start(act1)
    engine.record_action_end(act1)

    # Action 2 allowed
    ok2, reason2 = engine.check_action_safety(act2)
    assert ok2 is True
    assert reason2 is None
    engine.record_action_start(act2)
    engine.record_action_end(act2)

    # Action 3 rejected due to cluster rate limit
    ok3, reason3 = engine.check_action_safety(act3)
    assert ok3 is False
    assert "Cluster adaptation rate limit reached" in (reason3 or "")


def test_system_rate_limiting():
    """Verify system-level adaptation rate limits maintain isolation."""
    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            max_system_adaptations_per_window=2,
            system_window_seconds=60.0,
        )
    )

    act_a1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="sys_a")
    act_a2 = AdaptationAction(action_id="2", action_type="scale_up", target_system="sys_a")
    act_a3 = AdaptationAction(action_id="3", action_type="scale_up", target_system="sys_a")
    act_b1 = AdaptationAction(action_id="4", action_type="scale_up", target_system="sys_b")

    # Sys A action 1
    assert engine.check_action_safety(act_a1)[0] is True
    engine.record_action_start(act_a1)
    engine.record_action_end(act_a1)

    # Sys A action 2
    assert engine.check_action_safety(act_a2)[0] is True
    engine.record_action_start(act_a2)
    engine.record_action_end(act_a2)

    # Sys A action 3 rejected
    ok_a3, reason_a3 = engine.check_action_safety(act_a3)
    assert ok_a3 is False
    assert "System adaptation rate limit reached for 'sys_a'" in (reason_a3 or "")

    # Sys B action 1 allowed (system isolation)
    ok_b1, _ = engine.check_action_safety(act_b1)
    assert ok_b1 is True


def test_max_concurrent_cluster_adaptations():
    """Verify max concurrent in-flight cluster adaptations limit."""
    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            max_concurrent_cluster_adaptations=1,
            concurrency_mode="parallel",
        )
    )

    act1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="sys_a")
    act2 = AdaptationAction(action_id="2", action_type="scale_up", target_system="sys_b")

    # Start action 1
    assert engine.check_action_safety(act1)[0] is True
    engine.record_action_start(act1)
    assert engine.is_system_active("sys_a") is True

    # Action 2 rejected while action 1 in-flight
    ok2, reason2 = engine.check_action_safety(act2)
    assert ok2 is False
    assert "Cluster max concurrent adaptations reached" in (reason2 or "")

    # End action 1
    engine.record_action_end(act1)
    assert engine.is_system_active("sys_a") is False

    # Action 2 now allowed
    ok2_after, _ = engine.check_action_safety(act2)
    assert ok2_after is True


def test_blast_radius_mutual_exclusion():
    """Verify blast-radius mutual exclusion prevents concurrent dependent adaptations."""
    topo = SystemTopology()
    topo.add_dependency("gateway", "orders")
    topo.add_dependency("orders", "db")
    topo.add_node("analytics")  # Disconnected node

    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            concurrency_mode="blast_radius_isolated",
        )
    )

    act_orders = AdaptationAction(action_id="1", action_type="scale_up", target_system="orders")
    act_gw = AdaptationAction(action_id="2", action_type="throttle", target_system="gateway")
    act_db = AdaptationAction(action_id="3", action_type="reindex", target_system="db")
    act_analytics = AdaptationAction(action_id="4", action_type="sync", target_system="analytics")

    # Start adaptation on orders
    assert engine.check_action_safety(act_orders, topology=topo)[0] is True
    engine.record_action_start(act_orders, topology=topo)

    # Gateway rejected due to blast radius overlap with orders
    ok_gw, reason_gw = engine.check_action_safety(act_gw, topology=topo)
    assert ok_gw is False
    assert "Blast radius of 'gateway' overlaps with active adaptation" in (reason_gw or "")

    # Database rejected due to blast radius overlap with orders
    ok_db, reason_db = engine.check_action_safety(act_db, topology=topo)
    assert ok_db is False
    assert "Blast radius of 'db' overlaps with active adaptation" in (reason_db or "")

    # Analytics is independent; allowed
    ok_analytics, _ = engine.check_action_safety(act_analytics, topology=topo)
    assert ok_analytics is True

    # Complete orders adaptation
    engine.record_action_end(act_orders)

    # Gateway now allowed
    assert engine.check_action_safety(act_gw, topology=topo)[0] is True


def test_anti_flapping_cooldown_detection():
    """Verify flapping detection trips cooldown on rapid opposing actions."""
    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            flapping_window_seconds=60.0,
            flapping_threshold=3,
            flapping_cooldown_seconds=120.0,
        )
    )

    act_up1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="web")
    act_down1 = AdaptationAction(action_id="2", action_type="scale_down", target_system="web")
    act_up2 = AdaptationAction(action_id="3", action_type="scale_up", target_system="web")
    act_down2 = AdaptationAction(action_id="4", action_type="scale_down", target_system="web")
    act_clear = AdaptationAction(action_id="5", action_type="clear_cache", target_system="web")

    # Cycle 1: scale_up
    assert engine.check_action_safety(act_up1)[0] is True
    engine.record_action_start(act_up1)
    engine.record_action_end(act_up1)

    # Cycle 2: scale_down
    assert engine.check_action_safety(act_down1)[0] is True
    engine.record_action_start(act_down1)
    engine.record_action_end(act_down1)

    # Cycle 3: scale_up (triggers 3rd alternating point)
    assert engine.check_action_safety(act_up2)[0] is True
    engine.record_action_start(act_up2)
    engine.record_action_end(act_up2)

    # Cycle 4: scale_down trips flapping detector
    ok_down2, reason_down2 = engine.check_action_safety(act_down2)
    assert ok_down2 is False
    assert "Flapping detected on 'web'" in (reason_down2 or "")

    # Subsequent check is blocked by active cooldown
    ok_cooldown, reason_cooldown = engine.check_action_safety(act_down2)
    assert ok_cooldown is False
    assert "blocked due to flapping" in (reason_cooldown or "")

    # Opposing action is also in cooldown
    ok_up_cooldown, _ = engine.check_action_safety(act_up1)
    assert ok_up_cooldown is False

    # Non-opposing action is permitted
    ok_clear, _ = engine.check_action_safety(act_clear)
    assert ok_clear is True

    cooldowns = engine.get_active_cooldowns("web")
    assert "scale_down" in cooldowns
    assert "scale_up" in cooldowns


def test_mutual_exclusion_rules():
    """Verify declared mutual exclusion groups prevent conflicting adaptations."""
    engine = SafetyPolicyEngine(
        config=SafetyConfig(
            mutual_exclusions=[["compact_db", "reindex_db"]],
            concurrency_mode="parallel",
        )
    )

    act_compact = AdaptationAction(action_id="1", action_type="compact_db", target_system="db")
    act_reindex = AdaptationAction(action_id="2", action_type="reindex_db", target_system="db")
    act_vacuum = AdaptationAction(action_id="3", action_type="vacuum_db", target_system="db")

    # Start compact_db
    assert engine.check_action_safety(act_compact)[0] is True
    engine.record_action_start(act_compact)

    # reindex_db is rejected because it's mutually exclusive with active compact_db
    ok_reindex, reason_reindex = engine.check_action_safety(act_reindex)
    assert ok_reindex is False
    assert "mutually exclusive with active action 'compact_db'" in (reason_reindex or "")

    # vacuum_db is not in the exclusion group; allowed
    ok_vacuum, _ = engine.check_action_safety(act_vacuum)
    assert ok_vacuum is True

    engine.record_action_end(act_compact)
    # After completion, reindex_db is allowed
    assert engine.check_action_safety(act_reindex)[0] is True


@pytest.mark.asyncio
async def test_adaptation_pipeline_with_safety_engine():
    """Verify AdaptationPipeline enforces SafetyPolicyEngine rules during assessment."""
    mock_strategy = MagicMock()
    mock_strategy.requires_system_contract = False
    candidate_action = AdaptationAction(
        action_id="act-blocked",
        action_type="scale_up",
        target_system="order_svc",
    )
    mock_strategy.assess = AsyncMock(return_value=[candidate_action])
    mock_strategy.on_action_executed = AsyncMock()

    mock_connector = MagicMock()
    mock_connector.validate_action = AsyncMock(return_value=True)
    mock_connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="act-blocked",
            status=ExecutionStatus.SUCCESS,
            result_data={},
        )
    )

    # Create safety engine with 0 allowed cluster adaptations (blocks everything)
    safety_engine = SafetyPolicyEngine(
        config=SafetyConfig(
            max_cluster_adaptations_per_window=1,
        )
    )
    # Burn the single allowed slot
    dummy = AdaptationAction(action_id="dummy", action_type="dummy", target_system="other")
    safety_engine.record_action_start(dummy)
    safety_engine.record_action_end(dummy)

    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()
    mock_logger = MagicMock()
    mock_metrics = MagicMock()

    pipeline = AdaptationPipeline(
        strategy=mock_strategy,
        knowledge_store=None,
        world_model=None,
        event_bus=mock_event_bus,
        logger=mock_logger,
        config=PolarisConfig(),
        metrics=mock_metrics,
        safety_engine=safety_engine,
    )

    state = SystemState(
        system_id="order_svc",
        timestamp=datetime.now(timezone.utc),
        health_status=HealthStatus.HEALTHY,
        metrics={"latency": MetricValue(name="latency", value=100.0)},
    )

    executed = await pipeline.run(state, mock_connector)
    # Action was blocked by safety guardrail
    assert executed is False
    assert not mock_connector.execute_action.called

    # Reset safety engine and run again
    safety_engine.reset()
    executed2 = await pipeline.run(state, mock_connector)
    assert executed2 is True
    assert mock_connector.execute_action.called


def test_polaris_config_instantiates_safety_engine():
    """Verify Polaris coordinator instantiates SafetyPolicyEngine from YAML config."""
    cfg = PolarisConfig.from_dict(
        {
            "safety": {
                "enabled": True,
                "max_cluster_adaptations_per_window": 10,
                "concurrency_mode": "serialized",
            }
        }
    )
    polaris = Polaris(config=cfg)
    assert polaris.safety_policy_engine is not None
    assert polaris.safety_policy_engine.config.max_cluster_adaptations_per_window == 10
    assert polaris.safety_policy_engine.config.concurrency_mode == "serialized"
