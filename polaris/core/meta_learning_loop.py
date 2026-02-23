"""Meta-learning background loop.

Extracted from ``Polaris._meta_learning_loop`` so the meta-learning cycle
can be tested and reused independently of the monitoring loop.
"""

import asyncio
import random
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from polaris.core.meta_learning_transparency import MetaLearningTransparencyWriter

if TYPE_CHECKING:
    from polaris.abstractions import AdaptationStrategy, Logger, MetaLearner, MetricsCollector
    from polaris.core.registry import ConnectorRegistry
    from polaris.infrastructure.config import PolarisConfig


class MetaLearningLoop:
    """Background loop that periodically analyses performance and tunes the strategy.

    The loop sleeps for ``interval_seconds``, then for each registered system:

    1. Calls ``meta_learner.analyze_performance(system_id)``.
    2. Calls ``meta_learner.propose_strategy_updates(strategy, analysis)``.
    3. Validates and applies approved proposals via the meta-learner.
    """

    def __init__(
        self,
        meta_learner: "MetaLearner",
        strategy: "AdaptationStrategy",
        registry: "ConnectorRegistry",
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        interval_seconds: float,
        config: "PolarisConfig",
        transparency_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the meta-learning loop."""
        self._meta_learner = meta_learner
        self._strategy = strategy
        self._registry = registry
        self._logger = logger
        self._metrics = metrics
        self._interval_seconds = interval_seconds
        self._config = config
        self._running = False
        self._transparency_writer: Optional[MetaLearningTransparencyWriter] = None
        self._transparency_config = self._normalize_transparency_config(transparency_config)
        if self._transparency_config["enabled"]:
            self._transparency_writer = MetaLearningTransparencyWriter(
                output_path=self._transparency_config["output_path"],
                logger=self._logger,
            )

    async def run(self) -> None:
        """Run the meta-learning loop until cancelled."""
        self._running = True
        self._logger.info("Starting meta-learning loop")
        self._emit("polaris.meta_learning.started")

        while self._running:
            try:
                # Add ±10% jitter to prevent multi-instance thundering herd.
                jitter = self._interval_seconds * 0.1 * (random.random() * 2 - 1)
                await asyncio.sleep(max(0, self._interval_seconds + jitter))

                for system_id in self._registry.system_ids():
                    await self._run_for_system(system_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in meta-learning loop: {e}")
                self._emit("polaris.meta_learning.loop_errors")

        self._logger.info("Meta-learning loop stopped")
        self._emit("polaris.meta_learning.stopped")

    async def _run_for_system(self, system_id: str) -> None:
        """Run one meta-learning cycle for a single system."""
        cycle_id = str(uuid.uuid4())
        stage = "analysis"
        analysis: Any = None
        proposals: List[Any] = []
        validated: List[Any] = []
        applied: List[Any] = []
        error: Optional[Dict[str, str]] = None

        try:
            analysis = await self._meta_learner.analyze_performance(system_id)
            self._emit_tagged("polaris.meta_learning.analysis_completed", system_id)

            stage = "proposal_generation"
            proposals = await self._meta_learner.propose_strategy_updates(self._strategy, analysis)
            self._gauge_tagged(
                "polaris.meta_learning.proposals_generated", len(proposals), system_id
            )

            if not proposals:
                return

            stage = "validation"
            validated = await self._meta_learner.validate_proposals(proposals)
            self._gauge_tagged(
                "polaris.meta_learning.proposals_validated", len(validated), system_id
            )

            stage = "application"
            applied = await self._meta_learner.apply_proposals(self._strategy, validated)
            self._gauge_tagged("polaris.meta_learning.proposals_applied", len(applied), system_id)

            self._logger.info(f"Meta-learner applied {len(applied)} parameter updates")

        except Exception as e:
            error = {"stage": stage, "message": str(e)}
            self._logger.error(f"Error in meta-learning for {system_id}: {e}")
            self._emit_tagged("polaris.meta_learning.errors", system_id)
        finally:
            self._record_transparency(
                cycle_id=cycle_id,
                system_id=system_id,
                analysis=analysis,
                proposals=proposals,
                validated=validated,
                applied=applied,
                error=error,
            )

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _should_collect(self) -> bool:
        from polaris.core.component_builder import ComponentBuilder

        return ComponentBuilder.should_collect(self._config, "meta_learner", self._metrics)

    def _emit(self, metric: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.increment(metric)

    def _emit_tagged(self, metric: str, system_id: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.increment(metric, tags={"system_id": system_id})

    def _gauge_tagged(self, metric: str, value: float, system_id: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.gauge(metric, value, tags={"system_id": system_id})

    def _normalize_transparency_config(
        self, transparency_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        defaults = {"enabled": True, "output_path": "./logs/meta_learning_updates.jsonl"}
        if not isinstance(transparency_config, dict):
            return defaults

        enabled = transparency_config.get("enabled", defaults["enabled"])
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
        elif not isinstance(enabled, bool):
            enabled = defaults["enabled"]

        output_path = transparency_config.get("output_path", defaults["output_path"])
        if not isinstance(output_path, str) or not output_path.strip():
            output_path = defaults["output_path"]

        return {"enabled": bool(enabled), "output_path": output_path}

    def _record_transparency(
        self,
        *,
        cycle_id: str,
        system_id: str,
        analysis: Any,
        proposals: List[Any],
        validated: List[Any],
        applied: List[Any],
        error: Optional[Dict[str, str]],
    ) -> None:
        if not self._transparency_writer:
            return

        apply_details, applied_succeeded, applied_failed = self._extract_apply_details(applied)
        proposal_records = [
            self._proposal_to_record(
                proposal=proposal,
                index=index,
                apply_details=apply_details,
            )
            for index, proposal in enumerate(proposals)
        ]
        approved = sum(
            1
            for proposal_record in proposal_records
            if proposal_record.get("validation_status") == "approved"
        )
        if approved == 0 and validated:
            approved = len(validated)
        rejected = max(0, len(proposal_records) - approved)

        record = {
            "record_type": "meta_learning_cycle",
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "cycle_id": cycle_id,
            "system_id": system_id,
            "status": "error" if error else "completed",
            "error": error,
            "analysis": analysis,
            "proposals": proposal_records,
            "counts": {
                "generated": len(proposals),
                "approved": approved,
                "rejected": rejected,
                "applied": len(applied),
                "applied_succeeded": applied_succeeded,
                "applied_failed": applied_failed,
            },
        }

        try:
            self._transparency_writer.record_cycle(record)
        except Exception as exc:
            self._logger.warning(
                f"Failed to record meta-learning transparency cycle for {system_id}: {exc}"
            )

    def _extract_apply_details(
        self, applied: List[Any]
    ) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
        details: Dict[str, Dict[str, Any]] = {}
        succeeded = 0
        failed = 0

        for index, update in enumerate(applied):
            proposal_id = getattr(update, "proposal_id", None)
            success: Any = getattr(update, "success", None)
            error_message = getattr(update, "error_message", None)

            if isinstance(update, dict):
                proposal_id = update.get("proposal_id", proposal_id)
                success = update.get("success", success)
                error_message = update.get("error_message", error_message)

            if isinstance(success, bool):
                if success:
                    succeeded += 1
                else:
                    failed += 1
            else:
                success = None

            if proposal_id is None:
                proposal_id = f"applied-{index}"

            details[str(proposal_id)] = {
                "success": success,
                "error_message": error_message,
            }

        return details, succeeded, failed

    def _proposal_to_record(
        self,
        *,
        proposal: Any,
        index: int,
        apply_details: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        proposal_id = getattr(proposal, "proposal_id", None)
        if proposal_id is None and isinstance(proposal, dict):
            proposal_id = proposal.get("proposal_id")
        if proposal_id is None:
            proposal_id = f"proposal-{index}"
        proposal_id = str(proposal_id)

        status: Any = getattr(proposal, "status", None)
        if status is None and isinstance(proposal, dict):
            status = proposal.get("status")
        if hasattr(status, "value"):
            status = status.value
        elif status is not None:
            status = str(status)

        current_value = getattr(proposal, "current_value", None)
        proposed_value = getattr(proposal, "proposed_value", None)
        if isinstance(proposal, dict):
            current_value = proposal.get("current_value", current_value)
            proposed_value = proposal.get("proposed_value", proposed_value)

        applied_detail = apply_details.get(proposal_id, {})
        record = {
            "proposal_id": proposal_id,
            "parameter_path": (
                proposal.get("parameter_path")
                if isinstance(proposal, dict)
                else getattr(proposal, "parameter_path", None)
            ),
            "current_value": current_value,
            "proposed_value": proposed_value,
            "rationale": (
                proposal.get("rationale")
                if isinstance(proposal, dict)
                else getattr(proposal, "rationale", None)
            ),
            "expected_impact": (
                proposal.get("expected_impact")
                if isinstance(proposal, dict)
                else getattr(proposal, "expected_impact", None)
            ),
            "confidence": (
                proposal.get("confidence")
                if isinstance(proposal, dict)
                else getattr(proposal, "confidence", None)
            ),
            "created_at": (
                proposal.get("created_at")
                if isinstance(proposal, dict)
                else getattr(proposal, "created_at", None)
            ),
            "applied_at": (
                proposal.get("applied_at")
                if isinstance(proposal, dict)
                else getattr(proposal, "applied_at", None)
            ),
            "validation_status": status,
            "applied": proposal_id in apply_details,
            "apply_success": applied_detail.get("success"),
            "apply_error": applied_detail.get("error_message"),
        }

        if record["parameter_path"] is None and record["current_value"] is None:
            record["raw"] = proposal

        return record
