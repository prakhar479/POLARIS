r"""
SUAVE system connector — corrected to follow the SUAVE README exactly.

Key corrections vs. previous version
--------------------------------------
1. Uses /task/request and /task/cancel (suave_msgs/Task) for mission
   lifecycle, exactly as the README specifies. The Coordinate Mission node
   owns T1 (search) → T2 (follow) sequencing; Polaris only sends goals.

2. Does NOT directly activate lifecycle nodes on start_mission.
   Instead it requests task T1 ("search_pipeline") via /task/request, which
   causes Coordinate Mission to activate the correct nodes for that task.
   When the pipeline is found, Coordinate Mission transitions to T2
   automatically. Polaris can optionally send T2 explicitly.

3. mission_active detection is based on /pipeline/detected subscription
   plus internal _mission_running flag — NOT on SUAVE monitor diagnostics,
   because water_visibility and thruster_monitor nodes only publish AFTER
   the mission nodes are active (chicken-and-egg if you check for them first).

4. fd_all_thrusters / fd_recover_thrusters corrected:
   - Table V of the SEAMS paper: fd_all_thrusters → inactive (default, no
     recovery needed), fd_recover_thrusters → active (recovery running).
   - change_mode is only called for ADAPTATION, not to start the mission.
   - To trigger thruster recovery: change_mode(f_maintain_motion, fd_recover_thrusters)

5. Telemetry accumulates /diagnostics for up to 5 s to collect SUAVE monitor
   data (water_visibility, thruster_monitor), which publish at lower frequency
   than mavros. Only SUAVE monitor entries are parsed; mavros noise is ignored.

6. f_follow_pipeline stuck in 'activating': this happens when change_mode
   activates it BEFORE the pipeline is detected. Coordinate Mission handles
   this transition. Polaris should only request follow_pipeline task AFTER
   pipeline_detected == 1, or let Coordinate Mission do it automatically.

SUAVE ROS 2 interfaces used
----------------------------
  /task/request          (suave_msgs/srv/Task)             service  — start a task
  /task/cancel           (suave_msgs/srv/Task)             service  — cancel a task
  /pipeline/detected     (std_msgs/Bool)                   subscribe
  /diagnostics           (diagnostic_msgs/DiagnosticArray) subscribe
  /f_maintain_motion/change_mode      (system_modes_msgs/ChangeMode) service
  /f_generate_search_path/change_mode (system_modes_msgs/ChangeMode) service
  /f_follow_pipeline/change_mode      (system_modes_msgs/ChangeMode) service

Docker launch
-------------
  ros2 launch suave_missions mission.launch.py \\
      adaptation_manager:=polaris mission_type:=time_constrained_mission

suave_polaris.launch.py sets task_bridge:=False and starts rosbridge on 9090.
"""

import asyncio
import math
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

try:
    import roslibpy
except ImportError:
    roslibpy = None

from polaris.abstractions.connector import Connector
from polaris.abstractions.observability import Logger, MetricsCollector

if TYPE_CHECKING:
    import roslibpy as _roslibpy
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ROSBRIDGE_HOST = "localhost"
DEFAULT_ROSBRIDGE_PORT = 9090
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_SERVICE_TIMEOUT = 10.0  # task services can be slower than mode changes

# Task names as defined in suave_msgs/Task
TASK_SEARCH_PIPELINE = "search_pipeline"
TASK_INSPECT_PIPELINE = "inspect_pipeline"
TASK_FOLLOW_PIPELINE_LEGACY = "follow_pipeline"

# Valid modes per function node (Table V, SEAMS 2023 paper)
_VALID_MODES: Dict[str, List[str]] = {
    "f_maintain_motion": [
        "fd_all_thrusters",  # → inactive lifecycle state (normal operation)
        "fd_recover_thrusters",  # → active lifecycle state  (recovery running)
        "fd_unground",
    ],
    "f_generate_search_path": [
        "fd_spiral_low",
        "fd_spiral_medium",
        "fd_spiral_high",
        "fd_unground",  # → inactive
    ],
    "f_follow_pipeline": [
        "fd_follow_pipeline",  # → active
        "fd_unground",  # → inactive
    ],
}

# Action type constants
_ACTION_START_MISSION = "start_mission"
_ACTION_STOP_MISSION = "stop_mission"
_ACTION_CHANGE_MODE = "change_mode"

# Substrings that identify SUAVE monitor diagnostic entries (not mavros)
_SUAVE_MONITOR_SUBSTRINGS = (
    "water_visibility",
    "thruster_monitor",
    "thruster monitor",
)


class SUAVEConnector(Connector):
    """
    Connector for SUAVE (Self-Adaptive Underwater Vehicle Exemplar).

    Mission lifecycle follows the SUAVE README exactly:
      - Subscribe to /diagnostics for monitoring data
      - Use /task/request and /task/cancel for mission start/stop
      - Use /f_*/change_mode services only for adaptation reconfiguration

    Supported actions
    -----------------
    start_mission
        Request T1 (search_pipeline) via /task/request.
        Coordinate Mission activates f_generate_search_path + f_maintain_motion,
        then automatically transitions to T2 when pipeline is found.
        Parameters: { "task": "search_pipeline" }  (default if omitted)

    stop_mission
        Cancel the current task via /task/cancel.
        Parameters: { "task": "search_pipeline" }  (or whichever is running)

    change_mode
        Change one function node's mode — ADAPTATION ONLY, not mission start.
        Parameters: { "function_node": str, "mode": str }
        Valid combinations:
          f_maintain_motion      → fd_recover_thrusters | fd_all_thrusters | fd_unground
          f_generate_search_path → fd_spiral_low | fd_spiral_medium | fd_spiral_high | fd_unground
          f_follow_pipeline      → fd_follow_pipeline | fd_unground
    """

    def __init__(
        self,
        host: str = DEFAULT_ROSBRIDGE_HOST,
        port: int = DEFAULT_ROSBRIDGE_PORT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        service_timeout: float = DEFAULT_SERVICE_TIMEOUT,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize SUAVEConnector with rosbridge parameters."""
        if roslibpy is None:
            raise ImportError(
                "roslibpy is required for SUAVEConnector. " "Install it with: pip install roslibpy"
            )
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.service_timeout = service_timeout
        self._logger = logger
        self._metrics = metrics
        self._client: Optional["_roslibpy.Ros"] = None
        self._connected = False

        # Cached state updated by background subscriptions
        self._pipeline_detected: bool = False
        self._mission_running: bool = False
        self._pipeline_topic: Optional["_roslibpy.Topic"] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to rosbridge and set up subscriptions."""
        try:
            self._client = roslibpy.Ros(host=self.host, port=self.port)
            t = threading.Thread(target=self._client.run, daemon=True)
            t.start()

            deadline = time.monotonic() + self.connect_timeout
            while not self._client.is_connected:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"rosbridge did not connect within {self.connect_timeout}s")
                time.sleep(0.1)

            self._connected = True
            self._subscribe_pipeline_detected()

            if self._logger:
                self._logger.info("SUAVEConnector connected", host=self.host, port=self.port)
            if self._metrics:
                self._metrics.increment("polaris.connector.suave.connected")
            return True

        except Exception as exc:
            self._connected = False
            if self._logger:
                self._logger.error("SUAVEConnector connection failed", error=str(exc))
            if self._metrics:
                self._metrics.increment("polaris.connector.suave.connection_errors")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from rosbridge and clean up subscriptions."""
        try:
            if self._pipeline_topic is not None:
                try:
                    self._pipeline_topic.unsubscribe()
                except Exception:
                    pass
                self._pipeline_topic = None

            if self._client is not None:
                self._client.terminate()
                self._client = None
            self._connected = False
            if self._logger:
                self._logger.info("SUAVEConnector disconnected", host=self.host, port=self.port)
            if self._metrics:
                self._metrics.increment("polaris.connector.suave.disconnected")
            return True
        except Exception as exc:
            if self._logger:
                self._logger.error("SUAVEConnector disconnect error", error=str(exc))
            return False

    async def get_system_id(self) -> str:
        """Return the system ID for SUAVE connector."""
        return "suave"

    # ------------------------------------------------------------------
    # Background subscription
    # ------------------------------------------------------------------

    def _subscribe_pipeline_detected(self) -> None:
        """
        Subscribe to /pipeline/detected (std_msgs/Bool).

        This is the T1→T2 signal. True means pipeline found; Coordinate
        Mission will (or already has) switched to T2 automatically.
        We cache this so telemetry can report it without an extra poll,
        and so Polaris knows when it is safe to call change_mode on
        f_follow_pipeline if it wants to intervene.
        """
        if self._client is None:
            return

        self._pipeline_topic = roslibpy.Topic(self._client, "/pipeline/detected", "std_msgs/Bool")

        def _on_pipeline(msg: dict) -> None:
            detected = bool(msg.get("data", False))
            if detected and not self._pipeline_detected:
                if self._logger:
                    self._logger.info(
                        "SUAVEConnector: pipeline detected — T1 complete, "
                        "Coordinate Mission transitioning to T2"
                    )
            self._pipeline_detected = detected

        self._pipeline_topic.subscribe(_on_pipeline)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    async def collect_telemetry(self) -> SystemState:
        """
        Collect diagnostics from SUAVE monitor nodes.

        Accumulates /diagnostics for up to 5 s, keeping the latest entry
        per named status.  Only SUAVE monitor entries (water_visibility,
        thruster_monitor) are parsed; mavros entries are ignored.

        Metrics produced
        ----------------
        pipeline_detected           1/0 — from /pipeline/detected subscription
        mission_running             1/0 — tracked via task calls
        water_visibility            alias (metres)
        thruster_failure_detected   1 if any thruster reports failure
        diagnostics.suave_count     SUAVE monitor entries seen
        diagnostics.error_count     ERROR-level entries
        diagnostics.warn_count      WARN-level entries
        <node_name>.<key>           per-entry parsed values
        """
        if self._metrics:
            self._metrics.increment("polaris.connector.suave.telemetry_calls")

        if not self._connected:
            if self._logger:
                self._logger.warning("SUAVEConnector collect_telemetry called while not connected")
            return SystemState(
                system_id="suave",
                timestamp=datetime.now(timezone.utc),
                metrics={},
                health_status=HealthStatus.UNHEALTHY,
                metadata={"error": "Not connected"},
            )

        start_time = time.monotonic()
        loop = asyncio.get_event_loop()

        # Accumulate SUAVE-monitor entries by name for up to 5 s
        accumulated: Dict[str, dict] = {}
        stop_event = threading.Event()

        topic = roslibpy.Topic(self._client, "/diagnostics", "diagnostic_msgs/DiagnosticArray")

        def _on_diag(msg: dict) -> None:
            for s in msg.get("status", []):
                name = s.get("name", "")
                if any(sub in name.lower() for sub in _SUAVE_MONITOR_SUBSTRINGS):
                    accumulated[name] = s
            # Stop early once we have both monitor nodes
            has_vis = any("water_visibility" in n.lower() for n in accumulated)
            has_thr = any("thruster" in n.lower() for n in accumulated)
            if has_vis and has_thr:
                stop_event.set()

        topic.subscribe(_on_diag)

        def _wait() -> None:
            stop_event.wait(timeout=5.0)
            topic.unsubscribe()

        await loop.run_in_executor(None, _wait)

        now = datetime.now(timezone.utc)
        metrics: Dict[str, MetricValue] = {}

        # Always include cached state regardless of diagnostics availability
        metrics["pipeline_detected"] = MetricValue(
            name="pipeline_detected",
            value=1 if self._pipeline_detected else 0,
            unit="bool",
            timestamp=now,
        )
        metrics["mission_running"] = MetricValue(
            name="mission_running",
            value=1 if self._mission_running else 0,
            unit="bool",
            timestamp=now,
        )
        metrics["mission_active"] = MetricValue(
            name="mission_active",
            value=1 if self._mission_running else 0,
            unit="bool",
            timestamp=now,
        )

        if not accumulated:
            if self._logger:
                self._logger.warning(
                    "SUAVEConnector: no SUAVE monitor diagnostics received — "
                    "mission may not be running yet (this is normal before "
                    "start_mission is called)"
                )
            return SystemState(
                system_id="suave",
                timestamp=now,
                metrics=metrics,
                health_status=(
                    HealthStatus.HEALTHY if self._mission_running else HealthStatus.UNHEALTHY
                ),
                metadata={"warning": "No SUAVE monitor diagnostics received"},
            )

        error_count = 0
        warn_count = 0

        for raw_name, status in accumulated.items():
            raw_level = status.get("level", 0)
            try:
                level = (
                    int.from_bytes(raw_level, "little")
                    if isinstance(raw_level, (bytes, bytearray))
                    else int(raw_level)
                )
            except (TypeError, ValueError):
                level = 0

            if level == 2:
                error_count += 1
            elif level == 1:
                warn_count += 1

            safe_name = str(raw_name).replace(" ", "_").lower()

            for kv in status.get("values", []):
                key = str(kv.get("key", "")).replace(" ", "_").lower()
                raw_val = kv.get("value", "")
                metric_name = f"{safe_name}.{key}"

                try:
                    parsed = float(raw_val)
                    if not math.isfinite(parsed):
                        continue
                    value: float = int(parsed) if parsed == int(parsed) else parsed
                    metrics[metric_name] = MetricValue(
                        name=metric_name, value=value, unit="", timestamp=now
                    )
                except (ValueError, TypeError, OverflowError):
                    if isinstance(raw_val, str):
                        fval = (
                            0.0
                            if raw_val.strip().lower() in ("failure", "error", "false", "0")
                            else 1.0
                        )
                        metrics[metric_name] = MetricValue(
                            name=metric_name,
                            value=fval,
                            unit="status",
                            timestamp=now,
                        )

        metrics["diagnostics.suave_count"] = MetricValue(
            name="diagnostics.suave_count",
            value=len(accumulated),
            unit="count",
            timestamp=now,
        )
        metrics["diagnostics.error_count"] = MetricValue(
            name="diagnostics.error_count",
            value=error_count,
            unit="count",
            timestamp=now,
        )
        metrics["diagnostics.warn_count"] = MetricValue(
            name="diagnostics.warn_count",
            value=warn_count,
            unit="count",
            timestamp=now,
        )

        # Water visibility alias
        for candidate in (
            "water_visibility_observer.water_visibility",
            "water_visibility_observer_node.water_visibility",
        ):
            if candidate in metrics:
                metrics["water_visibility"] = MetricValue(
                    name="water_visibility",
                    value=metrics[candidate].value,
                    unit="m",
                    timestamp=now,
                )
                break

        # Thruster failure flag
        thruster_failed = any(
            v.value == 0.0
            for k, v in metrics.items()
            if "thruster_monitor" in k and "thruster_" in k
        )
        metrics["thruster_failure_detected"] = MetricValue(
            name="thruster_failure_detected",
            value=1 if thruster_failed else 0,
            unit="bool",
            timestamp=now,
        )

        health = HealthStatus.UNHEALTHY if error_count > 0 else HealthStatus.HEALTHY

        if self._metrics:
            self._metrics.histogram(
                "polaris.connector.suave.telemetry_duration_seconds",
                time.monotonic() - start_time,
            )

        if self._logger:
            self._logger.debug(
                "SUAVEConnector telemetry collected",
                suave_monitor_count=len(accumulated),
                error_count=error_count,
                pipeline_detected=self._pipeline_detected,
                mission_running=self._mission_running,
            )

        return SystemState(
            system_id="suave",
            timestamp=now,
            metrics=metrics,
            health_status=health,
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on the SUAVE system."""
        if not self._connected:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message="Not connected to SUAVE",
            )

        start_time = time.monotonic()
        action_type = action.action_type.lower()

        try:
            if action_type == _ACTION_START_MISSION:
                result = await self._execute_start_mission(action)
            elif action_type == _ACTION_STOP_MISSION:
                result = await self._execute_stop_mission(action)
            elif action_type == _ACTION_CHANGE_MODE:
                result = await self._execute_change_mode(action)
            else:
                if self._metrics:
                    self._metrics.increment("polaris.connector.suave.actions_unsupported")
                return ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.FAILED,
                    result_data={},
                    error_message=f"Unsupported action type: {action.action_type}",
                )

            if self._metrics:
                self._metrics.histogram(
                    "polaris.connector.suave.action_execution_duration_seconds",
                    time.monotonic() - start_time,
                    tags={"action_type": action_type},
                )
                self._metrics.increment(
                    "polaris.connector.suave.actions_executed",
                    tags={
                        "action_type": action_type,
                        "status": result.status.value,
                    },
                )
            return result

        except Exception as exc:
            if self._logger:
                self._logger.error(
                    "SUAVEConnector action execution failed",
                    action_type=action.action_type,
                    error=str(exc),
                )
            if self._metrics:
                self._metrics.increment(
                    "polaris.connector.suave.actions_executed",
                    tags={
                        "action_type": action_type,
                        "status": ExecutionStatus.FAILED.value,
                    },
                )
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=str(exc),
            )

    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate that an action is supported by SUAVE."""
        action_type = action.action_type.lower()
        if action_type in (_ACTION_START_MISSION, _ACTION_STOP_MISSION):
            return True
        if action_type == _ACTION_CHANGE_MODE:
            params = action.parameters or {}
            node = params.get("function_node", "")
            mode = params.get("mode", "")
            return node in _VALID_MODES and mode in _VALID_MODES[node]
        return False

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Return list of adaptation actions supported by SUAVE."""
        actions = [
            AdaptationAction(
                action_id="",
                action_type=_ACTION_START_MISSION,
                target_system="suave",
                parameters={"task": TASK_SEARCH_PIPELINE},
            ),
            AdaptationAction(
                action_id="",
                action_type=_ACTION_START_MISSION,
                target_system="suave",
                parameters={"task": TASK_INSPECT_PIPELINE},
            ),
            AdaptationAction(
                action_id="",
                action_type=_ACTION_STOP_MISSION,
                target_system="suave",
                parameters={"task": TASK_SEARCH_PIPELINE},
            ),
            AdaptationAction(
                action_id="",
                action_type=_ACTION_STOP_MISSION,
                target_system="suave",
                parameters={"task": TASK_INSPECT_PIPELINE},
            ),
        ]
        for node, modes in _VALID_MODES.items():
            for mode in modes:
                actions.append(
                    AdaptationAction(
                        action_id="",
                        action_type=_ACTION_CHANGE_MODE,
                        target_system="suave",
                        parameters={"function_node": node, "mode": mode},
                    )
                )
        return actions

    # ------------------------------------------------------------------
    # Internal — action helpers
    # ------------------------------------------------------------------

    async def _execute_start_mission(self, action: AdaptationAction) -> ExecutionResult:
        """
        Start the mission via /task/request (suave_msgs/Task).

        The README specifies the managing subsystem must use /task/request to
        request tasks. Coordinate Mission then activates the correct lifecycle
        nodes:
          T1 search_pipeline  → f_generate_search_path + f_maintain_motion
          T2 follow_pipeline  → f_follow_pipeline + f_maintain_motion

        Polaris requests T1 first.  Coordinate Mission transitions to T2
        automatically when the pipeline is detected. Polaris should NOT call
        change_mode on f_follow_pipeline before pipeline_detected == 1.
        """
        params = action.parameters or {}
        task_name = params.get("task", TASK_SEARCH_PIPELINE)
        if task_name == TASK_FOLLOW_PIPELINE_LEGACY:
            task_name = TASK_INSPECT_PIPELINE

        svc = roslibpy.Service(self._client, "/task/request", "suave_msgs/Task")
        req = roslibpy.ServiceRequest({"task_name": task_name})

        try:
            response = await self._call_service(svc, req)
        except Exception as exc:
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=f"/task/request failed: {exc}",
            )

        self._mission_running = True

        if self._logger:
            self._logger.info(
                "SUAVEConnector: task requested",
                task=task_name,
                response=response,
            )

        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={"task": task_name, "response": response},
        )

    async def _execute_stop_mission(self, action: AdaptationAction) -> ExecutionResult:
        """
        Cancel the current task via /task/cancel (suave_msgs/Task).

        Coordinate Mission will deactivate the function nodes it manages.
        Non-fatal if the service errors — the task may have already ended.
        """
        params = action.parameters or {}
        task_name = params.get("task", TASK_SEARCH_PIPELINE)
        if task_name == TASK_FOLLOW_PIPELINE_LEGACY:
            task_name = TASK_INSPECT_PIPELINE

        svc = roslibpy.Service(self._client, "/task/cancel", "suave_msgs/Task")
        req = roslibpy.ServiceRequest({"task_name": task_name})

        try:
            response = await self._call_service(svc, req)
        except Exception as exc:
            if self._logger:
                self._logger.warning(
                    "SUAVEConnector: /task/cancel failed (task may already be done)",
                    error=str(exc),
                )
            response = {"error": str(exc)}

        self._mission_running = False
        self._pipeline_detected = False

        if self._logger:
            self._logger.info(
                "SUAVEConnector: task cancelled",
                task=task_name,
                response=response,
            )

        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={"task": task_name, "response": response},
        )

    async def _execute_change_mode(self, action: AdaptationAction) -> ExecutionResult:
        """
        Change a function node's mode via system_modes ChangeMode service.

        This is the ADAPTATION interface only — do not use it to start the
        mission (use start_mission → /task/request instead).

        Key mode semantics (Table V, SEAMS 2023):
          f_maintain_motion fd_all_thrusters     → inactive (normal, no recovery)
          f_maintain_motion fd_recover_thrusters → active   (recovery running)
          f_generate_search_path fd_spiral_*     → active at that altitude
          f_generate_search_path fd_unground     → inactive
          f_follow_pipeline fd_follow_pipeline   → active
          f_follow_pipeline fd_unground          → inactive

        To trigger thruster recovery after a failure, call:
          change_mode(f_maintain_motion, fd_recover_thrusters)

        To adapt search altitude based on water visibility, call:
          change_mode(f_generate_search_path, fd_spiral_low/medium/high)
        """
        params = action.parameters or {}
        function_node = params.get("function_node", "")
        mode = params.get("mode", "")

        if function_node not in _VALID_MODES:
            if self._metrics:
                self._metrics.increment("polaris.connector.suave.actions_validation_failed")
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=(
                    f"Unknown function_node '{function_node}'. "
                    f"Valid: {list(_VALID_MODES.keys())}"
                ),
            )

        if mode not in _VALID_MODES[function_node]:
            if self._metrics:
                self._metrics.increment("polaris.connector.suave.actions_validation_failed")
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={},
                error_message=(
                    f"Unknown mode '{mode}' for '{function_node}'. "
                    f"Valid: {_VALID_MODES[function_node]}"
                ),
            )

        service_name = f"/{function_node}/change_mode"
        svc = roslibpy.Service(self._client, service_name, "system_modes_msgs/ChangeMode")
        req = roslibpy.ServiceRequest({"mode_name": mode})
        response = await self._call_service(svc, req)

        if self._logger:
            self._logger.info(
                "SUAVEConnector: mode changed",
                function_node=function_node,
                mode=mode,
                response=response,
            )

        return ExecutionResult(
            action_id=action.action_id,
            status=ExecutionStatus.SUCCESS,
            result_data={
                "service": service_name,
                "function_node": function_node,
                "mode": mode,
                "response": response,
            },
        )

    # ------------------------------------------------------------------
    # Internal — rosbridge service call helper
    # ------------------------------------------------------------------

    async def _call_service(
        self,
        service: "_roslibpy.Service",
        request: "_roslibpy.ServiceRequest",
    ) -> dict:
        """Wrap roslibpy's callback-based service call into an awaitable."""
        loop = asyncio.get_event_loop()

        def _blocking_call() -> dict:
            result: dict = {}
            event = threading.Event()
            error_holder: List[Optional[Exception]] = [None]

            def _on_success(response: dict) -> None:
                result.update(response)
                event.set()

            def _on_error(error: str) -> None:
                error_holder[0] = RuntimeError(f"Service call failed: {error}")
                event.set()

            service.call(request, _on_success, _on_error)

            if not event.wait(timeout=self.service_timeout):
                raise TimeoutError(
                    f"Service '{service.name}' timed out after {self.service_timeout}s"
                )
            if error_holder[0] is not None:
                raise error_holder[0]
            return result

        return await loop.run_in_executor(None, _blocking_call)
