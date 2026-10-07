"""SQLite-backed persistent knowledge store.

Zero external dependencies — uses Python's built-in ``sqlite3`` module. States and
actions survive process restarts.

Usage::

    store = SQLiteKnowledgeStore(db_path="./polaris_data.db")
    await store.store_state(state)
    states = await store.query_states(system_id, start, end)
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.core.models import (
    ActionWorkflow,
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
    WorkflowStatus,
)
from polaris.infrastructure.constants import DEFAULT_MAX_STATES_PER_SYSTEM

# SQL DDL

_CREATE_STATES = """
CREATE TABLE IF NOT EXISTS system_states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    system_id   TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    metrics     TEXT    NOT NULL,  -- JSON
    health      TEXT    NOT NULL,
    metadata    TEXT              -- JSON or NULL
);
CREATE INDEX IF NOT EXISTS idx_states_system_ts ON system_states (system_id, timestamp);
"""

_CREATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS adaptation_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id       TEXT    NOT NULL,
    target_system   TEXT    NOT NULL,
    action_type     TEXT    NOT NULL,
    parameters      TEXT    NOT NULL,  -- JSON
    created_at      TEXT,
    result_status   TEXT    NOT NULL,
    result_data     TEXT,              -- JSON or NULL
    result_error    TEXT,
    started_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_system ON adaptation_actions (target_system);
"""

_CREATE_TOPOLOGY = """
CREATE TABLE IF NOT EXISTS system_topology (
    id          TEXT    PRIMARY KEY,
    data        TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""

_CREATE_WORKFLOWS = """
CREATE TABLE IF NOT EXISTS system_workflows (
    workflow_id                 TEXT PRIMARY KEY,
    target_system               TEXT NOT NULL,
    action_type                 TEXT NOT NULL,
    status                      TEXT NOT NULL,
    verification_window_seconds REAL,
    error_message               TEXT,
    created_at                  TEXT,
    updated_at                  TEXT,
    action_json                 TEXT NOT NULL,
    result_json                 TEXT,
    rollback_json               TEXT
);
CREATE INDEX IF NOT EXISTS idx_workflows_system ON system_workflows (target_system);
"""


class SQLiteKnowledgeStore(KnowledgeStore):
    """Persistent knowledge store backed by SQLite.

    All blocking I/O is dispatched to a thread-pool executor so the asyncio event loop
    remains free.

    Args:
        db_path: Path to the SQLite database file.  The parent directory is created
            automatically.  Pass ``":memory:"`` for an in-process ephemeral store
            (useful for testing).
        max_states_per_system: How many states to retain per system.  Older rows are
            pruned automatically after each ``store_state`` call.
        logger: Optional structured logger.
        metrics: Optional metrics collector.
    """

    def __init__(
        self,
        db_path: str = "./polaris_data.db",
        max_states_per_system: int = DEFAULT_MAX_STATES_PER_SYSTEM,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize persistent sqllite knowledge store with capacity limits."""
        self._db_path = db_path
        self._max_states = max_states_per_system
        self._logger = logger
        self._metrics = metrics
        self._topology: Any = None
        # For :memory: databases we must reuse a single connection because each
        # sqlite3.connect(":memory:") call opens a brand-new, empty database.
        self._shared_conn: Optional[sqlite3.Connection] = None

        # Initialise the schema synchronously once at construction time.
        # This is intentionally sync so it completes before the first await.
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(db_path, check_same_thread=False)
            con = self._shared_conn
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(db_path, check_same_thread=False)
        try:
            con.executescript(_CREATE_STATES)
            con.executescript(_CREATE_ACTIONS)
            con.executescript(_CREATE_TOPOLOGY)
            con.executescript(_CREATE_WORKFLOWS)
            con.commit()
        finally:
            if self._shared_conn is None:
                con.close()

        if self._logger:
            self._logger.info("SQLiteKnowledgeStore initialised", db_path=db_path)

        if self._metrics:
            self._metrics.increment("polaris.knowledge.sqlite.initialized")

    # Helpers

    def _connect(self) -> sqlite3.Connection:
        """Return a database connection.

        For :memory: databases the shared persistent connection is returned. For file-
        based databases a fresh connection is opened (per-operation, safe for multi-
        threaded executor use).
        """
        if self._shared_conn is not None:
            return self._shared_conn
        return sqlite3.connect(self._db_path, check_same_thread=False)

    def _close(self, con: sqlite3.Connection) -> None:
        """Close *con* unless it is the shared in-memory connection."""
        if con is not self._shared_conn:
            con.close()

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Run a blocking callable in the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    # Serialisation

    @staticmethod
    def _serialise_state(state: SystemState) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for name, mv in state.metrics.items():
            metrics[name] = {
                "name": mv.name,
                "value": mv.value,
                "unit": mv.unit,
                "timestamp": mv.timestamp.isoformat() if mv.timestamp else None,
                "tags": mv.tags or {},
            }
        return {
            "system_id": state.system_id,
            "timestamp": state.timestamp.isoformat(),
            "metrics": metrics,
            "health": (
                state.health_status.value
                if hasattr(state.health_status, "value")
                else str(state.health_status)
            ),
            "metadata": state.metadata or {},
        }

    @staticmethod
    def _deserialise_state(row: sqlite3.Row) -> SystemState:
        raw_metrics: Dict[str, Any] = json.loads(row["metrics"])
        metrics: Dict[str, MetricValue] = {}
        for name, mv_data in raw_metrics.items():
            ts = datetime.fromisoformat(mv_data["timestamp"]) if mv_data.get("timestamp") else None
            metrics[name] = MetricValue(
                name=mv_data["name"],
                value=mv_data["value"],
                unit=mv_data.get("unit") or "",
                timestamp=ts,
                tags=mv_data.get("tags") or {},
            )
        try:
            health = HealthStatus(row["health"])
        except ValueError:
            health = HealthStatus.UNKNOWN

        meta: Optional[Dict[str, Any]] = None
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = None

        return SystemState(
            system_id=row["system_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metrics=metrics,
            health_status=health,
            metadata=meta,
        )

    # KnowledgeStore interface

    async def store_state(self, state: SystemState) -> None:
        """Persist a system state snapshot and prune old rows."""

        def _insert(state: SystemState) -> None:
            data = SQLiteKnowledgeStore._serialise_state(state)
            con = self._connect()
            try:
                con.execute(
                    "INSERT INTO system_states (system_id, timestamp, metrics, health, metadata)"
                    " VALUES (?,?,?,?,?)",
                    (
                        data["system_id"],
                        data["timestamp"],
                        json.dumps(data["metrics"]),
                        data["health"],
                        json.dumps(data["metadata"]) if data["metadata"] else None,
                    ),
                )
                # Prune oldest rows beyond the cap
                con.execute(
                    """
                    DELETE FROM system_states
                    WHERE system_id = ?
                      AND id NOT IN (
                          SELECT id FROM system_states
                          WHERE system_id = ?
                          ORDER BY timestamp DESC
                          LIMIT ?
                      )
                    """,
                    (data["system_id"], data["system_id"], self._max_states),
                )
                con.commit()
            finally:
                self._close(con)

        await self._run(_insert, state)

        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.sqlite.states_stored",
                tags={"system_id": state.system_id},
            )

    async def store_action(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Persist an adaptation action + its execution result."""

        def _insert(action: AdaptationAction, result: ExecutionResult) -> None:
            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO adaptation_actions
                        (action_id, target_system, action_type, parameters, created_at,
                         result_status, result_data, result_error, completed_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        action.action_id,
                        action.target_system,
                        action.action_type,
                        json.dumps(action.parameters or {}),
                        action.created_at.isoformat() if action.created_at else None,
                        (
                            result.status.value
                            if hasattr(result.status, "value")
                            else str(result.status)
                        ),
                        json.dumps(result.result_data) if result.result_data else None,
                        result.error_message,
                        result.completed_at.isoformat() if result.completed_at else None,
                    ),
                )
                con.commit()
            finally:
                self._close(con)

        await self._run(_insert, action, result)

        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.sqlite.actions_stored",
                tags={
                    "system_id": action.target_system,
                    "status": result.status.value if hasattr(result.status, "value") else "unknown",
                },
            )

    async def query_states(
        self, system_id: str, start_time: datetime, end_time: datetime
    ) -> List[SystemState]:
        """Return all stored states for a system within [start_time, end_time]."""

        def _query(system_id: str, start: str, end: str) -> List[sqlite3.Row]:
            con = self._connect()
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(
                    "SELECT * FROM system_states"
                    " WHERE system_id=? AND timestamp>=? AND timestamp<=?"
                    " ORDER BY timestamp ASC",
                    (system_id, start, end),
                )
                return cur.fetchall()
            finally:
                self._close(con)

        rows = await self._run(_query, system_id, start_time.isoformat(), end_time.isoformat())
        states = []
        for row in rows:
            try:
                states.append(self._deserialise_state(row))
            except Exception as exc:
                if self._logger:
                    self._logger.warning(
                        f"SQLiteKnowledgeStore: failed to deserialise state row: {exc}"
                    )

        if self._metrics:
            self._metrics.gauge(
                "polaris.knowledge.sqlite.state_query_results",
                len(states),
                tags={"system_id": system_id},
            )

        return states

    async def query_actions(
        self,
        system_id: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[Tuple[AdaptationAction, ExecutionResult]]:
        """Return adaptation history for a system, optionally filtered by time."""

        def _query(system_id: str, start: Optional[str], end: Optional[str]) -> List[sqlite3.Row]:
            con = self._connect()
            con.row_factory = sqlite3.Row
            try:
                if start and end:
                    cur = con.execute(
                        "SELECT * FROM adaptation_actions"
                        " WHERE target_system=? AND created_at>=? AND created_at<=?"
                        " ORDER BY created_at ASC",
                        (system_id, start, end),
                    )
                elif start:
                    cur = con.execute(
                        "SELECT * FROM adaptation_actions"
                        " WHERE target_system=? AND created_at>=?"
                        " ORDER BY created_at ASC",
                        (system_id, start),
                    )
                elif end:
                    cur = con.execute(
                        "SELECT * FROM adaptation_actions"
                        " WHERE target_system=? AND created_at<=?"
                        " ORDER BY created_at ASC",
                        (system_id, end),
                    )
                else:
                    cur = con.execute(
                        "SELECT * FROM adaptation_actions"
                        " WHERE target_system=?"
                        " ORDER BY created_at ASC",
                        (system_id,),
                    )
                return cur.fetchall()
            finally:
                self._close(con)

        rows = await self._run(
            _query,
            system_id,
            start_time.isoformat() if start_time else None,
            end_time.isoformat() if end_time else None,
        )

        results: List[Tuple[AdaptationAction, ExecutionResult]] = []
        for row in rows:
            try:
                action = AdaptationAction(
                    action_id=row["action_id"],
                    target_system=row["target_system"],
                    action_type=row["action_type"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    created_at=(
                        datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
                    ),
                )
                result = ExecutionResult(
                    action_id=row["action_id"],
                    status=ExecutionStatus(row["result_status"]),
                    result_data=json.loads(row["result_data"]) if row["result_data"] else {},
                    error_message=row["result_error"],
                    completed_at=(
                        datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                    ),
                )
                results.append((action, result))
            except Exception as exc:
                if self._logger:
                    self._logger.warning(
                        f"SQLiteKnowledgeStore: failed to deserialise action row: {exc}"
                    )

        return results

    async def get_latest_state(self, system_id: str) -> Optional[SystemState]:
        """Return the most recently stored state for a system."""

        def _query(system_id: str) -> Any:
            con = self._connect()
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(
                    "SELECT * FROM system_states WHERE system_id=? ORDER BY timestamp DESC LIMIT 1",
                    (system_id,),
                )
                return cur.fetchone()
            finally:
                self._close(con)

        row = await self._run(_query, system_id)
        if not row:
            return None
        try:
            return self._deserialise_state(row)
        except Exception as exc:
            if self._logger:
                self._logger.warning(
                    f"SQLiteKnowledgeStore: failed to deserialise latest state row: {exc}"
                )
            return None

    async def store_topology(self, topology: Any) -> None:
        """Store system topology graph persistently in SQLite."""
        self._topology = topology
        if topology is None:
            return

        def _persist(topo_data: str) -> None:
            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO system_topology (id, data, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
                    """,
                    ("current", topo_data, datetime.now(timezone.utc).isoformat()),
                )
                con.commit()
            finally:
                self._close(con)

        topo_dict = topology.to_dict() if hasattr(topology, "to_dict") else {}
        await self._run(_persist, json.dumps(topo_dict))

    async def get_topology(self) -> Any:
        """Retrieve system topology graph from SQLite."""
        if self._topology is not None:
            return self._topology

        def _query() -> Optional[str]:
            con = self._connect()
            try:
                cur = con.execute("SELECT data FROM system_topology WHERE id = 'current'")
                row = cur.fetchone()
                return str(row[0]) if row else None
            finally:
                self._close(con)

        raw_json = await self._run(_query)
        if raw_json:
            try:
                from polaris.core.topology import SystemTopology

                self._topology = SystemTopology.from_dict(json.loads(raw_json))
                return self._topology
            except Exception:
                return None
        return None

    async def store_workflow(self, workflow: ActionWorkflow) -> None:
        """Store action workflow lifecycle in SQLite."""

        def _insert() -> None:
            con = self._connect()
            try:
                act = workflow.action
                action_data = {
                    "action_id": act.action_id,
                    "action_type": act.action_type,
                    "target_system": act.target_system,
                    "parameters": act.parameters or {},
                }
                res_data = None
                if workflow.execution_result:
                    res_data = {
                        "action_id": workflow.execution_result.action_id,
                        "status": (
                            workflow.execution_result.status.value
                            if hasattr(workflow.execution_result.status, "value")
                            else str(workflow.execution_result.status)
                        ),
                        "result_data": workflow.execution_result.result_data,
                        "error_message": workflow.execution_result.error_message,
                        "execution_time_ms": workflow.execution_result.execution_time_ms,
                    }
                con.execute(
                    """
                    INSERT INTO system_workflows
                        (workflow_id, target_system, action_type, status,
                         verification_window_seconds, error_message, created_at, updated_at,
                         action_json, result_json, rollback_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        status=excluded.status,
                        error_message=excluded.error_message,
                        updated_at=excluded.updated_at,
                        result_json=excluded.result_json,
                        rollback_json=excluded.rollback_json
                    """,
                    (
                        workflow.workflow_id,
                        act.target_system,
                        act.action_type,
                        (
                            workflow.status.value
                            if hasattr(workflow.status, "value")
                            else str(workflow.status)
                        ),
                        workflow.verification_window_seconds,
                        workflow.error_message,
                        workflow.created_at.isoformat() if workflow.created_at else None,
                        workflow.updated_at.isoformat() if workflow.updated_at else None,
                        json.dumps(action_data),
                        json.dumps(res_data) if res_data else None,
                        None,
                    ),
                )
                con.commit()
            finally:
                self._close(con)

        await self._run(_insert)
        if self._metrics:
            self._metrics.increment(
                "polaris.knowledge.sqlite.workflows_stored",
                tags={
                    "system_id": workflow.action.target_system,
                    "status": (
                        workflow.status.value
                        if hasattr(workflow.status, "value")
                        else str(workflow.status)
                    ),
                },
            )

    async def query_workflows(self, system_id: str, limit: int = 100) -> List[ActionWorkflow]:
        """Query action workflows for a system from SQLite."""

        def _query() -> List[sqlite3.Row]:
            con = self._connect()
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(
                    """
                    SELECT * FROM system_workflows
                    WHERE target_system = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (system_id, max(1, limit)),
                )
                return cur.fetchall()
            finally:
                self._close(con)

        rows = await self._run(_query)
        workflows: List[ActionWorkflow] = []
        for row in reversed(rows):
            try:
                act_dict = json.loads(row["action_json"])
                action = AdaptationAction(
                    action_id=act_dict["action_id"],
                    action_type=act_dict["action_type"],
                    target_system=act_dict["target_system"],
                    parameters=act_dict.get("parameters", {}),
                )
                res = None
                if row["result_json"]:
                    res_dict = json.loads(row["result_json"])
                    res = ExecutionResult(
                        action_id=res_dict["action_id"],
                        status=ExecutionStatus(res_dict["status"]),
                        result_data=res_dict.get("result_data", {}),
                        error_message=res_dict.get("error_message"),
                        execution_time_ms=res_dict.get("execution_time_ms"),
                    )

                wf = ActionWorkflow(
                    action=action,
                    workflow_id=row["workflow_id"],
                    status=WorkflowStatus(row["status"]),
                    verification_window_seconds=float(row["verification_window_seconds"] or 0.0),
                    execution_result=res,
                    error_message=row["error_message"],
                    created_at=(
                        datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
                    ),
                    updated_at=(
                        datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
                    ),
                )
                workflows.append(wf)
            except Exception as exc:
                if self._logger:
                    self._logger.warning(f"Failed to deserialise workflow row: {exc}")
        return workflows
