#!/usr/bin/env python3
"""
SWIM Monitor Adapter for POLARIS POC

Subjects (configurable):
- publish to: POLARIS_TELEMETRY_SUBJECT (default: "polaris.telemetry.events")
- batch publish: POLARIS_TELEMETRY_BATCH_SUBJECT (default: "polaris.telemetry.events.batch")

"""

import asyncio
import json
import logging
from pathlib import Path
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from nats.aio.client import Client as NATS

from polaris.common import setup_logging, now_iso, jittered_backoff
from polaris.common.config import load_config, get_config

# load config (adjust path(s) as needed)
load_config(
    search_paths=[Path(
        "/home/prakhar/dev/prakhar479/POLARIS/polaris_poc/config/polaris_config.yaml")],
    required_keys=["SWIM_HOST", "SWIM_PORT", "NATS_URL"],
)


# --------------------------------- Data Model ---------------------------------


@dataclass
class TelemetryEvent:
    """Structured telemetry event for POLARIS."""
    name: str
    value: Any
    unit: str
    ts: str
    source: str = "swim_monitor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "ts": self.ts,
            "source": self.source,
        }


# ---------------------------- SWIM Async TCP Client ----------------------------


class SwimTCPClientAsync:
    """
    Async TCP client for SWIM External Control.

    - Fresh connection per command prevents 'stuck' sockets when SWIM is busy.
    - Per-command timeout and retries with jittered backoff.
    """

    def __init__(self, host: str, port: int, timeout_sec: float, logger: logging.Logger):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.logger = logger

    async def _send_recv(self, command: str) -> str:
        t0 = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout_sec,
            )
        except Exception as e:
            raise TimeoutError(f"connect failed: {e}") from e

        try:
            writer.write((command + "\n").encode())
            await asyncio.wait_for(writer.drain(), timeout=self.timeout_sec)

            line = await asyncio.wait_for(reader.readline(), timeout=self.timeout_sec)
            resp = line.decode(errors="replace").strip()
            return resp
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.debug("swim_tcp", extra={
                "phase": "io_complete",
                "cmd": command,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            })

    async def send_with_retries(
        self,
        command: str,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        ctx: Dict[str, Any],
    ) -> str:
        attempt = 0
        while True:
            try:
                self.logger.info("swim_send", extra={
                                 **ctx, "cmd": command, "attempt": attempt})
                resp = await self._send_recv(command)
                self.logger.info("swim_resp", extra={
                                 **ctx, "cmd": command, "resp": resp, "attempt": attempt})
                return resp
            except (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError) as e:
                if attempt >= max_retries:
                    self.logger.error("swim_failed", extra={
                                      **ctx, "cmd": command, "attempt": attempt, "error": str(e)})
                    raise
                delay = jittered_backoff(attempt, base_delay, max_delay)
                self.logger.warning("swim_retry", extra={
                                    **ctx, "cmd": command, "attempt": attempt, "error": str(e), "retry_in_sec": round(delay, 3)})
                await asyncio.sleep(delay)
                attempt += 1


# ------------------------------- Monitor Adapter (NATS) --------------------------------


class SwimMonitorAdapter:
    """Monitor adapter: collects metrics from SWIM and publishes telemetry to NATS."""

    # safety limit for outstanding immediate publish tasks
    MAX_IMMEDIATE_TASKS = 1000

    def __init__(self):
        # ---------------- Config (ENV) ----------------
        self.nats_url = get_config("NATS_URL", "nats://localhost:4222")
        self.swim_host = get_config("SWIM_HOST", "localhost")
        self.swim_port = get_config("SWIM_PORT", "4242", int)
        self.swim_cmd_timeout = get_config("SWIM_CMD_TIMEOUT", "10", float)
        self.swim_max_retries = get_config("SWIM_MAX_RETRIES", "2", int)
        self.retry_base_delay = get_config(
            "SWIM_RETRY_BASE_DELAY", "0.8", float)
        self.retry_max_delay = get_config("SWIM_RETRY_MAX_DELAY", "4.0", float)

        self.monitor_interval = get_config("MONITOR_INTERVAL", "5.0", float)
        self.metrics_concurrency = get_config("METRICS_CONCURRENCY", "1", int)

        self.telemetry_stream_subject = get_config(
            "POLARIS_TELEMETRY_STREAM_SUBJECT", "polaris.telemetry.events.stream")
        self.telemetry_batch_subject = get_config(
            "POLARIS_TELEMETRY_BATCH_SUBJECT", "polaris.telemetry.events.batch")
        self.telemetry_batch_size = get_config(
            "TELEMETRY_BATCH_SIZE", "100", int)
        self.telemetry_batch_max_wait = get_config(
            "TELEMETRY_BATCH_MAX_WAIT", "1.0", float)

        # streaming per-event flag: publish each event immediately if True (in addition to batch)
        self.telemetry_stream = get_config(
            "TELEMETRY_STREAM", "false").lower() == "true"

        # Always present: track immediate per-event publish tasks
        self._immediate_publish_tasks: List[asyncio.Task] = []

        # NATS connect guard to avoid concurrent connect() calls
        self._nats_connect_lock = asyncio.Lock()

        self.queue_maxsize = get_config(
            "TELEMETRY_QUEUE_MAXSIZE", "5000", int)  # 0 => unbounded

        # Dynamic metric list (JSON or comma-separated)
        default_metrics = "dimmer,get_active_servers,get_max_servers,get_servers,get_basic_rt,get_opt_rt,get_basic_throughput,get_opt_throughput,get_arrival_rate"
        metrics_env = get_config("SWIM_METRIC_COMMANDS", default_metrics)
        if metrics_env.strip().startswith("["):
            try:
                self.metric_commands: List[str] = json.loads(metrics_env)
            except Exception:
                self.metric_commands = default_metrics.split(",")
        else:
            self.metric_commands = [m.strip()
                                    for m in metrics_env.split(",") if m.strip()]

        # Utilization scan enabled
        self.utilization_scan = get_config(
            "UTILIZATION_SCAN", "true").lower() == "true"
        self.utilization_prefix = get_config(
            "UTILIZATION_PREFIX", "get_utilization server")
        self.utilization_metric_name = get_config(
            "UTILIZATION_METRIC_NAME", "total_utilization")

        # Units mapping (override via JSON in env if needed)
        units_default = {
            "dimmer": "ratio",
            "get_active_servers": "count",
            "get_max_servers": "count",
            "get_servers": "count",
            "get_basic_rt": "ms",
            "get_opt_rt": "ms",
            "average_response_time": "ms",
            "get_basic_throughput": "req/s",
            "get_opt_throughput": "req/s",
            "get_arrival_rate": "req/s",
            self.utilization_metric_name: "ratio",
        }
        units_env = get_config("METRIC_UNITS_JSON", "")
        try:
            self.metric_units = {**units_default, **
                                 (json.loads(units_env) if units_env else {})}
        except Exception:
            self.metric_units = units_default

        # --------------- Logging ---------------
        self.logger = setup_logging()
        self.logger.info("monitor_init", extra={
            "nats_url": self.nats_url,
            "swim_host": self.swim_host,
            "swim_port": self.swim_port,
            "swim_cmd_timeout": self.swim_cmd_timeout,
            "swim_max_retries": self.swim_max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "monitor_interval": self.monitor_interval,
            "metrics_concurrency": self.metrics_concurrency,
            "telemetry_stream_subject": self.telemetry_stream_subject,
            "telemetry_batch_subject": self.telemetry_batch_subject,
            "telemetry_batch_size": self.telemetry_batch_size,
            "telemetry_batch_max_wait": self.telemetry_batch_max_wait,
            "queue_maxsize": self.queue_maxsize,
            "telemetry_stream": self.telemetry_stream,
        })

        # --------------- NATS & Queues ---------------
        self.nc: Optional[NATS] = None
        self.telemetry_q: asyncio.Queue[TelemetryEvent] = asyncio.Queue(
            maxsize=self.queue_maxsize if self.queue_maxsize > 0 else 0
        )

        # --------------- SWIM Client ---------------
        self.swim = SwimTCPClientAsync(
            host=self.swim_host, port=self.swim_port, timeout_sec=self.swim_cmd_timeout, logger=self.logger
        )

        # Lifecycle
        self.running = False
        self.collect_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None

    # ---------------- NATS Connection ----------------

    async def _ensure_nats(self) -> None:
        """Ensure NATS connected; reconnect with backoff if needed.

        This method is protected by an asyncio.Lock so that only one coroutine
        attempts to connect at a time (prevents read() races).
        """
        # Fast path: already connected
        if self.nc is not None and getattr(self.nc, "is_connected", False):
            return

        async with self._nats_connect_lock:
            # double-check under lock
            if self.nc is not None and getattr(self.nc, "is_connected", False):
                return

            # create a new client object if necessary
            if self.nc is None:
                self.nc = NATS()

            attempt = 0
            while True:
                try:
                    await self.nc.connect(self.nats_url)
                    self.logger.info("nats_connected", extra={
                                     "nats_url": self.nats_url})
                    return
                except Exception as e:
                    # Defensive: try to close and discard the client so subsequent attempts recreate clean state.
                    try:
                        await self.nc.close()
                    except Exception:
                        pass
                    self.nc = None

                    delay = jittered_backoff(
                        attempt, self.retry_base_delay, self.retry_max_delay)
                    self.logger.warning("nats_connect_retry", extra={
                                        "attempt": attempt, "error": str(e), "retry_in_sec": round(delay, 3)})
                    await asyncio.sleep(delay)
                    attempt += 1
                    self.nc = NATS()

    # ---------------- SWIM Metric Helpers ----------------

    def _unit_for(self, metric_or_name: str) -> str:
        return self.metric_units.get(metric_or_name, "unknown")

    def _normalized_name(self, cmd: str) -> str:
        mapping = {
            "get_dimmer": "dimmer",
            "get_active_servers": "active_servers",
            "get_max_servers": "max_servers",
            "get_servers": "servers",
            "get_basic_rt": "basic_response_time",
            "get_opt_rt": "optional_response_time",
            "get_basic_throughput": "basic_throughput",
            "get_opt_throughput": "optional_throughput",
            "get_arrival_rate": "arrival_rate",
        }
        return mapping.get(cmd, cmd)

    async def _query_metric(self, cmd: str, cycle_ctx: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Return (normalized_name, value) or None on failure."""
        try:
            resp = await self.swim.send_with_retries(
                command=cmd,
                max_retries=self.swim_max_retries,
                base_delay=self.retry_base_delay,
                max_delay=self.retry_max_delay,
                ctx={**cycle_ctx, "metric_cmd": cmd},
            )
            try:
                val = float(resp)
            except Exception:
                self.logger.warning("metric_non_numeric", extra={
                                    **cycle_ctx, "metric_cmd": cmd, "resp": resp})
                return None
            return (self._normalized_name(cmd), val)
        except Exception as e:
            self.logger.warning("metric_query_failed", extra={
                                **cycle_ctx, "metric_cmd": cmd, "error": str(e)})
            return None

    async def _query_utilization_sum(self, active_servers: int, cycle_ctx: Dict[str, Any]) -> Optional[float]:
        total = 0.0
        for sid in range(1, max(0, int(active_servers)) + 1):
            cmd = f"{self.utilization_prefix}{sid}"
            try:
                resp = await self.swim.send_with_retries(
                    command=cmd,
                    max_retries=self.swim_max_retries,
                    base_delay=self.retry_base_delay,
                    max_delay=self.retry_max_delay,
                    ctx={**cycle_ctx, "metric_cmd": cmd},
                )
                try:
                    total += float(resp)
                except Exception:
                    self.logger.warning("util_non_numeric", extra={
                                        **cycle_ctx, "metric_cmd": cmd, "resp": resp})
            except Exception as e:
                self.logger.warning("util_query_failed", extra={
                                    **cycle_ctx, "metric_cmd": cmd, "error": str(e)})
        return total

    # ----------------- Collection → Stream -------------------------

    async def _publish_event(self, event: TelemetryEvent) -> None:
        """Publish a single telemetry event to the per-event subject."""
        try:
            await self._ensure_nats()
            assert self.nc is not None
            await self.nc.publish(self.telemetry_stream_subject, json.dumps(event.to_dict()).encode())
            self.logger.debug("telemetry_event_published", extra={
                              "name": event.name, "subject": self.telemetry_stream_subject})
        except Exception as e:
            self.logger.warning("telemetry_event_publish_failed", extra={
                                "error": str(e), "event": event.name})

    # ---------------- Collection → Queue (with a Stream Flag) ----------------

    async def _collect_once(self) -> int:
        """
        Collect metrics once and enqueue TelemetryEvents.
        Returns number of enqueued events.
        """
        cycle_id = str(uuid.uuid4())
        cycle_ctx = {"cycle_id": cycle_id}
        t0 = time.perf_counter()

        # Optional concurrency limit
        sem = asyncio.Semaphore(self.metrics_concurrency)

        async def _wrapped(cmd: str):
            async with sem:
                return await self._query_metric(cmd, cycle_ctx)

        # 1) Base metrics
        tasks = [asyncio.create_task(_wrapped(cmd))
                 for cmd in self.metric_commands]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        metrics: Dict[str, float] = {
            name: val for r in results if r is not None for name, val in [r]}

        # 2) Derived avg response time
        if all(k in metrics for k in ("basic_response_time", "optional_response_time", "basic_throughput", "optional_throughput")):
            tput = metrics["basic_throughput"] + metrics["optional_throughput"]
            if tput > 0:
                avg_rt = (
                    metrics["basic_response_time"] * metrics["basic_throughput"] +
                    metrics["optional_response_time"] *
                    metrics["optional_throughput"]
                ) / tput
                metrics["average_response_time"] = avg_rt

        # 3) Utilization sum if enabled
        if self.utilization_scan and "active_servers" in metrics:
            util_sum = await self._query_utilization_sum(int(metrics["active_servers"]), cycle_ctx)
            if util_sum is not None:
                metrics[self.utilization_metric_name] = util_sum

        # 4) Enqueue telemetry events + optional immediate publish
        enqueued = 0
        ts = now_iso()
        for metric_name, value in metrics.items():
            unit = self._unit_for(metric_name)
            evt = TelemetryEvent(
                name=f"swim.{metric_name}", value=value, unit=unit, ts=ts, source="swim_monitor")
            try:
                self.telemetry_q.put_nowait(evt)
                enqueued += 1
            except asyncio.QueueFull:
                self.logger.error("telemetry_queue_full", extra={
                                  "cycle_id": cycle_id, "dropped_metric": metric_name})
                break

            # streaming mode: schedule immediate per-event publish (fire-and-forget tracked)
            if self.telemetry_stream:
                try:
                    task = asyncio.create_task(self._publish_event(evt))
                    self._immediate_publish_tasks.append(task)
                    if len(self._immediate_publish_tasks) > self.MAX_IMMEDIATE_TASKS:
                        self.logger.warning("too_many_immediate_publish_tasks", extra={
                                            "count": len(self._immediate_publish_tasks)})

                    def _on_done(t: asyncio.Task):
                        try:
                            self._immediate_publish_tasks.remove(t)
                        except ValueError:
                            pass
                    task.add_done_callback(_on_done)
                except Exception as e:
                    self.logger.warning(
                        "immediate_publish_task_failed", extra={"error": str(e)})

        elapsed = time.perf_counter() - t0
        self.logger.info("collect_cycle_done", extra={
            "cycle_id": cycle_id,
            "metrics_collected": len(metrics),
            "events_enqueued": enqueued,
            "elapsed_ms": round(elapsed * 1000, 3),
            "queue_size": self.telemetry_q.qsize(),
        })
        return enqueued

    async def _collector_loop(self):
        self.logger.info("collector_started")
        try:
            while self.running:
                try:
                    await self._collect_once()
                except Exception as e:
                    self.logger.exception(
                        "collector_error", extra={"error": str(e)})
                await asyncio.sleep(self.monitor_interval)
        except asyncio.CancelledError:
            self.logger.info("collector_cancelled")
        finally:
            self.logger.info("collector_stopped")

    # ---------------- Queue → NATS (Batch Publisher) ----------------

    async def _flush_batch(self, batch: List[TelemetryEvent]) -> None:
        """Publish a batch to NATS; falls back to per-item if batch publish fails."""
        if not batch:
            return

        await self._ensure_nats()
        assert self.nc is not None

        payload = {"batch_ts": now_iso(), "count": len(
            batch), "events": [e.to_dict() for e in batch]}
        try:
            await self.nc.publish(self.telemetry_batch_subject, json.dumps(payload).encode())
            self.logger.info("telemetry_batch_published", extra={"count": len(
                batch), "subject": self.telemetry_batch_subject, "queue_size_after": self.telemetry_q.qsize()})
            return
        except Exception as e:
            self.logger.warning(
                "telemetry_batch_publish_failed", extra={"error": str(e)})

        # Fallback: publish items individually
        published = 0
        for e in batch:
            try:
                await self.nc.publish(self.telemetry_stream_subject, json.dumps(e.to_dict()).encode())
                published += 1
            except Exception as ex:
                self.logger.error("telemetry_publish_failed", extra={
                                  "error": str(ex), "event_name": e.name})
                break

        self.logger.info("telemetry_fallback_publish_done", extra={
            "published": published, "attempted": len(batch), "subject": self.telemetry_stream_subject, "queue_size_after": self.telemetry_q.qsize()
        })

    async def _publisher_loop(self):
        self.logger.info("publisher_started", extra={
                         "batch_size": self.telemetry_batch_size, "batch_max_wait": self.telemetry_batch_max_wait})
        try:
            while self.running:
                batch: List[TelemetryEvent] = []

                # Always take at least one (blocking if empty)
                try:
                    first = await asyncio.wait_for(self.telemetry_q.get(), timeout=self.telemetry_batch_max_wait)
                    batch.append(first)
                except asyncio.TimeoutError:
                    # no items ready to flush — loop to wait again
                    continue

                # Collect up to batch_size - 1 more items without waiting too long
                t_start = time.perf_counter()
                while len(batch) < self.telemetry_batch_size:
                    remaining = self.telemetry_batch_max_wait - \
                        (time.perf_counter() - t_start)
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self.telemetry_q.get(), timeout=max(0.0, remaining))
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

                try:
                    await self._flush_batch(batch)
                finally:
                    # mark done for every dequeued item
                    for _ in batch:
                        self.telemetry_q.task_done()

        except asyncio.CancelledError:
            self.logger.info("publisher_cancelled")
        finally:
            self.logger.info("publisher_stopped")

    # ---------------- Lifecycle ----------------

    async def start(self):
        self.logger.info("monitor_starting")
        # Try to establish NATS early to reduce race/handshake attempts later
        try:
            await self._ensure_nats()
        except Exception as e:
            self.logger.warning("initial_nats_connect_failed",
                                extra={"error": str(e)})

        self.running = True
        self.collect_task = asyncio.create_task(self._collector_loop())
        # always run publisher loop for batch behaviour
        self.publish_task = asyncio.create_task(self._publisher_loop())
        self.logger.info("monitor_started", extra={
                         "status": "running", "telemetry_stream": self.telemetry_stream})

    async def drain_and_stop(self, drain_timeout: float = 10.0):
        """Graceful shutdown: stop intake, drain queue, and close NATS."""
        self.logger.info("monitor_stopping")
        self.running = False

        # Cancel collector first (no more enqueue)
        if self.collect_task and not self.collect_task.done():
            self.collect_task.cancel()
            try:
                await self.collect_task
            except asyncio.CancelledError:
                pass

        # Drain Batch queue (let publisher flush)
        try:
            await asyncio.wait_for(self.telemetry_q.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("telemetry_drain_timeout", extra={
                                "remaining": self.telemetry_q.qsize()})

        # Stop publisher
        if self.publish_task and not self.publish_task.done():
            self.publish_task.cancel()
            try:
                await self.publish_task
            except asyncio.CancelledError:
                pass

        # Drain Stream immediate publish tasks (if any)
        if self._immediate_publish_tasks:
            self.logger.info("waiting_immediate_publish_tasks", extra={
                             "count": len(self._immediate_publish_tasks)})
            try:
                done, pending = await asyncio.wait(self._immediate_publish_tasks, timeout=2.0)
            except Exception:
                for t in list(self._immediate_publish_tasks):
                    t.cancel()
                pending = []
                done = []
            for t in pending:
                try:
                    t.cancel()
                except Exception:
                    pass
            self._immediate_publish_tasks.clear()

        # Close NATS
        try:
            if self.nc:
                await self.nc.close()
        except Exception:
            pass

        self.logger.info("monitor_stopped")


# ----------------------------------- Main -------------------------------------


async def main():
    adapter = SwimMonitorAdapter()
    await adapter.start()

    # Signal handling for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        adapter.logger.info("signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _signal_handler())

    await stop_event.wait()
    await adapter.drain_and_stop(drain_timeout=10.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
