"""THREAD-inspired recursive agentic adaptation strategy for POLARIS.

https://arxiv.org/pdf/2405.17402

This strategy implements a practical version of the THREAD paper's join-synchronized
recursive spawning pattern. A parent reasoning thread can spawn a child thread,
wait for child completion, ingest a compact child return payload, and continue.

Design notes:
- Root thread produces the final adaptation decision.
- Child threads produce compact return payloads for their parent.
- Tools are executed through the shared ToolRegistry infrastructure.
- Safety controls prevent runaway recursion and context growth.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import WorldModel
from polaris.core.models import AdaptationAction, SystemState
from polaris.infrastructure.constants import DEFAULT_MAX_TOKENS_REASONING
from polaris.infrastructure.llm import LLMClient, LLMMessage
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.tools import ToolDependencies, ToolRegistry, get_builtin_tools


class ActionBlock(BaseModel):
    """A block defining an action to be executed."""

    type: str = Field(description="The name or type of the action to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters required for the action"
    )


class SpawnBlock(BaseModel):
    """A block requesting recursive child-thread execution."""

    objective: str = Field(description="Sub-problem to solve in a child thread")
    context_hint: Optional[str] = Field(
        default=None,
        description="Optional extra context for the child thread",
    )


class ThreadFinalBlock(BaseModel):
    """A final output block for either root or child threads."""

    needs_adaptation: Optional[bool] = Field(
        default=None,
        description="Set by root thread for adaptation decision",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for the root decision or child summary",
    )
    actions: List[ActionBlock] = Field(
        default_factory=list,
        description="Root-thread proposed actions",
    )
    action: Optional[ActionBlock] = Field(
        default=None,
        description="Deprecated single action field for compatibility",
    )
    return_payload: Optional[str] = Field(
        default=None,
        description="Compact payload for child-to-parent return",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize with backward-compatibility for singular action."""
        super().__init__(**data)
        if self.action is not None and not self.actions:
            self.actions = [self.action]


class ThreadAgenticResponse(BaseModel):
    """Structured response schema for each thread step."""

    tool: Optional[str] = Field(
        default=None,
        description="Tool name to call. Leave null when spawning or finalizing.",
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments for the selected tool",
    )
    spawn: Optional[SpawnBlock] = Field(
        default=None,
        description="Spawn request for a child thread",
    )
    final: Optional[ThreadFinalBlock] = Field(
        default=None,
        description="Final output for this thread",
    )


@dataclass
class _ThreadResult:
    """Result returned from a thread execution."""

    return_payload: str
    final: Optional[ThreadFinalBlock] = None


@dataclass
class _ThreadRuntime:
    """Runtime state shared across recursive threads in one assess call."""

    total_threads: int = 0
    max_depth_reached: int = 0
    spawn_count: int = 0
    spawn_denied_depth: int = 0
    spawn_denied_budget: int = 0
    spawn_repeat_blocked: int = 0
    child_timeouts: int = 0
    tool_calls: int = 0
    spawn_signature_counts: Dict[str, int] = field(default_factory=dict)


class ThreadAgenticStrategy(AdaptationStrategy):
    """Recursive THREAD-style adaptation strategy with join synchronization."""

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        connector_getter: Optional[Callable[[str], Optional["Connector"]]] = None,
        steps_limit: int = 4,
        temperature: float = 0.1,
        max_thread_depth: int = 3,
        max_total_threads: int = 16,
        child_timeout_seconds: float = 20.0,
        max_repeated_spawns: int = 2,
        max_tool_result_chars: int = 1200,
        max_child_payload_chars: int = 800,
        phi_mode: str = "last_line",
        phi_max_lines: int = 6,
        listen_token: str = "=>",
        return_token: str = "<=",
        allowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        per_system_prompts: Optional[Dict[str, str]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        """Initialize the THREAD strategy.

        Args:
            llm_client: LLM client for structured generation.
            knowledge_store: Historical state/action store.
            world_model: World model for prediction tools.
            connector_getter: Optional connector provider for connector-aware tools.
            steps_limit: Max step count per thread.
            temperature: LLM temperature.
            max_thread_depth: Maximum child depth under root.
            max_total_threads: Global thread budget per assess call.
            child_timeout_seconds: Timeout for each child thread run.
            max_repeated_spawns: Guardrail against repeated identical spawn requests.
            max_tool_result_chars: Max serialized tool payload size injected to model.
            max_child_payload_chars: Max child payload size propagated via psi.
            phi_mode: Parent-to-child context mapping mode: last_line or recent_lines.
            phi_max_lines: Number of lines used when phi_mode is recent_lines.
            listen_token: Token prefix used for child feedback framing.
            return_token: Token suffix used for child feedback framing.
            allowed_tools: Enabled tool names.
            system_prompt: Optional prompt template override.
            per_system_prompts: Optional per-system prompt overrides.
            logger: Optional structured logger.
            metrics: Optional metrics collector.
        """
        self.llm = llm_client
        self.knowledge_store = knowledge_store
        self.world_model = world_model
        self._get_connector = connector_getter

        self.steps_limit = max(1, int(steps_limit))
        self.temperature = float(temperature)
        self.max_thread_depth = max(0, int(max_thread_depth))
        self.max_total_threads = max(1, int(max_total_threads))
        self.child_timeout_seconds = max(0.1, float(child_timeout_seconds))
        self.max_repeated_spawns = max(1, int(max_repeated_spawns))
        self.max_tool_result_chars = max(200, int(max_tool_result_chars))
        self.max_child_payload_chars = max(100, int(max_child_payload_chars))
        self.phi_mode = str(phi_mode or "last_line")
        self.phi_max_lines = max(1, int(phi_max_lines))
        self.listen_token = str(listen_token or "=>")
        self.return_token = str(return_token or "<=")

        self.allowed_tools = allowed_tools or [
            "get_recent_states",
            "summarize_metric_trends",
            "list_metric_fields",
            "compute_metric_math",
            "get_world_model_insights",
            "predict_outcome",
            "get_action_history",
            "list_supported_actions",
        ]

        self._system_prompt_template = system_prompt
        self._per_system_prompts = per_system_prompts or {}
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()

        self._tool_registry = ToolRegistry(metrics=self.metrics)
        all_tools = get_builtin_tools()
        if self.allowed_tools:
            for tool in all_tools:
                if tool.name in self.allowed_tools:
                    self._tool_registry.register(tool)
        else:
            self._tool_registry.register_all(all_tools)

        self._adaptation_count = 0
        self._success_count = 0

    async def assess(
        self,
        state: SystemState,
        context: AdaptationContext,
    ) -> List[AdaptationAction]:
        """Assess system state using recursive THREAD-style reasoning."""
        if self.logger:
            self.logger.debug("ThreadAgentic assessment started", system_id=state.system_id)

        self.metrics.increment(
            "polaris.strategy.thread_agentic.assessments",
            tags={"system_id": state.system_id},
        )

        start = datetime.now(timezone.utc)
        runtime = _ThreadRuntime()
        try:
            root_input = self._initial_user_prompt(state, context)
            result = await self._run_thread(
                state=state,
                context=context,
                system_id=state.system_id,
                thread_input=root_input,
                depth=0,
                lineage=(),
                runtime=runtime,
            )

            self.metrics.gauge(
                "polaris.strategy.thread_agentic.max_depth",
                runtime.max_depth_reached,
                tags={"system_id": state.system_id},
            )
            self.metrics.gauge(
                "polaris.strategy.thread_agentic.total_threads",
                runtime.total_threads,
                tags={"system_id": state.system_id},
            )
            self.metrics.gauge(
                "polaris.strategy.thread_agentic.tool_calls",
                runtime.tool_calls,
                tags={"system_id": state.system_id},
            )

            final = result.final
            if final is None:
                return []

            needs_adaptation = final.needs_adaptation
            if needs_adaptation is None:
                needs_adaptation = bool(final.actions)

            if not needs_adaptation:
                return []

            if not final.actions:
                if self.logger:
                    self.logger.warning(
                        "ThreadAgentic decision needs adaptation but actions are empty",
                        system_id=state.system_id,
                    )
                return []

            reasoning = final.reasoning or "Thread strategy recommended adaptation"
            proposed: List[AdaptationAction] = []
            for action_block in final.actions:
                if not action_block.type:
                    continue
                proposed.append(
                    AdaptationAction(
                        action_id=str(uuid.uuid4()),
                        action_type=action_block.type,
                        target_system=state.system_id,
                        parameters={
                            **action_block.parameters,
                            "llm_reasoning": reasoning,
                            "thread_depth": runtime.max_depth_reached,
                            "thread_count": runtime.total_threads,
                        },
                    )
                )

            for action in proposed:
                self.metrics.increment(
                    "polaris.strategy.thread_agentic.actions_proposed",
                    tags={
                        "system_id": state.system_id,
                        "action_type": action.action_type,
                    },
                )

            return proposed
        finally:
            self.metrics.histogram(
                "polaris.strategy.thread_agentic.assess_duration_seconds",
                (datetime.now(timezone.utc) - start).total_seconds(),
                tags={"system_id": state.system_id},
            )

    async def _run_thread(
        self,
        state: SystemState,
        context: AdaptationContext,
        system_id: str,
        thread_input: str,
        depth: int,
        lineage: Tuple[str, ...],
        runtime: _ThreadRuntime,
    ) -> _ThreadResult:
        """Run a single recursive thread using join synchronization."""
        runtime.total_threads += 1
        runtime.max_depth_reached = max(runtime.max_depth_reached, depth)

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=self._system_prompt(system_id, depth)),
            LLMMessage(
                role="user",
                content=self._thread_user_input(
                    thread_input=thread_input,
                    depth=depth,
                    lineage=lineage,
                ),
            ),
        ]
        transcript_lines: List[str] = [thread_input]

        for step in range(self.steps_limit):
            self.metrics.gauge(
                "polaris.strategy.thread_agentic.step",
                step + 1,
                tags={"system_id": system_id, "depth": str(depth)},
            )

            llm_start = datetime.now(timezone.utc)
            response = await self.llm.generate(
                messages,
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS_REASONING,
                response_schema=ThreadAgenticResponse,
            )
            self.metrics.histogram(
                "polaris.strategy.thread_agentic.llm_call_duration_seconds",
                (datetime.now(timezone.utc) - llm_start).total_seconds(),
                tags={"system_id": system_id, "depth": str(depth)},
            )

            parsed = self._parse_json(response.content)
            if not isinstance(parsed, dict):
                break

            try:
                structured = ThreadAgenticResponse.model_validate(parsed)
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        "ThreadAgentic validation error",
                        error=str(e),
                        depth=depth,
                    )
                break

            if structured.final is not None:
                return _ThreadResult(
                    return_payload=self._build_return_payload(structured.final),
                    final=structured.final,
                )

            if structured.spawn is not None:
                child_feedback = await self._handle_spawn(
                    state=state,
                    context=context,
                    system_id=system_id,
                    depth=depth,
                    lineage=lineage,
                    runtime=runtime,
                    transcript_lines=transcript_lines,
                    spawn=structured.spawn,
                )

                if child_feedback is not None:
                    transcript_lines.append(child_feedback)
                    messages.append(
                        LLMMessage(
                            role="user",
                            content=json.dumps({"child_feedback": child_feedback}),
                        )
                    )
                continue

            tool = structured.tool
            args = structured.args or {}
            if tool:
                tool_result = await self._execute_tool(tool, args, state, context)
                runtime.tool_calls += 1
                compact_result = self._compact_json(tool_result, self.max_tool_result_chars)
                transcript_lines.append(f"tool {tool}: {compact_result}")
                messages.append(
                    LLMMessage(
                        role="user",
                        content=json.dumps({"tool_result": {"tool": tool, "data": tool_result}}),
                    )
                )
                continue

            break

        self.metrics.increment(
            "polaris.strategy.thread_agentic.step_limit_reached",
            tags={"system_id": system_id, "depth": str(depth)},
        )
        if depth == 0:
            return _ThreadResult(return_payload="root_thread_no_final")
        return _ThreadResult(return_payload="child_thread_no_final")

    async def _handle_spawn(
        self,
        state: SystemState,
        context: AdaptationContext,
        system_id: str,
        depth: int,
        lineage: Tuple[str, ...],
        runtime: _ThreadRuntime,
        transcript_lines: List[str],
        spawn: SpawnBlock,
    ) -> Optional[str]:
        """Handle spawning a child thread and returning psi-framed payload."""
        objective = (spawn.objective or "").strip()
        if not objective:
            return self._psi("spawn_error: empty objective")

        if depth >= self.max_thread_depth:
            runtime.spawn_denied_depth += 1
            self.metrics.increment(
                "polaris.strategy.thread_agentic.spawn_denied_depth",
                tags={"system_id": system_id, "depth": str(depth)},
            )
            return self._psi("spawn_denied: max_thread_depth_reached")

        if runtime.total_threads >= self.max_total_threads:
            runtime.spawn_denied_budget += 1
            self.metrics.increment(
                "polaris.strategy.thread_agentic.spawn_denied_budget",
                tags={"system_id": system_id, "depth": str(depth)},
            )
            return self._psi("spawn_denied: max_total_threads_reached")

        signature = self._spawn_signature(depth, objective, lineage)
        count = runtime.spawn_signature_counts.get(signature, 0) + 1
        runtime.spawn_signature_counts[signature] = count
        if count > self.max_repeated_spawns:
            runtime.spawn_repeat_blocked += 1
            self.metrics.increment(
                "polaris.strategy.thread_agentic.spawn_repeat_blocked",
                tags={"system_id": system_id, "depth": str(depth)},
            )
            return self._psi("spawn_denied: repeated_spawn_blocked")

        runtime.spawn_count += 1
        self.metrics.increment(
            "polaris.strategy.thread_agentic.spawned",
            tags={"system_id": system_id, "depth": str(depth)},
        )

        child_input = self._phi(
            transcript_lines=transcript_lines,
            objective=objective,
            context_hint=spawn.context_hint,
        )

        child_lineage = lineage + (objective[:96],)
        try:
            child_result = await asyncio.wait_for(
                self._run_thread(
                    state=state,
                    context=context,
                    system_id=system_id,
                    thread_input=child_input,
                    depth=depth + 1,
                    lineage=child_lineage,
                    runtime=runtime,
                ),
                timeout=self.child_timeout_seconds,
            )
            return self._psi(child_result.return_payload)
        except asyncio.TimeoutError:
            runtime.child_timeouts += 1
            self.metrics.increment(
                "polaris.strategy.thread_agentic.child_timeout",
                tags={"system_id": system_id, "depth": str(depth + 1)},
            )
            return self._psi("child_timeout")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "ThreadAgentic child execution failed",
                    error=str(e),
                    depth=depth + 1,
                )
            return self._psi(f"child_error: {type(e).__name__}")

    async def _execute_tool(
        self,
        tool: str,
        args: Dict[str, Any],
        state: SystemState,
        context: AdaptationContext,
    ) -> Dict[str, Any]:
        """Execute a strategy tool with connector/world/knowledge dependencies."""
        if tool not in self.allowed_tools:
            from polaris.tools import ToolError

            return ToolError(
                code="tool_not_allowed",
                message=f"tool_not_allowed: {tool}",
                recoverable=True,
            ).to_dict()

        connector = None
        if callable(self._get_connector):
            try:
                connector = self._get_connector(state.system_id)
            except Exception:
                connector = None

        deps = ToolDependencies(
            knowledge_store=self.knowledge_store,
            world_model=self.world_model,
            connector=connector,
            logger=self.logger,
            metrics=self.metrics,
        )

        try:
            result = await self._tool_registry.execute(
                tool_name=tool,
                args=args,
                state=state,
                context=context,
                deps=deps,
            )
            self.metrics.increment(
                "polaris.strategy.thread_agentic.tool_called",
                tags={"tool": tool, "system_id": state.system_id},
            )
            return result
        except Exception as e:
            self.metrics.increment(
                "polaris.strategy.thread_agentic.tool_error",
                tags={"tool": tool, "system_id": state.system_id},
            )
            if self.logger:
                self.logger.error("ThreadAgentic tool execution error", tool=tool, error=str(e))
            return {"error": f"tool_error: {type(e).__name__}: {str(e)}"}

    def _system_prompt(self, system_id: str, depth: int) -> str:
        """Build the system prompt for a thread depth."""
        if system_id and self._per_system_prompts:
            override = self._per_system_prompts.get(system_id)
            if override:
                try:
                    return override.format(
                        system_id=system_id,
                        depth=depth,
                        allowed_tools=", ".join(self.allowed_tools),
                        listen_token=self.listen_token,
                        return_token=self.return_token,
                    )
                except Exception:
                    return override

        if self._system_prompt_template:
            try:
                return self._system_prompt_template.format(
                    system_id=system_id,
                    depth=depth,
                    allowed_tools=", ".join(self.allowed_tools),
                    listen_token=self.listen_token,
                    return_token=self.return_token,
                )
            except Exception:
                return self._system_prompt_template

        return (
            "You are a recursive adaptation reasoning thread. "
            "Operate step-by-step and output strict JSON only. "
            "Each step must use exactly one of these forms:\n"
            '1) {"tool": "name", "args": {...}}\n'
            '2) {"spawn": {"objective": "...", "context_hint": "..."}}\n'
            '3) {"final": {"return_payload": "...", "needs_adaptation": true|false, '
            '"reasoning": "...", "actions": [{"type": "...", "parameters": {...}}]}}\n'
            "Root thread (depth 0) must decide adaptation using needs_adaptation and actions. "
            "Child threads should return concise return_payload summaries to parent. "
            "When parent receives child output, it is framed with tokens "
            f"{self.listen_token} and {self.return_token}. "
            f"Current depth: {depth}. Allowed tools: {', '.join(self.allowed_tools)}."
        )

    def _thread_user_input(self, thread_input: str, depth: int, lineage: Tuple[str, ...]) -> str:
        """Build user message for thread execution."""
        payload = {
            "thread": {
                "depth": depth,
                "lineage": list(lineage),
                "input": thread_input,
            }
        }
        return json.dumps(payload)

    def _initial_user_prompt(self, state: SystemState, context: AdaptationContext) -> str:
        """Build root thread input from current system state."""
        metrics = []
        for k, v in state.metrics.items():
            try:
                metrics.append({"name": k, "value": v.value, "unit": v.unit})
            except Exception:
                metrics.append(
                    {
                        "name": k,
                        "value": str(getattr(v, "value", None)),
                        "unit": getattr(v, "unit", None),
                    }
                )

        data = {
            "system_id": state.system_id,
            "health": getattr(state.health_status, "value", "unknown"),
            "timestamp": state.timestamp.isoformat(),
            "metrics": metrics,
            "world_model_insights": context.world_model_insights or {},
        }
        return json.dumps({"current_state": data})

    def _phi(
        self,
        transcript_lines: List[str],
        objective: str,
        context_hint: Optional[str],
    ) -> str:
        """Parent-to-child context mapper (phi)."""
        if self.phi_mode == "recent_lines":
            basis = "\n".join(transcript_lines[-self.phi_max_lines :])
        else:
            basis = transcript_lines[-1] if transcript_lines else ""

        parts = [f"Sub-problem: {objective}"]
        if context_hint:
            parts.append(f"Hint: {context_hint}")
        if basis:
            parts.append(f"Parent context:\n{basis}")
        return "\n\n".join(parts)

    def _psi(self, child_payload: str) -> str:
        """Child-to-parent payload mapper (psi)."""
        compact = (child_payload or "").strip()
        if len(compact) > self.max_child_payload_chars:
            compact = compact[: self.max_child_payload_chars] + "..."
        return f"{self.listen_token}{compact}{self.return_token}"

    def _spawn_signature(self, depth: int, objective: str, lineage: Tuple[str, ...]) -> str:
        """Build a stable signature for repeated-spawn detection."""
        lineage_str = "|".join(lineage[-2:])
        return f"d={depth}|{lineage_str}|{objective.strip().lower()}"

    def _build_return_payload(self, final: ThreadFinalBlock) -> str:
        """Build a compact payload returned to the parent thread."""
        if final.return_payload:
            return final.return_payload
        if final.reasoning:
            return final.reasoning
        if final.actions:
            action_types = [a.type for a in final.actions if a.type]
            if action_types:
                return f"actions: {', '.join(action_types)}"
        return "thread_done"

    def _compact_json(self, value: Any, max_chars: int) -> str:
        """Serialize and truncate payload for bounded context growth."""
        try:
            text = json.dumps(value, ensure_ascii=True)
        except Exception:
            text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _parse_json(self, content: str) -> Any:
        """Parse JSON from potentially noisy model output."""
        raw = (content or "").strip()
        if not raw:
            return {}

        fence_match = re.search(
            r"```(?:json)?\s*(?P<body>.*?)\s*```",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates: List[str] = []
        if fence_match:
            candidates.append(fence_match.group("body").strip())
        candidates.append(raw)

        for cand in candidates:
            parsed = self._extract_first_json_value(cand)
            if parsed is not None:
                return parsed

        if self.logger:
            self.logger.warning(
                "ThreadAgentic received malformed JSON",
                content_preview=raw[:500],
            )
        return {}

    def _extract_first_json_value(self, text: str) -> Optional[Any]:
        """Extract first JSON value from text while respecting strings/escapes."""
        s = (text or "").strip()
        if not s:
            return None

        try:
            return json.loads(s)
        except Exception:
            pass

        start = None
        for i, ch in enumerate(s):
            if ch in "[{":
                start = i
                break
        if start is None:
            return None

        opening = s[start]
        closing = "]" if opening == "[" else "}"
        depth = 0
        in_str = False
        escape = False

        for j in range(start, len(s)):
            ch = s[j]
            if in_str:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == opening:
                depth += 1
                continue

            if ch == closing:
                depth -= 1
                if depth == 0:
                    snippet = s[start : j + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        return None

        return None

    async def on_action_executed(self, action: AdaptationAction, result: Any) -> None:
        """Track execution outcomes for strategy performance metrics."""
        self._adaptation_count += 1
        ok = hasattr(result, "status") and getattr(result.status, "value", None) == "success"
        if ok:
            self._success_count += 1

        self.metrics.increment(
            "polaris.strategy.thread_agentic.actions_executed",
            tags={
                "system_id": action.target_system,
                "action_type": action.action_type,
                "status": getattr(getattr(result, "status", None), "value", "unknown"),
            },
        )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Expose tunable parameters for the meta-learner."""
        return {
            "temperature": ParameterSpec(
                current_value=self.temperature,
                type=float,
                min_value=0.0,
                max_value=2.0,
                description="LLM sampling temperature",
                kind="llm_temperature",
            ),
            "steps_limit": ParameterSpec(
                current_value=self.steps_limit,
                type=int,
                min_value=1,
                max_value=20,
                description="Max reasoning steps per thread",
                kind="agent_steps_limit",
            ),
            "max_thread_depth": ParameterSpec(
                current_value=self.max_thread_depth,
                type=int,
                min_value=0,
                max_value=8,
                description="Maximum recursive thread depth",
                kind="agent_steps_limit",
            ),
            "max_total_threads": ParameterSpec(
                current_value=self.max_total_threads,
                type=int,
                min_value=1,
                max_value=128,
                description="Maximum total thread count per assessment",
                kind="agent_steps_limit",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter in-place."""
        if parameter_path == "temperature":
            self.temperature = float(new_value)
            return True
        if parameter_path == "steps_limit":
            self.steps_limit = max(1, int(new_value))
            return True
        if parameter_path == "max_thread_depth":
            self.max_thread_depth = max(0, int(new_value))
            return True
        if parameter_path == "max_total_threads":
            self.max_total_threads = max(1, int(new_value))
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply hot-reload configuration updates."""
        if not isinstance(config, dict):
            return  # type: ignore[unreachable]

        for key in (
            "temperature",
            "steps_limit",
            "max_thread_depth",
            "max_total_threads",
        ):
            if key in config:
                await self.update_parameter(key, config[key])

        if "child_timeout_seconds" in config:
            self.child_timeout_seconds = max(0.1, float(config["child_timeout_seconds"]))
        if "max_repeated_spawns" in config:
            self.max_repeated_spawns = max(1, int(config["max_repeated_spawns"]))
        if "max_tool_result_chars" in config:
            self.max_tool_result_chars = max(200, int(config["max_tool_result_chars"]))
        if "max_child_payload_chars" in config:
            self.max_child_payload_chars = max(100, int(config["max_child_payload_chars"]))
        if "phi_mode" in config:
            self.phi_mode = str(config["phi_mode"] or "last_line")
        if "phi_max_lines" in config:
            self.phi_max_lines = max(1, int(config["phi_max_lines"]))
        if "listen_token" in config:
            self.listen_token = str(config["listen_token"] or "=>")
        if "return_token" in config:
            self.return_token = str(config["return_token"] or "<=")

        if "system_prompt" in config:
            self._system_prompt_template = config["system_prompt"]
        if "per_system_prompts" in config and isinstance(config["per_system_prompts"], dict):
            self._per_system_prompts = config["per_system_prompts"]

        if "tools" in config and isinstance(config["tools"], dict):
            enabled = config["tools"].get("enabled")
            if isinstance(enabled, list):
                self.allowed_tools = enabled
                self._tool_registry = ToolRegistry(metrics=self.metrics)
                for tool in get_builtin_tools():
                    if tool.name in self.allowed_tools:
                        self._tool_registry.register(tool)

        resil = config.get("resilience")
        if resil and hasattr(self.llm, "update_resilience"):
            try:
                llm_any: Any = self.llm
                llm_any.update_resilience(resil)
            except Exception as e:
                if self.logger:
                    self.logger.warning("ThreadAgentic resilience update failed", error=str(e))

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy-level outcome metrics."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }
