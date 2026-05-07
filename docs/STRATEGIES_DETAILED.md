# Polaris Strategy Deep Dive

This document explains every built-in strategy in detail, with special focus on:

- Agentic tool-calling behavior
- Hybrid with threshold-first patterns
- Practical selection guidance for users

## 1. Canonical Strategy Schema

All strategies use the same top-level shape:

```yaml
strategy:
  type: "<strategy_name>"
  params: ...
```

Valid built-in strategy types:

- threshold
- llm_reasoning
- agentic_llm
- thread_agentic
- multi_agent
- hybrid

## 2. Strategy Selection

- Recommended default for all deployments: hybrid.
- Mandatory design rule: always keep a reactive deterministic path via threshold as the first stage.
- threshold: deterministic baseline, and the required reactive branch in hybrid.
- llm_reasoning: single-shot semantic branch, typically as a secondary hybrid stage.
- agentic_llm: tool-assisted branch in hybrid for evidence-driven decisions.
- thread_agentic: recursive branch in hybrid for decomposition-heavy decisions.
- multi_agent: committee branch in hybrid for safety-governed decisions.
- hybrid: the orchestrator strategy that should be used for production recommendations.

## 3. Threshold Strategy (Deterministic Baseline)

## 3.1 How it works

Threshold strategy compares live metrics against configured bounds and emits adaptation proposals when bounds are crossed.

## 3.2 Core params

```yaml
strategy:
  type: "threshold"
  params:
    thresholds:
      average_response_time:
        high: 750.0
      average_utilization:
        high: 0.8
        low: 0.3
    cooldown_seconds: 45
    action_templates:
      average_response_time:
        high:
          type: "scale_up"
```

## 3.3 Validation rules

- thresholds must be a mapping.
- high and low must be numeric when provided.
- high must be greater than low when both are present.
- cooldown_seconds must be integer >= 0.

## 3.4 When to use

- Always include threshold in hybrid as the first reactive guardrail stage.
- Use threshold-only mode mainly for smoke tests and baseline comparisons.

## 3.5 How action is chosen on threshold breach

When a threshold is crossed, Polaris chooses the action through configured
`action_templates` (not from the breach alone):

- First it evaluates cooldown. If still in cooldown, no action is emitted.
- It scans configured metrics and checks `high` / `low` bounds.
- On breach, it resolves template in this order:
  1. metric-specific template (`action_templates.<metric>.<high|low>`)
  2. default fallback (`action_templates.default.<high|low>`)
- It emits the first breached metric action in that assessment cycle.
- If a breach occurs but no matching template exists, this is a configuration error.

Practical implication: define explicit templates for every threshold path you care
about (or provide a safe `default` fallback), otherwise breached thresholds cannot
be translated into executable actions.

## 4. LLM Reasoning Strategy (Single-Shot Strict JSON)

## 4.1 How it works

One LLM call per assessment. It must return strict JSON containing decision and actions.

Required output semantics:

- needs_adaptation: boolean
- reasoning: non-empty string
- actions: non-empty list when adaptation is needed

## 4.2 Core params

```yaml
strategy:
  type: "llm_reasoning"
  params:
    provider: "google"
    model: "gemini-3-flash-preview"
    temperature: 0.1
    system_description: "SWIM server pool"
    adaptation_goals: "Maintain latency and utilization"
    per_system_prompts:
      swim: |
        You are controlling SWIM. Use only connector-supported action names.
```

## 4.3 Strict contract behavior

LLM strategies are contract-first. If connector action contract is missing, the system iteration fails rather than executing untrusted actions.

## 4.4 When to use

- Use this as a secondary branch inside hybrid, after threshold.
- Prefer this branch when single-call reasoning is enough and you want lower orchestration overhead.

## 5. Agentic LLM Strategy (Tool Calling)

## 5.1 How it works

Agentic strategy runs a bounded reasoning loop up to steps_limit.
At each step, model can:

- call a Polaris tool to gather information, or
- produce final adaptation decision/actions.

It supports two operation modes:

- JSON text mode
- native tool-calling mode (provider API function-calls)

`tools` can be configured either as a simple list or as `tools.enabled: [...]`.
Configured tool names are validated against the registered tool factory names.

## 5.2 Core params

```yaml
strategy:
  type: "agentic_llm"
  params:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.1
    steps_limit: 3
    decision_cooldown_seconds: 60
    # JSON-text mode: prompt instructs the LLM to reply in strict JSON.
    per_system_prompts:
      swim: |
        You are controlling SWIM. Use only supported actions.
        Reply in strict JSON only.
    tools:
      enabled:
        - get_recent_states
        - summarize_metric_trends
        - list_metric_fields
        - compute_metric_math
        - get_world_model_insights
        - predict_outcome
        - get_action_history
        - list_supported_actions

    # Native tool calling mode (recommended over JSON-text mode).
    # When native_tools is present, system_prompt replaces per_system_prompts
    # and the LLM must respond by calling a declared function.
    system_prompt: |
      Use the provided functions to respond. Do not emit free-form JSON.
    native_tools:
      - type: "function"
        function:
          name: "get_recent_states"
          description: "Query recent system states."
          parameters:
            type: "object"
            properties:
              window_seconds:
                type: "integer"
    native_tools_unsupported_policy: skip_cycle   # skip_cycle | json_fallback | strict_fail
```

## 5.3 Built-in tools and purpose

- get_recent_states: fetch recent telemetry timeline.
- summarize_metric_trends: trend summary for one metric.
- list_metric_fields: discover metrics present in recent data.
- compute_metric_math: safe formula/stat operations over metrics.
- get_world_model_insights: access model-driven insights.
- predict_outcome: estimate action outcomes before applying.
- get_action_history: avoid repeats and oscillation.
- list_supported_actions: discover canonical connector actions.

## 5.4 Native tool calling mode

When `native_tools` is provided in config:

- model receives function definitions.
- strategy consumes returned tool calls.
- if provider returns tool_calls as none/empty, strategy falls back to JSON-text parse path.
- if provider does not implement native tool calling, behavior follows
  `native_tools_unsupported_policy`:
  - `skip_cycle` (default)
  - `json_fallback`
  - `strict_fail`

### Prompt key: `system_prompt` vs `per_system_prompts`

| Mode | Prompt key | LLM response expected |
|---|---|---|
| JSON-text (default) | `per_system_prompts.<system_id>` | Strict JSON object |
| Native tool calling | `system_prompt` | Function call(s) |

When `native_tools` is present, use `system_prompt`. The prompt should instruct the LLM
to use the declared functions rather than emit JSON text.

### Native tool groups

Native tool entries serve two distinct roles:

1. **Polaris analysis tools** — must also be listed in `tools.enabled`.
   Examples: `get_recent_states`, `summarize_metric_trends`, `get_action_history`.
2. **Adaptation action functions** — connector-specific actions declared only in
   `native_tools` (not in `tools.enabled`).
   Examples for SWIM: `scale_up`, `scale_down`, `set_dimmer`, `no_adaptation`.

Polaris validates at startup that every name in `tools.enabled` has a matching entry in
`native_tools`. Missing entries cause startup failure; run `polaris doctor` to check.

### SWIM example — minimal native_tools block

```yaml
strategy:
  type: "agentic_llm"
  params:
    provider: "groq"
    temperature: 0.05
    steps_limit: 3
    native_tools_unsupported_policy: skip_cycle
    system_prompt: |
      Use the provided functions to respond.
      Call get_recent_states or get_action_history first, then decide.
      NEVER scale_down when average_response_time > 400 ms.
    tools:
      enabled:
        - get_recent_states
        - get_action_history
    native_tools:
      # ── Analysis tools (must mirror tools.enabled) ──────────────────
      - type: "function"
        function:
          name: "get_recent_states"
          description: "Query recent system states."
          parameters:
            type: "object"
            properties:
              window_seconds:
                type: "integer"
                minimum: 1
                maximum: 3600
                description: "Lookback window in seconds."
      - type: "function"
        function:
          name: "get_action_history"
          description: "Query historical adaptation actions."
          parameters:
            type: "object"
            properties:
              window_seconds:
                type: "integer"
                minimum: 1
                maximum: 2592000
                description: "Lookback window in seconds."
      # ── Adaptation actions (SWIM-specific) ───────────────────────────
      - type: "function"
        function:
          name: "scale_up"
          description: "Add one server to the SWIM pool."
          parameters:
            type: "object"
            properties:
              reasoning:
                type: "string"
                description: "Why scaling up is needed."
            required:
              - "reasoning"
      - type: "function"
        function:
          name: "scale_down"
          description: "Remove one server (only when load is sustainably low)."
          parameters:
            type: "object"
            properties:
              reasoning:
                type: "string"
                description: "Why scaling down is safe."
            required:
              - "reasoning"
      - type: "function"
        function:
          name: "set_dimmer"
          description: "Set optional-content ratio [0.10, 1.0]."
          parameters:
            type: "object"
            properties:
              value:
                type: "number"
                minimum: 0.10
                maximum: 1.0
                description: "Target dimmer value."
              reasoning:
                type: "string"
                description: "Why this value is appropriate."
            required:
              - "value"
              - "reasoning"
      - type: "function"
        function:
          name: "no_adaptation"
          description: "Signal that no adaptation is needed this cycle."
          parameters:
            type: "object"
            properties:
              reasoning:
                type: "string"
                description: "Why no changes are needed."
            required:
              - "reasoning"
```

A complete, ready-to-run SWIM native-tool-calling config is available at
[config/swim.yaml](../config/swim.yaml).

## 5.5 Reliability controls

- steps_limit prevents unbounded loops.
- decision_cooldown_seconds prevents over-frequent decisions.
- resilience settings provide backoff/retry/rate-limit controls.

## 5.6 When to use

- Use this as a secondary branch inside hybrid, after threshold.
- Prefer this branch when you want evidence-backed reasoning via tools and bounded planning.

## 6. Thread Agentic Strategy (Recursive)

## 6.1 How it works

Thread-agentic strategy recursively spawns child reasoning threads for subproblems and joins results.

It is designed for complex decision decomposition with bounded recursion.

## 6.2 Core params

```yaml
strategy:
  type: "thread_agentic"
  params:
    provider: "google"
    temperature: 0.1
    steps_limit: 4
    max_thread_depth: 3
    max_total_threads: 16
    child_timeout_seconds: 20
    max_repeated_spawns: 2
    assessment_cooldown_seconds: 0
    phi_mode: "last_line"
    phi_max_lines: 6
```

## 6.3 Safety and cost controls

- max_thread_depth bounds recursion depth.
- max_total_threads bounds total spawned threads.
- child_timeout_seconds caps blocking wait on children.
- max_repeated_spawns limits repeated subproblem loops.

## 6.4 When to use

- Use this as a secondary branch inside hybrid, after threshold.
- Prefer this branch for decomposition-heavy decisions where recursion materially improves plan quality.

## 7. Multi-Agent Strategy (Committee)

## 7.1 How it works

Three-stage pipeline:

1. Diagnostician identifies anomalies and root causes.
2. Planner proposes action plan.
3. SafetyValidator approves/rejects/filters plan.

## 7.2 Core params

```yaml
strategy:
  type: "multi_agent"
  params:
    provider: "google"
    temperature: 0.1
    steps_limit: 3
    system_description: "Managed system"
    diagnostician:
      temperature: 0.0
      steps_limit: 4
    planner:
      provider: "openai"
      temperature: 0.2
    validator:
      temperature: 0.0
```

## 7.3 Role specialization guidance

- Diagnostician: low temperature, evidence-oriented.
- Planner: moderate temperature, solution generation.
- Validator: low temperature, conservative risk filtering.

## 7.4 When to use

- Use this as a secondary branch inside hybrid, after threshold.
- Prefer this branch when governance and safety validation are top priorities.

## 8. Hybrid Strategy (Detailed)

## 8.1 How it works

Hybrid combines sub-strategies and selects output by configured mode.

Selection modes:

- first: first valid strategy result wins (ordered by priority/sequence).
- priority: explicit weight/priority bias.
- confidence: confidence-oriented selection with min_confidence gate.

## 8.2 Core params

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    min_confidence: 0.7
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 0.9
        params:
          thresholds:
            average_response_time:
              high: 750.0
          cooldown_seconds: 45
      - type: "agentic_llm"
        priority: 0.6
        params:
          provider: "google"
          steps_limit: 3
          temperature: 0.1
          tools:
            enabled:
              - get_recent_states
              - summarize_metric_trends
              - predict_outcome
              - list_supported_actions
```

## 8.3 Recommended hybrid compositions (always reactive first)

Use one of these compositions, always with threshold first:

1. Hybrid: threshold + agentic_llm

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 0.9
        params:
          thresholds:
            average_response_time:
              high: 750.0
          cooldown_seconds: 45
      - type: "agentic_llm"
        priority: 0.6
        params:
          provider: "google"
          steps_limit: 3
          temperature: 0.1
```

2. Hybrid: threshold + multi_agent

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 0.9
        params:
          thresholds:
            average_response_time:
              high: 750.0
          cooldown_seconds: 45
      - type: "multi_agent"
        priority: 0.6
        params:
          provider: "google"
          temperature: 0.1
          steps_limit: 3
```

3. Hybrid: threshold + thread_agentic

```yaml
strategy:
  type: "hybrid"
  params:
    selection_mode: "first"
    cooldown_seconds: 120
    strategies:
      - type: "threshold"
        priority: 0.9
        params:
          thresholds:
            average_response_time:
              high: 750.0
          cooldown_seconds: 45
      - type: "thread_agentic"
        priority: 0.6
        params:
          provider: "google"
          steps_limit: 4
          max_thread_depth: 3
          max_total_threads: 16
```

Reactive-path rule for all three:

- Keep threshold first and stricter than advanced branch triggers.
- Let advanced branch refine decisions when threshold does not fire or when nuanced adaptation is required.

## 8.4 Validation rules

- strategies list must be non-empty.
- each sub-strategy must include type and params.
- unknown extra keys in sub-strategy blocks are rejected.

## 9. Resilience Configuration for LLM Strategies

Common resilience fields:

- rps
- burst
- concurrency
- max_retries
- base_backoff_ms
- max_backoff_ms
- optional keys_env_var

Use this for:

- rate limit management
- transient failure recovery
- key rotation scenarios

## 10. Prompting Best Practices for All LLM Strategies

- Explicitly list allowed action names.
- Explicitly forbid invented action types.
- Include operating constraints (resource bounds, safety rules).
- Instruct non-oscillation behavior.
- Require concise evidence-backed reasoning.
- Keep output contract strict and machine-parseable.

## 11. Decision Guide

If unsure, start with:

1. hybrid with threshold + agentic_llm (default recommendation)
2. hybrid with threshold + multi_agent (for stronger safety governance)
3. hybrid with threshold + thread_agentic (for decomposition-heavy planning)

Always keep the reactive threshold path in hybrid as the first stage.

## 12. Related Docs

- Main guide: [POLARIS_COMPLETE_DOCUMENTATION.md](POLARIS_COMPLETE_DOCUMENTATION.md)
- SWIM walkthrough: [SWIM_EXAMPLE_GUIDE.md](SWIM_EXAMPLE_GUIDE.md)
- Config reference baseline: [config/default.yaml](../config/default.yaml)
