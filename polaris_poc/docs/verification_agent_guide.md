# POLARIS Verification Agent Guide

## Overview

The **Verification Agent** is a critical safety component in the POLARIS framework that validates control actions before execution. It ensures that all adaptation actions comply with safety constraints, organizational policies, and system invariants, preventing potentially harmful or policy-violating actions from being executed.

## Architecture

### Core Components

1. **Verification Adapter** (`polaris.adapters.verification`)
   - Main verification logic and constraint evaluation
   - NATS message handling for verification requests/results
   - Integration with Digital Twin for predictive verification

2. **Constraint Engine**
   - Evaluates safety and resource constraints
   - Supports complex boolean expressions
   - Provides detailed violation reporting

3. **Policy Engine**
   - Enforces organizational policies
   - Rule-based evaluation system
   - Configurable severity levels

4. **Digital Twin Integration**
   - Predictive "what-if" analysis
   - Simulation-based safety verification
   - Future state impact assessment

### Data Flow

```
Kernel/Controller → Verification Request → Verification Adapter
                                              ↓
                                         Constraint Check
                                              ↓
                                         Policy Check
                                              ↓
                                      Digital Twin Check
                                              ↓
                                      Verification Result → Execution Adapter
```

## Configuration

### Framework Configuration

Add verification configuration to `polaris_config.yaml`:

```yaml
verification:
  input_subject: "polaris.verification.requests"
  output_subject: "polaris.verification.results"
  policy_subject: "polaris.verification.policies"
  metrics_subject: "polaris.verification.metrics"
  default_timeout_sec: 30
  max_concurrent_verifications: 5
  enable_by_default: true
```

### Plugin Configuration

Add verification section to your plugin's `config.yaml`:

```yaml
verification:
  constraints:
    - id: "max_servers_limit"
      type: "resource"
      condition: "action_type != 'ADD_SERVER' or (current_state.get('active_servers', 0) + params.get('count', 1)) <= current_state.get('max_servers', 10)"
      severity: "critical"
      description: "Cannot exceed maximum server limit"
      suggested_fix: "Remove existing servers before adding new ones"
      
  policies:
    - id: "change_management"
      name: "Change Management Policy"
      rules:
        - condition: "action_type != 'REMOVE_SERVER' or action.get('approved_by') is not None"
          severity: "medium"
          description: "Server removal requires approval"
          
  settings:
    default_timeout_sec: 30
    max_concurrent: 5
    enable_digital_twin: true
```

## Constraint Types

### Safety Constraints
Critical system safety requirements that must never be violated.

**Example:**
```yaml
- id: "min_servers_requirement"
  type: "safety"
  condition: "action_type != 'REMOVE_SERVER' or (current_state.get('active_servers', 1) - params.get('count', 1)) >= 1"
  severity: "critical"
  description: "Must maintain at least one active server"
```

### Resource Constraints
Limits on system resources and capacity.

**Example:**
```yaml
- id: "memory_limit"
  type: "resource"
  condition: "action_type != 'ADD_SERVER' or current_state.get('memory_usage', 0) + params.get('memory_per_server', 1024) <= 8192"
  severity: "high"
  description: "Cannot exceed memory limit"
```

### Policy Constraints
Organizational policies and compliance requirements.

**Example:**
```yaml
- id: "business_hours_policy"
  type: "policy"
  condition: "action.get('priority') == 'urgent' or (datetime.now().hour >= 9 and datetime.now().hour <= 17)"
  severity: "medium"
  description: "Non-urgent changes should be made during business hours"
```

## Verification Levels

### Basic Verification
- Fast constraint checking only
- Suitable for time-critical decisions
- Timeout: ~5 seconds

### Policy Verification
- Includes organizational policy checks
- Moderate verification depth
- Timeout: ~15 seconds

### Formal Verification
- Includes Digital Twin simulation
- Comprehensive safety analysis
- Timeout: ~30 seconds

### Comprehensive Verification
- All verification methods
- Maximum safety assurance
- Timeout: ~60 seconds

## Expression Language

Constraints use a safe expression language with the following context:

### Available Variables
- `action_type`: Type of the action (e.g., "ADD_SERVER")
- `params`: Action parameters dictionary
- `current_state`: Current system state from monitoring
- `action`: Complete action object

### Supported Operators
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Logical: `and`, `or`, `not`
- Membership: `in`, `not in`
- Arithmetic: `+`, `-`, `*`, `/`

### Example Expressions
```python
# Check server count limits
"current_state.get('active_servers', 0) + params.get('count', 1) <= 10"

# Validate parameter ranges
"0.0 <= params.get('value', 1.0) <= 1.0"

# Complex conditions
"action_type == 'REMOVE_SERVER' and current_state.get('utilization', 0) < 0.8"
```

## Usage

### Starting the Verification Adapter

```bash
# Start verification adapter with SWIM plugin
python src/scripts/start_component.py verification --plugin-dir extern

# With custom configuration
python src/scripts/start_component.py verification --plugin-dir extern --config custom_config.yaml

# With debug logging
python src/scripts/start_component.py verification --plugin-dir extern --log-level DEBUG
```

### Integration with Kernel

The kernel automatically integrates with verification when enabled:

```python
# In kernel configuration
self.enable_verification = True
self.verification_timeout = 30.0

# Actions are automatically verified before execution
await self.execute_action_with_verification(action, context)
```

### Manual Verification Requests

You can also send verification requests directly via NATS:

```python
verification_request = {
    "request_id": "unique-request-id",
    "action": {
        "action_id": "action-123",
        "action_type": "ADD_SERVER",
        "params": {"count": 1}
    },
    "context": {
        "active_servers": 3,
        "max_servers": 10
    },
    "verification_level": "basic",
    "timeout_sec": 30.0
}

await nats_client.publish_json("polaris.verification.requests", verification_request)
```

## Monitoring and Metrics

The verification adapter publishes metrics for monitoring:

### Key Metrics
- `verification_completed`: Individual verification results
- `constraint_violations`: Constraint violation counts
- `policy_violations`: Policy violation counts
- `verification_latency`: Processing time statistics

### Health Monitoring
```bash
# Monitor verification metrics
python src/scripts/nats_spy.py --subjects "polaris.verification.metrics"

# Monitor verification results
python src/scripts/nats_spy.py --subjects "polaris.verification.results"
```

## Best Practices

### Constraint Design
1. **Keep constraints simple** - Complex expressions are harder to debug
2. **Use appropriate severity levels** - Critical for safety, high for resources, medium for policies
3. **Provide clear descriptions** - Help operators understand violations
4. **Include suggested fixes** - Guide remediation efforts

### Performance Optimization
1. **Order constraints by likelihood** - Put most likely violations first
2. **Use appropriate verification levels** - Don't over-verify time-critical actions
3. **Set reasonable timeouts** - Balance safety with responsiveness
4. **Monitor verification latency** - Identify performance bottlenecks

### Safety Considerations
1. **Fail-safe defaults** - Reject actions when verification fails
2. **Comprehensive testing** - Test all constraint combinations
3. **Regular policy reviews** - Keep policies current and relevant
4. **Audit verification decisions** - Log all verification results

## Troubleshooting

### Common Issues

#### Verification Timeouts
```
Symptom: Actions rejected due to timeout
Cause: Complex constraints or slow Digital Twin
Solution: Increase timeout or simplify constraints
```

#### Constraint Evaluation Errors
```
Symptom: Constraint evaluation failures
Cause: Invalid expression syntax or missing context
Solution: Validate expressions and ensure context availability
```

#### Policy Violations
```
Symptom: Unexpected policy violations
Cause: Outdated policies or missing approval metadata
Solution: Review policy rules and action metadata
```

### Debug Mode

Enable debug logging for detailed verification information:

```bash
python src/scripts/start_component.py verification --plugin-dir extern --log-level DEBUG
```

### Testing Constraints

Use the validation mode to test constraint configurations:

```bash
python src/scripts/start_component.py verification --plugin-dir extern --validate-only
```

## Advanced Features

### Dynamic Policy Updates

Policies can be updated at runtime via NATS:

```python
policy_update = {
    "operation": "add_policy",
    "policy": {
        "id": "new_policy",
        "name": "New Policy",
        "rules": [...]
    }
}

await nats_client.publish_json("polaris.verification.policies", policy_update)
```

### Custom Verification Plugins

Extend verification with custom plugins:

```python
class CustomVerificationPlugin:
    async def verify_action(self, action, context):
        # Custom verification logic
        return violations, recommendations
```

### Formal Verification Integration

For systems requiring formal guarantees, integrate with model checkers:

```yaml
verification:
  integrations:
    formal_verification:
      tool: "prism"
      model_path: "models/system_model.prism"
      properties:
        - "P>=0.99 [F response_time <= 1.0]"
        - "P>=0.95 [G servers >= 1]"
```

## API Reference

### VerificationRequest
```python
class VerificationRequest:
    request_id: str
    action: ControlAction
    context: Dict[str, Any]
    verification_level: VerificationLevel
    timeout_sec: float
    requester: str
```

### VerificationResult
```python
class VerificationResult:
    request_id: str
    action_id: str
    approved: bool
    confidence: float
    violations: List[ConstraintViolation]
    recommendations: List[str]
    verification_time_ms: float
```

### ConstraintViolation
```python
class ConstraintViolation:
    constraint_id: str
    constraint_type: ConstraintType
    severity: str
    description: str
    suggested_fix: Optional[str]
    metadata: Dict[str, Any]
```

## Examples

See the `extern/` directory for complete SWIM plugin configuration examples, including comprehensive verification constraints and policies.