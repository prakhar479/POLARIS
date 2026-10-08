# WildFire Adapter - Quick Start Guide

## Installation & Setup

### Prerequisites
- Python 3.7+
- Flask
- Requests (for testing)

### Install Dependencies
```bash
cd /path/to/polaris/wildfire
pip install flask requests
```

---

## Running the Adapter

### Start the Server
```bash
cd wildfire
python3 adapter.py
```

Expected output:
```
2026-02-02 21:25:11,063 - adapter - INFO - SimulationManager initialized
2026-02-02 21:25:11,065 - adapter - INFO - Starting WildFire Adapter Server...
 * Running on http://0.0.0.0:5000
```

### Verify Server is Running
```bash
curl http://localhost:5000/health
```

Response:
```json
{
    "status": "healthy",
    "timestamp": "2026-02-02T21:30:45.123456",
    "active_sessions": 0,
    "current_session": null
}
```

---

## Running Tests

### Full Test Suite
```bash
python3 test_adapter.py
```

Expected output:
```
============================================================
WILDFIRE ADAPTER TEST SUITE
============================================================

--- Testing Health Check ---
  ✓ Health endpoint responds
  ✓ Health response has status
  ...

============================================================
TEST SUMMARY: 63/63 passed
============================================================
```

### Run Specific Test Category
Modify `test_adapter.py` to run only specific test methods:

```python
suite = AdapterTestSuite()
suite.test_health_check()
suite.test_session_management()
suite.print_summary()
```

---

## Basic Workflow

### 1. Create a Session
```bash
curl -X POST http://localhost:5000/api/v1/sessions
```

Response:
```json
{
    "session_id": "6dc8b003",
    "message": "Simulation session created",
    "num_agents": 2,
    "grid_size": {"height": 50, "width": 50}
}
```

Save the `session_id` for later use.

### 2. Set as Current Session
```bash
curl -X PUT http://localhost:5000/api/v1/sessions/6dc8b003/current
```

### 3. Reset Simulation
```bash
curl -X POST http://localhost:5000/api/v1/sim/reset
```

### 4. Get Initial State
```bash
curl http://localhost:5000/api/v1/sim/state | jq .
```

### 5. Execute Actions
```bash
curl -X POST http://localhost:5000/api/v1/sim/action \
  -H "Content-Type: application/json" \
  -d '[{"uav": 0, "move": "north"}]'
```

### 6. Check Metrics
```bash
curl http://localhost:5000/api/v1/sim/metrics | jq .
```

### 7. Check Agent Info
```bash
curl http://localhost:5000/api/v1/sim/agents | jq .
```

---

## Python Client Example

### Simple Client Script
```python
import requests
import json

BASE_URL = "http://localhost:5000"

# Create session
r = requests.post(f"{BASE_URL}/api/v1/sessions")
session_id = r.json()["session_id"]
print(f"Created session: {session_id}")

# Set current
requests.put(f"{BASE_URL}/api/v1/sessions/{session_id}/current")

# Reset
requests.post(f"{BASE_URL}/api/v1/sim/reset")

# Get config
r = requests.get(f"{BASE_URL}/api/v1/sim/config")
config = r.json()
print(f"Agents: {config['num_agents']}")

# Execute actions for 10 steps
for step in range(10):
    actions = [
        {"uav": 0, "move": "north"},
        {"uav": 1, "move": "east"}
    ]
    r = requests.post(f"{BASE_URL}/api/v1/sim/action", json=actions)

    if step % 5 == 0:
        r = requests.get(f"{BASE_URL}/api/v1/sim/metrics")
        metrics = r.json()["metrics"]
        print(f"Step {step}: Burning cells: {metrics['fire_cells_burning']}")

# Get final metrics
r = requests.get(f"{BASE_URL}/api/v1/sim/metrics")
print(f"\nFinal metrics: {json.dumps(r.json(), indent=2)}")
```

### Client Class with Convenience Methods
```python
class WildFireClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_id = None

    def create_session(self):
        """Create and set current session"""
        r = requests.post(f"{self.base_url}/api/v1/sessions")
        self.session_id = r.json()["session_id"]
        requests.put(f"{self.base_url}/api/v1/sessions/{self.session_id}/current")
        return self.session_id

    def reset(self):
        """Reset simulation"""
        return requests.post(f"{self.base_url}/api/v1/sim/reset").json()

    def step(self, actions):
        """Execute actions and step simulation"""
        return requests.post(
            f"{self.base_url}/api/v1/sim/action",
            json=actions
        ).json()

    def get_metrics(self):
        """Get simulation metrics"""
        return requests.get(
            f"{self.base_url}/api/v1/sim/metrics"
        ).json()

    def pause(self):
        """Pause simulation"""
        return requests.post(f"{self.base_url}/api/v1/sim/pause").json()

    def resume(self):
        """Resume simulation"""
        return requests.post(f"{self.base_url}/api/v1/sim/resume").json()

# Usage
client = WildFireClient()
client.create_session()
client.reset()

for step in range(100):
    actions = [{"uav": 0, "move": "north"}]
    client.step(actions)

    if step % 10 == 0:
        metrics = client.get_metrics()["metrics"]
        print(f"Step {step}: {metrics['fire_cells_burning']} burning cells")
```

---

## Common Tasks

### Task 1: Run a Single Experiment
```bash
# Create and initialize
SESSION=$(curl -s -X POST http://localhost:5000/api/v1/sessions | jq -r '.session_id')
curl -X PUT http://localhost:5000/api/v1/sessions/$SESSION/current
curl -X POST http://localhost:5000/api/v1/sim/reset

# Run for 50 steps
for i in {1..50}; do
    curl -s -X POST http://localhost:5000/api/v1/sim/action \
      -H "Content-Type: application/json" \
      -d '[{"uav": 0, "move": "north"}, {"uav": 1, "move": "east"}]' > /dev/null
done

# Get final result
curl http://localhost:5000/api/v1/sim/metrics | jq '.metrics | {timestep, mr1_values, mr2_value}'
```

### Task 2: Compare Multiple Strategies
```python
import requests

def run_strategy(actions_fn, num_steps=50):
    """Run a strategy and return final metrics"""
    # Create session
    r = requests.post("http://localhost:5000/api/v1/sessions")
    session_id = r.json()["session_id"]
    requests.put(f"http://localhost:5000/api/v1/sessions/{session_id}/current")

    # Reset
    requests.post("http://localhost:5000/api/v1/sim/reset")

    # Execute strategy
    for step in range(num_steps):
        actions = actions_fn(step)
        requests.post("http://localhost:5000/api/v1/sim/action", json=actions)

    # Get metrics
    r = requests.get("http://localhost:5000/api/v1/sim/metrics")
    return r.json()["metrics"]

# Strategy 1: Always north
def strategy_north(step):
    return [{"uav": 0, "move": "north"}, {"uav": 1, "move": "north"}]

# Strategy 2: Alternating
def strategy_alternate(step):
    move = "north" if step % 2 == 0 else "east"
    return [{"uav": 0, "move": move}, {"uav": 1, "move": move}]

# Compare
metrics1 = run_strategy(strategy_north)
metrics2 = run_strategy(strategy_alternate)

print(f"Strategy 1 MR1: {metrics1['mr1_values']}")
print(f"Strategy 2 MR1: {metrics2['mr1_values']}")
```

### Task 3: Interactive Debugging
```bash
# Start session
SESSION=$(curl -s -X POST http://localhost:5000/api/v1/sessions | jq -r '.session_id')
curl -X PUT http://localhost:5000/api/v1/sessions/$SESSION/current
curl -X POST http://localhost:5000/api/v1/sim/reset

# Pause before executing risky action
curl -X POST http://localhost:5000/api/v1/sim/pause

# Inspect current state
curl http://localhost:5000/api/v1/sim/state | jq '.states'
curl http://localhost:5000/api/v1/sim/agents | jq '.agents.agents[0]'

# Resume when ready
curl -X POST http://localhost:5000/api/v1/sim/resume

# Execute action
curl -X POST http://localhost:5000/api/v1/sim/action \
  -H "Content-Type: application/json" \
  -d '[{"uav": 0, "move": "north"}]'

# Check result
curl http://localhost:5000/api/v1/sim/metrics | jq '.metrics.agent_positions'
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill existing process
pkill -f "python3 adapter.py"

# Try with different port (edit adapter.py):
app.run(host='0.0.0.0', port=5001)
```

### Connection Refused
```bash
# Check server is running
curl http://localhost:5000/health

# If not running, start it
python3 adapter.py
```

### Action Returns 500 Error
```bash
# Check action format is correct
# Valid format: [{"uav": 0, "move": "north"}]
# Invalid: {"uav": 0, "move": "north"}  (missing list)

# Check UAV index is valid (0 to NUM_AGENTS-1)
# With NUM_AGENTS=2, valid indices are 0 and 1

# Check move is valid: north, south, east, west, hold
```

### Test Failures
```bash
# Run tests with output
python3 test_adapter.py 2>&1

# Check adapter logs
tail -f /tmp/adapter.log

# Verify adapter is responsive
curl http://localhost:5000/health
```

---

## Configuration Changes

### Increase Number of Agents
Edit `common_fixed_variables.py`:
```python
NUM_AGENTS = 5  # Changed from 2
```

Then restart adapter and test:
```bash
python3 test_adapter.py
```

### Change Grid Size
Edit `common_fixed_variables.py`:
```python
WIDTH = 100   # Changed from 50
HEIGHT = 100  # Changed from 50
```

---

## Performance Tips

1. **Use Batch Actions**: Instead of single actions repeatedly
   ```python
   # Slower
   for i in range(100):
       requests.post("/api/v1/sim/action", json=[...])

   # Faster
   actions = [[...] for _ in range(100)]
   requests.post("/api/v1/sim/batch-actions", json={"actions": actions})
   ```

2. **Use Multiple Sessions**: For parallel experiments
   ```python
   sessions = [create_session() for _ in range(10)]
   # Run all in parallel threads
   ```

3. **Query Metrics Sparingly**: Only when needed
   ```python
   # Don't query every step
   for step in range(1000):
       step_simulation()

   # Only query at checkpoints
   if step % 100 == 0:
       check_metrics()
   ```

---

## API Response Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Action executed |
| 201 | Created | Session created |
| 400 | Bad Request | Invalid action format |
| 404 | Not Found | Session doesn't exist |
| 500 | Server Error | Unexpected failure |

---

## Next Steps

1. Read [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete endpoint reference
2. Review [test_adapter.py](test_adapter.py) for integration examples
3. Modify [common_fixed_variables.py](common_fixed_variables.py) for your simulation parameters

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test cases in `test_adapter.py`
3. Check adapter logs for error messages
4. Verify configuration in `common_fixed_variables.py`
