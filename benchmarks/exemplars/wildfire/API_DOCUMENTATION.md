# WildFire Adapter REST API Documentation

## Overview

The WildFire Adapter is a Flask-based REST API server that provides external control and monitoring of WildFire simulations. It enables detailed management of multi-agent simulations through comprehensive REST endpoints with session management, metrics retrieval, and precise action execution.

**Server**: Runs on `http://localhost:5000` by default

---

## Key Features

✅ **Multi-Session Management** - Create, list, switch between, and delete independent simulation sessions
✅ **Precise Action Control** - Execute individual UAV actions with full validation
✅ **Batch Execution** - Run multiple timesteps with predefined action sequences
✅ **Rich Metrics** - Access MR1, MR2, fire spread data, and agent positions
✅ **Simulation Control** - Pause, resume, reset, and step through simulations
✅ **Comprehensive Logging** - Full error tracking and debugging information
✅ **Robust Validation** - Input validation with detailed error messages

---

## Base URL

```
http://localhost:5000
```

---

## Health & Status Endpoints

### 1. Health Check
**Endpoint**: `GET /health`

Check if the adapter server is running and get active session info.

**Response** (200):
```json
{
    "status": "healthy",
    "timestamp": "2026-02-02T21:30:45.123456",
    "active_sessions": 2,
    "current_session": "6dc8b003"
}
```

---

## Session Management Endpoints

### 2. Create Session
**Endpoint**: `POST /api/v1/sessions`

Create a new simulation session.

**Response** (201):
```json
{
    "session_id": "6dc8b003",
    "message": "Simulation session created",
    "num_agents": 2,
    "grid_size": {"height": 50, "width": 50}
}
```

**Error** (500):
```json
{"error": "Error message describing the failure"}
```

---

### 3. List Sessions
**Endpoint**: `GET /api/v1/sessions`

List all active sessions.

**Response** (200):
```json
{
    "total": 2,
    "current": "6dc8b003",
    "sessions": ["6dc8b003", "a1b2c3d4"]
}
```

---

### 4. Get Session Info
**Endpoint**: `GET /api/v1/sessions/<session_id>`

Get detailed information about a specific session.

**Response** (200):
```json
{
    "session_id": "6dc8b003",
    "is_current": true,
    "paused": false,
    "timestep": 5,
    "total_steps": 10,
    "metrics": {
        "timestep": 5,
        "num_agents": 2,
        "mr1_values": [0.5, 0.6],
        "mr2_value": 2,
        "fire_cells_burning": 15,
        "fire_cells_total": 2500,
        "agent_positions": [
            {"id": 0, "position": [25, 30]},
            {"id": 1, "position": [26, 31]}
        ]
    }
}
```

**Error** (404):
```json
{"error": "Session xyz not found"}
```

---

### 5. Set Current Session
**Endpoint**: `PUT /api/v1/sessions/<session_id>/current`

Set the active session for subsequent API calls.

**Response** (200):
```json
{"message": "Current session set to 6dc8b003"}
```

**Error** (404):
```json
{"error": "Session xyz not found"}
```

---

### 6. Delete Session
**Endpoint**: `DELETE /api/v1/sessions/<session_id>`

Delete a simulation session and free its resources.

**Response** (200):
```json
{"message": "Session 6dc8b003 deleted"}
```

---

## Simulation Control Endpoints

### 7. Initialize Simulation (Deprecated)
**Endpoint**: `GET /api/v1/sim/init`

**⚠️ Deprecated**: Use `POST /api/v1/sessions` instead.

Creates a new session for backwards compatibility.

**Response** (200):
```json
{
    "message": "Simulation initialized",
    "session_id": "6dc8b003",
    "num_agents": 2
}
```

---

### 8. Reset Simulation
**Endpoint**: `POST /api/v1/sim/reset`

Reset the current simulation to initial state (timestep 0).

**Response** (200):
```json
{
    "message": "Simulation reset",
    "timestep": 0,
    "num_agents": 2
}
```

**Error** (400):
```json
{"error": "No session ID provided and no current session set"}
```

---

### 9. Pause Simulation
**Endpoint**: `POST /api/v1/sim/pause`

Pause the simulation. No actions can be executed while paused.

**Response** (200):
```json
{
    "message": "Simulation paused",
    "paused": true,
    "timestep": 5
}
```

---

### 10. Resume Simulation
**Endpoint**: `POST /api/v1/sim/resume`

Resume a paused simulation.

**Response** (200):
```json
{
    "message": "Simulation resumed",
    "paused": false,
    "timestep": 5
}
```

---

### 11. Step Simulation
**Endpoint**: `POST /api/v1/sim/step`

Execute a single simulation step without any agent actions (uses hold for all agents).

**Response** (200):
```json
{
    "message": "Step executed",
    "timestep": 6
}
```

**Error** (400):
```json
{"error": "Simulation is paused"}
```

---

## State & Observation Endpoints

### 12. Get State
**Endpoint**: `GET /api/v1/sim/state`

Get the observation state from all agents (burning cell observations in each UAV's field of view).

**Response** (200):
```json
{
    "timestep": 5,
    "total_steps": 10,
    "paused": false,
    "num_agents": 2,
    "states": [
        [0, 1, 0, 0, 1, 0, 0, 0, 0, ..., 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, ..., 0]
    ]
}
```

**Format**: Each state is a list of burning cell detections in the UAV's observation radius.

---

### 13. Get Metrics
**Endpoint**: `GET /api/v1/sim/metrics`

Get comprehensive simulation metrics.

**Response** (200):
```json
{
    "timestep": 5,
    "metrics": {
        "timestep": 5,
        "num_agents": 2,
        "mr1_values": [0.5, 0.6],
        "mr2_value": 2,
        "fire_cells_burning": 15,
        "fire_cells_total": 2500,
        "agent_positions": [
            {"id": 0, "position": [25, 30]},
            {"id": 1, "position": [26, 31]}
        ]
    }
}
```

**Metrics Description**:
- `mr1_values`: Detection reward for each agent (normalized burning cell count)
- `mr2_value`: Collision avoidance metric
- `fire_cells_burning`: Current burning cell count
- `fire_cells_total`: Total fire cells in grid
- `agent_positions`: Current positions of all UAVs

---

## Agent Query Endpoints

### 14. Get All Agents
**Endpoint**: `GET /api/v1/sim/agents`

Get detailed information about all agents.

**Response** (200):
```json
{
    "timestep": 5,
    "agents": {
        "total_agents": 2,
        "agents": [
            {
                "agent_id": 0,
                "index": 0,
                "position": [25, 30],
                "selected_direction": 3
            },
            {
                "agent_id": 1,
                "index": 1,
                "position": [26, 31],
                "selected_direction": 2
            }
        ]
    }
}
```

---

### 15. Get Specific Agent
**Endpoint**: `GET /api/v1/sim/agents/<agent_index>`

Get information about a specific agent by index.

**Parameters**:
- `agent_index` (int): Agent index (0-based)

**Response** (200):
```json
{
    "timestep": 5,
    "agent": {
        "agent_id": 0,
        "index": 0,
        "position": [25, 30],
        "selected_direction": 3
    }
}
```

**Error** (400):
```json
{"error": "Invalid agent index 999"}
```

---

## Action Execution Endpoints

### 16. Execute Single Action
**Endpoint**: `POST /api/v1/sim/action`

Execute one or more actions and step the simulation once.

**Request Body**:
```json
[
    {"uav": 0, "move": "north"},
    {"uav": 1, "move": "hold"}
]
```

**Valid Moves**: `"north"`, `"south"`, `"east"`, `"west"`, `"hold"`

**Response** (200):
```json
{
    "message": "Action executed successfully",
    "timestep": 6,
    "applied": 2
}
```

**Error** (400) - Invalid Move:
```json
{"error": "Expected a list of actions"}
```

**Error** (400) - Invalid UAV:
```json
{
    "message": "Actions applied with errors",
    "applied": 1,
    "total": 2,
    "errors": [
        "Action 1: Invalid UAV index 999. Valid range: 0-1"
    ]
}
```

**Error** (400) - Paused Simulation:
```json
{"error": "Simulation is paused. Call /sim/resume first."}
```

---

### 17. Execute Batch Actions
**Endpoint**: `POST /api/v1/sim/batch-actions`

Execute multiple timesteps with predefined actions sequences.

**Request Body**:
```json
{
    "actions": [
        [{"uav": 0, "move": "north"}, {"uav": 1, "move": "east"}],
        [{"uav": 0, "move": "east"}, {"uav": 1, "move": "north"}],
        [{"uav": 0, "move": "south"}, {"uav": 1, "move": "hold"}]
    ]
}
```

**Response** (200):
```json
{
    "message": "Batch execution completed",
    "steps_executed": 3,
    "results": [
        {
            "step": 0,
            "timestep": 1,
            "applied": 2,
            "errors": []
        },
        {
            "step": 1,
            "timestep": 2,
            "applied": 2,
            "errors": []
        },
        {
            "step": 2,
            "timestep": 3,
            "applied": 2,
            "errors": []
        }
    ]
}
```

**Note**: Batch execution stops if simulation becomes paused.

---

## Configuration Endpoint

### 18. Get Configuration
**Endpoint**: `GET /api/v1/sim/config`

Get simulation configuration and constants.

**Response** (200):
```json
{
    "num_agents": 2,
    "grid_height": 50,
    "grid_width": 50,
    "observation_radius": 8,
    "valid_moves": ["east", "south", "west", "north", "hold"],
    "constants": {
        "burning_rate": 1,
        "fire_spread_speed": 2,
        "fuel_limits": {
            "upper": 10,
            "lower": 7
        }
    }
}
```

---

## Common Error Responses

### 404 - Not Found
```json
{
    "error": "Endpoint not found",
    "path": "/api/v1/invalid"
}
```

### 500 - Internal Server Error
```json
{"error": "Internal server error"}
```

---

## Action Specification

### Supported Actions
| Move | Code | Direction |
|------|------|-----------|
| north | 3 | Up (positive Y) |
| south | 1 | Down (negative Y) |
| east | 0 | Right (positive X) |
| west | 2 | Left (negative X) |
| hold | - | Stay in place, keep previous direction |

### Action Validation Rules
1. ✅ UAV index must be in range [0, NUM_AGENTS-1]
2. ✅ Move must be one of the valid moves
3. ✅ Actions list cannot be empty (although individual agents can hold)
4. ✅ All required fields (`uav`, `move`) must be present

---

## Workflow Examples

### Example 1: Create Session and Execute Single Step

```bash
# Create a new session
curl -X POST http://localhost:5000/api/v1/sessions

# Set as current (if not already)
curl -X PUT http://localhost:5000/api/v1/sessions/6dc8b003/current

# Reset simulation
curl -X POST http://localhost:5000/api/v1/sim/reset

# Get initial state
curl -X GET http://localhost:5000/api/v1/sim/state

# Execute action
curl -X POST http://localhost:5000/api/v1/sim/action \
  -H "Content-Type: application/json" \
  -d '[{"uav": 0, "move": "north"}]'

# Check final state
curl -X GET http://localhost:5000/api/v1/sim/state
```

### Example 2: Multi-Session Management

```bash
# Create session 1
SESSION1=$(curl -X POST http://localhost:5000/api/v1/sessions | jq -r .session_id)

# Create session 2
SESSION2=$(curl -X POST http://localhost:5000/api/v1/sessions | jq -r .session_id)

# Switch to session 1
curl -X PUT http://localhost:5000/api/v1/sessions/$SESSION1/current

# Do work in session 1...

# Switch to session 2
curl -X PUT http://localhost:5000/api/v1/sessions/$SESSION2/current

# Do work in session 2...

# List all sessions
curl -X GET http://localhost:5000/api/v1/sessions

# Delete session 1
curl -X DELETE http://localhost:5000/api/v1/sessions/$SESSION1
```

### Example 3: Batch Execution with Pause/Resume

```bash
# Execute batch of actions
curl -X POST http://localhost:5000/api/v1/sim/batch-actions \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      [{"uav": 0, "move": "north"}],
      [{"uav": 0, "move": "east"}],
      [{"uav": 0, "move": "south"}]
    ]
  }'

# Pause simulation
curl -X POST http://localhost:5000/api/v1/sim/pause

# Inspect metrics
curl -X GET http://localhost:5000/api/v1/sim/metrics

# Resume and continue
curl -X POST http://localhost:5000/api/v1/sim/resume
```

---

## Implementation Details

### Direction Encoding
- Direction values 0-3 correspond to move vectors:
  - **0 (EAST)**: Move right (+X)
  - **1 (SOUTH)**: Move down (-Y)
  - **2 (WEST)**: Move left (-X)
  - **3 (NORTH)**: Move up (+Y)

### Hold Action
- **hold** preserves the agent's current direction
- Multiple consecutive holds keep the agent stationary with no direction change

### Agent Ordering
- Agents are indexed in the order they were created (0, 1, 2, ...)
- Agent positions are grid coordinates [X, Y]
- Direction in response is the numeric code (0-3)

### Metrics
- **MR1 (Detection Reward)**: Cumulative normalized burning cell count per agent
- **MR2 (Collision Avoidance)**: Count of inter-agent proximity violations

---

## Logging

The adapter logs all operations at INFO level. Enable DEBUG logging to track individual action processing:

```python
import logging
logging.getLogger('adapter').setLevel(logging.DEBUG)
```

---

## Performance Notes

- Each `/action` call steps the simulation once
- Batch actions (`/batch-actions`) execute multiple steps sequentially
- State and metrics calls are read-only and don't affect simulation
- Session switching is O(1) with no performance penalty

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial release with comprehensive REST API |
