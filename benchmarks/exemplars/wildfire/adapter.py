import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import agents
import flask
import wildfire_model
from common_fixed_variables import *
from flask import jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = flask.Flask(__name__)

# Action mapping - maps string directions to numeric codes
# Note: We only support 4 directions that directly map to move vectors
# For "hold", we'll keep the previous direction instead of changing it
ACTION_MAP = {
    "east": 0,
    "south": 1,
    "west": 2,
    "north": 3,
}

VALID_MOVES = list(ACTION_MAP.keys()) + ["hold"]


@dataclass
class SimulationMetrics:
    """Container for simulation metrics"""

    timestep: int
    num_agents: int
    mr1_values: List[float]
    mr2_value: float
    fire_cells_burning: int
    fire_cells_total: int
    agent_positions: List[Dict[str, Any]]


class ExternalControlledWildFireModel(wildfire_model.WildFireModel):
    """Extended WildFireModel for external control via REST API"""

    def __init__(self):
        super().__init__()
        self.paused = False
        self.total_steps = 0

    def step(self):
        """Override step to NOT generate random actions"""
        if self.paused:
            logger.info("Step called but simulation is paused, skipping...")
            return

        self.datacollector.collect(self)

        # Calculate metrics
        if sum(isinstance(i, agents.UAV) for i in self.schedule.agents) > 0:
            state = self.state()
            self.MR1(state)
            self.MR2()
            self.set_drone_dirs()

        self.evaluation_timesteps_counter += 1
        self.total_steps += 1
        self.schedule.step()
        logger.debug(f"Step {self.total_steps} completed")

    def apply_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply external actions to UAVs.

        Args:
            actions: List of action dicts [{"uav": 0, "move": "north"}, ...]

        Returns:
            Dict with validation results and errors (if any)

        Raises:
            `ValueError`: If actions format is invalid
        """
        if not isinstance(actions, list):
            raise ValueError("Actions must be a list")

        if self.new_direction is None:
            self.new_direction = [0] * self.NUM_AGENTS  # Default to direction 0 (east)

        # Keep previous directions by default (for "hold" functionality)
        current_directions = self.new_direction.copy()
        errors = []
        applied_count = 0

        for idx, action in enumerate(actions):
            try:
                if not isinstance(action, dict):
                    errors.append(f"Action {idx}: Expected dict, got {type(action)}")
                    continue

                uav_idx = action.get("uav")
                move_str = action.get("move")

                # Validation
                if uav_idx is None:
                    errors.append(f"Action {idx}: Missing 'uav' field")
                    continue

                if move_str is None:
                    errors.append(f"Action {idx}: Missing 'move' field")
                    continue

                if not isinstance(uav_idx, int) or uav_idx < 0 or uav_idx >= self.NUM_AGENTS:
                    errors.append(
                        f"Action {idx}: Invalid UAV index {uav_idx}. Valid range: 0-{self.NUM_AGENTS-1}"
                    )
                    continue

                if move_str not in VALID_MOVES:
                    errors.append(
                        f"Action {idx}: Invalid move '{move_str}'. Valid moves: {VALID_MOVES}"
                    )
                    continue

                # Apply the action
                if move_str == "hold":
                    # Keep the current direction (do nothing)
                    pass
                else:
                    # Update direction to the new one
                    current_directions[uav_idx] = ACTION_MAP[move_str]

                applied_count += 1

            except Exception as e:
                errors.append(f"Action {idx}: Error processing - {str(e)}")

        self.new_direction = current_directions

        return {
            "applied": applied_count,
            "total": len(actions),
            "errors": errors,
            "success": len(errors) == 0,
        }

    def get_agent_info(self, agent_idx: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed information about agents"""
        uavs = [a for a in self.schedule.agents if isinstance(a, agents.UAV)]

        if agent_idx is not None:
            if not isinstance(agent_idx, int) or agent_idx < 0 or agent_idx >= len(uavs):
                raise ValueError(f"Invalid agent index {agent_idx}")

            agent = uavs[agent_idx]
            return {
                "agent_id": agent.unique_id,
                "index": agent_idx,
                "position": agent.pos,
                "selected_direction": agent.selected_dir,
            }

        # Return all agents
        return {
            "total_agents": len(uavs),
            "agents": [
                {
                    "agent_id": agent.unique_id,
                    "index": i,
                    "position": agent.pos,
                    "selected_direction": agent.selected_dir,
                }
                for i, agent in enumerate(uavs)
            ],
        }

    def get_metrics(self) -> SimulationMetrics:
        """Get comprehensive simulation metrics"""
        # Count burning cells
        burning_count = 0
        total_count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, agents.Fire):
                total_count += 1
                if agent.is_burning():
                    burning_count += 1

        # Get agent positions
        agent_positions = [
            {"id": a.unique_id, "position": a.pos}
            for a in self.schedule.agents
            if isinstance(a, agents.UAV)
        ]

        return SimulationMetrics(
            timestep=self.evaluation_timesteps_counter,
            num_agents=self.NUM_AGENTS,
            mr1_values=self.MR1_LIST.copy(),
            mr2_value=self.MR2_VALUE,
            fire_cells_burning=burning_count,
            fire_cells_total=total_count,
            agent_positions=agent_positions,
        )


class SimulationManager:
    """Manages multiple simulation sessions"""

    def __init__(self):
        self.sessions: Dict[str, ExternalControlledWildFireModel] = {}
        self.current_session: Optional[str] = None
        logger.info("SimulationManager initialized")

    def create_session(self) -> str:
        """Create a new simulation session"""
        session_id = str(uuid.uuid4())[:8]
        try:
            self.sessions[session_id] = ExternalControlledWildFireModel()
            self.current_session = session_id
            logger.info(f"Created new session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def get_session(self, session_id: Optional[str] = None) -> ExternalControlledWildFireModel:
        """Get a simulation session"""
        if session_id is None:
            session_id = self.current_session

        if session_id is None:
            raise ValueError("No session ID provided and no current session set")

        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        return self.sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Delete a simulation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session == session_id:
                self.current_session = None
            logger.info(f"Deleted session: {session_id}")

    def list_sessions(self) -> Dict[str, Any]:
        """List all sessions"""
        return {
            "total": len(self.sessions),
            "current": self.current_session,
            "sessions": list(self.sessions.keys()),
        }


# Global manager
manager = SimulationManager()


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_sessions": len(manager.sessions),
                "current_session": manager.current_session,
            }
        ),
        200,
    )


@app.route("/api/v1/sessions", methods=["POST"])
def create_session():
    """Create a new simulation session"""
    try:
        session_id = manager.create_session()
        model = manager.get_session(session_id)
        return (
            jsonify(
                {
                    "session_id": session_id,
                    "message": "Simulation session created",
                    "num_agents": model.NUM_AGENTS,
                    "grid_size": {"height": HEIGHT, "width": WIDTH},
                }
            ),
            201,
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions"""
    return jsonify(manager.list_sessions()), 200


@app.route("/api/v1/sessions/<session_id>", methods=["GET"])
def get_session_info(session_id):
    """Get information about a specific session"""
    try:
        model = manager.get_session(session_id)
        metrics = model.get_metrics()
        return (
            jsonify(
                {
                    "session_id": session_id,
                    "is_current": manager.current_session == session_id,
                    "paused": model.paused,
                    "timestep": model.evaluation_timesteps_counter,
                    "total_steps": model.total_steps,
                    "metrics": asdict(metrics),
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/v1/sessions/<session_id>/current", methods=["PUT"])
def set_current_session(session_id):
    """Set the current active session"""
    try:
        manager.get_session(session_id)  # Validate session exists
        manager.current_session = session_id
        logger.info(f"Set current session to: {session_id}")
        return jsonify({"message": f"Current session set to {session_id}"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/v1/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a simulation session"""
    try:
        manager.delete_session(session_id)
        return jsonify({"message": f"Session {session_id} deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/sim/init", methods=["GET"])
def init_simulation():
    """Initialize simulation on current session (deprecated, use POST /sessions)"""
    try:
        session_id = manager.create_session()
        model = manager.get_session(session_id)
        return (
            jsonify(
                {
                    "message": "Simulation initialized",
                    "session_id": session_id,
                    "num_agents": model.NUM_AGENTS,
                }
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Error initializing simulation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/sim/reset", methods=["POST"])
def reset_simulation():
    """Reset the current simulation to initial state"""
    try:
        model = manager.get_session()
        model.reset()
        model.paused = False
        logger.info("Simulation reset")
        return (
            jsonify({"message": "Simulation reset", "timestep": 0, "num_agents": model.NUM_AGENTS}),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/state", methods=["GET"])
def get_state():
    """Get observation state from all agents"""
    try:
        model = manager.get_session()
        states = model.state()
        return (
            jsonify(
                {
                    "timestep": model.evaluation_timesteps_counter,
                    "total_steps": model.total_steps,
                    "paused": model.paused,
                    "num_agents": len(states),
                    "states": states,
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/metrics", methods=["GET"])
def get_metrics():
    """Get comprehensive metrics from simulation"""
    try:
        model = manager.get_session()
        metrics = model.get_metrics()
        return jsonify({"timestep": metrics.timestep, "metrics": asdict(metrics)}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/agents", methods=["GET"])
def get_agents():
    """Get detailed agent information"""
    try:
        model = manager.get_session()
        agent_info = model.get_agent_info()
        return jsonify({"timestep": model.evaluation_timesteps_counter, "agents": agent_info}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/agents/<int:agent_idx>", methods=["GET"])
def get_agent(agent_idx):
    """Get information about a specific agent"""
    try:
        model = manager.get_session()
        agent_info = model.get_agent_info(agent_idx)
        return jsonify({"timestep": model.evaluation_timesteps_counter, "agent": agent_info}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/action", methods=["POST"])
def take_action():
    """Execute action(s) and step simulation"""
    try:
        model = manager.get_session()

        if model.paused:
            return jsonify({"error": "Simulation is paused. Call /sim/resume first."}), 400

        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of actions"}), 400

        # Validate and apply actions
        result = model.apply_actions(data)

        if not result["success"]:
            return (
                jsonify(
                    {
                        "message": "Actions applied with errors",
                        "applied": result["applied"],
                        "total": result["total"],
                        "errors": result["errors"],
                    }
                ),
                400,
            )

        # Step simulation
        model.step()

        return (
            jsonify(
                {
                    "message": "Action executed successfully",
                    "timestep": model.evaluation_timesteps_counter,
                    "applied": result["applied"],
                }
            ),
            200,
        )

    except ValueError as e:
        logger.error(f"Error in action: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in action: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/sim/batch-actions", methods=["POST"])
def batch_actions():
    """Execute multiple timesteps with predefined actions"""
    try:
        model = manager.get_session()
        data = request.get_json()

        if data is None or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        actions_list = data.get("actions", [])
        if not isinstance(actions_list, list):
            return jsonify({"error": "Expected 'actions' to be a list"}), 400

        results = []
        for step_idx, actions in enumerate(actions_list):
            if model.paused:
                logger.warning(f"Batch execution paused at step {step_idx}")
                break

            result = model.apply_actions(actions if isinstance(actions, list) else [])
            model.step()
            results.append(
                {
                    "step": step_idx,
                    "timestep": model.evaluation_timesteps_counter,
                    "applied": result["applied"],
                    "errors": result["errors"],
                }
            )

        return (
            jsonify(
                {
                    "message": "Batch execution completed",
                    "steps_executed": len(results),
                    "results": results,
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/pause", methods=["POST"])
def pause_simulation():
    """Pause the simulation"""
    try:
        model = manager.get_session()
        model.paused = True
        logger.info("Simulation paused")
        return (
            jsonify(
                {
                    "message": "Simulation paused",
                    "paused": True,
                    "timestep": model.evaluation_timesteps_counter,
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/resume", methods=["POST"])
def resume_simulation():
    """Resume the simulation"""
    try:
        model = manager.get_session()
        model.paused = False
        logger.info("Simulation resumed")
        return (
            jsonify(
                {
                    "message": "Simulation resumed",
                    "paused": False,
                    "timestep": model.evaluation_timesteps_counter,
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/step", methods=["POST"])
def step_simulation():
    """Execute a single simulation step without actions"""
    try:
        model = manager.get_session()

        if model.paused:
            return jsonify({"error": "Simulation is paused"}), 400

        model.step()
        return (
            jsonify({"message": "Step executed", "timestep": model.evaluation_timesteps_counter}),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/sim/config", methods=["GET"])
def get_config():
    """Get simulation configuration"""
    try:
        model = manager.get_session()
        return (
            jsonify(
                {
                    "num_agents": model.NUM_AGENTS,
                    "grid_height": HEIGHT,
                    "grid_width": WIDTH,
                    "observation_radius": UAV_OBSERVATION_RADIUS,
                    "valid_moves": VALID_MOVES,
                    "constants": {
                        "burning_rate": BURNING_RATE,
                        "fire_spread_speed": FIRE_SPREAD_SPEED,
                        "fuel_limits": {"upper": FUEL_UPPER_LIMIT, "lower": FUEL_BOTTOM_LIMIT},
                    },
                }
            ),
            200,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "path": request.path}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting WildFire Adapter Server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
