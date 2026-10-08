import json
import subprocess
import sys
import threading
import time

import requests


class AdapterTestSuite:
    """Comprehensive test suite for WildFire Adapter"""

    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_id = None
        self.test_results = {"passed": 0, "failed": 0, "errors": []}

    def test(self, name, condition, error_msg=""):
        """Helper to track test results"""
        if condition:
            print(f"  ✓ {name}")
            self.test_results["passed"] += 1
        else:
            print(f"  ✗ {name}")
            self.test_results["failed"] += 1
            self.test_results["errors"].append(f"{name}: {error_msg}")

    def print_summary(self):
        """Print test summary"""
        total = self.test_results["passed"] + self.test_results["failed"]
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.test_results['passed']}/{total} passed")
        print("=" * 60)
        if self.test_results["errors"]:
            print("\nFailures:")
            for error in self.test_results["errors"]:
                print(f"  - {error}")
        return self.test_results["failed"] == 0

    # ====================================================================
    # TEST GROUPS
    # ====================================================================

    def test_health_check(self):
        """Test /health endpoint"""
        print("\n--- Testing Health Check ---")
        try:
            resp = requests.get(f"{self.base_url}/health")
            self.test("Health endpoint responds", resp.status_code == 200)
            self.test("Health response has status", "status" in resp.json())
            self.test("Health status is healthy", resp.json()["status"] == "healthy")
        except Exception as e:
            self.test("Health endpoint", False, str(e))

    def test_session_management(self):
        """Test session CRUD operations"""
        print("\n--- Testing Session Management ---")

        # Test 1: Create session
        try:
            resp = requests.post(f"{self.base_url}/api/v1/sessions")
            self.test("Create session returns 201", resp.status_code == 201)

            data = resp.json()
            self.test("Create session returns session_id", "session_id" in data)
            self.test("Create session has num_agents", "num_agents" in data)

            self.session_id = data["session_id"]

        except Exception as e:
            self.test("Create session", False, str(e))
            return

        # Test 2: List sessions
        try:
            resp = requests.get(f"{self.base_url}/api/v1/sessions")
            self.test("List sessions returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("List sessions has total", "total" in data)
            self.test("List sessions includes our session", self.session_id in data["sessions"])

        except Exception as e:
            self.test("List sessions", False, str(e))

        # Test 3: Get specific session info
        try:
            resp = requests.get(f"{self.base_url}/api/v1/sessions/{self.session_id}")
            self.test("Get session info returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("Session info has timestep", "timestep" in data)
            self.test("Session info has paused flag", "paused" in data)
            self.test("Session info has metrics", "metrics" in data)

        except Exception as e:
            self.test("Get session info", False, str(e))

        # Test 4: Set current session
        try:
            resp = requests.put(f"{self.base_url}/api/v1/sessions/{self.session_id}/current")
            self.test("Set current session returns 200", resp.status_code == 200)

        except Exception as e:
            self.test("Set current session", False, str(e))

        # Test 5: Get invalid session
        try:
            resp = requests.get(f"{self.base_url}/api/v1/sessions/invalid_id")
            self.test("Invalid session returns 404", resp.status_code == 404)

        except Exception as e:
            self.test("Invalid session handling", False, str(e))

    def test_simulation_initialization(self):
        """Test simulation initialization"""
        print("\n--- Testing Simulation Initialization ---")

        try:
            # Create a fresh session for this test
            resp = requests.post(f"{self.base_url}/api/v1/sessions")
            session_id = resp.json()["session_id"]

            # Get config
            requests.put(f"{self.base_url}/api/v1/sessions/{session_id}/current")
            resp = requests.get(f"{self.base_url}/api/v1/sim/config")
            self.test("Get config returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("Config has num_agents", "num_agents" in data)
            self.test("Config has grid dimensions", "grid_height" in data and "grid_width" in data)
            self.test("Config has valid_moves", "valid_moves" in data)
            self.test(
                "Valid moves are correct",
                set(data["valid_moves"]) == {"east", "south", "west", "north", "hold"},
            )

        except Exception as e:
            self.test("Simulation initialization", False, str(e))

    def test_state_retrieval(self):
        """Test state retrieval"""
        print("\n--- Testing State Retrieval ---")

        try:
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            self.test("Get state returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("State has timestep", "timestep" in data)
            self.test("State has num_agents", "num_agents" in data)
            self.test("State has states array", "states" in data)
            self.test("State array matches num_agents", len(data["states"]) == data["num_agents"])

        except Exception as e:
            self.test("State retrieval", False, str(e))

    def test_metrics_retrieval(self):
        """Test metrics retrieval"""
        print("\n--- Testing Metrics Retrieval ---")

        try:
            resp = requests.get(f"{self.base_url}/api/v1/sim/metrics")
            self.test("Get metrics returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("Metrics has timestep", "timestep" in data)
            self.test("Metrics has metrics object", "metrics" in data)

            metrics = data["metrics"]
            self.test("Metrics has num_agents", "num_agents" in metrics)
            self.test("Metrics has mr1_values", "mr1_values" in metrics)
            self.test("Metrics has mr2_value", "mr2_value" in metrics)
            self.test("Metrics has fire_cells_burning", "fire_cells_burning" in metrics)
            self.test("Metrics has agent_positions", "agent_positions" in metrics)

        except Exception as e:
            self.test("Metrics retrieval", False, str(e))

    def test_agent_queries(self):
        """Test agent information queries"""
        print("\n--- Testing Agent Queries ---")

        try:
            # Get all agents
            resp = requests.get(f"{self.base_url}/api/v1/sim/agents")
            self.test("Get all agents returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("Agents response has agents key", "agents" in data)

            agents_info = data["agents"]
            self.test("Agents info has total_agents", "total_agents" in agents_info)

            # Get specific agent
            if agents_info.get("total_agents", 0) > 0:
                resp = requests.get(f"{self.base_url}/api/v1/sim/agents/0")
                self.test("Get specific agent returns 200", resp.status_code == 200)

                agent = resp.json()["agent"]
                self.test("Agent has position", "position" in agent)
                self.test("Agent has selected_direction", "selected_direction" in agent)

                # Test invalid agent index
                resp = requests.get(f"{self.base_url}/api/v1/sim/agents/999")
                self.test("Invalid agent index returns 400", resp.status_code == 400)

        except Exception as e:
            self.test("Agent queries", False, str(e))

    def test_single_action(self):
        """Test single action execution"""
        print("\n--- Testing Single Action ---")

        try:
            # Reset first
            requests.post(f"{self.base_url}/api/v1/sim/reset")

            # Get initial state
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            initial_timestep = resp.json()["timestep"]

            # Send action
            actions = [{"uav": 0, "move": "north"}]
            resp = requests.post(f"{self.base_url}/api/v1/sim/action", json=actions)
            self.test("Action returns 200", resp.status_code == 200)

            # Verify timestep incremented
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            new_timestep = resp.json()["timestep"]
            self.test("Timestep incremented", new_timestep > initial_timestep)

        except Exception as e:
            self.test("Single action", False, str(e))

    def test_invalid_actions(self):
        """Test invalid action handling"""
        print("\n--- Testing Invalid Actions ---")

        try:
            requests.post(f"{self.base_url}/api/v1/sim/reset")

            # Invalid move
            resp = requests.post(
                f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "invalid"}]
            )
            self.test("Invalid move returns 400", resp.status_code == 400)

            # Invalid UAV index
            resp = requests.post(
                f"{self.base_url}/api/v1/sim/action", json=[{"uav": 999, "move": "north"}]
            )
            self.test("Invalid UAV index returns 400", resp.status_code == 400)

            # Missing fields
            resp = requests.post(f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0}])
            self.test("Missing move field returns 400", resp.status_code == 400)

            # Not a list
            resp = requests.post(
                f"{self.base_url}/api/v1/sim/action", json={"uav": 0, "move": "north"}
            )
            self.test("Non-list action returns 400", resp.status_code == 400)

        except Exception as e:
            self.test("Invalid action handling", False, str(e))

    def test_batch_actions(self):
        """Test batch action execution"""
        print("\n--- Testing Batch Actions ---")

        try:
            requests.post(f"{self.base_url}/api/v1/sim/reset")

            # Execute batch
            batch = {
                "actions": [
                    [{"uav": 0, "move": "north"}],
                    [{"uav": 0, "move": "south"}],
                    [{"uav": 0, "move": "east"}],
                ]
            }
            resp = requests.post(f"{self.base_url}/api/v1/sim/batch-actions", json=batch)
            self.test("Batch execution returns 200", resp.status_code == 200)

            data = resp.json()
            self.test("Batch returns steps_executed", "steps_executed" in data)
            self.test("Batch executed 3 steps", data["steps_executed"] == 3)

        except Exception as e:
            self.test("Batch actions", False, str(e))

    def test_simulation_control(self):
        """Test pause/resume functionality"""
        print("\n--- Testing Simulation Control ---")

        try:
            requests.post(f"{self.base_url}/api/v1/sim/reset")

            # Pause simulation
            resp = requests.post(f"{self.base_url}/api/v1/sim/pause")
            self.test("Pause returns 200", resp.status_code == 200)
            self.test("Pause sets paused flag", resp.json()["paused"] == True)

            # Try to act while paused
            resp = requests.post(
                f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "north"}]
            )
            self.test("Action while paused returns 400", resp.status_code == 400)

            # Resume simulation
            resp = requests.post(f"{self.base_url}/api/v1/sim/resume")
            self.test("Resume returns 200", resp.status_code == 200)
            self.test("Resume clears paused flag", resp.json()["paused"] == False)

            # Now action should work
            resp = requests.post(
                f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "north"}]
            )
            self.test("Action after resume works", resp.status_code == 200)

        except Exception as e:
            self.test("Simulation control", False, str(e))

    def test_reset(self):
        """Test reset functionality"""
        print("\n--- Testing Reset ---")

        try:
            # Do some steps
            requests.post(f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "north"}])
            requests.post(f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "south"}])

            # Check timestep
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            timestep_before = resp.json()["timestep"]
            self.test("Timestep > 0 after actions", timestep_before > 0)

            # Reset
            resp = requests.post(f"{self.base_url}/api/v1/sim/reset")
            self.test("Reset returns 200", resp.status_code == 200)

            # Check timestep
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            timestep_after = resp.json()["timestep"]
            self.test("Reset clears timestep", timestep_after == 0)

        except Exception as e:
            self.test("Reset functionality", False, str(e))

    def test_step_without_actions(self):
        """Test stepping without actions"""
        print("\n--- Testing Step Without Actions ---")

        try:
            requests.post(f"{self.base_url}/api/v1/sim/reset")

            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            initial_timestep = resp.json()["timestep"]

            resp = requests.post(f"{self.base_url}/api/v1/sim/step")
            self.test("Step returns 200", resp.status_code == 200)

            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            new_timestep = resp.json()["timestep"]
            self.test("Step increments timestep", new_timestep > initial_timestep)

        except Exception as e:
            self.test("Step without actions", False, str(e))

    def test_session_isolation(self):
        """Test that sessions are isolated"""
        print("\n--- Testing Session Isolation ---")

        try:
            # Create session 1
            resp = requests.post(f"{self.base_url}/api/v1/sessions")
            session1_id = resp.json()["session_id"]
            requests.put(f"{self.base_url}/api/v1/sessions/{session1_id}/current")

            # Do some steps
            requests.post(f"{self.base_url}/api/v1/sim/reset")
            for _ in range(3):
                requests.post(
                    f"{self.base_url}/api/v1/sim/action", json=[{"uav": 0, "move": "north"}]
                )

            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            session1_timestep = resp.json()["timestep"]

            # Create session 2
            resp = requests.post(f"{self.base_url}/api/v1/sessions")
            session2_id = resp.json()["session_id"]
            requests.put(f"{self.base_url}/api/v1/sessions/{session2_id}/current")

            # Check session 2 starts fresh
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            session2_timestep = resp.json()["timestep"]
            self.test("Session 2 starts at timestep 0", session2_timestep == 0)

            # Switch back to session 1
            requests.put(f"{self.base_url}/api/v1/sessions/{session1_id}/current")
            resp = requests.get(f"{self.base_url}/api/v1/sim/state")
            session1_timestep_restored = resp.json()["timestep"]
            self.test("Session 1 state preserved", session1_timestep_restored == session1_timestep)

        except Exception as e:
            self.test("Session isolation", False, str(e))

    def test_deprecated_endpoints(self):
        """Test backwards compatibility with deprecated endpoints"""
        print("\n--- Testing Deprecated Endpoints ---")

        try:
            # Create session via old /init endpoint
            resp = requests.get(f"{self.base_url}/api/v1/sim/init")
            self.test("Deprecated /init returns 200", resp.status_code == 200)

        except Exception as e:
            self.test("Deprecated endpoints", False, str(e))

    def run_all_tests(self):
        """Run all test suites"""
        print("=" * 60)
        print("WILDFIRE ADAPTER TEST SUITE")
        print("=" * 60)

        self.test_health_check()
        self.test_session_management()
        self.test_simulation_initialization()
        self.test_state_retrieval()
        self.test_metrics_retrieval()
        self.test_agent_queries()
        self.test_single_action()
        self.test_invalid_actions()
        self.test_batch_actions()
        self.test_simulation_control()
        self.test_reset()
        self.test_step_without_actions()
        self.test_session_isolation()
        self.test_deprecated_endpoints()

        return self.print_summary()


def run_tests():
    """Start adapter server and run tests"""
    print("Starting adapter server...")

    # Start the adapter in a separate process
    process = subprocess.Popen(
        [sys.executable, "adapter.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        # Wait for server to start
        time.sleep(3)

        # Check if server started successfully
        try:
            requests.get("http://localhost:5000/health")
        except requests.exceptions.ConnectionError:
            print("ERROR: Failed to connect to adapter server")
            outs, errs = process.communicate(timeout=1)
            print("Server output:", outs)
            print("Server errors:", errs)
            return False

        # Run test suite
        suite = AdapterTestSuite()
        success = suite.run_all_tests()

        return success

    except Exception as e:
        print(f"Test execution failed: {e}")
        return False
    finally:
        print("\nTerminating adapter server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
