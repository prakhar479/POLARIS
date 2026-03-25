"""Example: Configuration validation and error handling.

Shows how configuration validation works and handles various error scenarios.
"""

import asyncio

from polaris import Polaris
from polaris.infrastructure.config import load_config


async def test_valid_config():
    """Test with valid configuration."""
    print("=== Testing Valid Configuration ===")
    try:
        polaris = Polaris(config_path="config/default.yaml")
        print("✅ Valid configuration loaded successfully")
        print(f" - Systems: {len(polaris.config.systems)}")
        print(f" - Strategy: {polaris.config.strategy.type}")
        print(f" - Monitoring interval: {polaris._monitoring_interval}s")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def test_invalid_configs():
    """Test various invalid configurations."""
    print("\n=== Testing Invalid Configurations ===")

    # Test 1: Invalid connector type
    print("\n1. Testing invalid connector type...")
    try:
        config_data = {
            "systems": [{"id": "test", "connector_type": "invalid_connector", "enabled": True}]
        }
        from polaris.infrastructure.config import PolarisConfig

        _ = PolarisConfig.from_dict(config_data)
        print("❌ Should have failed with invalid connector type")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 2: Empty system ID
    print("\n2. Testing empty system ID...")
    try:
        config_data = {"systems": [{"id": "", "connector_type": "swim", "enabled": True}]}
        _ = PolarisConfig.from_dict(config_data)
        print("❌ Should have failed with empty system ID")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 3: Invalid strategy type
    print("\n3. Testing invalid strategy type...")
    try:
        config_data = {"strategy": {"type": "invalid_strategy"}}
        _ = PolarisConfig.from_dict(config_data)
        print("❌ Should have failed with invalid strategy type")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 4: Invalid threshold configuration
    print("\n4. Testing invalid threshold configuration...")
    try:
        config_data = {
            "strategy": {
                "type": "threshold",
                "threshold": {
                    "thresholds": {"cpu_usage": {"high": 20.0, "low": 80.0}}
                },  # High < Low (invalid)
            }
        }
        PolarisConfig.from_dict(config_data)
        print("❌ Should have failed with invalid threshold values")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")

    # Test 5: Invalid port number
    print("\n5. Testing invalid port number...")
    try:
        config_data = {
            "systems": [
                {
                    "id": "test",
                    "connector_type": "swim",
                    "connection": {"port": 99999},  # Invalid port
                }
            ]
        }
        _ = PolarisConfig.from_dict(config_data)
        print("❌ Should have failed with invalid port number")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")


async def test_environment_variables():
    """Test environment variable handling."""
    print("\n=== Testing Environment Variable Handling ===")

    # Test missing environment variable
    print("\n1. Testing missing environment variable...")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
    systems:
    - id: "test"
        connector_type: "swim"
        connection:
        host: "${MISSING_VAR}"
        port: 4242
    """
        )
        temp_config = f.name

    try:
        _ = load_config(temp_config)
        print("❌ Should have failed with missing environment variable")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    finally:
        os.unlink(temp_config)


async def main():
    """Run all configuration validation tests."""
    print("Polaris Configuration Validation Tests")
    print("=" * 50)

    await test_valid_config()
    await test_invalid_configs()
    await test_environment_variables()

    print("\n" + "=" * 50)
    print("Configuration validation tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
