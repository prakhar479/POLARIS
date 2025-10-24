import asyncio
import json
from nats.aio.client import Client as NATS

async def test_model_switch():
    nc = NATS()
    await nc.connect("nats://localhost:4222")
    
    # Switch to yolov5s model
    action = {
        "action_id": "test-001",
        "action_type": "SWITCH_MODEL_YOLOV5S",  # Must match config action type
        "system_name": "switch_yolo",  # Must match system_name in config.yaml
        "params": {"model": "yolov5s"},  # Use 'params' not 'parameters'
        "timestamp": "2024-01-01T00:00:00Z",
        "source": "quickstart_test"
    }
    
    await nc.publish("polaris.execution.actions", json.dumps(action).encode())
    print("✓ Action sent: Switch to yolov5s")
    print(f"  Action details: {json.dumps(action, indent=2)}")
    
    await asyncio.sleep(2)  # Give more time for execution
    await nc.close()
    print("✓ Test completed")

asyncio.run(test_model_switch())