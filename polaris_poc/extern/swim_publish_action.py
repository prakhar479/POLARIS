#!/usr/bin/env python3
"""
NATS Action Publisher Script for POLARIS

A standalone command-line tool to publish `ControlAction` events to a NATS server,
designed to test the POLARIS ExecutionAdapter.

Usage Examples:
  # 1. Add a server
  python publish_action.py ADD_SERVER

  # 2. Remove a server
  python publish_action.py REMOVE_SERVER

  # 3. Set the dimmer (QoS) to 0.5
  python publish_action.py SET_DIMMER --value 0.5 (or check alias with ADJUST_QOS --value 0.5)

  # 4. Use a different NATS server URL
  python publish_action.py ADD_SERVER --nats_url "nats://my-nats-server:4222"

  # 5. Publish to a different subject
  python publish_action.py REMOVE_SERVER --subject "custom.actions"
"""

import asyncio
import json
import argparse
import uuid
from typing import Dict, Any

from nats.aio.client import Client as NATS

async def main():
    """
    Main function to parse arguments, construct a ControlAction,
    and publish it to the NATS server.
    """
    parser = argparse.ArgumentParser(
        description="Publish a ControlAction event to a NATS server.",
        formatter_class=argparse.RawTextHelpFormatter  # To preserve newlines in help text
    )

    # --- Positional Arguments ---
    parser.add_argument(
        "action_type",
        choices=["ADD_SERVER", "REMOVE_SERVER", "SET_DIMMER", "ADJUST_QOS"],
        help="The type of action to publish."
    )

    # --- Optional Arguments ---
    parser.add_argument(
        "--value",
        type=float,
        help="The floating point value for SET_DIMMER or ADJUST_QOS actions (e.g., 0.5)."
    )
    parser.add_argument(
        "--nats_url",
        default="nats://localhost:4222",
        help="The URL of the NATS server."
    )
    parser.add_argument(
        "--subject",
        default="polaris.execution.actions",
        help="The NATS subject to publish the action to."
    )

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.action_type in ["SET_DIMMER", "ADJUST_QOS"] and args.value is None:
        parser.error(f"--value <float> is required for the '{args.action_type}' action.")

    # --- Construct the ControlAction Payload ---
    params: Dict[str, Any] = {}
    if args.value is not None:
        params["value"] = args.value

    action_payload = {
        "action_type": args.action_type,
        "params": params,
        "action_id": str(uuid.uuid4()),
        "source": "manual_publisher_script",
        # Timestamp will be added by the pydantic model on the receiving end
    }

    # Convert to JSON bytes
    json_payload = json.dumps(action_payload, indent=2).encode('utf-8')

    # --- Connect to NATS and Publish ---
    nc = NATS()
    try:
        print(f"Connecting to NATS at {args.nats_url}...")
        await nc.connect(args.nats_url)
        print("Connection successful.")

        print("-" * 40)
        print(f"Publishing to subject: '{args.subject}'")
        print("Payload:")
        print(json_payload.decode())
        print("-" * 40)

        await nc.publish(args.subject, json_payload)
        await nc.flush()  # Ensure the message is sent before closing

        print("✅ Action event published successfully!")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if nc.is_connected:
            await nc.close()
            print("NATS connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")