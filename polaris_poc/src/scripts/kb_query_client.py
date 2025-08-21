#!/usr/bin/env python3
"""
Knowledge Base Query Client

A comprehensive client for querying the POLARIS Knowledge Base Service.
Provides interactive and command-line interfaces for accessing stored data.

Usage:
    python kb_query_client.py stats                    # Get KB statistics
    python kb_query_client.py telemetry --limit 10     # Get recent telemetry
    python kb_query_client.py observations --limit 5   # Get recent observations
    python kb_query_client.py search --keyword swim    # Search by keyword
    python kb_query_client.py interactive              # Interactive mode
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional

import nats


class KnowledgeBaseClient:
    """Client for querying the POLARIS Knowledge Base Service."""

    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc = None

    async def connect(self):
        """Connect to NATS."""
        try:
            self.nc = await nats.connect(self.nats_url)
            print(f"✅ Connected to NATS at {self.nats_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to NATS: {e}")
            return False

    async def disconnect(self):
        """Disconnect from NATS."""
        if self.nc:
            await self.nc.close()
            print("🔌 Disconnected from NATS")

    async def get_stats(self) -> Optional[Dict]:
        """Get Knowledge Base statistics."""
        try:
            response = await self.nc.request(
                "polaris.knowledge.stats", b"", timeout=5.0
            )
            stats = json.loads(response.data.decode())
            return stats
        except Exception as e:
            print(f"❌ Stats query failed: {e}")
            return None

    async def query_data(self, query: Dict) -> Optional[Dict]:
        """Execute a general query against the Knowledge Base."""
        try:
            response = await self.nc.request(
                "polaris.knowledge.query", json.dumps(query).encode(), timeout=10.0
            )
            result = json.loads(response.data.decode())
            return result
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None

    async def get_telemetry(
        self, limit: int = 10, metric_name: str = None
    ) -> List[Dict]:
        """Get recent telemetry events."""
        query = {
            "query_type": "structured",
            "data_types": ["raw_telemetry_event"],
            "limit": limit,
        }

        if metric_name:
            query["filters"] = {"metric_name": metric_name}

        result = await self.query_data(query)
        if result and result.get("success"):
            return result.get("results", [])
        return []

    async def get_observations(
        self, limit: int = 10, metric_name: str = None
    ) -> List[Dict]:
        """Get recent observations."""
        query = {
            "query_type": "structured",
            "data_types": ["observation"],
            "limit": limit,
        }

        if metric_name:
            query["filters"] = {"metric_name": metric_name}

        result = await self.query_data(query)
        if result and result.get("success"):
            return result.get("results", [])
        return []

    async def search_by_keyword(self, keyword: str, limit: int = 20) -> List[Dict]:
        """Search entries by keyword."""
        query = {
            "query_type": "natural_language",
            "query_text": keyword,
            "limit": limit,
        }

        result = await self.query_data(query)
        if result and result.get("success"):
            return result.get("results", [])
        return []

    def format_entry(self, entry: Dict, show_details: bool = False) -> str:
        """Format a KB entry for display."""
        entry_type = entry.get("data_type", "UNKNOWN")
        metric_name = entry.get("metric_name", "unknown")
        metric_value = entry.get("metric_value", "N/A")
        timestamp = entry.get("timestamp", "N/A")
        source = entry.get("source", "unknown")

        # Format timestamp if it's a datetime string
        if isinstance(timestamp, str) and "T" in timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamp = dt.strftime("%H:%M:%S")
            except:
                pass

        basic_info = f"[{entry_type}] {metric_name}: {metric_value} (from {source} at {timestamp})"

        if show_details:
            summary = entry.get("summary", "")
            tags = entry.get("tags", [])
            details = f"\n    Summary: {summary}\n    Tags: {', '.join(tags)}"
            return basic_info + details

        return basic_info

    async def display_stats(self):
        """Display Knowledge Base statistics."""
        print("\n📊 Knowledge Base Statistics")
        print("=" * 40)

        stats = await self.get_stats()
        if not stats:
            return

        # Service stats
        service_stats = stats.get("service", {})
        kb_stats = stats.get("knowledge_base", {})
        uptime = stats.get("uptime_seconds", 0)

        print(f"Service Uptime: {uptime:.0f} seconds")
        print(f"Events Processed: {service_stats.get('events_processed', 0)}")
        print(f"Queries Served: {service_stats.get('queries_served', 0)}")
        print(f"KB Total Entries: {kb_stats.get('total_permanent_entries', 0)}")
        print(f"Buffered Events: {kb_stats.get('total_buffered_events', 0)}")
        print(f"Active Buffers: {kb_stats.get('active_telemetry_buffers', 0)}")
        print(f"Unique Tags: {kb_stats.get('unique_tags', 0)}")
        print(f"Indexed Keywords: {kb_stats.get('indexed_keywords', 0)}")

        # Show data type breakdown
        data_type_counts = kb_stats.get("data_type_counts", {})
        if data_type_counts:
            print("Data Type Breakdown:")
            for dtype, count in data_type_counts.items():
                print(f"  - {dtype}: {count}")

    async def display_telemetry(self, limit: int = 10, metric_name: str = None):
        """Display recent telemetry events."""
        print(f"\n📡 Recent Telemetry Events (limit: {limit})")
        if metric_name:
            print(f"    Filtered by metric: {metric_name}")
        print("=" * 60)

        events = await self.get_telemetry(limit, metric_name)
        if not events:
            print("No telemetry events found.")
            return

        for event in events:
            print(f"  {self.format_entry(event)}")

    async def display_observations(self, limit: int = 10, metric_name: str = None):
        """Display recent observations."""
        print(f"\n🔍 Recent Observations (limit: {limit})")
        if metric_name:
            print(f"    Filtered by metric: {metric_name}")
        print("=" * 60)

        observations = await self.get_observations(limit, metric_name)
        if not observations:
            print("No observations found.")
            return

        for obs in observations:
            print(f"  {self.format_entry(obs, show_details=True)}")

    async def display_search_results(self, keyword: str, limit: int = 20):
        """Display search results."""
        print(f"\n🔍 Search Results for '{keyword}' (limit: {limit})")
        print("=" * 60)

        results = await self.search_by_keyword(keyword, limit)
        if not results:
            print("No results found.")
            return

        for result in results:
            print(f"  {self.format_entry(result)}")

    async def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🤖 Knowledge Base Interactive Query Mode")
        print(
            "Commands: stats, telemetry [limit], observations [limit], search <keyword>, quit"
        )
        print("=" * 70)

        while True:
            try:
                command = input("\nkb> ").strip().split()
                if not command:
                    continue

                cmd = command[0].lower()

                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "stats":
                    await self.display_stats()
                elif cmd == "telemetry":
                    limit = int(command[1]) if len(command) > 1 else 10
                    await self.display_telemetry(limit)
                elif cmd == "observations":
                    limit = int(command[1]) if len(command) > 1 else 10
                    await self.display_observations(limit)
                elif cmd == "search":
                    if len(command) < 2:
                        print("Usage: search <keyword>")
                        continue
                    keyword = command[1]
                    await self.display_search_results(keyword)
                else:
                    print(
                        "Unknown command. Try: stats, telemetry, observations, search, quit"
                    )

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n👋 Goodbye!")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query the POLARIS Knowledge Base Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                              # Show statistics
  %(prog)s telemetry --limit 20               # Show 20 recent telemetry events
  %(prog)s observations --limit 5             # Show 5 recent observations
  %(prog)s search --keyword swim               # Search for entries with 'swim'
  %(prog)s interactive                        # Start interactive mode
        """,
    )

    parser.add_argument(
        "command",
        choices=["stats", "telemetry", "observations", "search", "interactive"],
        help="Command to execute",
    )

    parser.add_argument(
        "--limit", type=int, default=10, help="Limit number of results (default: 10)"
    )

    parser.add_argument("--keyword", help="Keyword to search for")

    parser.add_argument("--metric", help="Filter by specific metric name")

    parser.add_argument(
        "--nats-url",
        default="nats://localhost:4222",
        help="NATS server URL (default: nats://localhost:4222)",
    )

    args = parser.parse_args()

    # Create client and connect
    client = KnowledgeBaseClient(args.nats_url)

    if not await client.connect():
        sys.exit(1)

    try:
        # Execute command
        if args.command == "stats":
            await client.display_stats()
        elif args.command == "telemetry":
            await client.display_telemetry(args.limit, args.metric)
        elif args.command == "observations":
            await client.display_observations(args.limit, args.metric)
        elif args.command == "search":
            if not args.keyword:
                print("❌ --keyword is required for search command")
                sys.exit(1)
            await client.display_search_results(args.keyword, args.limit)
        elif args.command == "interactive":
            await client.interactive_mode()

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
