#!/usr/bin/env python3
"""
POLARIS Component Startup Examples

This script demonstrates various ways to start POLARIS components
using the enhanced start_component.py script.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, wait_time=2):
    """Run a command and display the result."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        # For demonstration, we'll just show what would be run
        # In practice, you'd use subprocess.run() for actual execution
        print(f"Would execute: {' '.join(cmd)}")
        time.sleep(wait_time)
        print("✅ Command completed successfully")
    except Exception as e:
        print(f"❌ Command failed: {e}")


def main():
    """Demonstrate various startup scenarios."""
    
    print("🚀 POLARIS Component Startup Examples")
    print("This script demonstrates various startup scenarios")
    print("Note: Commands are shown for demonstration - not actually executed")
    
    # Get the script directory
    script_dir = Path(__file__).parent.parent / "src" / "scripts"
    start_script = script_dir / "start_component.py"
    
    # Example 1: Basic component validation
    run_command([
        sys.executable, str(start_script), "help"
    ], "Show detailed component information")
    
    # Example 2: Configuration validation
    run_command([
        sys.executable, str(start_script), "digital-twin", "--validate-only"
    ], "Validate Digital Twin configuration")
    
    run_command([
        sys.executable, str(start_script), "agentic-reasoner", "--validate-only"
    ], "Validate Agentic Reasoner configuration")
    
    # Example 3: Dry run mode
    run_command([
        sys.executable, str(start_script), "digital-twin", "--dry-run"
    ], "Digital Twin dry run (initialize but don't start)")
    
    # Example 4: Digital Twin with different world models
    run_command([
        sys.executable, str(start_script), "digital-twin", "--world-model", "bayesian"
    ], "Start Digital Twin with Bayesian world model")
    
    run_command([
        sys.executable, str(start_script), "digital-twin", "--world-model", "gemini", "--health-check"
    ], "Start Digital Twin with Gemini model and health check")
    
    # Example 5: Agentic Reasoner with improved features
    run_command([
        sys.executable, str(start_script), "agentic-reasoner", "--use-bayesian-world-model"
    ], "Start Agentic Reasoner with Bayesian world model")
    
    run_command([
        sys.executable, str(start_script), "agentic-reasoner", 
         "--timeout-config", "robust", "--monitor-performance"
    ], "Start Agentic Reasoner with robust timeouts and monitoring")
    
    # Example 6: Adapter components
    run_command([
        sys.executable, str(start_script), "monitor", 
         "--plugin-dir", "extern", "--log-level", "DEBUG"
    ], "Start Monitor adapter with debug logging")
    
    # Example 7: Complete framework validation
    run_command([
        sys.executable, str(start_script), "all", 
         "--plugin-dir", "extern", "--validate-only"
    ], "Validate complete framework configuration")
    
    # Example 8: Custom component order
    run_command([
        sys.executable, str(start_script), "all", 
         "--plugin-dir", "extern", "--start-order", 
         "digital-twin", "agentic-reasoner", "monitor", "--dry-run"
    ], "Custom startup order (dry run)")
    
    print(f"\n{'='*60}")
    print("📋 Production Deployment Example")
    print('='*60)
    
    production_commands = [
        "# 1. Start infrastructure services",
        f"{sys.executable} {start_script} knowledge-base",
        f"{sys.executable} {start_script} digital-twin --world-model bayesian",
        f"{sys.executable} {start_script} kernel",
        "",
        "# 2. Start system adapters", 
        f"{sys.executable} {start_script} monitor --plugin-dir extern",
        f"{sys.executable} {start_script} execution --plugin-dir extern",
        "",
        "# 3. Start reasoning agents",
        f"{sys.executable} {start_script} agentic-reasoner --use-bayesian-world-model --timeout-config robust",
        f"{sys.executable} {start_script} meta-learner",
    ]
    
    for cmd in production_commands:
        print(cmd)
    
    print(f"\n{'='*60}")
    print("🔍 Monitoring and Health Checks")
    print('='*60)
    
    monitoring_commands = [
        "# Health checks",
        f"{sys.executable} {start_script} digital-twin --health-check",
        f"{sys.executable} {start_script} agentic-reasoner --validate-only",
        "",
        "# Performance monitoring",
        f"{sys.executable} {start_script} agentic-reasoner --monitor-performance",
        "",
        "# Configuration validation",
        f"{sys.executable} {start_script} all --plugin-dir extern --validate-only",
    ]
    
    for cmd in monitoring_commands:
        print(cmd)
    
    print(f"\n{'='*60}")
    print("🎯 Use Case Examples")
    print('='*60)
    
    use_cases = {
        "High Performance (Fast, Deterministic)": [
            f"{sys.executable} {start_script} digital-twin --world-model bayesian",
            f"{sys.executable} {start_script} agentic-reasoner --use-bayesian-world-model --timeout-config fast"
        ],
        "High Reliability (Robust, Fault Tolerant)": [
            f"{sys.executable} {start_script} digital-twin --world-model gemini",
            f"{sys.executable} {start_script} agentic-reasoner --timeout-config robust --monitor-performance"
        ],
        "Development/Testing (Mock Components)": [
            f"{sys.executable} {start_script} digital-twin --world-model mock --dry-run",
            f"{sys.executable} {start_script} agentic-reasoner --validate-only"
        ],
        "Hybrid (LLM + Statistical)": [
            f"{sys.executable} {start_script} digital-twin --world-model bayesian",
            f"{sys.executable} {start_script} reasoner --reasoning-mode hybrid",
            f"{sys.executable} {start_script} agentic-reasoner --use-improved-grpc"
        ]
    }
    
    for use_case, commands in use_cases.items():
        print(f"\n{use_case}:")
        for cmd in commands:
            print(f"  {cmd}")
    
    print(f"\n{'='*60}")
    print("✅ Examples completed!")
    print("📖 See docs/COMPONENT_STARTUP_GUIDE.md for detailed documentation")
    print("🔧 Use 'python start_component.py help' for component information")
    print('='*60)


if __name__ == "__main__":
    main()