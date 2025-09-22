#!/usr/bin/env python3
"""
Environment Setup Script for SWIM POLARIS Adaptation System

Provides automated environment setup, dependency installation,
configuration generation, and system validation.
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_templates import ConfigurationTemplateGenerator, Environment


class EnvironmentSetup:
    """Handles environment setup and configuration."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.config_dir = self.base_dir / "config"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        
        print(f"Setting up environment in: {self.base_dir}")
    
    def create_directory_structure(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.config_dir,
            self.config_dir / "ablation_configs",
            self.logs_dir,
            self.results_dir,
            self.base_dir / "scripts",
            self.base_dir / "src"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def check_python_dependencies(self) -> List[str]:
        """Check for required Python packages."""
        required_packages = [
            "asyncio",
            "yaml", 
            "pydantic",
            "nats-py",
            "watchdog",
            "jsonschema",
            "psutil"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "yaml":
                    import yaml
                elif package == "nats-py":
                    import nats
                elif package == "asyncio":
                    import asyncio
                elif package == "pydantic":
                    import pydantic
                elif package == "watchdog":
                    import watchdog
                elif package == "jsonschema":
                    import jsonschema
                elif package == "psutil":
                    import psutil
                
                print(f"✓ {package} is available")
            
            except ImportError:
                print(f"✗ {package} is missing")
                missing_packages.append(package)
        
        return missing_packages
    
    def install_python_dependencies(self, missing_packages: List[str]) -> bool:
        """Install missing Python packages."""
        if not missing_packages:
            return True
        
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        
        # Map package names to pip names
        pip_names = {
            "yaml": "PyYAML",
            "nats-py": "nats-py",
            "pydantic": "pydantic",
            "watchdog": "watchdog",
            "jsonschema": "jsonschema",
            "psutil": "psutil"
        }
        
        pip_packages = [pip_names.get(pkg, pkg) for pkg in missing_packages]
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + pip_packages)
            
            print("✓ All packages installed successfully")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install packages: {e}")
            return False
    
    def check_external_dependencies(self) -> Dict[str, bool]:
        """Check for external system dependencies."""
        dependencies = {}
        
        # Check for NATS server
        nats_available = shutil.which("nats-server") is not None
        dependencies["nats-server"] = nats_available
        print(f"{'✓' if nats_available else '✗'} NATS server: {'available' if nats_available else 'not found'}")
        
        # Check for Docker (optional)
        docker_available = shutil.which("docker") is not None
        dependencies["docker"] = docker_available
        print(f"{'✓' if docker_available else '○'} Docker: {'available' if docker_available else 'not found (optional)'}")
        
        # Check for Git
        git_available = shutil.which("git") is not None
        dependencies["git"] = git_available
        print(f"{'✓' if git_available else '○'} Git: {'available' if git_available else 'not found (optional)'}")
        
        return dependencies
    
    def generate_configurations(self, environments: List[str]) -> None:
        """Generate configuration files for specified environments."""
        generator = ConfigurationTemplateGenerator()
        
        for env_name in environments:
            try:
                env = Environment(env_name)
                config = generator.generate_base_config(env)
                
                # Save configuration
                config_file = self.config_dir / f"{env_name}_config.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                print(f"✓ Generated {env_name} configuration: {config_file}")
                
            except ValueError:
                print(f"✗ Unknown environment: {env_name}")
    
    def generate_ablation_configs(self) -> None:
        """Generate ablation study configurations."""
        ablation_configs = {
            "full_system": {
                "description": "Full system with all components enabled",
                "components": {
                    "world_model": True,
                    "knowledge_base": True,
                    "learning_engine": True,
                    "statistical_reasoning": True,
                    "causal_reasoning": True,
                    "experience_based_reasoning": True,
                    "reactive_strategy": True,
                    "predictive_strategy": True
                },
                "study_parameters": {
                    "duration": 3600,
                    "warmup_period": 300,
                    "cooldown_period": 300,
                    "metrics_export_interval": 60
                }
            },
            
            "no_learning": {
                "description": "System without learning engine",
                "components": {
                    "world_model": True,
                    "knowledge_base": True,
                    "learning_engine": False,
                    "statistical_reasoning": True,
                    "causal_reasoning": True,
                    "experience_based_reasoning": False,
                    "reactive_strategy": True,
                    "predictive_strategy": True
                },
                "study_parameters": {
                    "duration": 3600,
                    "warmup_period": 300,
                    "cooldown_period": 300,
                    "metrics_export_interval": 60
                }
            },
            
            "no_world_model": {
                "description": "System without world model",
                "components": {
                    "world_model": False,
                    "knowledge_base": True,
                    "learning_engine": True,
                    "statistical_reasoning": True,
                    "causal_reasoning": False,
                    "experience_based_reasoning": True,
                    "reactive_strategy": True,
                    "predictive_strategy": False
                },
                "study_parameters": {
                    "duration": 3600,
                    "warmup_period": 300,
                    "cooldown_period": 300,
                    "metrics_export_interval": 60
                }
            },
            
            "reactive_only": {
                "description": "Only reactive adaptation strategy",
                "components": {
                    "world_model": True,
                    "knowledge_base": True,
                    "learning_engine": False,
                    "statistical_reasoning": True,
                    "causal_reasoning": False,
                    "experience_based_reasoning": False,
                    "reactive_strategy": True,
                    "predictive_strategy": False
                },
                "study_parameters": {
                    "duration": 3600,
                    "warmup_period": 300,
                    "cooldown_period": 300,
                    "metrics_export_interval": 60
                }
            },
            
            "no_reasoning": {
                "description": "System without advanced reasoning",
                "components": {
                    "world_model": True,
                    "knowledge_base": True,
                    "learning_engine": True,
                    "statistical_reasoning": True,
                    "causal_reasoning": False,
                    "experience_based_reasoning": False,
                    "reactive_strategy": True,
                    "predictive_strategy": True
                },
                "study_parameters": {
                    "duration": 3600,
                    "warmup_period": 300,
                    "cooldown_period": 300,
                    "metrics_export_interval": 60
                }
            }
        }
        
        ablation_dir = self.config_dir / "ablation_configs"
        
        for config_name, config_data in ablation_configs.items():
            # Create full configuration by merging with base
            full_config = {
                "ablation": config_data,
                # Add any base configuration overrides here
                "logging": {
                    "level": "INFO",
                    "handlers": [
                        {
                            "type": "file",
                            "path": f"logs/{config_name}.log",
                            "level": "DEBUG"
                        }
                    ]
                }
            }
            
            config_file = ablation_dir / f"{config_name}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False, indent=2)
            
            print(f"✓ Generated ablation config: {config_file}")
    
    def create_startup_scripts(self) -> None:
        """Create convenience startup scripts."""
        scripts_dir = self.base_dir / "scripts"
        
        # Create shell script for Unix systems
        if os.name != 'nt':
            startup_script = scripts_dir / "start.sh"
            with open(startup_script, 'w') as f:
                f.write(f"""#!/bin/bash
# SWIM POLARIS Adaptation System Startup Script

cd "{self.base_dir}"

echo "Starting SWIM POLARIS Adaptation System..."

# Check if NATS server is running
if ! pgrep -x "nats-server" > /dev/null; then
    echo "Starting NATS server..."
    nats-server &
    sleep 2
fi

# Start the system
python3 scripts/start_system.py "$@"
""")
            startup_script.chmod(0o755)
            print(f"✓ Created startup script: {startup_script}")
        
        # Create batch script for Windows
        if os.name == 'nt':
            startup_script = scripts_dir / "start.bat"
            with open(startup_script, 'w') as f:
                f.write(f"""@echo off
REM SWIM POLARIS Adaptation System Startup Script

cd /d "{self.base_dir}"

echo Starting SWIM POLARIS Adaptation System...

REM Check if NATS server is running
tasklist /FI "IMAGENAME eq nats-server.exe" 2>NUL | find /I /N "nats-server.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting NATS server...
    start /B nats-server
    timeout /t 2 /nobreak >NUL
)

REM Start the system
python scripts\\start_system.py %*
""")
            print(f"✓ Created startup script: {startup_script}")
    
    def create_environment_file(self) -> None:
        """Create environment configuration file."""
        env_file = self.base_dir / ".env"
        
        env_content = f"""# SWIM POLARIS Adaptation System Environment Configuration

# System Configuration
SWIM_POLARIS_CONFIG_DIR={self.config_dir}
SWIM_POLARIS_LOGS_DIR={self.logs_dir}
SWIM_POLARIS_RESULTS_DIR={self.results_dir}

# SWIM Connection
SWIM_HOST=localhost
SWIM_PORT=4242

# NATS Configuration
NATS_SERVERS=nats://localhost:4222

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Development Settings
ENVIRONMENT=development
DEBUG=false

# Optional: Database URLs (for production)
# SWIM_POLARIS_DB_URL=postgresql://user:pass@localhost/swim_polaris
# SWIM_POLARIS_METRICS_DB_URL=postgresql://user:pass@localhost/swim_polaris_metrics
# SWIM_POLARIS_REDIS_URL=redis://localhost:6379

# Optional: Observability
# JAEGER_ENDPOINT=http://localhost:14268/api/traces
# PROMETHEUS_ENDPOINT=http://localhost:9090
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✓ Created environment file: {env_file}")
    
    def validate_setup(self) -> bool:
        """Validate the setup."""
        print("\nValidating setup...")
        
        # Check directory structure
        required_dirs = [self.config_dir, self.logs_dir, self.results_dir]
        for directory in required_dirs:
            if directory.exists():
                print(f"✓ Directory exists: {directory}")
            else:
                print(f"✗ Directory missing: {directory}")
                return False
        
        # Check configuration files
        base_config = self.config_dir / "base_config.yaml"
        if base_config.exists():
            print(f"✓ Base configuration exists: {base_config}")
        else:
            print(f"✗ Base configuration missing: {base_config}")
            return False
        
        # Check ablation configs
        ablation_dir = self.config_dir / "ablation_configs"
        if ablation_dir.exists() and list(ablation_dir.glob("*.yaml")):
            print(f"✓ Ablation configurations exist: {len(list(ablation_dir.glob('*.yaml')))} files")
        else:
            print(f"✗ Ablation configurations missing")
            return False
        
        print("✓ Setup validation passed")
        return True
    
    def print_next_steps(self) -> None:
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("SETUP COMPLETE - NEXT STEPS")
        print("="*60)
        
        print("\n1. Start NATS server (if not already running):")
        print("   nats-server")
        
        print("\n2. Start SWIM system (if using external SWIM):")
        print("   # Follow SWIM documentation to start the system")
        
        print("\n3. Start POLARIS system:")
        if os.name != 'nt':
            print(f"   cd {self.base_dir}")
            print("   ./scripts/start.sh")
        else:
            print(f"   cd {self.base_dir}")
            print("   scripts\\start.bat")
        
        print("\n4. Or use Python directly:")
        print("   python scripts/start_system.py --config config/base_config.yaml")
        
        print("\n5. Run ablation studies:")
        print("   python scripts/run_ablation_study.py --list")
        print("   python scripts/run_ablation_study.py --study full_system")
        
        print("\n6. Monitor system status:")
        print("   python scripts/system_status.py --continuous")
        
        print("\nConfiguration files are in:", self.config_dir)
        print("Logs will be written to:", self.logs_dir)
        print("Results will be saved to:", self.results_dir)
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup SWIM POLARIS Environment")
    parser.add_argument("--base-dir", "-d",
                       help="Base directory for setup")
    parser.add_argument("--environments", "-e",
                       nargs="+",
                       default=["development", "testing", "production"],
                       help="Environments to generate configs for")
    parser.add_argument("--skip-deps",
                       action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--skip-configs",
                       action="store_true",
                       help="Skip configuration generation")
    parser.add_argument("--validate-only",
                       action="store_true",
                       help="Only validate existing setup")
    
    args = parser.parse_args()
    
    # Create setup manager
    setup = EnvironmentSetup(args.base_dir)
    
    if args.validate_only:
        success = setup.validate_setup()
        sys.exit(0 if success else 1)
    
    try:
        print("Setting up SWIM POLARIS Adaptation System environment...")
        
        # Create directory structure
        setup.create_directory_structure()
        
        # Check and install dependencies
        if not args.skip_deps:
            print("\nChecking Python dependencies...")
            missing_packages = setup.check_python_dependencies()
            
            if missing_packages:
                if not setup.install_python_dependencies(missing_packages):
                    print("Failed to install dependencies")
                    sys.exit(1)
            
            print("\nChecking external dependencies...")
            setup.check_external_dependencies()
        
        # Generate configurations
        if not args.skip_configs:
            print("\nGenerating configurations...")
            setup.generate_configurations(args.environments)
            setup.generate_ablation_configs()
        
        # Create additional files
        print("\nCreating additional files...")
        setup.create_startup_scripts()
        setup.create_environment_file()
        
        # Validate setup
        if setup.validate_setup():
            setup.print_next_steps()
        else:
            print("Setup validation failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()