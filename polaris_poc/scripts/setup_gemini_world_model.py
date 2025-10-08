#!/usr/bin/env python3
"""
Setup script for Gemini World Model

This script helps users set up the Gemini World Model with proper
configuration and API key management.
"""

import os
import sys
from pathlib import Path
import yaml


def check_api_key():
    """Check if Gemini API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ GEMINI_API_KEY is configured (length: {len(api_key)})")
        return True
    else:
        print("❌ GEMINI_API_KEY environment variable not found")
        print("\nTo get a Gemini API key:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as an environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("4. Or add it to your .env file")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
        return True
    except ImportError:
        print("❌ google-generativeai package not found")
        print("Install it with: pip install google-generativeai")
        return False


def update_configuration():
    """Update the main configuration to use Gemini World Model."""
    config_path = Path("src/config/polaris_config.yaml")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update world model configuration
        if 'digital_twin' in config and 'world_model' in config['digital_twin']:
            current_impl = config['digital_twin']['world_model'].get('implementation', 'mock')
            
            if current_impl != 'gemini':
                config['digital_twin']['world_model']['implementation'] = 'gemini'
                config['digital_twin']['world_model']['config_path'] = 'gemini_world_model.yaml'
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                print(f"✅ Updated configuration to use Gemini World Model (was: {current_impl})")
            else:
                print("✅ Configuration already uses Gemini World Model")
        else:
            print("❌ Configuration structure not as expected")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False


def verify_gemini_config():
    """Verify the Gemini World Model configuration."""
    config_path = Path("src/config/gemini_world_model.yaml")
    
    if not config_path.exists():
        print(f"❌ Gemini configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        if config.get('implementation') != 'gemini':
            print("❌ Gemini configuration has wrong implementation type")
            return False
        
        gemini_config = config.get('config', {})
        if not gemini_config.get('api_key_env'):
            print("❌ Gemini configuration missing api_key_env")
            return False
        
        print("✅ Gemini World Model configuration is valid")
        print(f"   Model: {gemini_config.get('model', 'default')}")
        print(f"   Temperature: {gemini_config.get('temperature', 'default')}")
        print(f"   Max tokens: {gemini_config.get('max_tokens', 'default')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to verify Gemini configuration: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Gemini World Model Setup")
    print("=" * 40)
    print("This script helps you set up the Gemini World Model for POLARIS.")
    print()
    
    all_checks_passed = True
    
    # Check 1: Dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        all_checks_passed = False
    print()
    
    # Check 2: API Key
    print("2. Checking API key...")
    if not check_api_key():
        all_checks_passed = False
    print()
    
    # Check 3: Configuration
    print("3. Checking configuration...")
    if not verify_gemini_config():
        all_checks_passed = False
    print()
    
    # Check 4: Update main config
    print("4. Updating main configuration...")
    if not update_configuration():
        all_checks_passed = False
    print()
    
    # Summary
    if all_checks_passed:
        print("🎉 Setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Test the implementation:")
        print("   python tests/test_gemini_world_model.py")
        print()
        print("2. Run the meta-learning demo:")
        print("   python examples/gemini_meta_learning_demo.py")
        print()
        print("3. Start the Digital Twin with Gemini:")
        print("   python src/scripts/start_component.py digital-twin")
        print()
        print("4. Start the Agentic Reasoner:")
        print("   python src/scripts/start_component.py agentic-reasoner")
        
    else:
        print("💥 Setup incomplete. Please fix the issues above.")
        print()
        print("Common solutions:")
        print("- Install dependencies: pip install google-generativeai")
        print("- Set API key: export GEMINI_API_KEY='your-key'")
        print("- Check configuration files exist and are valid")
    
    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)