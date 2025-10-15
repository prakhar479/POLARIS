#!/usr/bin/env python3
"""
Test script for interactive API key management.

This script demonstrates the interactive API key functionality
for Gemini-based POLARIS components.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_api_key_manager():
    """Test the API key manager functionality."""
    print("🔑 Testing Interactive API Key Manager")
    print("=" * 50)
    
    try:
        from polaris.common.api_key_manager import (
            APIKeyManager, 
            get_gemini_api_key_interactive,
            validate_gemini_environment
        )
        
        print("✅ API Key Manager imported successfully")
        
        # Test environment validation
        print("\n📋 Testing environment validation...")
        validation = validate_gemini_environment("Test Component")
        
        print(f"Valid: {validation['valid']}")
        print(f"API Key Found: {validation['api_key_found']}")
        print(f"Source: {validation['source']}")
        
        if validation['issues']:
            print("Issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        # Test interactive API key retrieval (only if no key found)
        if not validation['api_key_found']:
            print("\n🔐 Testing interactive API key retrieval...")
            print("Note: This will prompt for API key input")
            
            # Uncomment the next line to test interactive prompt
            # api_key = get_gemini_api_key_interactive("Test Component")
            # print(f"API Key obtained: {'Yes' if api_key else 'No'}")
            
            print("Interactive test skipped (uncomment to test)")
        else:
            print("\n✅ API key already available, skipping interactive test")
        
        print("\n🧪 Testing API key manager methods...")
        manager = APIKeyManager()
        
        # Test validation method
        result = manager.validate_environment("Test Component")
        print(f"Manager validation result: {result['valid']}")
        
        print("\n✅ All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install required packages: pip install keyring cryptography")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True


def test_component_startup_with_api_key():
    """Test component startup with API key handling."""
    print("\n🚀 Testing Component Startup with API Key")
    print("=" * 50)
    
    try:
        # Test the start_component.py helper functions
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))
        
        from start_component import (
            get_api_key_for_component,
            validate_api_key_environment
        )
        
        print("✅ Start component helpers imported successfully")
        
        # Test API key retrieval (non-interactive)
        print("\n📋 Testing non-interactive API key retrieval...")
        api_key = get_api_key_for_component("Test Component", interactive=False)
        print(f"API Key found: {'Yes' if api_key else 'No'}")
        
        # Test environment validation
        print("\n📋 Testing environment validation...")
        validation = validate_api_key_environment("Test Component", interactive=False)
        print(f"Environment valid: {validation['valid']}")
        
        if validation['issues']:
            print("Issues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        print("\n✅ Component startup tests completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True


def show_usage_examples():
    """Show usage examples for the API key manager."""
    print("\n📖 Usage Examples")
    print("=" * 50)
    
    examples = [
        "# Set API key in environment (recommended)",
        "export GEMINI_API_KEY='your-api-key-here'",
        "",
        "# Start components (will prompt for API key if not found)",
        "python start_component.py digital-twin --world-model gemini",
        "python start_component.py agentic-reasoner",
        "python start_component.py reasoner",
        "",
        "# Validate environment without starting",
        "python start_component.py digital-twin --validate-only",
        "python start_component.py agentic-reasoner --validate-only",
        "",
        "# Use in Python code",
        "from polaris.common.api_key_manager import get_gemini_api_key_interactive",
        "api_key = get_gemini_api_key_interactive('My Component')",
        "",
        "# Environment validation",
        "from polaris.common.api_key_manager import validate_gemini_environment",
        "result = validate_gemini_environment('My Component')",
        "if result['valid']:",
        "    print('API key is available')",
    ]
    
    for example in examples:
        print(example)


def main():
    """Run all tests."""
    print("🧪 POLARIS Interactive API Key Manager Tests")
    print("=" * 60)
    
    # Test 1: API Key Manager
    success1 = test_api_key_manager()
    
    # Test 2: Component Startup Integration
    success2 = test_component_startup_with_api_key()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed!")
        print("\n💡 To test interactive API key prompt:")
        print("1. Remove GEMINI_API_KEY from environment")
        print("2. Run: python start_component.py digital-twin --world-model gemini")
        print("3. Follow the interactive prompts")
    else:
        print("❌ Some tests failed")
        print("\n🔧 To fix issues:")
        print("1. Install dependencies: pip install keyring cryptography")
        print("2. Set GEMINI_API_KEY environment variable")
    
    print("\n📚 Documentation:")
    print("- See docs/COMPONENT_STARTUP_GUIDE.md for detailed usage")
    print("- API key manager source: src/polaris/common/api_key_manager.py")


if __name__ == "__main__":
    main()