"""
Simple API Key Manager (fallback without keyring dependencies).

This module provides basic API key management without requiring
keyring or cryptography dependencies.
"""

import os
import getpass
import logging
from typing import Optional, Dict, Any


def get_gemini_api_key_simple(component_name: str = "POLARIS Component") -> Optional[str]:
    """
    Simple API key prompt without keyring dependencies.
    
    Args:
        component_name: Name of the component requesting the key
        
    Returns:
        API key string or None if not available
    """
    # First, try environment variables
    for env_var in ["GEMINI_API_KEY", "API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"]:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    
    # Interactive prompt
    print(f"\n🔑 {component_name} requires a Gemini API key")
    print("=" * 60)
    print("The Gemini API key was not found in your environment variables.")
    print("You can obtain a free API key from: https://makersuite.google.com/app/apikey")
    print()
    print("Environment variables checked:")
    print("  - GEMINI_API_KEY (recommended)")
    print("  - API_KEY")
    print("  - GOOGLE_API_KEY") 
    print("  - GENAI_API_KEY")
    print()
    
    # Ask if user wants to continue
    while True:
        choice = input("Would you like to enter your API key now? (y/n/help): ").lower().strip()
        
        if choice in ['n', 'no']:
            print("❌ Cannot proceed without API key. Exiting...")
            return None
        elif choice in ['help', 'h']:
            _print_api_key_help()
            continue
        elif choice in ['y', 'yes', '']:
            break
        else:
            print("Please enter 'y' for yes, 'n' for no, or 'help' for more information.")
    
    # Get API key securely
    while True:
        try:
            print("\n🔐 Please enter your Gemini API key:")
            print("(Input will be hidden for security)")
            api_key = getpass.getpass("API Key: ").strip()
            
            if not api_key:
                print("❌ Empty API key entered. Please try again.")
                continue
            
            # Basic validation
            if not _validate_gemini_api_key_format(api_key):
                print("⚠️  API key format looks unusual. Gemini API keys typically start with 'AIza'")
                confirm = input("Continue anyway? (y/n): ").lower().strip()
                if confirm not in ['y', 'yes']:
                    continue
            
            # Set for current session
            os.environ["GEMINI_API_KEY"] = api_key
            print("✅ API key set for current session")
            print("💡 To make it permanent, add this to your shell profile:")
            print(f"   export GEMINI_API_KEY='{api_key}'")
            
            return api_key
                    
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def _validate_gemini_api_key_format(api_key: str) -> bool:
    """Basic validation of Gemini API key format."""
    return (
        len(api_key) >= 30 and 
        len(api_key) <= 50 and
        api_key.startswith("AIza")
    )


def _print_api_key_help():
    """Print detailed help about getting a Gemini API key."""
    print("\n" + "=" * 60)
    print("🔑 How to Get a Gemini API Key")
    print("=" * 60)
    print()
    print("1. Visit: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key (starts with 'AIza...')")
    print()
    print("💡 Setting up the API key:")
    print()
    print("Option 1 - Environment Variable (Recommended):")
    print("  Linux/Mac: export GEMINI_API_KEY='your-api-key-here'")
    print("  Windows:   set GEMINI_API_KEY=your-api-key-here")
    print()
    print("Option 2 - Add to your shell profile:")
    print("  echo 'export GEMINI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc")
    print("  source ~/.bashrc")
    print()
    print("Option 3 - Use .env file in project root:")
    print("  GEMINI_API_KEY=your-api-key-here")
    print()
    print("📋 Free Tier Limits:")
    print("  - 60 requests per minute")
    print("  - 1,500 requests per day")
    print("  - Sufficient for development and testing")
    print()
    print("=" * 60)


def validate_gemini_environment_simple(component_name: str) -> Dict[str, Any]:
    """Simple validation without keyring dependencies."""
    result = {
        "valid": False,
        "api_key_found": False,
        "source": None,
        "issues": [],
        "recommendations": []
    }
    
    # Check for API key
    for env_var in ["GEMINI_API_KEY", "API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"]:
        api_key = os.getenv(env_var)
        if api_key:
            result["api_key_found"] = True
            result["valid"] = True
            result["source"] = f"{env_var} environment variable"
            break
    
    if not result["api_key_found"]:
        result["issues"].append("Gemini API key not found")
        result["recommendations"].append("Set GEMINI_API_KEY environment variable or run with interactive mode")
    
    return result