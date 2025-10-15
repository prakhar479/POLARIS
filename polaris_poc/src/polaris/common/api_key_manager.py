"""
API Key Management Utilities for POLARIS Components.

This module provides interactive API key management for components that require
external API access, with secure input handling and optional persistence.
"""

import os
import sys
import getpass
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import keyring
from cryptography.fernet import Fernet
import base64


class APIKeyManager:
    """Manages API keys for POLARIS components with interactive prompts and secure storage."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_dir = Path.home() / ".polaris"
        self.config_file = self.config_dir / "api_keys.json"
        
    def get_gemini_api_key(self, 
                          interactive: bool = True, 
                          save_option: bool = True,
                          component_name: str = "POLARIS") -> Optional[str]:
        """
        Get Gemini API key with interactive prompt if not found in environment.
        
        Args:
            interactive: Whether to prompt user interactively
            save_option: Whether to offer to save the key
            component_name: Name of the component requesting the key
            
        Returns:
            API key string or None if not available
        """
        # First, try environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.logger.debug("Found GEMINI_API_KEY in environment")
            return api_key
        
        # Try alternative environment variable names
        alt_names = ["API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"]
        for alt_name in alt_names:
            api_key = os.getenv(alt_name)
            if api_key:
                self.logger.info(f"Found API key in environment variable: {alt_name}")
                return api_key
        
        # Try to load from secure storage
        api_key = self._load_from_secure_storage("gemini_api_key")
        if api_key:
            self.logger.info("Loaded Gemini API key from secure storage")
            return api_key
        
        # If not interactive, return None
        if not interactive:
            self.logger.warning("Gemini API key not found and interactive mode disabled")
            return None
        
        # Interactive prompt
        return self._interactive_gemini_key_prompt(save_option, component_name)
    
    def _interactive_gemini_key_prompt(self, save_option: bool, component_name: str) -> Optional[str]:
        """Interactive prompt for Gemini API key."""
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
                self._print_api_key_help()
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
                if not self._validate_gemini_api_key_format(api_key):
                    print("⚠️  API key format looks unusual. Gemini API keys typically start with 'AIza'")
                    confirm = input("Continue anyway? (y/n): ").lower().strip()
                    if confirm not in ['y', 'yes']:
                        continue
                
                # Test the API key
                print("🔍 Testing API key...")
                if self._test_gemini_api_key(api_key):
                    print("✅ API key is valid!")
                    break
                else:
                    print("❌ API key test failed. Please check your key and try again.")
                    retry = input("Try again? (y/n): ").lower().strip()
                    if retry not in ['y', 'yes']:
                        return None
                    
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
        
        # Offer to save the key
        if save_option:
            self._offer_to_save_key(api_key)
        
        return api_key
    
    def _validate_gemini_api_key_format(self, api_key: str) -> bool:
        """Basic validation of Gemini API key format."""
        # Gemini API keys typically start with "AIza" and are about 39 characters
        return (
            len(api_key) >= 30 and 
            len(api_key) <= 50 and
            api_key.startswith("AIza")
        )
    
    def _test_gemini_api_key(self, api_key: str) -> bool:
        """Test if the Gemini API key works."""
        try:
            # Set the API key temporarily
            original_key = os.environ.get("GEMINI_API_KEY")
            os.environ["GEMINI_API_KEY"] = api_key
            
            # Try to import and test
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Create a simple model instance and test
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Hello")
            
            # Restore original environment
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key
            else:
                os.environ.pop("GEMINI_API_KEY", None)
            
            return response and response.text
            
        except Exception as e:
            self.logger.debug(f"API key test failed: {e}")
            # Restore original environment
            if 'original_key' in locals():
                if original_key:
                    os.environ["GEMINI_API_KEY"] = original_key
                else:
                    os.environ.pop("GEMINI_API_KEY", None)
            return False
    
    def _offer_to_save_key(self, api_key: str):
        """Offer to save the API key for future use."""
        print("\n💾 Save API Key Options:")
        print("1. Save to secure system keyring (recommended)")
        print("2. Save to environment variable for this session only")
        print("3. Don't save (you'll need to enter it again next time)")
        
        while True:
            choice = input("Choose option (1/2/3): ").strip()
            
            if choice == "1":
                if self._save_to_secure_storage("gemini_api_key", api_key):
                    print("✅ API key saved to secure keyring")
                else:
                    print("❌ Failed to save to keyring, setting for session only")
                    os.environ["GEMINI_API_KEY"] = api_key
                break
            elif choice == "2":
                os.environ["GEMINI_API_KEY"] = api_key
                print("✅ API key set for this session")
                print("💡 To make it permanent, add this to your shell profile:")
                print(f"   export GEMINI_API_KEY='{api_key}'")
                break
            elif choice == "3":
                os.environ["GEMINI_API_KEY"] = api_key  # Set for current session anyway
                print("✅ API key set for current session only")
                break
            else:
                print("Please enter 1, 2, or 3")
    
    def _save_to_secure_storage(self, key_name: str, api_key: str) -> bool:
        """Save API key to secure storage."""
        try:
            keyring.set_password("polaris", key_name, api_key)
            return True
        except Exception as e:
            self.logger.debug(f"Failed to save to keyring: {e}")
            return False
    
    def _load_from_secure_storage(self, key_name: str) -> Optional[str]:
        """Load API key from secure storage."""
        try:
            return keyring.get_password("polaris", key_name)
        except Exception as e:
            self.logger.debug(f"Failed to load from keyring: {e}")
            return None
    
    def _print_api_key_help(self):
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
        print("🔒 Security Notes:")
        print("  - Never commit API keys to version control")
        print("  - Use environment variables or secure key management")
        print("  - The API key will be stored securely if you choose to save it")
        print()
        print("📋 Free Tier Limits:")
        print("  - 60 requests per minute")
        print("  - 1,500 requests per day")
        print("  - Sufficient for development and testing")
        print()
        print("=" * 60)
    
    def clear_stored_keys(self):
        """Clear all stored API keys."""
        try:
            keyring.delete_password("polaris", "gemini_api_key")
            print("✅ Cleared stored Gemini API key")
        except Exception:
            pass
    
    def validate_environment(self, component_name: str) -> Dict[str, Any]:
        """Validate API key environment for a component."""
        result = {
            "valid": False,
            "api_key_found": False,
            "source": None,
            "issues": [],
            "recommendations": []
        }
        
        # Check for API key
        api_key = self.get_gemini_api_key(interactive=False)
        if api_key:
            result["api_key_found"] = True
            result["valid"] = True
            
            # Determine source
            if os.getenv("GEMINI_API_KEY"):
                result["source"] = "GEMINI_API_KEY environment variable"
            elif self._load_from_secure_storage("gemini_api_key"):
                result["source"] = "secure keyring storage"
            else:
                result["source"] = "alternative environment variable"
        else:
            result["issues"].append("Gemini API key not found")
            result["recommendations"].append("Set GEMINI_API_KEY environment variable or run with interactive mode")
        
        return result


# Global instance for easy access
api_key_manager = APIKeyManager()


def get_gemini_api_key_interactive(component_name: str = "POLARIS Component") -> Optional[str]:
    """
    Convenience function to get Gemini API key with interactive prompt.
    
    Args:
        component_name: Name of the component requesting the key
        
    Returns:
        API key string or None if not available
    """
    return api_key_manager.get_gemini_api_key(
        interactive=True,
        save_option=True,
        component_name=component_name
    )


def validate_gemini_environment(component_name: str) -> Dict[str, Any]:
    """
    Validate Gemini API environment for a component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Validation result dictionary
    """
    return api_key_manager.validate_environment(component_name)