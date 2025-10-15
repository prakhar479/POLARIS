# Interactive API Key Management Guide

## 🔑 Overview

POLARIS now includes an interactive API key management system that automatically prompts users for their Gemini API key when it's not found in the environment. This eliminates the need to manually set environment variables and provides a user-friendly experience for getting started with Gemini-based components.

## ✨ Features

### **Automatic Detection**
- Checks multiple environment variables (`GEMINI_API_KEY`, `API_KEY`, `GOOGLE_API_KEY`, `GENAI_API_KEY`)
- Loads from secure system keyring if previously saved
- Falls back to interactive prompt when needed

### **Interactive Prompts**
- User-friendly prompts with clear instructions
- Secure password input (hidden from terminal)
- API key validation with live testing
- Help system with detailed setup instructions

### **Secure Storage Options**
- Save to system keyring (recommended)
- Set for current session only
- Guidance for permanent environment variable setup

### **Smart Integration**
- Seamlessly integrated with all Gemini-based components
- Works with Digital Twin (Gemini world model)
- Works with Agentic Reasoner
- Works with basic Reasoner agent
- Works with Meta-Learner agent

## 🚀 Quick Start

### **Automatic Setup**
Simply start any Gemini-based component without setting up the API key first:

```bash
# Digital Twin with Gemini world model
python start_component.py digital-twin --world-model gemini

# Agentic Reasoner
python start_component.py agentic-reasoner

# Basic Reasoner
python start_component.py reasoner
```

If no API key is found, you'll see an interactive prompt like this:

```
🔑 Digital Twin (Gemini World Model) requires a Gemini API key
============================================================
The Gemini API key was not found in your environment variables.
You can obtain a free API key from: https://makersuite.google.com/app/apikey

Environment variables checked:
  - GEMINI_API_KEY (recommended)
  - API_KEY
  - GOOGLE_API_KEY
  - GENAI_API_KEY

Would you like to enter your API key now? (y/n/help): y

🔐 Please enter your Gemini API key:
(Input will be hidden for security)
API Key: [hidden input]

🔍 Testing API key...
✅ API key is valid!

💾 Save API Key Options:
1. Save to secure system keyring (recommended)
2. Save to environment variable for this session only
3. Don't save (you'll need to enter it again next time)
Choose option (1/2/3): 1

✅ API key saved to secure keyring
```

## 📋 Getting Your API Key

### **Step 1: Visit Google AI Studio**
1. Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key (starts with 'AIza...')

### **Step 2: Use the Key**
You have several options:

#### **Option A: Let POLARIS prompt you (Recommended)**
Just start any component - POLARIS will ask for the key when needed.

#### **Option B: Set environment variable**
```bash
# Linux/Mac
export GEMINI_API_KEY='your-api-key-here'

# Windows
set GEMINI_API_KEY=your-api-key-here
```

#### **Option C: Add to shell profile**
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### **Option D: Use .env file**
Create a `.env` file in your project root:
```
GEMINI_API_KEY=your-api-key-here
```

## 🔧 Component Integration

### **Digital Twin with Gemini World Model**
```bash
# Will prompt for API key if not found
python start_component.py digital-twin --world-model gemini

# Validate environment (shows API key status)
python start_component.py digital-twin --world-model gemini --validate-only

# Health check (tests API connectivity)
python start_component.py digital-twin --world-model gemini --health-check
```

### **Agentic Reasoner**
```bash
# Standard startup with interactive API key
python start_component.py agentic-reasoner

# With Bayesian world model (still needs API key for reasoning)
python start_component.py agentic-reasoner --use-bayesian-world-model

# Validation mode
python start_component.py agentic-reasoner --validate-only
```

### **Basic Reasoner Agent**
```bash
# Interactive API key prompt
python start_component.py reasoner

# With custom reasoning mode
python start_component.py reasoner --reasoning-mode hybrid
```

### **Meta-Learner Agent**
```bash
# Interactive API key prompt
python start_component.py meta-learner
```

## 🔒 Security Features

### **Secure Input**
- API key input is hidden from terminal using `getpass`
- No API key echoing in logs or console output
- Secure memory handling during input

### **Validation**
- Format validation (checks for typical Gemini API key patterns)
- Live API testing to verify the key works
- Clear error messages for invalid keys

### **Storage Options**
1. **System Keyring (Recommended)**: Uses OS-level secure storage
2. **Session Only**: Temporary environment variable
3. **No Storage**: Manual entry each time

### **Environment Variable Priority**
1. `GEMINI_API_KEY` (primary)
2. `API_KEY` (fallback)
3. `GOOGLE_API_KEY` (alternative)
4. `GENAI_API_KEY` (alternative)
5. Secure keyring storage
6. Interactive prompt

## 🛠️ Advanced Usage

### **Programmatic Access**
```python
from polaris.common.api_key_manager import get_gemini_api_key_interactive

# Get API key with interactive prompt
api_key = get_gemini_api_key_interactive("My Component")

# Non-interactive (returns None if not found)
from polaris.common.api_key_manager import APIKeyManager
manager = APIKeyManager()
api_key = manager.get_gemini_api_key(interactive=False)
```

### **Environment Validation**
```python
from polaris.common.api_key_manager import validate_gemini_environment

result = validate_gemini_environment("My Component")
print(f"Valid: {result['valid']}")
print(f"Source: {result['source']}")
print(f"Issues: {result['issues']}")
```

### **Custom Component Integration**
```python
from polaris.common.api_key_manager import APIKeyManager

class MyGeminiComponent:
    def __init__(self):
        self.api_manager = APIKeyManager()
        
    def initialize(self):
        # Get API key with interactive prompt
        api_key = self.api_manager.get_gemini_api_key(
            interactive=True,
            save_option=True,
            component_name="My Custom Component"
        )
        
        if not api_key:
            raise Exception("API key required")
        
        # Use the API key...
```

## 🔍 Troubleshooting

### **Common Issues**

#### **"API Key Manager not available"**
```bash
# Install required dependencies
pip install keyring cryptography
```

#### **"API key test failed"**
- Check your internet connection
- Verify the API key is correct
- Ensure you have API quota remaining
- Try generating a new API key

#### **"Failed to save to keyring"**
- Keyring storage may not be available on your system
- Choose option 2 (session only) instead
- Set environment variable manually

#### **"Permission denied" errors**
- Run with appropriate permissions
- Check if keyring service is running
- Try alternative storage options

### **Validation Commands**
```bash
# Check API key status
python start_component.py digital-twin --world-model gemini --validate-only

# Test API connectivity
python start_component.py digital-twin --world-model gemini --health-check

# Debug mode
python start_component.py digital-twin --world-model gemini --log-level DEBUG
```

### **Manual Cleanup**
```python
# Clear stored API keys
from polaris.common.api_key_manager import APIKeyManager
manager = APIKeyManager()
manager.clear_stored_keys()
```

## 📊 Free Tier Limits

### **Gemini API Free Tier**
- **60 requests per minute**
- **1,500 requests per day**
- **Sufficient for development and testing**

### **Monitoring Usage**
- Monitor your usage at [Google AI Studio](https://makersuite.google.com/)
- Set up billing alerts if needed
- Consider rate limiting in production applications

## 🎯 Best Practices

### **Development**
- Use interactive prompts for quick setup
- Save to keyring for convenience
- Use validation mode to check setup

### **Production**
- Set environment variables in deployment configuration
- Use secure secret management systems
- Monitor API usage and quotas
- Implement proper error handling

### **Security**
- Never commit API keys to version control
- Use environment variables or secure storage
- Rotate API keys regularly
- Monitor for unauthorized usage

## 📚 Examples

### **Complete Workflow Example**
```bash
# 1. Start without API key set
python start_component.py digital-twin --world-model gemini

# 2. Follow interactive prompts:
#    - Enter API key when prompted
#    - Choose to save to keyring
#    - Component starts successfully

# 3. Future runs use saved key automatically
python start_component.py agentic-reasoner
# No prompt needed - uses saved key

# 4. Validate environment
python start_component.py digital-twin --validate-only
# Shows: ✅ API key found in secure keyring storage
```

### **Batch Validation Example**
```bash
# Validate all Gemini-based components
python start_component.py digital-twin --world-model gemini --validate-only
python start_component.py agentic-reasoner --validate-only
python start_component.py reasoner --validate-only
python start_component.py meta-learner --validate-only
```

### **Testing Example**
```bash
# Test the interactive API key system
python examples/test_interactive_api_key.py
```

## 🔮 Future Enhancements

### **Planned Features**
- Support for multiple API providers
- API key rotation and management
- Usage monitoring and alerts
- Team/organization key sharing
- Configuration templates with embedded keys

### **Integration Improvements**
- Docker container support
- Kubernetes secret integration
- CI/CD pipeline integration
- Cloud provider secret managers

## 🏁 Summary

The interactive API key management system makes POLARIS more user-friendly by:

- **Eliminating setup friction** - no need to manually configure environment variables
- **Providing clear guidance** - step-by-step instructions for getting API keys
- **Ensuring security** - secure input and storage options
- **Offering flexibility** - multiple storage and configuration options
- **Improving reliability** - validation and testing of API keys

This feature makes POLARIS accessible to users of all technical levels while maintaining security and flexibility for production deployments.