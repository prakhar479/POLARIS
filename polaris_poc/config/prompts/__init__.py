"""
Prompt loading module for POLARIS Agentic Reasoner.

Provides system-agnostic prompt loading based on managed system type.
"""

from typing import Tuple, Callable
import logging


def load_system_prompts(system_name: str, logger: logging.Logger = None) -> Tuple[Callable, Callable]:
    """
    Load system-specific prompts based on managed system name.
    
    Args:
        system_name: Name of the managed system (e.g., "swim", "switch", "switch_yolo")
        logger: Optional logger instance
        
    Returns:
        Tuple of (get_system_prompt_func, get_user_prompt_template_func)
        
    Raises:
        ValueError: If system name is not supported
    """
    if logger is None:
        logger = logging.getLogger("PromptLoader")
    
    # Normalize system name
    system_name_lower = system_name.lower().replace("-", "_").replace(" ", "_")
    
    # Map system names to their prompt modules
    if system_name_lower in ["swim", "swim_system"]:
        logger.info(f"Loading SWIM prompts for system: {system_name}")
        from .swim_agentic_reasoner_prompts import (
            get_swim_system_prompt,
            get_swim_user_prompt_template
        )
        return get_swim_system_prompt, get_swim_user_prompt_template
        
    elif system_name_lower in ["switch", "switch_yolo", "switch_system"]:
        logger.info(f"Loading SWITCH prompts for system: {system_name}")
        from .switch_agentic_reasoner_prompts import (
            get_switch_system_prompt,
            get_switch_user_prompt_template
        )
        return get_switch_system_prompt, get_switch_user_prompt_template
    
    else:
        logger.warning(f"Unknown system name: {system_name}, falling back to generic prompts")
        # Fallback to base prompts with minimal system-specific content
        from .base_agentic_reasoner_prompts import (
            get_base_system_prompt_template,
            get_tools_description,
            format_base_user_prompt
        )
        
        def get_generic_system_prompt(tools_available: bool = True) -> str:
            tools_desc = get_tools_description(tools_available)
            system_content = f"""
# ============================================================================
# GENERIC SYSTEM CONFIGURATION
# ============================================================================

System: {system_name}

This is a generic configuration. For optimal performance, please configure
system-specific prompts for your managed system.

Decision Making Process:
1. Analyze the input data to understand the current situation
2. Determine what additional information you need
3. Use tools to gather that information (CRITICAL step)
4. Wait for tool results and analyze them
5. Synthesize all information to make an adaptation decision
6. Generate the appropriate control action

Control Actions:
- System-specific actions will be defined by your system configuration
- NO_ACTION: Take no action if system is operating optimally

System Constraints:
- System-specific constraints will be defined by your system configuration
"""
            return get_base_system_prompt_template(tools_desc, system_content)
        
        def get_generic_user_prompt_template() -> str:
            return """
# CURRENT SYSTEM STATE

**Timestamp**: {timestamp}
**System**: {system_name}

## Telemetry Events
{telemetry_events}

## Recent Actions
{recent_actions}

---

**Your Task**: Analyze this system state and determine the optimal adaptation action.
"""
        
        return get_generic_system_prompt, get_generic_user_prompt_template


def get_supported_systems():
    """
    Get list of supported system names for prompt loading.
    
    Returns:
        List of supported system names
    """
    return [
        "swim",
        "swim_system",
        "switch",
        "switch_yolo",
        "switch_system"
    ]
