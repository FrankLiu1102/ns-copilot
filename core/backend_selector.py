"""
Backend Selector for NS-Copilot

Determines which LLM backend to use based on context:
- When has_dataset=False: Use cloud LLM (currently GPT-4, easily replaceable)
- When has_dataset=True: Use local Ollama

This design makes it easy to swap cloud LLM backends in the future
(e.g., GPT-4 → Claude, Gemini, or custom model)
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from neuro_copilot.core.config import (
    OPENAI_MODEL,
    USE_GPT4_FOR_NO_DATASET
)


@dataclass
class BackendConfig:
    """Configuration for LLM backend"""
    name: str  # "gpt4", "claude", "ollama", etc.
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0


def select_backend(has_dataset: bool, task_type: str) -> BackendConfig:
    """
    Select appropriate LLM backend based on context
    
    Args:
        has_dataset: Whether user provided dataset
        task_type: "QA" | "DA" | "ER"
        
    Returns:
        BackendConfig for selected backend
        
    Strategy:
    - No dataset + (QA/DA/ER) → Cloud LLM (GPT-4)
    - Has dataset + DA → Local Ollama (for dataset-specific analysis)
    - Has dataset + (QA/ER) → Local Ollama
    """
    # When no dataset: use cloud LLM for all tasks
    # Read API key at runtime so web UI config save takes effect immediately
    if not has_dataset and USE_GPT4_FOR_NO_DATASET:
        return BackendConfig(
            name="gpt4",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=OPENAI_MODEL,
            timeout=60.0
        )
    
    # When has dataset: use local Ollama
    # (Dataset-specific tasks need local tools/execution)
    return BackendConfig(
        name="ollama",
        base_url="http://localhost:11434",
        model_name="llama3.1:8b",
        timeout=120.0
    )


def get_cloud_llm_client(config: BackendConfig):
    """
    Get client for cloud LLM backend
    
    This function centralizes cloud LLM initialization,
    making it easy to swap backends in the future:
    
    Future swap example:
    ```python
    if config.name == "gpt4":
        from neuro_copilot.core.gpt4_client import GPT4Client
        return GPT4Client(api_key=config.api_key, model=config.model_name)
    elif config.name == "claude":
        from neuro_copilot.core.claude_client import ClaudeClient
        return ClaudeClient(api_key=config.api_key, model=config.model_name)
    elif config.name == "gemini":
        from neuro_copilot.core.gemini_client import GeminiClient
        return GeminiClient(api_key=config.api_key, model=config.model_name)
    ```
    """
    if config.name == "gpt4":
        from neuro_copilot.core.gpt4_client import GPT4Client
        return GPT4Client(
            api_key=config.api_key,
            model=config.model_name,
            timeout=config.timeout
        )
    
    # Future: Add more backends here
    # elif config.name == "claude":
    #     from neuro_copilot.core.claude_client import ClaudeClient
    #     return ClaudeClient(...)
    
    else:
        raise ValueError(f"Unknown cloud LLM backend: {config.name}")


def is_cloud_backend(config: BackendConfig) -> bool:
    """Check if backend is a cloud LLM (vs local Ollama)"""
    return config.name in ["gpt4", "claude", "gemini"]


# Example usage:
if __name__ == "__main__":
    print("Backend Selection Examples:")
    print("="*60)
    
    # No dataset → Cloud LLM
    config = select_backend(has_dataset=False, task_type="QA")
    print(f"\nNo dataset, QA task → {config.name} ({config.model_name})")
    
    # Has dataset → Local Ollama
    config = select_backend(has_dataset=True, task_type="DA")
    print(f"Has dataset, DA task → {config.name} ({config.model_name})")
    
    print("\n" + "="*60)
    print("To swap cloud LLM backend:")
    print("1. Update config.py: CLOUD_LLM_BACKEND = 'claude'")
    print("2. Update select_backend() to return claude config")
    print("3. Add claude_client.py with ClaudeClient class")
    print("4. Update get_cloud_llm_client() to handle 'claude'")
