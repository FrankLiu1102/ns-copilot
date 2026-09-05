"""
GPT-4 Client for NS-Copilot

Provides a clean interface to OpenAI GPT-4 API for inference tasks.
Used when has_dataset == False to leverage cloud-based reasoning.
"""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from neuro_copilot.core.config import OPENAI_MODEL


@dataclass
class GPT4Response:
    """GPT-4 response wrapper"""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float


class GPT4Client:
    """
    OpenAI GPT-4 client for NS-Copilot
    
    Design principles:
    - Minimal prompt engineering (natural QA only)
    - No dataset-specific constraints
    - Clean separation from local LLM pipeline
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0):
        """
        Initialize GPT-4 client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from config)
            model: Model name (defaults to OPENAI_MODEL from config)
            timeout: API request timeout in seconds (default: 60s)
        """
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        # Read at runtime so web UI config save takes effect immediately
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or OPENAI_MODEL
        self.timeout = timeout
        
        if not self.api_key or self.api_key == "":
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key argument."
            )
        
        # Initialize OpenAI client with timeout
        self.client = OpenAI(api_key=self.api_key, timeout=timeout)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> GPT4Response:
        """
        Generate response from GPT-4
        
        Args:
            prompt: User prompt (question + context)
            system_prompt: System prompt (optional, defaults to minimal instruction)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            GPT4Response with content and metadata
        """
        # Default system prompt (minimal, no dataset-specific instructions)
        if system_prompt is None:
            system_prompt = (
                "You are a helpful research assistant. "
                "Answer questions based on the provided context."
            )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI API with timing and timeout
        start_time = time.perf_counter()
        
        try:
            # Use OpenAI SDK's timeout parameter (thread-safe, works in Gradio)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        # Extract response
        content = response.choices[0].message.content
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        return GPT4Response(
            content=content,
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency
        )


def extract_classification_label(response: str) -> str:
    """
    Simplified answer extraction logic for classification tasks
    
    ⚠️ CRITICAL: This function must be used consistently across:
    - GPT-4 backend
    - Llama 3 baseline
    - Any future models
    
    Strategy (SIMPLIFIED per paper methodology):
    1. Strip and lowercase the response
    2. Check if starts with yes/no/maybe
    3. If not found → fallback to "maybe"
    
    ❌ NO heuristic / sentiment / regex scanning
    ✅ ONLY startswith check
    
    Args:
        response: Free-form text response from any LLM
        
    Returns:
        One of: "yes", "no", "maybe"
    """
    # Normalize: strip whitespace and convert to lowercase
    response = response.strip().lower()
    
    # Simple startswith check (as per paper methodology)
    if response.startswith("yes"):
        return "yes"
    elif response.startswith("no"):
        return "no"
    elif response.startswith("maybe"):
        return "maybe"
    else:
        # Fallback: no explicit label found
        return "maybe"


# Convenience function for quick testing
def quick_test():
    """Quick test of GPT-4 client"""
    client = GPT4Client()
    
    test_prompt = "What is the capital of France?"
    response = client.generate(test_prompt)
    
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})")
    print(f"Latency: {response.latency:.2f}s")
    
    # Test extraction
    label = extract_classification_label(response.content)
    print(f"Extracted label: {label}")


if __name__ == "__main__":
    quick_test()
