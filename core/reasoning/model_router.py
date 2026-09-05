"""
Model Router for Debate MoE

Provides a unified interface to call multiple LLM APIs:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3 Opus, Sonnet, etc.)
- Ollama (Llama 3.1, Mistral, etc.)

The router automatically detects which API to use based on model name.
"""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import API keys from config (with fallback to env vars)
try:
    from neuro_copilot.core.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_BASE_URL
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None
    OLLAMA_BASE_URL = None

# Lazy imports for API clients
_openai_client = None
_anthropic_client = None


@dataclass
class ModelResponse:
    """Unified response from any LLM API"""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    provider: str  # openai, anthropic, ollama


class ModelRouter:
    """
    Unified router for multiple LLM APIs.
    
    Supports:
    - OpenAI: gpt-4, gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Ollama: llama3.1:8b, llama3.1:70b, mistral:7b, etc.
    
    Usage:
        router = ModelRouter()
        response = router.call(
            model="gpt-4o",
            prompt="What is dopamine?",
            system_prompt="You are a neuroscience expert.",
            temperature=0.7
        )
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        default_timeout: float = 60.0
    ):
        """
        Initialize the model router.
        
        Args:
            openai_api_key: OpenAI API key (defaults to config or OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to config or ANTHROPIC_API_KEY env var)
            ollama_base_url: Ollama server URL (defaults to http://localhost:11434)
            default_timeout: Default timeout for API calls in seconds
        """
        # Priority: explicit arg > env var (updated by web UI save) > config
        self.openai_api_key = (
            openai_api_key or
            os.getenv("OPENAI_API_KEY") or
            (OPENAI_API_KEY if _CONFIG_AVAILABLE else None)
        )
        self.anthropic_api_key = (
            anthropic_api_key or
            os.getenv("ANTHROPIC_API_KEY") or
            (ANTHROPIC_API_KEY if _CONFIG_AVAILABLE else None)
        )
        self.ollama_base_url = (
            ollama_base_url or 
            (OLLAMA_BASE_URL if _CONFIG_AVAILABLE else None) or 
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.default_timeout = default_timeout
        
        # Lazy-loaded clients
        self._openai_client = None
        self._anthropic_client = None
    
    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
            
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            self._openai_client = OpenAI(
                api_key=self.openai_api_key,
                timeout=self.default_timeout
            )
        return self._openai_client
    
    def _get_anthropic_client(self):
        """Lazy load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
            
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.anthropic_api_key,
                timeout=self.default_timeout
            )
        return self._anthropic_client
    
    def _detect_provider(self, model: str) -> str:
        """
        Detect which provider to use based on model name.
        
        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-opus", "llama3.1:8b")
            
        Returns:
            Provider name: "openai", "anthropic", or "ollama"
        """
        model_lower = model.lower()
        
        # OpenAI models
        if any(prefix in model_lower for prefix in ["gpt-", "gpt4", "o1-", "o3-"]):
            return "openai"
        
        # Anthropic models
        if "claude" in model_lower:
            return "anthropic"
        
        # Ollama models (typically have : in name like llama3.1:8b)
        if ":" in model or any(name in model_lower for name in ["llama", "mistral", "mixtral", "qwen", "phi"]):
            return "ollama"
        
        # Default to OpenAI for unknown models
        return "openai"
    
    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: Optional[float] = None
    ) -> ModelResponse:
        """
        Call an LLM model through the appropriate API.
        
        Args:
            model: Model name (auto-detects provider)
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout (uses default if not specified)
            
        Returns:
            ModelResponse with content and metadata
        """
        provider = self._detect_provider(model)
        timeout = timeout or self.default_timeout
        
        if provider == "openai":
            return self._call_openai(model, prompt, system_prompt, temperature, max_tokens, timeout)
        elif provider == "anthropic":
            return self._call_anthropic(model, prompt, system_prompt, temperature, max_tokens, timeout)
        elif provider == "ollama":
            return self._call_ollama(model, prompt, system_prompt, temperature, max_tokens, timeout)
        else:
            raise ValueError(f"Unknown provider for model: {model}")
    
    def _call_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float
    ) -> ModelResponse:
        """Call OpenAI API."""
        client = self._get_openai_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        latency = time.perf_counter() - start_time
        
        return ModelResponse(
            content=response.choices[0].message.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency=latency,
            provider="openai"
        )
    
    def _call_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float
    ) -> ModelResponse:
        """Call Anthropic API."""
        client = self._get_anthropic_client()
        
        # Anthropic uses different message format
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.perf_counter()
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = client.messages.create(**kwargs)
        
        latency = time.perf_counter() - start_time
        
        # Extract content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        
        return ModelResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency=latency,
            provider="anthropic"
        )
    
    def _call_ollama(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float
    ) -> ModelResponse:
        """Call Ollama API."""
        import requests
        
        # Build the prompt with system message
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        start_time = time.perf_counter()
        
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        
        latency = time.perf_counter() - start_time
        result = response.json()
        
        # Ollama doesn't provide exact token counts, estimate from response
        prompt_tokens = result.get("prompt_eval_count", len(full_prompt.split()) * 2)
        completion_tokens = result.get("eval_count", len(result["response"].split()) * 2)
        
        return ModelResponse(
            content=result["response"],
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency=latency,
            provider="ollama"
        )
    
    def is_available(self, provider: str) -> bool:
        """
        Check if a provider is available (has API key configured).
        
        Args:
            provider: "openai", "anthropic", or "ollama"
            
        Returns:
            True if the provider is configured and available
        """
        if provider == "openai":
            return bool(self.openai_api_key)
        elif provider == "anthropic":
            return bool(self.anthropic_api_key)
        elif provider == "ollama":
            # Check if Ollama server is reachable
            try:
                import requests
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
                return response.status_code == 200
            except Exception:
                return False
        return False
    
    def list_available_providers(self) -> list:
        """List all available providers."""
        providers = []
        for provider in ["openai", "anthropic", "ollama"]:
            if self.is_available(provider):
                providers.append(provider)
        return providers
