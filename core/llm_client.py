# neuro_copilot/llm_client.py
"""
Unified LLM client supporting multiple backends:
- OpenAI (GPT-4, GPT-4-turbo, GPT-4o)
- Ollama (Llama 3.1, local models)

The backend is selected via LLM_BACKEND environment variable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from neuro_copilot.core.config import (
    LLM_BACKEND,
    OPENAI_BASE_URL,
    OLLAMA_BASE_URL,
)


@dataclass
class LLMResponse:
    raw_text: str
    json_obj: Optional[Dict[str, Any]]
    model: str
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None
    total_duration_ms: Optional[int] = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class OllamaClientError(LLMClientError):
    """Exception for Ollama-specific errors."""
    pass


class OpenAIClientError(LLMClientError):
    """Exception for OpenAI-specific errors."""
    pass


def _http_post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_s: float = 60.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generic HTTP POST with JSON payload."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    req = urllib.request.Request(
        url,
        data=data,
        headers=req_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise LLMClientError(f"HTTPError {e.code}: {e.reason}\n{detail}") from e
    except urllib.error.URLError as e:
        raise LLMClientError(f"URLError: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMClientError(f"Failed to decode JSON response: {e}") from e


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON object extraction.
    If the model outputs extra text, try to locate the outermost {...}.
    """
    text = text.strip()
    if not text:
        return None

    # Try direct parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try bracket extraction
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1].strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _generate_json_openai(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    schema_hint: Optional[str] = None,
) -> LLMResponse:
    """
    Call OpenAI Chat Completions API for JSON generation.
    Uses response_format: {"type": "json_object"} for structured output.
    """
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    
    strict_instr = (
        "You must output ONLY a single valid JSON object. "
        "No markdown, no code fences, no extra commentary."
    )
    if schema_hint:
        strict_instr += f" The JSON must follow this schema hint: {schema_hint}"
    
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "system", "content": strict_instr})
    messages.append({"role": "user", "content": user_prompt.strip()})
    
    # Models that support response_format: json_object
    # (gpt-4-turbo, gpt-4o, gpt-4-1106-preview, gpt-3.5-turbo-1106, etc.)
    JSON_MODE_MODELS = (
        "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4-turbo-2024",
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024",
        "gpt-4-1106-preview", "gpt-4-0125-preview",
        "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
        "gpt-5", "gpt-5.1",
    )
    
    # Check if model supports JSON mode (prefix match for dated versions)
    supports_json_mode = any(
        model.startswith(m) or model == m 
        for m in JSON_MODE_MODELS
    )
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "seed": 42,  # Deterministic sampling for reproducibility
    }

    # Only add response_format for models that support it
    if supports_json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    # Read at runtime so web UI config save takes effect immediately
    api_key = os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    last_err: Optional[Exception] = None
    start_time = time.time()
    
    for attempt in range(retries + 1):
        try:
            resp = _http_post_json(url, payload, timeout_s=timeout_s, headers=headers)
            
            # OpenAI returns: choices[0].message.content, usage.prompt_tokens, usage.completion_tokens
            choices = resp.get("choices", [])
            if not choices:
                raise OpenAIClientError("No choices in response")
            
            raw_text = choices[0].get("message", {}).get("content", "").strip()
            json_obj = _extract_json_object(raw_text)
            
            usage = resp.get("usage", {})
            duration_ms = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                raw_text=raw_text,
                json_obj=json_obj,
                model=model,
                prompt_tokens=usage.get("prompt_tokens"),
                eval_tokens=usage.get("completion_tokens"),
                total_duration_ms=duration_ms,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise
    
    raise OpenAIClientError(f"Failed after retries. Last error: {last_err}")


def _generate_json_ollama(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    seed: Optional[int] = 0,
    schema_hint: Optional[str] = None,
) -> LLMResponse:
    """
    Call Ollama /api/generate for JSON generation.
    Uses format: "json" for structured output.
    """
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/generate"

    strict_instr = (
        "You must output ONLY a single valid JSON object. "
        "No markdown, no code fences, no extra commentary."
    )
    if schema_hint:
        strict_instr += f" The JSON must follow this schema hint: {schema_hint}"

    # Compose prompt for local models
    prompt_parts = []
    if system_prompt.strip():
        prompt_parts.append(f"[SYSTEM]\n{system_prompt.strip()}\n")
    prompt_parts.append(f"[INSTRUCTION]\n{strict_instr}\n")
    prompt_parts.append(f"[USER]\n{user_prompt.strip()}\n")
    full_prompt = "\n".join(prompt_parts).strip()

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(temperature),
        },
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = _http_post_json(url, payload, timeout_s=timeout_s)

            raw_text = (resp.get("response") or "").strip()
            json_obj = _extract_json_object(raw_text)

            total_duration_ns = resp.get("total_duration")
            total_ms = int(total_duration_ns / 1e6) if isinstance(total_duration_ns, int) else None

            return LLMResponse(
                raw_text=raw_text,
                json_obj=json_obj,
                model=model,
                prompt_tokens=resp.get("prompt_eval_count"),
                eval_tokens=resp.get("eval_count"),
                total_duration_ms=total_ms,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise

    raise OllamaClientError(f"Failed after retries. Last error: {last_err}")


def generate_json(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    base_url: str = "http://localhost:11434",  # Legacy parameter, ignored when using config
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    seed: Optional[int] = 0,
    schema_hint: Optional[str] = None,
    backend: Optional[str] = None,  # Override LLM_BACKEND if specified
) -> LLMResponse:
    """
    Unified LLM JSON generation supporting multiple backends.
    
    Automatically selects backend based on LLM_BACKEND config:
    - "openai": Uses OpenAI Chat Completions API (GPT-4, GPT-4-turbo, etc.)
    - "ollama": Uses Ollama local API (Llama 3.1, etc.)
    
    Args:
        model: Model name (e.g., "gpt-4", "llama3.1:8b")
        user_prompt: User message
        system_prompt: System instructions
        base_url: Legacy parameter (ignored, use config instead)
        timeout_s: Request timeout in seconds
        retries: Number of retry attempts
        retry_backoff_s: Backoff multiplier between retries
        temperature: Sampling temperature
        seed: Random seed (Ollama only)
        schema_hint: Optional JSON schema description
        backend: Override LLM_BACKEND config ("openai" or "ollama")
    
    Returns:
        LLMResponse with raw_text, json_obj, and usage stats
    """
    selected_backend = (backend or LLM_BACKEND).lower()
    
    if selected_backend == "openai":
        return _generate_json_openai(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            temperature=temperature,
            schema_hint=schema_hint,
        )
    else:
        # Default to Ollama
        return _generate_json_ollama(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            temperature=temperature,
            seed=seed,
            schema_hint=schema_hint,
        )


# =========================================================================
# Plain-text generation (no JSON mode)
# =========================================================================

def _generate_text_openai(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> LLMResponse:
    """
    Call OpenAI Chat Completions API for free-form text (NO JSON mode).
    Used by CodeGenerator which needs raw Python code, not JSON.
    """
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_completion_tokens": int(max_tokens),  # OpenAI now uses max_completion_tokens
        "seed": 42,  # Deterministic sampling for reproducibility
    }
    # Deliberately NO response_format — we want free-form text

    # Read at runtime so web UI config save takes effect immediately
    api_key = os.getenv("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}

    last_err: Optional[Exception] = None
    start_time = time.time()

    for attempt in range(retries + 1):
        try:
            resp = _http_post_json(url, payload, timeout_s=timeout_s, headers=headers)
            choices = resp.get("choices", [])
            if not choices:
                raise OpenAIClientError("No choices in response")

            raw_text = choices[0].get("message", {}).get("content", "").strip()
            usage = resp.get("usage", {})
            duration_ms = int((time.time() - start_time) * 1000)

            return LLMResponse(
                raw_text=raw_text,
                json_obj=None,
                model=model,
                prompt_tokens=usage.get("prompt_tokens"),
                eval_tokens=usage.get("completion_tokens"),
                total_duration_ms=duration_ms,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise

    raise OpenAIClientError(f"Failed after retries. Last error: {last_err}")


def _generate_text_ollama(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    seed: Optional[int] = 0,
    max_tokens: int = 4000,
) -> LLMResponse:
    """Call Ollama API for free-form text (no JSON format constraint)."""
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/generate"

    full_prompt = ""
    if system_prompt.strip():
        full_prompt = f"[SYSTEM]\n{system_prompt.strip()}\n\n[USER]\n{user_prompt.strip()}"
    else:
        full_prompt = user_prompt.strip()

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    # Deliberately NO "format": "json"

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = _http_post_json(url, payload, timeout_s=timeout_s)
            raw_text = (resp.get("response") or "").strip()
            total_duration_ns = resp.get("total_duration")
            total_ms = int(total_duration_ns / 1e6) if isinstance(total_duration_ns, int) else None

            return LLMResponse(
                raw_text=raw_text,
                json_obj=None,
                model=model,
                prompt_tokens=resp.get("prompt_eval_count"),
                eval_tokens=resp.get("eval_count"),
                total_duration_ms=total_ms,
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise

    raise OllamaClientError(f"Failed after retries. Last error: {last_err}")


def generate_text(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    timeout_s: float = 120.0,
    retries: int = 2,
    retry_backoff_s: float = 1.5,
    temperature: float = 0.2,
    seed: Optional[int] = 0,
    max_tokens: int = 4000,
    backend: Optional[str] = None,
) -> LLMResponse:
    """
    Unified LLM text generation (NO JSON mode).

    Use this instead of ``generate_json`` when you need free-form text
    output (e.g., code generation, explanations).

    Returns:
        LLMResponse with raw_text (json_obj will be None).
    """
    selected_backend = (backend or LLM_BACKEND).lower()

    if selected_backend == "openai":
        return _generate_text_openai(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        return _generate_text_ollama(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )


def check_ollama_health(base_url: str = "http://localhost:11434", timeout_s: float = 5.0) -> bool:
    """
    Quick health check: GET / should return 200 with a small body.
    """
    url = base_url.rstrip("/") + "/"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False