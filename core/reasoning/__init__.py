"""
Reasoning Module for NS-Copilot

This module implements a Debate-style Mixture of Experts (MoE) reasoning system
where multiple LLM experts (Advocate, Critic, Synthesizer, Judge) collaborate
through structured debate to produce high-quality answers.

Architecture:
    - ModelRouter: Unified interface for multiple LLM APIs (OpenAI, Anthropic, Ollama)
    - Experts: Role-specific LLM wrappers (Advocate, Critic, Synthesizer, Judge)
    - DebateMoE: Core debate engine that orchestrates multi-round discussions
    - Prompts: Role-specific prompt templates for neuroscience QA

Usage:
    from neuro_copilot.core.reasoning import DebateMoE, DebateConfig
    
    config = DebateConfig(max_rounds=3)
    engine = DebateMoE(config)
    result = engine.debate(question="...", context="...")
"""

from .debate_moe import DebateMoE, DebateConfig, DebateResult, DebateRound
from .experts import (
    BaseExpert, 
    ExpertConfig,
    AdvocateExpert,
    CriticExpert,
    SynthesizerExpert,
    JudgeExpert
)
from .model_router import ModelRouter

__all__ = [
    # Core engine
    "DebateMoE",
    "DebateConfig", 
    "DebateResult",
    "DebateRound",
    # Experts
    "BaseExpert",
    "ExpertConfig",
    "AdvocateExpert",
    "CriticExpert",
    "SynthesizerExpert",
    "JudgeExpert",
    # Router
    "ModelRouter",
]
