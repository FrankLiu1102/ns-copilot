"""
Expert Roles for Debate MoE

Implements the four expert roles:
- Advocate: Proposes and defends answers based on evidence
- Critic: Challenges answers and identifies weaknesses
- Synthesizer: Integrates perspectives from the debate
- Judge: Makes final determination and assigns confidence
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .model_router import ModelRouter, ModelResponse
from . import prompts


class ExpertRole(Enum):
    """Enumeration of expert roles in the debate."""
    ADVOCATE = "advocate"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    JUDGE = "judge"


@dataclass
class ExpertConfig:
    """Configuration for an expert."""
    role: ExpertRole
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1500
    
    @classmethod
    def default_advocate(cls) -> "ExpertConfig":
        return cls(
            role=ExpertRole.ADVOCATE,
            model=os.getenv("ADVOCATE_MODEL", "gpt-4o"),
            temperature=0.7
        )
    
    @classmethod
    def default_critic(cls) -> "ExpertConfig":
        return cls(
            role=ExpertRole.CRITIC,
            model=os.getenv("CRITIC_MODEL", "gpt-4o"),
            temperature=0.7
        )
    
    @classmethod
    def default_synthesizer(cls) -> "ExpertConfig":
        return cls(
            role=ExpertRole.SYNTHESIZER,
            model=os.getenv("SYNTHESIZER_MODEL", "gpt-4o"),
            temperature=0.6
        )
    
    @classmethod
    def default_judge(cls) -> "ExpertConfig":
        return cls(
            role=ExpertRole.JUDGE,
            model=os.getenv("JUDGE_MODEL", "gpt-4o"),
            temperature=0.3  # Lower temperature for more deterministic judgment
        )


@dataclass
class ExpertResponse:
    """Response from an expert."""
    role: ExpertRole
    content: str
    model: str
    tokens_used: int
    latency: float
    
    # Parsed sections (optional)
    sections: Dict[str, str] = field(default_factory=dict)


class BaseExpert:
    """
    Base class for debate experts.
    
    Each expert has a specific role and system prompt that guides
    their behavior in the debate.
    """
    
    def __init__(
        self,
        config: ExpertConfig,
        router: Optional[ModelRouter] = None
    ):
        """
        Initialize the expert.
        
        Args:
            config: Expert configuration (role, model, temperature)
            router: Model router for API calls (creates new one if not provided)
        """
        self.config = config
        self.router = router or ModelRouter()
        self._system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this expert's role."""
        role_prompts = {
            ExpertRole.ADVOCATE: prompts.ADVOCATE_SYSTEM_PROMPT,
            ExpertRole.CRITIC: prompts.CRITIC_SYSTEM_PROMPT,
            ExpertRole.SYNTHESIZER: prompts.SYNTHESIZER_SYSTEM_PROMPT,
            ExpertRole.JUDGE: prompts.JUDGE_SYSTEM_PROMPT,
        }
        return role_prompts.get(self.config.role, "")
    
    def respond(
        self,
        prompt: str,
        **kwargs
    ) -> ExpertResponse:
        """
        Generate a response from this expert.
        
        Args:
            prompt: The prompt to respond to
            **kwargs: Additional arguments passed to the model
            
        Returns:
            ExpertResponse with content and metadata
        """
        # Merge default config with any overrides
        temperature = kwargs.pop("temperature", self.config.temperature)
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        
        # Call the model
        model_response = self.router.call(
            model=self.config.model,
            prompt=prompt,
            system_prompt=self._system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Parse sections from response
        sections = self._parse_sections(model_response.content)
        
        return ExpertResponse(
            role=self.config.role,
            content=model_response.content,
            model=model_response.model,
            tokens_used=model_response.total_tokens,
            latency=model_response.latency,
            sections=sections
        )
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse structured sections from the response."""
        sections = {}
        
        # Define expected sections per role
        role_sections = {
            ExpertRole.ADVOCATE: ["POSITION", "EVIDENCE", "REASONING", "CONFIDENCE"],
            ExpertRole.CRITIC: ["STRENGTHS", "WEAKNESSES", "CHALLENGES", "ALTERNATIVES"],
            ExpertRole.SYNTHESIZER: ["CONSENSUS", "RESOLUTION", "SYNTHESIS", "UNCERTAINTIES"],
            ExpertRole.JUDGE: ["FINAL_ANSWER", "KEY_EVIDENCE", "CONFIDENCE_SCORE", "DISSENTING_VIEW"],
        }
        
        for section_name in role_sections.get(self.config.role, []):
            section_content = prompts.extract_section(content, section_name)
            if section_content:
                sections[section_name] = section_content
        
        return sections


class AdvocateExpert(BaseExpert):
    """
    Advocate expert that proposes and defends answers.
    
    The Advocate's job is to:
    1. Provide a clear, evidence-based answer
    2. Support claims with specific evidence from context
    3. Defend valid points against criticism
    4. Revise answer when criticism is valid
    """
    
    def __init__(self, router: Optional[ModelRouter] = None, config: Optional[ExpertConfig] = None):
        super().__init__(
            config=config or ExpertConfig.default_advocate(),
            router=router
        )
    
    def initial_response(self, question: str, context: str) -> ExpertResponse:
        """Generate the initial answer to the question."""
        prompt = prompts.get_advocate_initial_prompt(question, context)
        return self.respond(prompt)
    
    def rebuttal(
        self,
        question: str,
        context: str,
        previous_response: str,
        critic_response: str
    ) -> ExpertResponse:
        """Respond to critic's challenges."""
        prompt = prompts.get_advocate_rebuttal_prompt(
            question, context, previous_response, critic_response
        )
        return self.respond(prompt)


class CriticExpert(BaseExpert):
    """
    Critic expert that challenges and scrutinizes answers.
    
    The Critic's job is to:
    1. Identify logical gaps and unsupported claims
    2. Challenge questionable assumptions
    3. Propose alternative interpretations
    4. Help improve answer quality through constructive criticism
    """
    
    def __init__(self, router: Optional[ModelRouter] = None, config: Optional[ExpertConfig] = None):
        super().__init__(
            config=config or ExpertConfig.default_critic(),
            router=router
        )
    
    def critique(
        self,
        question: str,
        context: str,
        advocate_response: str
    ) -> ExpertResponse:
        """Critique the Advocate's response."""
        prompt = prompts.get_critic_prompt(question, context, advocate_response)
        return self.respond(prompt)


class SynthesizerExpert(BaseExpert):
    """
    Synthesizer expert that integrates debate perspectives.
    
    The Synthesizer's job is to:
    1. Identify points of agreement
    2. Resolve disagreements based on evidence strength
    3. Create a balanced, integrated perspective
    4. Highlight remaining uncertainties
    """
    
    def __init__(self, router: Optional[ModelRouter] = None, config: Optional[ExpertConfig] = None):
        super().__init__(
            config=config or ExpertConfig.default_synthesizer(),
            router=router
        )
    
    def synthesize(
        self,
        question: str,
        context: str,
        advocate_response: str,
        critic_response: str,
        advocate_rebuttal: str
    ) -> ExpertResponse:
        """Synthesize the debate into an integrated perspective."""
        prompt = prompts.get_synthesizer_prompt(
            question, context, advocate_response, critic_response, advocate_rebuttal
        )
        return self.respond(prompt)


class JudgeExpert(BaseExpert):
    """
    Judge expert that makes the final determination.
    
    The Judge's job is to:
    1. Evaluate all arguments based on evidence quality
    2. Determine the best-supported answer
    3. Assign a confidence score
    4. Note any valid dissenting views
    """
    
    def __init__(self, router: Optional[ModelRouter] = None, config: Optional[ExpertConfig] = None):
        super().__init__(
            config=config or ExpertConfig.default_judge(),
            router=router
        )
    
    def judge(
        self,
        question: str,
        context: str,
        advocate_response: str,
        critic_response: str,
        advocate_rebuttal: str,
        synthesizer_response: str
    ) -> ExpertResponse:
        """Make final judgment on the debate."""
        prompt = prompts.get_judge_prompt(
            question, context, advocate_response, critic_response,
            advocate_rebuttal, synthesizer_response
        )
        return self.respond(prompt)
    
    def get_final_answer(self, response: ExpertResponse) -> str:
        """Extract the final answer from Judge's response."""
        if "FINAL_ANSWER" in response.sections:
            return response.sections["FINAL_ANSWER"]
        
        # Fallback: try to extract from content
        final_answer = prompts.extract_section(response.content, "FINAL_ANSWER")
        if final_answer:
            return final_answer
        
        # Last resort: return first paragraph
        lines = response.content.strip().split("\n")
        return lines[0] if lines else response.content
    
    def get_confidence(self, response: ExpertResponse) -> float:
        """Extract confidence score from Judge's response."""
        return prompts.parse_confidence_score(response.content)


def create_expert_panel(
    router: Optional[ModelRouter] = None,
    advocate_config: Optional[ExpertConfig] = None,
    critic_config: Optional[ExpertConfig] = None,
    synthesizer_config: Optional[ExpertConfig] = None,
    judge_config: Optional[ExpertConfig] = None
) -> Dict[str, BaseExpert]:
    """
    Create a complete panel of debate experts.
    
    Args:
        router: Shared model router (creates new one if not provided)
        *_config: Optional custom configs for each expert
        
    Returns:
        Dictionary mapping role names to expert instances
    """
    router = router or ModelRouter()
    
    return {
        "advocate": AdvocateExpert(router, advocate_config),
        "critic": CriticExpert(router, critic_config),
        "synthesizer": SynthesizerExpert(router, synthesizer_config),
        "judge": JudgeExpert(router, judge_config),
    }
