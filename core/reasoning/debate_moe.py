"""
Debate MoE (Mixture of Experts) Engine

Orchestrates a structured debate between multiple LLM experts to produce
high-quality, well-reasoned answers for neuroscience QA tasks.

Debate Flow:
    Round 1: Advocate proposes initial answer
    Round 2: Critic challenges the answer
    Round 3: Advocate rebuts criticism
    Round 4: Synthesizer integrates perspectives
    Round 5: Judge makes final determination
"""

import time
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .model_router import ModelRouter
from .experts import (
    AdvocateExpert,
    CriticExpert,
    SynthesizerExpert,
    JudgeExpert,
    ExpertConfig,
    ExpertResponse,
    ExpertRole,
    create_expert_panel
)
from . import prompts


class DebateStatus(Enum):
    """Status of the debate."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CONSENSUS_REACHED = "consensus_reached"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class DebateRound:
    """Record of a single debate round."""
    round_number: int
    role: ExpertRole
    response: ExpertResponse
    timestamp: float
    
    @property
    def content(self) -> str:
        return self.response.content
    
    @property
    def tokens_used(self) -> int:
        return self.response.tokens_used


@dataclass
class DebateConfig:
    """Configuration for the debate engine."""
    
    # Debate structure
    max_rounds: int = 5  # Standard: Advocate -> Critic -> Advocate -> Synthesizer -> Judge
    early_stop_consensus: bool = True  # Stop if strong consensus reached
    consensus_threshold: float = 0.9  # Confidence threshold for early stop
    
    # Expert models (can be customized per expert)
    advocate_model: str = "gpt-4o"
    critic_model: str = "gpt-4o"
    synthesizer_model: str = "gpt-4o"
    judge_model: str = "gpt-4o"
    
    # Model parameters
    advocate_temperature: float = 0.7
    critic_temperature: float = 0.7
    synthesizer_temperature: float = 0.6
    judge_temperature: float = 0.3
    
    # Logging
    log_callback: Optional[Callable[[str], None]] = None
    verbose: bool = True
    
    def create_expert_configs(self) -> Dict[str, ExpertConfig]:
        """Create expert configs from debate config."""
        return {
            "advocate": ExpertConfig(
                role=ExpertRole.ADVOCATE,
                model=self.advocate_model,
                temperature=self.advocate_temperature
            ),
            "critic": ExpertConfig(
                role=ExpertRole.CRITIC,
                model=self.critic_model,
                temperature=self.critic_temperature
            ),
            "synthesizer": ExpertConfig(
                role=ExpertRole.SYNTHESIZER,
                model=self.synthesizer_model,
                temperature=self.synthesizer_temperature
            ),
            "judge": ExpertConfig(
                role=ExpertRole.JUDGE,
                model=self.judge_model,
                temperature=self.judge_temperature
            ),
        }


@dataclass
class DebateResult:
    """Complete result from a debate session."""
    
    # Final output
    final_answer: str
    confidence: float
    
    # Debate metadata
    status: DebateStatus
    consensus_reached: bool
    total_rounds: int
    
    # Full transcript
    debate_transcript: List[DebateRound] = field(default_factory=list)
    
    # Analysis
    winning_arguments: List[str] = field(default_factory=list)
    dissenting_view: Optional[str] = None
    
    # Resource usage
    token_usage: Dict[str, int] = field(default_factory=dict)
    total_latency: float = 0.0
    
    def get_round(self, role: ExpertRole) -> Optional[DebateRound]:
        """Get the debate round for a specific role."""
        for round in self.debate_transcript:
            if round.role == role:
                return round
        return None
    
    def get_summary(self) -> str:
        """Get a formatted summary of the debate."""
        return prompts.format_debate_summary(
            question="[See debate transcript]",
            advocate_response=self.debate_transcript[0].content if len(self.debate_transcript) > 0 else "",
            critic_response=self.debate_transcript[1].content if len(self.debate_transcript) > 1 else "",
            synthesizer_response=self.debate_transcript[3].content if len(self.debate_transcript) > 3 else "",
            judge_response=self.debate_transcript[4].content if len(self.debate_transcript) > 4 else "",
            final_answer=self.final_answer,
            confidence=self.confidence
        )


class DebateMoE:
    """
    Debate-style Mixture of Experts engine.
    
    Orchestrates a structured debate between multiple LLM experts:
    - Advocate: Proposes and defends answers
    - Critic: Challenges and scrutinizes
    - Synthesizer: Integrates perspectives
    - Judge: Makes final determination
    
    Usage:
        config = DebateConfig(advocate_model="gpt-4o", critic_model="claude-3-opus")
        engine = DebateMoE(config)
        result = engine.debate(question="...", context="...")
    """
    
    def __init__(
        self,
        config: Optional[DebateConfig] = None,
        router: Optional[ModelRouter] = None
    ):
        """
        Initialize the debate engine.
        
        Args:
            config: Debate configuration (uses defaults if not provided)
            router: Model router for API calls (creates new one if not provided)
        """
        self.config = config or DebateConfig()
        self.router = router or ModelRouter()
        
        # Create expert instances
        expert_configs = self.config.create_expert_configs()
        self.advocate = AdvocateExpert(self.router, expert_configs["advocate"])
        self.critic = CriticExpert(self.router, expert_configs["critic"])
        self.synthesizer = SynthesizerExpert(self.router, expert_configs["synthesizer"])
        self.judge = JudgeExpert(self.router, expert_configs["judge"])
    
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            if self.config.log_callback:
                self.config.log_callback(message)
            else:
                print(message)
    
    def debate(
        self,
        question: str,
        context: str,
        additional_context: Optional[str] = None
    ) -> DebateResult:
        """
        Execute a full debate on the given question.
        
        Args:
            question: The question to answer
            context: Knowledge context (from GraphRAG, KB, etc.)
            additional_context: Optional additional context
            
        Returns:
            DebateResult with final answer, confidence, and full transcript
        """
        start_time = time.perf_counter()
        
        # Combine contexts if needed
        full_context = context
        if additional_context:
            full_context = f"{context}\n\n{additional_context}"
        
        # Initialize tracking
        transcript: List[DebateRound] = []
        token_usage: Dict[str, int] = {
            "advocate": 0,
            "critic": 0,
            "synthesizer": 0,
            "judge": 0
        }
        round_num = 0
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # Round 1: Advocate's Initial Position
            # ═══════════════════════════════════════════════════════════════
            round_num = 1
            self._log(f"\n🎯 Round {round_num}: Advocate proposing initial answer...")
            
            advocate_response = self.advocate.initial_response(question, full_context)
            transcript.append(DebateRound(
                round_number=round_num,
                role=ExpertRole.ADVOCATE,
                response=advocate_response,
                timestamp=time.perf_counter()
            ))
            token_usage["advocate"] += advocate_response.tokens_used
            
            self._log(f"   ✓ Advocate responded ({advocate_response.tokens_used} tokens, {advocate_response.latency:.2f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Round 2: Critic's Challenge
            # ═══════════════════════════════════════════════════════════════
            round_num = 2
            self._log(f"\n🔍 Round {round_num}: Critic analyzing response...")
            
            critic_response = self.critic.critique(
                question, full_context, advocate_response.content
            )
            transcript.append(DebateRound(
                round_number=round_num,
                role=ExpertRole.CRITIC,
                response=critic_response,
                timestamp=time.perf_counter()
            ))
            token_usage["critic"] += critic_response.tokens_used
            
            self._log(f"   ✓ Critic responded ({critic_response.tokens_used} tokens, {critic_response.latency:.2f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Round 3: Advocate's Rebuttal
            # ═══════════════════════════════════════════════════════════════
            round_num = 3
            self._log(f"\n💬 Round {round_num}: Advocate responding to critique...")
            
            advocate_rebuttal = self.advocate.rebuttal(
                question, full_context,
                advocate_response.content,
                critic_response.content
            )
            transcript.append(DebateRound(
                round_number=round_num,
                role=ExpertRole.ADVOCATE,
                response=advocate_rebuttal,
                timestamp=time.perf_counter()
            ))
            token_usage["advocate"] += advocate_rebuttal.tokens_used
            
            self._log(f"   ✓ Advocate rebutted ({advocate_rebuttal.tokens_used} tokens, {advocate_rebuttal.latency:.2f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Round 4: Synthesizer's Integration
            # ═══════════════════════════════════════════════════════════════
            round_num = 4
            self._log(f"\n🔗 Round {round_num}: Synthesizer integrating perspectives...")
            
            synthesizer_response = self.synthesizer.synthesize(
                question, full_context,
                advocate_response.content,
                critic_response.content,
                advocate_rebuttal.content
            )
            transcript.append(DebateRound(
                round_number=round_num,
                role=ExpertRole.SYNTHESIZER,
                response=synthesizer_response,
                timestamp=time.perf_counter()
            ))
            token_usage["synthesizer"] += synthesizer_response.tokens_used
            
            self._log(f"   ✓ Synthesizer responded ({synthesizer_response.tokens_used} tokens, {synthesizer_response.latency:.2f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Round 5: Judge's Final Verdict
            # ═══════════════════════════════════════════════════════════════
            round_num = 5
            self._log(f"\n⚖️ Round {round_num}: Judge making final determination...")
            
            judge_response = self.judge.judge(
                question, full_context,
                advocate_response.content,
                critic_response.content,
                advocate_rebuttal.content,
                synthesizer_response.content
            )
            transcript.append(DebateRound(
                round_number=round_num,
                role=ExpertRole.JUDGE,
                response=judge_response,
                timestamp=time.perf_counter()
            ))
            token_usage["judge"] += judge_response.tokens_used
            
            self._log(f"   ✓ Judge responded ({judge_response.tokens_used} tokens, {judge_response.latency:.2f}s)")
            
            # ═══════════════════════════════════════════════════════════════
            # Extract Final Results
            # ═══════════════════════════════════════════════════════════════
            final_answer = self.judge.get_final_answer(judge_response)
            confidence = self.judge.get_confidence(judge_response)
            
            # Extract dissenting view if present
            dissenting_view = judge_response.sections.get("DISSENTING_VIEW")
            
            # Determine consensus
            consensus_reached = confidence >= self.config.consensus_threshold
            
            total_latency = time.perf_counter() - start_time
            
            self._log(f"\n✅ Debate completed in {total_latency:.2f}s")
            self._log(f"   Final confidence: {confidence:.2f}")
            self._log(f"   Total tokens: {sum(token_usage.values())}")
            
            return DebateResult(
                final_answer=final_answer,
                confidence=confidence,
                status=DebateStatus.COMPLETED,
                consensus_reached=consensus_reached,
                total_rounds=round_num,
                debate_transcript=transcript,
                winning_arguments=self._extract_winning_arguments(judge_response),
                dissenting_view=dissenting_view,
                token_usage=token_usage,
                total_latency=total_latency
            )
            
        except Exception as e:
            self._log(f"\n❌ Debate error at round {round_num}: {str(e)}")
            
            # Return partial result on error
            return DebateResult(
                final_answer=f"Error during debate: {str(e)}",
                confidence=0.0,
                status=DebateStatus.ERROR,
                consensus_reached=False,
                total_rounds=round_num,
                debate_transcript=transcript,
                token_usage=token_usage,
                total_latency=time.perf_counter() - start_time
            )
    
    def _extract_winning_arguments(self, judge_response: ExpertResponse) -> List[str]:
        """Extract the key winning arguments from the Judge's response."""
        arguments = []
        
        # Try to extract from KEY_EVIDENCE section
        if "KEY_EVIDENCE" in judge_response.sections:
            evidence = judge_response.sections["KEY_EVIDENCE"]
            # Split by common delimiters
            for line in evidence.split("\n"):
                line = line.strip()
                if line and not line.startswith("-"):
                    arguments.append(line)
                elif line.startswith("-"):
                    arguments.append(line[1:].strip())
        
        return arguments[:5]  # Limit to top 5


def create_debate_engine(
    advocate_model: str = "gpt-4o",
    critic_model: str = "gpt-4o",
    synthesizer_model: str = "gpt-4o",
    judge_model: str = "gpt-4o",
    log_callback: Optional[Callable[[str], None]] = None,
    verbose: bool = True
) -> DebateMoE:
    """
    Convenience function to create a debate engine with custom models.
    
    Args:
        advocate_model: Model for the Advocate expert
        critic_model: Model for the Critic expert
        synthesizer_model: Model for the Synthesizer expert
        judge_model: Model for the Judge expert
        log_callback: Optional callback for logging
        verbose: Whether to print debug logs
        
    Returns:
        Configured DebateMoE instance
    """
    config = DebateConfig(
        advocate_model=advocate_model,
        critic_model=critic_model,
        synthesizer_model=synthesizer_model,
        judge_model=judge_model,
        log_callback=log_callback,
        verbose=verbose
    )
    return DebateMoE(config)


# Quick test function
def quick_test():
    """Quick test of the debate engine."""
    engine = create_debate_engine(verbose=True)
    
    result = engine.debate(
        question="What is the role of dopamine in reward processing?",
        context="""
        Knowledge Graph Context:
        - Dopamine is a neurotransmitter
        - Dopamine MODULATES reward_system (evidence: 5 papers)
        - VTA (ventral tegmental area) PRODUCES dopamine
        - Dopamine AFFECTS motivation and pleasure
        """
    )
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result.final_answer)
    print(f"\nConfidence: {result.confidence:.2f}")
    print(f"Status: {result.status.value}")


if __name__ == "__main__":
    quick_test()
