"""
Prompt Templates for Debate MoE Experts

Each expert role has a specific system prompt and response format designed
for neuroscience QA tasks. The prompts are designed to:
1. Encourage evidence-based reasoning
2. Maintain focus on the provided context
3. Enable structured debate between experts
"""

from typing import Dict, List, Optional


# =============================================================================
# System Prompts for Each Expert Role
# =============================================================================

ADVOCATE_SYSTEM_PROMPT = """You are the ADVOCATE expert in a neuroscience debate panel.

Your role is to:
1. Provide a clear, well-reasoned answer to the question based on the provided context
2. Support your answer with specific evidence from the context (cite relevant concepts, relationships, or paper findings)
3. Build a strong case for your position
4. Be confident but open to refinement based on valid criticism

Guidelines:
- Always ground your answer in the provided knowledge graph context and literature
- When multiple interpretations exist, choose the most well-supported one
- Clearly state your reasoning chain
- Acknowledge when evidence is indirect or limited

Response format:
[POSITION]: State your clear answer/position
[EVIDENCE]: List the key evidence supporting your position
[REASONING]: Explain your reasoning chain
[CONFIDENCE]: Rate your confidence (High/Medium/Low) and explain why"""


CRITIC_SYSTEM_PROMPT = """You are the CRITIC expert in a neuroscience debate panel.

Your role is to:
1. Carefully examine the Advocate's answer for weaknesses
2. Identify logical gaps, unsupported claims, or missing evidence
3. Challenge assumptions that may not hold
4. Propose alternative interpretations if the evidence supports them

Guidelines:
- Be constructive, not adversarial - your goal is to improve the answer quality
- Focus on substantive issues, not minor details
- If the Advocate's answer is well-supported, acknowledge its strengths before noting weaknesses
- Consider whether the context supports alternative conclusions

Response format:
[STRENGTHS]: What the Advocate got right
[WEAKNESSES]: Logical gaps, unsupported claims, or missing considerations
[CHALLENGES]: Specific questions or challenges for the Advocate
[ALTERNATIVES]: Alternative interpretations worth considering (if any)"""


SYNTHESIZER_SYSTEM_PROMPT = """You are the SYNTHESIZER expert in a neuroscience debate panel.

Your role is to:
1. Review the debate between Advocate and Critic
2. Identify points of agreement and disagreement
3. Synthesize a balanced perspective that incorporates valid points from both sides
4. Highlight remaining uncertainties or areas needing further evidence

Guidelines:
- Weigh the strength of arguments from both sides fairly
- Don't simply split the difference - evaluate which arguments are better supported
- Integrate insights from both perspectives where possible
- Clearly distinguish between what is well-established vs. uncertain

Response format:
[CONSENSUS]: Points where both sides agree
[RESOLUTION]: How to resolve the key disagreements
[SYNTHESIS]: Your integrated perspective
[UNCERTAINTIES]: What remains uncertain or debatable"""


JUDGE_SYSTEM_PROMPT = """You are the JUDGE expert in a neuroscience debate panel.

Your role is to:
1. Evaluate the entire debate and synthesized perspective
2. Determine the best answer based on evidence quality and reasoning strength
3. Provide a final, authoritative answer to the original question
4. Assign a confidence score based on the strength of the supporting evidence

Guidelines:
- Base your judgment primarily on evidence quality, not rhetorical skill
- Consider how well each argument is grounded in the provided context
- If the debate revealed genuine uncertainty, reflect this in your confidence score
- Provide a clear, actionable answer that addresses the original question

Response format:
[FINAL_ANSWER]: Your definitive answer to the question
[KEY_EVIDENCE]: The most important evidence supporting this answer
[CONFIDENCE_SCORE]: A score from 0.0 to 1.0 with justification
[DISSENTING_VIEW]: Note any valid alternative perspective that couldn't be ruled out"""


# =============================================================================
# Turn-specific Prompts
# =============================================================================

def get_advocate_initial_prompt(question: str, context: str) -> str:
    """Generate the initial prompt for the Advocate."""
    return f"""Based on the following neuroscience knowledge context, answer the question.

## Knowledge Context
{context}

## Question
{question}

Provide your answer following the response format in your instructions."""


def get_critic_prompt(question: str, context: str, advocate_response: str) -> str:
    """Generate the prompt for the Critic to challenge the Advocate."""
    return f"""Review the Advocate's answer to this neuroscience question and provide your critique.

## Knowledge Context
{context}

## Question
{question}

## Advocate's Response
{advocate_response}

Analyze this response and provide your critique following your response format."""


def get_advocate_rebuttal_prompt(
    question: str, 
    context: str, 
    previous_response: str,
    critic_response: str
) -> str:
    """Generate the prompt for the Advocate to respond to criticism."""
    return f"""The Critic has challenged your answer. Review their critique and respond.

## Knowledge Context
{context}

## Question
{question}

## Your Previous Response
{previous_response}

## Critic's Challenges
{critic_response}

Respond to the valid criticisms, defend your well-supported points, and revise your answer if needed.

Updated response format:
[ACKNOWLEDGMENT]: Valid points from the Critic
[DEFENSE]: Defense of your well-supported claims
[REVISION]: Any revisions to your original answer
[UPDATED_POSITION]: Your refined position after considering the critique"""


def get_synthesizer_prompt(
    question: str,
    context: str,
    advocate_response: str,
    critic_response: str,
    advocate_rebuttal: str
) -> str:
    """Generate the prompt for the Synthesizer."""
    return f"""Synthesize the debate between Advocate and Critic on this neuroscience question.

## Knowledge Context
{context}

## Question
{question}

## Debate History

### Advocate's Initial Response
{advocate_response}

### Critic's Challenges
{critic_response}

### Advocate's Rebuttal
{advocate_rebuttal}

Synthesize these perspectives following your response format."""


def get_judge_prompt(
    question: str,
    context: str,
    advocate_response: str,
    critic_response: str,
    advocate_rebuttal: str,
    synthesizer_response: str
) -> str:
    """Generate the prompt for the Judge to make final determination."""
    return f"""As the final Judge, evaluate this neuroscience debate and provide the definitive answer.

## Knowledge Context
{context}

## Original Question
{question}

## Full Debate Transcript

### Round 1: Advocate's Initial Position
{advocate_response}

### Round 2: Critic's Challenge
{critic_response}

### Round 3: Advocate's Rebuttal
{advocate_rebuttal}

### Round 4: Synthesizer's Integration
{synthesizer_response}

Based on the entire debate, provide your final judgment following your response format."""


# =============================================================================
# Utility Functions
# =============================================================================

def extract_section(response: str, section_name: str) -> Optional[str]:
    """
    Extract a specific section from a structured response.
    
    Args:
        response: Full response text
        section_name: Section to extract (e.g., "FINAL_ANSWER", "CONFIDENCE_SCORE")
        
    Returns:
        Content of the section, or None if not found
    """
    import re
    
    # Match [SECTION_NAME]: content until next [SECTION] or end
    pattern = rf'\[{section_name}\]:\s*(.*?)(?=\n\[|$)'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return None


def parse_confidence_score(response: str) -> float:
    """
    Extract confidence score from Judge's response.
    
    Args:
        response: Judge's full response
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    import re
    
    # Try to extract from [CONFIDENCE_SCORE] section
    section = extract_section(response, "CONFIDENCE_SCORE")
    if section:
        # Look for a number like 0.8, 0.85, 80%, etc.
        match = re.search(r'(\d+\.?\d*)', section)
        if match:
            score = float(match.group(1))
            # Normalize if given as percentage
            if score > 1:
                score = score / 100
            return min(1.0, max(0.0, score))
    
    # Fallback: look for confidence indicators in text
    response_lower = response.lower()
    if "high confidence" in response_lower:
        return 0.85
    elif "medium confidence" in response_lower:
        return 0.65
    elif "low confidence" in response_lower:
        return 0.45
    
    # Default
    return 0.5


def format_debate_summary(
    question: str,
    advocate_response: str,
    critic_response: str,
    synthesizer_response: str,
    judge_response: str,
    final_answer: str,
    confidence: float
) -> str:
    """
    Format the complete debate into a summary for logging/audit.
    """
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DEBATE MOE SUMMARY                            ║
╚══════════════════════════════════════════════════════════════════╝

QUESTION: {question}

────────────────────────────────────────────────────────────────────
ROUND 1 - ADVOCATE'S INITIAL POSITION
────────────────────────────────────────────────────────────────────
{advocate_response[:500]}{'...' if len(advocate_response) > 500 else ''}

────────────────────────────────────────────────────────────────────
ROUND 2 - CRITIC'S CHALLENGE
────────────────────────────────────────────────────────────────────
{critic_response[:500]}{'...' if len(critic_response) > 500 else ''}

────────────────────────────────────────────────────────────────────
ROUND 3 - SYNTHESIZER'S INTEGRATION
────────────────────────────────────────────────────────────────────
{synthesizer_response[:500]}{'...' if len(synthesizer_response) > 500 else ''}

────────────────────────────────────────────────────────────────────
ROUND 4 - JUDGE'S VERDICT
────────────────────────────────────────────────────────────────────
{judge_response[:500]}{'...' if len(judge_response) > 500 else ''}

════════════════════════════════════════════════════════════════════
FINAL ANSWER (Confidence: {confidence:.2f})
════════════════════════════════════════════════════════════════════
{final_answer}
"""
