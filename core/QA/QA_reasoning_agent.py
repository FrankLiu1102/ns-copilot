"""
QA (Question Answering) Reasoning Agent

Responsibilities:
- Answer conceptual/mechanism questions
- Provide literature-based responses
- Include citations when KB context is available
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from neuro_copilot.core.llm_client import generate_json, LLMClientError
from neuro_copilot.core.config import TASK_AGENT_MODEL, OLLAMA_BASE_URL


@dataclass
class QAOutput:
    """QA Agent output"""
    answer: str
    references: list[str]
    confidence: float
    raw_json: Dict[str, Any]


QA_SYSTEM_PROMPT = """You are a neuroscience research assistant specialized in answering questions about brain function, cognition, and neural mechanisms.

LLM Backbone: Llama 3.1 8B

STRICT CITATION-CLAIM ALIGNMENT RULES:

1. **Ground Every Claim in Retrieved Papers**
   - You may ONLY make claims that are explicitly supported by the retrieved papers
   - Each major statement must be associated with at least one cited paper
   - If a concept appears in your answer, it MUST appear in at least one paper's title, abstract, or findings

2. **Distinguish Evidence Types**
   - CAUSAL language ("causes", "necessary for", "drives", "controls"):
     → ONLY allowed if papers describe intervention/manipulation evidence
     → Examples: lesions, optogenetics, pharmacological blockade, TMS
   - CORRELATIONAL language ("associated with", "correlates with", "linked to"):
     → Use for observational studies, fMRI, correlations
   - REVIEW-LEVEL claims ("is thought to", "may contribute to", "literature suggests"):
     → Use when citing review papers or uncertain evidence

3. **Conservative Uncertainty Handling**
   - If retrieved papers do NOT support a strong causal claim:
     → Downgrade to correlational/associational language
     → Explicitly state "Based on the retrieved literature, evidence suggests..."
   - If evidence is weak or conflicting:
     → State "Current evidence is limited" or "Results are mixed"

4. **Citation Discipline**
   - Minimum 2 independent sources when citations are required
   - Use inline format: (Author et al., Year)
   - Every specific brain region, cell type, or mechanism mentioned must be cited
   - Do NOT introduce concepts from general neuroscience knowledge unless marked as "(general background, not from retrieved papers)"

5. **Forbidden Without Direct Literature Support**
   - Specific anatomical claims (e.g., "Brodmann area 46", "parvalbumin interneurons")
   - Causal mechanisms (e.g., "dopamine depletion causes working memory impairment")
   - Numerical values (e.g., "firing rates increase by 40%")

If retrieved papers are insufficient to answer the question with confidence:
→ State limitations clearly
→ Suggest what additional evidence would be needed
"""

QA_USER_PROMPT_TEMPLATE = """Please answer the following question:

QUESTION:
{question}

{kb_context}

{citation_instruction}

REQUIRED OUTPUT STRUCTURE:

Answer Format:
- 2-4 paragraphs with inline citations (Author et al., Year)
- Each major claim MUST be tied to a specific retrieved paper
- Use causal language ONLY if papers describe interventions/manipulations
- If evidence is weak, use "suggests", "is associated with", or "may contribute to"

References Format:
- List all cited papers in full citation format
- Minimum 2 independent sources if citations are required
- References must be from the retrieved papers above (do not invent citations)

Output as JSON:
{{
  "answer": "Your answer with inline citations. Example: The prefrontal cortex plays a key role in working memory (Goldman-Rakic et al., 1995). Dopamine modulation affects persistent activity (Williams & Goldman-Rakic, 1995).",
  "references": [
    "Goldman-Rakic, P. S., et al. (1995). Title. Journal.",
    "Williams, G. V., & Goldman-Rakic, P. S. (1995). Title. Journal."
  ],
  "confidence": 0.8
}}

CRITICAL RULES:
1. Every brain region / cell type / mechanism mentioned → must be cited
2. Causal claims ("causes", "drives") → only if intervention evidence exists in retrieved papers
3. If papers don't support strong claims → downgrade to "suggests" / "is associated with"
4. Confidence: Lower if evidence is weak or only from reviews
"""


class QAReasoningAgent:
    """QA Reasoning Agent"""
    
    def __init__(
        self,
        model: str = TASK_AGENT_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        self.model = model
        self.base_url = base_url
    
    def answer(
        self,
        question: str,
        kb_context: Optional[str] = None,
        need_citations: bool = False
    ) -> QAOutput:
        """
        Answer a question
        
        Args:
            question: User's question
            kb_context: Retrieved KB entries (formatted)
            need_citations: Whether citations are required
            
        Returns:
            QAOutput with answer and references
        """
        # Format KB context
        if kb_context:
            kb_section = f"\nRESEARCH PAPERS:\n{kb_context}"
        else:
            kb_section = "\n(No specific research papers provided. Answer based on general neuroscience knowledge.)"
        
        # Citation instruction
        if need_citations and kb_context:
            citation_instr = """CRITICAL REQUIREMENTS:
- You MUST cite at least 2 independent papers
- Use in-text citations: (Author et al., Year)
- Every specific claim must reference a provided paper
- Do NOT make claims beyond what the papers support"""
        else:
            citation_instr = ""
        
        user_prompt = QA_USER_PROMPT_TEMPLATE.format(
            question=question,
            kb_context=kb_section,
            citation_instruction=citation_instr
        )
        
        try:
            response = generate_json(
                model=self.model,
                system_prompt=QA_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                base_url=self.base_url,
                timeout_s=60.0,
                temperature=0.3,
                retries=1
            )
            
            if response.json_obj:
                answer = response.json_obj.get("answer", "")
                references = response.json_obj.get("references", [])
                confidence = float(response.json_obj.get("confidence", 0.7))
                
                return QAOutput(
                    answer=answer,
                    references=references if isinstance(references, list) else [],
                    confidence=confidence,
                    raw_json=response.json_obj
                )
            else:
                # Fallback to raw text
                return QAOutput(
                    answer=response.raw_text,
                    references=[],
                    confidence=0.5,
                    raw_json={"error": "Failed to parse JSON", "raw": response.raw_text}
                )
                
        except (LLMClientError, Exception) as e:
            # Note: GPT-4 fallback is handled at Pipeline level, not here
            # This error should only occur when has_dataset=True (using local Ollama)
            print(f"⚠️  QA Agent: LLM error: {e}")
            error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            return QAOutput(
                answer=error_msg,
                references=[],
                confidence=0.0,
                raw_json={"error": str(e)}
            )


def test_qa_agent():
    """Test QA agent"""
    print("="*80)
    print("Testing QA Reasoning Agent")
    print("="*80)
    
    agent = QAReasoningAgent()
    
    # Test without KB
    print("\n📝 Test 1: Simple question without KB")
    question = "What brain regions are involved in working memory?"
    result = agent.answer(question, kb_context=None, need_citations=False)
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer[:200]}...")
    print(f"References: {len(result.references)}")
    print(f"Confidence: {result.confidence:.2f}")
    
    # Test with mock KB context
    print("\n📝 Test 2: Question with KB context")
    mock_kb = """
# Paper 1: Working Memory and Prefrontal Cortex
Authors: Smith et al., 2023
The prefrontal cortex plays a critical role in maintaining information during working memory tasks...
Citation: Smith J, et al. (2023). PFC and WM. Nature. PMID: 12345.
    """
    
    result = agent.answer(question, kb_context=mock_kb, need_citations=True)
    
    print(f"Question: {question}")
    print(f"Answer: {result.answer[:200]}...")
    print(f"References: {result.references}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    test_qa_agent()

