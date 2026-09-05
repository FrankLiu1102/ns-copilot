"""
ER (Experiment Recommendation) Reasoning Agent

Responsibilities:
- Recommend experimental designs
- Suggest control conditions and sample sizes
- Provide methodological guidance
- Include citations to relevant experimental papers
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from neuro_copilot.core.llm_client import generate_json, LLMClientError
from neuro_copilot.core.config import TASK_AGENT_MODEL, OLLAMA_BASE_URL


@dataclass
class EROutput:
    """ER Agent output"""
    recommendation: str
    references: list[str]
    confidence: float
    raw_json: Dict[str, Any]


ER_SYSTEM_PROMPT = """You are a neuroscience research methodologist specialized in experimental design.

LLM Backbone: Llama 3.1 8B

STRICT EXPERIMENTAL DESIGN & CITATION RULES:

1. **Method-Evidence Alignment**
   - Every proposed manipulation/technique MUST be grounded in retrieved papers
   - If you propose "optogenetics" → must cite papers using optogenetics
   - If you propose "6-OHDA lesions" → must cite papers using 6-OHDA in the same species
   - If you propose "fMRI" → must cite papers using fMRI for similar questions
   - Do NOT recommend methods not demonstrated in the retrieved literature

2. **Causal Manipulation Requirements**
   - If proposing causal intervention (TMS, optogenetics, lesions, drugs, DBS):
     → Requires ≥2 references using that intervention
     → References must be from species-appropriate studies
     → If only review papers available → downgrade to "observational design" or flag insufficient evidence
   
3. **Species-Method Consistency**
   - Cell-type–specific claims (e.g., "target parvalbumin interneurons"):
     → Requires cellular-resolution evidence (electrophysiology, 2-photon, optogenetics)
     → Human fMRI/review papers do NOT support cell-type claims
   - Human studies can support:
     → Systems-level manipulations (TMS, tDCS, pharmacology)
     → Brain region associations (fMRI, EEG)
   - Animal studies can support:
     → Causal cellular mechanisms
     → Circuit-level manipulations

4. **Citation Discipline**
   - Every section (Hypothesis / Methods / Controls / Analysis) must reference literature
   - Minimum 2 independent sources for causal designs
   - Use inline format: (Author et al., Year)
   - If retrieved papers are insufficient → state "Evidence base is limited; recommend pilot study"

5. **Conservative Fallback**
   - If causal manipulation lacks support in retrieved papers:
     → Switch to correlational/observational design
     → Explicitly state: "Given current evidence, we recommend an observational approach..."
   - If species mismatch (e.g., want mouse circuit but only have human fMRI papers):
     → Flag mismatch: "Retrieved papers focus on human imaging; mouse circuit evidence is needed"

6. **Forbidden Without Direct Literature Support**
   - Specific surgical procedures not demonstrated in papers
   - Statistical power calculations not grounded in cited effect sizes
   - Analysis methods not used in similar studies

Provide structured recommendations:
- Research Question & Hypothesis (with citations)
- Experimental Design (with citations)
- Control Conditions (with citations)
- Sample Size (with citations if possible)
- Methods & Procedures (with citations)
- Expected Outcomes & Analysis (with citations)
- References (full citations from retrieved papers)
"""

ER_USER_PROMPT_TEMPLATE = """Please recommend an experimental design for the following request:

REQUEST:
{request}

{kb_context}

{citation_instruction}

REQUIRED OUTPUT STRUCTURE:

Recommendation Format (each section with inline citations):

1. Research Question & Hypothesis
   - State hypothesis clearly
   - Cite papers supporting the theoretical background (Author et al., Year)

2. Experimental Design
   - Between/within-subjects, groups, conditions
   - Cite papers using similar designs (Author et al., Year)

3. Control Conditions
   - Specify controls
   - Cite papers justifying control choices

4. Sample Size & Power
   - Estimate sample size
   - Cite effect sizes from literature if available

5. Methods & Procedures
   - Every proposed technique/manipulation must be cited
   - If proposing causal intervention → must cite ≥2 papers using that intervention
   - Ensure species-method consistency

6. Expected Outcomes & Analysis
   - Statistical approach
   - Cite similar analysis methods from papers

References Format:
- List all cited papers in full citation format
- Minimum 2 independent sources for causal designs
- Must be from retrieved papers (do not invent citations)

Output as JSON:
{{
  "recommendation": "Structured experimental design with sections as above. Example: Research Question: We hypothesize dopamine depletion impairs working memory (Cools et al., 2001). Design: We will use a 6-OHDA lesion model (Smith et al., 2020; Jones et al., 2021)...",
  "references": [
    "Cools, R., et al. (2001). Title. Journal.",
    "Smith, A., et al. (2020). Title. Journal."
  ],
  "confidence": 0.8
}}

CRITICAL RULES:
1. Every proposed method → must be demonstrated in retrieved papers
2. Causal manipulation → requires ≥2 supporting references with intervention evidence
3. If retrieved papers insufficient → downgrade to observational design OR flag limitation
4. Species-method consistency → cell-type claims need cellular evidence, not fMRI reviews
5. Confidence: Lower if evidence base is thin or only reviews available
"""


class ERReasoningAgent:
    """ER Reasoning Agent"""
    
    def __init__(
        self,
        model: str = TASK_AGENT_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        self.model = model
        self.base_url = base_url
    
    def recommend(
        self,
        request: str,
        kb_context: Optional[str] = None,
        need_citations: bool = False
    ) -> EROutput:
        """
        Recommend experimental design
        
        Args:
            request: User's experiment request
            kb_context: Retrieved KB entries (formatted)
            need_citations: Whether citations are required
            
        Returns:
            EROutput with recommendation and references
        """
        # Format KB context
        if kb_context:
            kb_section = f"\nRELEVANT EXPERIMENTAL PAPERS:\n{kb_context}"
        else:
            kb_section = "\n(No specific experimental papers provided. Recommend based on general methodological principles.)"
        
        # Citation instruction
        if need_citations and kb_context:
            citation_instr = """CRITICAL REQUIREMENTS:
- You MUST cite at least 2 independent papers
- Causal manipulations (rTMS, optogenetics, lesions, drugs) require ≥2 supporting references
- Aging/disease populations require ≥2 supporting references
- Ensure method-species consistency (e.g., cell-type claims need cellular evidence, not just fMRI)
- Use in-text citations: (Author et al., Year)"""
        else:
            citation_instr = ""
        
        user_prompt = ER_USER_PROMPT_TEMPLATE.format(
            request=request,
            kb_context=kb_section,
            citation_instruction=citation_instr
        )
        
        try:
            response = generate_json(
                model=self.model,
                system_prompt=ER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                base_url=self.base_url,
                timeout_s=60.0,
                temperature=0.3,
                retries=1
            )
            
            if response.json_obj:
                recommendation = response.json_obj.get("recommendation", "")
                references = response.json_obj.get("references", [])
                confidence = float(response.json_obj.get("confidence", 0.7))
                
                return EROutput(
                    recommendation=recommendation,
                    references=references if isinstance(references, list) else [],
                    confidence=confidence,
                    raw_json=response.json_obj
                )
            else:
                # Fallback to raw text
                return EROutput(
                    recommendation=response.raw_text,
                    references=[],
                    confidence=0.5,
                    raw_json={"error": "Failed to parse JSON", "raw": response.raw_text}
                )
                
        except (LLMClientError, Exception) as e:
            error_msg = f"I apologize, but I encountered an error while generating recommendations: {str(e)}"
            return EROutput(
                recommendation=error_msg,
                references=[],
                confidence=0.0,
                raw_json={"error": str(e)}
            )


def test_er_agent():
    """Test ER agent"""
    print("="*80)
    print("Testing ER Reasoning Agent")
    print("="*80)
    
    agent = ERReasoningAgent()
    
    # Test without KB
    print("\n📝 Test 1: Simple request without KB")
    request = "How do I design an experiment to test working memory capacity in mice?"
    result = agent.recommend(request, kb_context=None, need_citations=False)
    
    print(f"Request: {request}")
    print(f"Recommendation: {result.recommendation[:300]}...")
    print(f"References: {len(result.references)}")
    print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    test_er_agent()

