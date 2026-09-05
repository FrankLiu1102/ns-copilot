"""
DA (Data Analysis) Reasoning Agent

Responsibilities:
- If NO dataset: provide methodological guidance
- If dataset provided: delegate to existing encoding/decoding workflow
- This is a SKELETON version - full implementation in next phase
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from neuro_copilot.core.llm_client import generate_json, LLMClientError
from neuro_copilot.core.config import TASK_AGENT_MODEL, OLLAMA_BASE_URL


@dataclass
class DAOutput:
    """DA Agent output"""
    response: str
    needs_dataset: bool
    analysis_type: Optional[str] = None
    references: list[str] = None
    confidence: float = 0.7
    raw_json: Dict[str, Any] = None
    # NEW: Execution diagnosis for user review
    needs_user_review: bool = False
    diagnosis_summary: str = ""
    diagnosis_recommendations: list[str] = None
    suggested_action: str = ""  # "replan" | "adjust_params" | "accept"
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
        if self.raw_json is None:
            self.raw_json = {}
        if self.diagnosis_recommendations is None:
            self.diagnosis_recommendations = []


DA_SYSTEM_PROMPT = """You are a neuroscience data analysis consultant.

When NO dataset is provided:
- Explain HOW to perform the requested analysis
- Describe required data structure and formats
- List metrics and statistical methods
- Provide step-by-step methodological guidance

When dataset IS provided (future phase):
- Will delegate to existing encoding/decoding tools
- Not implemented yet in this phase

Be specific about:
- Data requirements (format, structure, sample size)
- Analysis methods (PCA, decoding, GLM, etc.)
- Metrics and validation approaches
- Common pitfalls and best practices
"""

DA_USER_PROMPT_TEMPLATE = """The user requests the following data analysis:

REQUEST:
{request}

DATASET PROVIDED: {has_dataset}

{guidance_instruction}

Output your response as JSON:
{{
  "response": "Your methodological guidance or analysis results",
  "needs_dataset": true,
  "analysis_type": "decoding|encoding|pca|other",
  "confidence": 0.8
}}

Guidelines:
- If NO dataset: Explain methodology, data requirements, and analysis steps
- If dataset provided: In this phase, still provide guidance (execution not implemented yet)
- Be specific about data format, metrics, and procedures
"""


class DAReasoningAgent:
    """DA Reasoning Agent (Skeleton)"""
    
    def __init__(
        self,
        model: str = TASK_AGENT_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        self.model = model
        self.base_url = base_url
    
    def analyze_or_guide(
        self,
        request: str,
        has_dataset: bool = False,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> DAOutput:
        """
        Provide analysis or methodological guidance
        
        Args:
            request: User's analysis request
            has_dataset: Whether dataset is provided
            dataset_info: Dataset metadata (if provided)
            
        Returns:
            DAOutput with guidance or analysis results
        """
        # In this skeleton phase, we primarily provide guidance
        if not has_dataset:
            guidance_instr = """
Since NO dataset is provided, explain:
1. What data is needed (format, fields, structure)
2. How to perform this analysis (methods, steps)
3. What metrics to use (accuracy, R², correlation, etc.)
4. How to interpret results
5. Common considerations and best practices
"""
        else:
            guidance_instr = """
Dataset is provided. In this SKELETON phase, still provide methodological guidance.
(Full dataset execution will be implemented in next phase)

Explain how the analysis would be performed:
1. Data preprocessing steps
2. Analysis method and parameters
3. Expected outputs and metrics
4. Interpretation guidance
"""
        
        user_prompt = DA_USER_PROMPT_TEMPLATE.format(
            request=request,
            has_dataset="YES" if has_dataset else "NO",
            guidance_instruction=guidance_instr
        )
        
        try:
            response = generate_json(
                model=self.model,
                system_prompt=DA_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                base_url=self.base_url,
                timeout_s=60.0,
                temperature=0.3,
                retries=1
            )
            
            if response.json_obj:
                resp_text = response.json_obj.get("response", "")
                needs_dataset = response.json_obj.get("needs_dataset", not has_dataset)
                analysis_type = response.json_obj.get("analysis_type", "unknown")
                confidence = float(response.json_obj.get("confidence", 0.7))
                
                return DAOutput(
                    response=resp_text,
                    needs_dataset=needs_dataset,
                    analysis_type=analysis_type,
                    confidence=confidence,
                    raw_json=response.json_obj
                )
            else:
                # Fallback to raw text
                return DAOutput(
                    response=response.raw_text,
                    needs_dataset=not has_dataset,
                    analysis_type="unknown",
                    confidence=0.5,
                    raw_json={"error": "Failed to parse JSON", "raw": response.raw_text}
                )
                
        except (LLMClientError, Exception) as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            return DAOutput(
                response=error_msg,
                needs_dataset=not has_dataset,
                confidence=0.0,
                raw_json={"error": str(e)}
            )
    
    def execute_analysis(self, request: str, dataset_info: Dict[str, Any]) -> DAOutput:
        """
        Execute actual data analysis (PLACEHOLDER for next phase)
        
        In this skeleton phase, this just returns a message that execution
        will be implemented in the next phase.
        """
        return DAOutput(
            response=(
                "Dataset execution is not implemented in this skeleton phase. "
                "This will delegate to existing encoding/decoding workflow in the next phase. "
                f"For now, here's methodological guidance...\n\n"
                f"{self.analyze_or_guide(request, has_dataset=True).response}"
            ),
            needs_dataset=False,
            analysis_type="placeholder",
            confidence=0.5
        )


def test_da_agent():
    """Test DA agent"""
    print("="*80)
    print("Testing DA Reasoning Agent (Skeleton)")
    print("="*80)
    
    agent = DAReasoningAgent()
    
    # Test without dataset
    print("\n📝 Test 1: Analysis request WITHOUT dataset")
    request = "I want to decode cue location from neural activity using a classifier"
    result = agent.analyze_or_guide(request, has_dataset=False)
    
    print(f"Request: {request}")
    print(f"Response: {result.response[:300]}...")
    print(f"Needs Dataset: {result.needs_dataset}")
    print(f"Analysis Type: {result.analysis_type}")
    print(f"Confidence: {result.confidence:.2f}")
    
    # Test with dataset (skeleton)
    print("\n📝 Test 2: Analysis request WITH dataset (skeleton)")
    result = agent.analyze_or_guide(request, has_dataset=True)
    
    print(f"Request: {request}")
    print(f"Response: {result.response[:300]}...")
    print(f"Needs Dataset: {result.needs_dataset}")


if __name__ == "__main__":
    test_da_agent()





