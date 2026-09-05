"""
Router Agent - Task Classification and Keyword Extraction

Responsibilities:
1. Classify task type (QA / DA / ER)
2. Extract keywords for KB search
3. Extract basic constraints (e.g., citations required)
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from neuro_copilot.core.llm_client import generate_json, LLMClientError
from neuro_copilot.core.config import ROUTER_MODEL


@dataclass
class RouterOutput:
    """Router classification output"""
    task_type: str  # QA | DA | ER
    domain: str  # working_memory | parkinson | alzheimer | unknown
    keywords: List[str]
    constraints: Dict[str, Any]
    confidence: float
    raw_json: Dict[str, Any]


# Router system prompt — disabled (DA-only mode, no LLM routing or domain classification)
ROUTER_SYSTEM_PROMPT = """You are a task classification router. Classify as DA (Data Analysis)."""

ROUTER_USER_PROMPT_TEMPLATE = """Classify this user request:

USER REQUEST:
{user_idea}

DATASET PROVIDED: {has_dataset}

Output ONLY a JSON object with this structure:
{{
  "task_type": "QA|DA|ER",
  "domain": "working_memory|parkinson|alzheimer|unknown",
  "keywords": ["keyword1", "keyword2", ...],
  "constraints": {{
    "need_citations": true
  }},
  "confidence": 0.0
}}

Reasoning:
- Check if request matches DA patterns (analysis keywords + dataset)
- Check if request matches ER patterns (design/protocol keywords)
- Otherwise → QA
- Identify domain based on keywords (parkinson/alzheimer/working_memory)
- If no clear domain or multiple domains → unknown
- Extract 3-5 relevant scientific keywords
- Set need_citations=true if user wants references"""


class RouterAgent:
    """Router Agent for task classification"""
    
    def __init__(
        self,
        model: str = ROUTER_MODEL
    ):
        self.model = model
    
    def _rule_based_fallback(self, user_idea: str, has_dataset: bool) -> RouterOutput:
        """Rule-based classification as fallback"""
        idea_lower = user_idea.lower()
        
        # DA keywords
        da_keywords = [
            'pca', 'decode', 'decoding', 'encode', 'encoding', 'accuracy', 
            'classifier', 'regression', 'train', 'test', 'dimensionality',
            'model', 'predict', 'correlation', 'glm', 'anova'
        ]
        
        # ER keywords
        er_keywords = [
            'experiment', 'design', 'control', 'sample size', 'protocol',
            'intervention', 'study', 'procedure', 'method', 'setup',
            'how to conduct', 'how to run'
        ]
        
        # Domain keywords — disabled to avoid dataset-specific routing
        # parkinson_keywords = [
        #     'parkinson', 'pd', 'dopamine', 'dopaminergic', 'substantia nigra',
        #     'basal ganglia', 'striatum', 'alpha-synuclein', 'l-dopa', 'levodopa', 'updrs'
        # ]
        # alzheimer_keywords = [
        #     'alzheimer', 'ad', 'amyloid', 'beta-amyloid', 'tau',
        #     'hippocampus', 'dementia', 'memory loss'
        # ]
        # wm_keywords = [
        #     'working memory', 'wm', 'pfc', 'dlpfc', 'prefrontal cortex',
        #     'odr', 'delay period', 'persistent activity', 'spatial memory', 'short-term memory'
        # ]
        
        # Count matches for task type
        da_count = sum(1 for kw in da_keywords if kw in idea_lower)
        er_count = sum(1 for kw in er_keywords if kw in idea_lower)
        
        # Classify task type
        if da_count > 0 and has_dataset:
            task_type = "DA"
        elif er_count >= 2:
            task_type = "ER"
        else:
            task_type = "QA"
        
        # Domain classification disabled — no dataset-specific routing
        domain = "unknown"
        
        # Extract keywords (simple tokenization)
        words = user_idea.lower().split()
        important_words = [
            w for w in words 
            if len(w) > 4 and w not in {'about', 'would', 'could', 'should', 'their', 'there'}
        ]
        keywords = important_words[:5] if important_words else ["neuroscience"]
        
        # Check if citations needed
        need_citations = any(phrase in idea_lower for phrase in [
            'paper', 'reference', 'study', 'citation', 'literature', 'research'
        ])
        
        return RouterOutput(
            task_type=task_type,
            domain=domain,
            keywords=keywords,
            constraints={"need_citations": need_citations},
            confidence=0.5,
            raw_json={
                "task_type": task_type,
                "domain": domain,
                "keywords": keywords,
                "constraints": {"need_citations": need_citations},
                "confidence": 0.5,
                "fallback": True
            }
        )
    
    def route(self, user_idea: str, has_dataset: bool = False) -> RouterOutput:
        """
        Route user request to appropriate task type
        
        Args:
            user_idea: User's question/request
            has_dataset: Whether dataset is provided
            
        Returns:
            RouterOutput with classification and extracted info
        """
        # Strategy: When no dataset, skip Ollama and use rule-based classification
        # This avoids connection errors and is fast enough for routing
        if not has_dataset:
            print("⚠️  Router: No dataset provided, using rule-based classification (skip Ollama)")
            return self._rule_based_fallback(user_idea, has_dataset)
        
        user_prompt = ROUTER_USER_PROMPT_TEMPLATE.format(
            user_idea=user_idea,
            has_dataset="YES" if has_dataset else "NO"
        )
        
        try:
            # Try LLM classification (only when has_dataset=True)
            response = generate_json(
                model=self.model,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_s=30.0,
                temperature=0.2,
                retries=1
            )
            
            if response.json_obj:
                # Validate and normalize task_type
                task_type = response.json_obj.get("task_type", "QA").upper()
                if task_type not in ["QA", "DA", "ER"]:
                    task_type = "QA"
                
                # Validate and normalize domain
                domain = response.json_obj.get("domain", "unknown").lower()
                if domain not in ["working_memory", "parkinson", "alzheimer", "unknown"]:
                    domain = "unknown"
                
                keywords = response.json_obj.get("keywords", [])
                if not isinstance(keywords, list):
                    keywords = []
                keywords = [str(k) for k in keywords if k]
                
                constraints = response.json_obj.get("constraints", {})
                if not isinstance(constraints, dict):
                    constraints = {}
                if "need_citations" not in constraints:
                    constraints["need_citations"] = False
                
                confidence = float(response.json_obj.get("confidence", 0.7))
                
                return RouterOutput(
                    task_type=task_type,
                    domain=domain,
                    keywords=keywords,
                    constraints=constraints,
                    confidence=confidence,
                    raw_json=response.json_obj
                )
            else:
                # JSON parsing failed, retry once
                print("⚠️  Router: First attempt failed to parse JSON, retrying...")
                
                response = generate_json(
                    model=self.model,
                    system_prompt=ROUTER_SYSTEM_PROMPT,
                    user_prompt=user_prompt + "\n\nREMINDER: Output ONLY valid JSON, no other text.",
                    timeout_s=30.0,
                    temperature=0.1,
                    retries=0
                )
                
                if response.json_obj:
                    task_type = response.json_obj.get("task_type", "QA").upper()
                    if task_type not in ["QA", "DA", "ER"]:
                        task_type = "QA"
                    
                    domain = response.json_obj.get("domain", "unknown").lower()
                    if domain not in ["working_memory", "parkinson", "alzheimer", "unknown"]:
                        domain = "unknown"
                    
                    keywords = response.json_obj.get("keywords", [])
                    if not isinstance(keywords, list):
                        keywords = []
                    
                    constraints = response.json_obj.get("constraints", {})
                    if not isinstance(constraints, dict):
                        constraints = {}
                    
                    confidence = float(response.json_obj.get("confidence", 0.6))
                    
                    return RouterOutput(
                        task_type=task_type,
                        domain=domain,
                        keywords=keywords,
                        constraints=constraints,
                        confidence=confidence,
                        raw_json=response.json_obj
                    )
                else:
                    # Both attempts failed, use fallback
                    print("⚠️  Router: LLM parsing failed, using rule-based fallback")
                    return self._rule_based_fallback(user_idea, has_dataset)
        
        except (LLMClientError, Exception) as e:
            print(f"⚠️  Router: LLM error ({e}), using rule-based fallback")
            return self._rule_based_fallback(user_idea, has_dataset)


def test_router():
    """Test router agent"""
    print("="*80)
    print("Testing Router Agent")
    print("="*80)
    
    router = RouterAgent()
    
    test_cases = [
        ("What brain regions are involved in working memory?", False),
        ("I want to decode cue location from neural activity", True),
        ("How do I design an experiment to test working memory in mice?", False),
    ]
    
    for idea, has_ds in test_cases:
        print(f"\n📝 Input: '{idea}'")
        print(f"   Dataset: {'YES' if has_ds else 'NO'}")
        
        result = router.route(idea, has_ds)
        
        print(f"   → Task Type: {result.task_type}")
        print(f"   → Keywords: {result.keywords}")
        print(f"   → Need Citations: {result.constraints.get('need_citations')}")
        print(f"   → Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    test_router()


