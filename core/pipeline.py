"""
Unified Pipeline for NS-Copilot

User → Task Agent → Controller → User

This release is data-analysis (DA) only. The router, KB retrieval, and the
QA/ER task agents are not instantiated (see __init__).

Performance Optimizations:
- Fine-grained timing instrumentation
- Optional in-memory caching for controller results
- Developer mode gated logging
"""

import os
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass
from functools import lru_cache

from neuro_copilot.core.router.router_agent import RouterOutput  # Only RouterOutput needed for DA
# Disabled: QA/ER/KB imports (DA-only mode)
# from neuro_copilot.core.router.router_agent import RouterAgent
# from neuro_copilot.core.kb.kb_client import KBClient, KBEntry
# from neuro_copilot.core.kb.graph_integration import (
#     get_cached_graph_client,
#     extract_concepts_from_question,
#     build_graph_context,
# )
# from neuro_copilot.core.kb.context_fusion import ContextFusion
# from neuro_copilot.core.QA.QA_reasoning_agent import QAReasoningAgent, QAOutput
# from neuro_copilot.core.ER.ER_reasoning_agent import ERReasoningAgent, EROutput
from neuro_copilot.core.DA.DA_reasoning_agent import DAReasoningAgent, DAOutput
from neuro_copilot.core.DA.planner_agent import run_planner_agent
from neuro_copilot.core.DA.analysis_agent import (
    run_analysis_agent,
    AnalysisResult,
    AnalysisAgentConfig,
    format_analysis_for_display,
    get_visualization_paths,
)
from neuro_copilot.core.DA.schema_parser import (
    SchemaParser,
    ParsedSchema,
    parse_schema,
    build_data_summary_from_raw,
)
from neuro_copilot.core.config import ANALYSIS_MODEL
from neuro_copilot.core.DA.dataloader import load_dataset_into_state
from neuro_copilot.core.state import NeuroGlobalState
from neuro_copilot.core.controller.controller_agent import ControllerAgent, ControllerOutput, ExecutionDiagnosis
from neuro_copilot.core.DA.code_generator import CodeGenerator, GeneratedCode
from neuro_copilot.core.DA.code_executor import CodeExecutor, ExecutionResult as CodeExecutionResult
from neuro_copilot.core.config import USE_GPT4_FOR_NO_DATASET, ENABLE_DEBATE_MOE

# Debate MoE import (lazy load)
try:
    from neuro_copilot.core.reasoning import DebateMoE, DebateConfig
    from neuro_copilot.core.config import (
        DEBATE_MAX_ROUNDS,
        DEBATE_CONSENSUS_THRESHOLD,
        ADVOCATE_MODEL,
        CRITIC_MODEL,
        SYNTHESIZER_MODEL,
        JUDGE_MODEL,
        ADVOCATE_TEMPERATURE,
        CRITIC_TEMPERATURE,
        SYNTHESIZER_TEMPERATURE,
        JUDGE_TEMPERATURE,
    )
    DEBATE_MOE_AVAILABLE = True
except ImportError:
    DEBATE_MOE_AVAILABLE = False
    DebateMoE = None
    DebateConfig = None

# GPT-4 backend (lazy import)
try:
    from neuro_copilot.core.gpt4_client import GPT4Client
    GPT4_AVAILABLE = True
except ImportError:
    GPT4_AVAILABLE = False
    GPT4Client = None


@dataclass
class PipelineOutput:
    """Complete pipeline output"""
    # Final output
    response: str
    references: list[str]
    
    # Debug info
    router_output: RouterOutput
    kb_entries: list  # KBEntry type disabled in DA-only mode
    task_agent_output: Any  # QAOutput | EROutput | DAOutput
    controller_output: ControllerOutput
    
    # Metadata
    task_type: str
    domain: str  # working_memory | parkinson | alzheimer | unknown
    status: str  # PASS | RETRIEVE_MORE | REWRITE_REQUIRED | NEED_DATA
    
    # Performance metrics
    timing: Optional[Dict[str, float]] = None
    
    # Analysis Agent results (DA tasks only)
    analysis_result: Optional[AnalysisResult] = None
    visualization_paths: Optional[list[str]] = None
    

def parse_schema_from_prompt(user_idea: str) -> Optional[Dict[str, Any]]:
    """
    Parse dataset schema from user prompt if provided.
    
    This function performs RAW SCHEMA EXTRACTION - no semantic interpretation.
    The schema is stored as-is for downstream LLM tools to interpret.
    
    Looks for a 'Dataset Schema:' section in the prompt and extracts the raw
    file → column → description mappings.
    
    Example prompt section:
        Dataset Schema:
          data_file_1.csv:
            - id: Record ID
            - event_time: Event timestamp relative to start
            - category: Category label
          data_file_2.csv:
            - item_id: Item identifier
            - timestamp: Absolute timestamp in seconds
    
    Returns:
        Dict with raw file-level schema, e.g.:
        {
            "data_file_1.csv": {
                "id": "Record ID",
                "event_time": "Event timestamp relative to start",
                "category": "Category label"
            },
            "data_file_2.csv": {
                "item_id": "Item identifier",
                "timestamp": "Absolute timestamp in seconds"
            },
            "_raw_text": "Dataset Schema:\n  data_file_1.csv:\n    - id: Record ID\n..."
        }
        Or None if no schema found.
    """
    import re
    
    # Check if there's a schema section
    schema_match = re.search(r'Dataset Schema:', user_idea, re.IGNORECASE)
    if not schema_match:
        return None
    
    # Extract the raw schema text until next major section
    remaining_text = user_idea[schema_match.start():]
    
    # Find end of schema section (next major heading or end of text)
    end_match = re.search(r'\n[A-Z][a-zA-Z\s-]+:\s*\n', remaining_text[20:])  # Skip "Dataset Schema:"
    if end_match:
        raw_schema_text = remaining_text[:20 + end_match.start()]
    else:
        raw_schema_text = remaining_text
    
    schema = {"_raw_text": raw_schema_text}
    
    # Parse file-level sections
    lines = raw_schema_text.split('\n')
    current_file = None
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        
        # Check for file header (e.g., "trials.csv:" or "spike_times.csv:")
        file_match = re.match(r'^(\w+\.csv):?\s*$', stripped, re.IGNORECASE)
        if file_match:
            current_file = file_match.group(1)
            schema[current_file] = {}
            continue
        
        # Check for column definition within a file section
        if current_file:
            # Match "- column_name: description" pattern
            col_match = re.match(r'-\s*(\w+):\s*(.+)', stripped)
            if col_match:
                col_name = col_match.group(1)
                description = col_match.group(2)
                schema[current_file][col_name] = description
    
    # Remove _raw_text from count check
    file_sections = {k: v for k, v in schema.items() if k != "_raw_text"}
    
    return schema if file_sections else None


class NeuroCopilotPipeline:
    """
    Unified pipeline for all task types with performance optimizations.
    
    Features:
    - Fine-grained timing instrumentation
    - Optional in-memory caching (router, KB, controller)
    - Developer mode for verbose logging
    """
    
    def __init__(self, enable_cache: bool = True, developer_mode: bool = False, use_gpt4: bool = None):
        """
        Initialize pipeline with optional performance features
        
        Args:
            enable_cache: Enable in-memory caching for controller results
            developer_mode: Enable verbose logging and timing breakdown
            use_gpt4: Override GPT-4 backend setting (None = use config default)
        """
        # DA-only mode: only initialize components needed for DA execution
        # self.router = RouterAgent()  # Disabled: DA mode, no routing needed
        # self.kb_client = KBClient()  # Disabled: DA doesn't use KB
        # self.qa_agent = QAReasoningAgent()  # Disabled: DA only
        # self.er_agent = ERReasoningAgent()  # Disabled: DA only
        self.da_agent = DAReasoningAgent()
        self.controller = ControllerAgent()

        # Disabled for DA-only mode (kept as None to avoid AttributeError)
        self._graph_client = None
        self._context_fusion = None
        self.enable_online_search = False
        self.online_search_top_k = 0
        
        # Performance features
        self.enable_cache = enable_cache
        self.developer_mode = developer_mode
        
        # GPT-4 backend (initialized on-demand)
        self.use_gpt4 = use_gpt4 if use_gpt4 is not None else USE_GPT4_FOR_NO_DATASET
        self._gpt4_client = None
        
        # Debate MoE (initialized on-demand)
        # Note: We check this dynamically in process() to support runtime config changes
        self._debate_engine = None
        
        # Simple in-memory caches (cleared on restart)
        self._router_cache = {}  # user_idea_hash -> RouterOutput
        self._kb_cache = {}      # (domain, keywords_hash) -> list[KBEntry]
        self._controller_cache = {}  # (task_type, response_hash) -> ControllerOutput
    
    def _get_gpt4_client(self):
        """Lazy initialization of GPT-4 client"""
        if self._gpt4_client is None:
            if not GPT4_AVAILABLE:
                raise RuntimeError(
                    "GPT-4 backend not available. "
                    "Install openai package: pip install openai"
                )
            self._gpt4_client = GPT4Client()
        return self._gpt4_client
    
    def _get_graph_client(self):
        """Lazy initialization of Graph KB client"""
        if self._graph_client is None:
            self._graph_client = get_cached_graph_client()
        return self._graph_client
    
    def _get_context_fusion(self):
        """Lazy initialization of context fusion"""
        if self._context_fusion is None:
            self._context_fusion = ContextFusion(
                enable_graph=True,
                enable_online=self.enable_online_search,
                online_top_k=self.online_search_top_k,
            )
        return self._context_fusion
    
    def _get_debate_engine(self):
        """Lazy initialization of Debate MoE engine"""
        if self._debate_engine is None:
            if not DEBATE_MOE_AVAILABLE:
                raise RuntimeError(
                    "Debate MoE not available. Check reasoning module imports."
                )
            
            # Create config from environment variables
            config = DebateConfig(
                max_rounds=DEBATE_MAX_ROUNDS,
                consensus_threshold=DEBATE_CONSENSUS_THRESHOLD,
                advocate_model=ADVOCATE_MODEL,
                critic_model=CRITIC_MODEL,
                synthesizer_model=SYNTHESIZER_MODEL,
                judge_model=JUDGE_MODEL,
                advocate_temperature=ADVOCATE_TEMPERATURE,
                critic_temperature=CRITIC_TEMPERATURE,
                synthesizer_temperature=SYNTHESIZER_TEMPERATURE,
                judge_temperature=JUDGE_TEMPERATURE,
                log_callback=self._log,
                verbose=self.developer_mode,
            )
            self._debate_engine = DebateMoE(config)
        return self._debate_engine
    
    def _hash_text(self, text: str) -> str:
        """Create fast hash of text for cache keys"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def _log(self, message: str):
        """Print log only if developer_mode is enabled"""
        if self.developer_mode:
            print(message)
    
    def _save_qa_log(
        self,
        user_idea: str,
        task_type: str,
        domain: str,
        kb_context: Optional[str],
        response: str,
        task_output: Any,
        timing: Dict[str, float],
        debate_used: bool,
        debate_enabled: bool,
    ):
        """Save QA execution log to outputs/qa_run_logs/"""
        from datetime import datetime
        import os
        
        # Create timestamp-based folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent.parent / "outputs" / "qa_run_logs" / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build audit log
        audit_log = {
            "timestamp": datetime.now().isoformat(),
            "user_question": user_idea,
            "task_type": task_type,
            "domain": domain,
            "timing": timing,
            "debate_moe": {
                "enabled_in_config": debate_enabled,
                "actually_used": debate_used,
                "reason_not_used": None if debate_used else (
                    "Task type is not QA" if task_type != "QA" else
                    "Debate MoE not enabled in config"
                )
            },
            "context": {
                "kb_context_length": len(kb_context) if kb_context else 0,
                "kb_context_preview": (kb_context[:1000] + "...") if kb_context and len(kb_context) > 1000 else kb_context,
            },
            "response": {
                "length": len(response),
                "preview": (response[:2000] + "...") if len(response) > 2000 else response,
            }
        }
        
        # Add debate-specific info if debate was used
        if debate_used and hasattr(task_output, 'debate_result') and task_output.debate_result:
            dr = task_output.debate_result
            audit_log["debate_details"] = {
                "total_rounds": dr.total_rounds,
                "confidence": dr.confidence,
                "consensus_reached": dr.consensus_reached,
                "token_usage": dr.token_usage,
                "total_latency": dr.total_latency,
                "winning_arguments": dr.winning_arguments,
                "dissenting_view": dr.dissenting_view,
                "transcript": [
                    {
                        "round": r.round_number,
                        "role": r.role.value,
                        "content": r.content,
                        "tokens": r.tokens_used,
                        "latency": r.response.latency,
                    }
                    for r in dr.debate_transcript
                ]
            }
        elif hasattr(task_output, 'raw_json'):
            audit_log["gpt4_details"] = task_output.raw_json
        
        # Save audit log
        audit_path = log_dir / "audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=2, ensure_ascii=False)
        
        self._log(f"   📝 QA log saved to: {log_dir}")
    
    def process(
        self,
        user_idea: str,
        has_dataset: bool = False,
        dataset_info: Optional[Dict[str, Any]] = None,
        preprocess_rules: Optional[Dict[str, Any]] = None,
        task_type_override: Optional[str] = None,
    ) -> PipelineOutput:
        """
        Process user request through complete pipeline with timing instrumentation
        
        Args:
            user_idea: User's question/request
            has_dataset: Whether dataset is provided
            dataset_info: Dataset metadata (optional)
            
        Returns:
            PipelineOutput with complete results, debug info, and timing metrics
        """
        timing = {}
        t_start = time.perf_counter()
        debate_used = False  # Track if Debate MoE was actually used
        # Check debate_moe dynamically to support runtime config changes
        enable_debate_moe = (
            os.environ.get("ENABLE_DEBATE_MOE", "false").lower() == "true"
            and DEBATE_MOE_AVAILABLE
        )
        
        # Banner (always show)
        print("\n" + "="*80)
        print("🚀 NS-Copilot PIPELINE")
        print("="*80)
        
        # Normalize dataset_info / preprocess rules
        if dataset_info is None:
            dataset_info = {}
        if preprocess_rules is not None:
            dataset_info["preprocess_rules"] = preprocess_rules

        # Step 1: Task type — default to DA (Router disabled for this version)
        t_router_start = time.perf_counter()
        self._log("\n📍 Step 1: Task Classification")

        from neuro_copilot.core.router.router_agent import RouterOutput
        task_type = task_type_override if task_type_override in ["QA", "DA", "ER"] else "DA"
        router_output = RouterOutput(
            task_type=task_type,
            domain="unknown",
            keywords=user_idea.lower().split()[:5],
            constraints={"need_citations": False},
            confidence=1.0,
            raw_json={"task_type": task_type, "domain": "unknown", "router_disabled": True},
        )
        domain = "unknown"
        need_citations = False

        t_router_end = time.perf_counter()
        timing['router'] = t_router_end - t_router_start

        self._log(f"   Task Type: {task_type} (default DA, Router disabled)")
        
        # Step 2: KB Retrieval — disabled (DA-only mode, no domain-specific KB)
        t_kb_start = time.perf_counter()
        self._log(f"\n📚 Step 2: Knowledge Base Retrieval — Skipped (DA mode)")
        kb_entries = []
        kb_context = None
        
        t_kb_end = time.perf_counter()
        timing['kb_retrieval'] = t_kb_end - t_kb_start
        
        # GPT-4 Backend Selection (when has_dataset == False)
        # ⚠️ This bypasses local Task Agents and KB retrieval for cloud-based reasoning
        use_gpt4_backend = (
            self.use_gpt4 and 
            not has_dataset and 
            task_type in ["QA", "ER", "DA"]
        )
        
        if use_gpt4_backend:
            self._log(f"\n🌐 Step 3: GPT-4 Backend (task_type={task_type}, no dataset)")
            # Keep Graph KB context but clear paper entries for GPT-4 path
            kb_entries = []
            # Note: kb_context is preserved if Graph KB enrichment was applied
        
        # Step 3: Task Agent (or GPT-4 Backend)
        t_agent_start = time.perf_counter()
        
        if use_gpt4_backend:
            # GPT-4 Backend Path (with optional Debate MoE)
            
            # Create dummy task_output class for compatibility
            from dataclasses import dataclass
            @dataclass
            class DummyGPT4Output:
                answer: str = None
                recommendation: str = None
                response: str = None
                references: list = None
                confidence: float = 0.0
                raw_json: dict = None
                debate_result: Any = None  # Store debate transcript if used
            
            try:
                # ═══════════════════════════════════════════════════════════════
                # DEBATE MOE PATH: Multi-expert collaborative reasoning
                # ═══════════════════════════════════════════════════════════════
                if enable_debate_moe and task_type == "QA":
                    self._log(f"   🎭 Using Debate MoE for reasoning")
                    debate_used = True
                    
                    debate_engine = self._get_debate_engine()
                    
                    # Run the debate with context
                    debate_result = debate_engine.debate(
                        question=user_idea,
                        context=kb_context or "No additional context provided."
                    )
                    
                    response = debate_result.final_answer
                    references = []  # Could extract from debate if needed
                    confidence = debate_result.confidence
                    
                    self._log(f"   ✅ Debate completed: {debate_result.total_rounds} rounds")
                    self._log(f"   📊 Confidence: {confidence:.2f}")
                    self._log(f"   🎯 Consensus: {'Yes' if debate_result.consensus_reached else 'No'}")
                    self._log(f"   💰 Total tokens: {sum(debate_result.token_usage.values())}")
                    self._log(f"   ⏱️  Debate latency: {debate_result.total_latency:.2f}s")
                    
                    # Store debate transcript in raw_json for audit
                    debate_audit = {
                        "mode": "debate_moe",
                        "total_rounds": debate_result.total_rounds,
                        "confidence": confidence,
                        "consensus_reached": debate_result.consensus_reached,
                        "token_usage": debate_result.token_usage,
                        "latency": debate_result.total_latency,
                        "winning_arguments": debate_result.winning_arguments,
                        "dissenting_view": debate_result.dissenting_view,
                        "transcript": [
                            {
                                "round": r.round_number,
                                "role": r.role.value,
                                "content_preview": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                                "tokens": r.tokens_used
                            }
                            for r in debate_result.debate_transcript
                        ]
                    }
                    
                    if task_type == "QA":
                        task_output = DummyGPT4Output(
                            answer=response, 
                            references=references, 
                            confidence=confidence, 
                            raw_json=debate_audit,
                            debate_result=debate_result
                        )
                    elif task_type == "ER":
                        task_output = DummyGPT4Output(
                            recommendation=response, 
                            references=references, 
                            confidence=confidence, 
                            raw_json=debate_audit,
                            debate_result=debate_result
                        )
                    elif task_type == "DA":
                        task_output = DummyGPT4Output(
                            response=response, 
                            references=references, 
                            confidence=confidence, 
                            raw_json=debate_audit,
                            debate_result=debate_result
                        )
                
                # ═══════════════════════════════════════════════════════════════
                # SINGLE MODEL PATH: Original GPT-4 direct reasoning
                # ═══════════════════════════════════════════════════════════════
                else:
                    self._log(f"   Using GPT-4 for QA (single model)")
                    
                    gpt4_client = self._get_gpt4_client()
                    
                    # Build prompt with Graph KB context if available
                    if kb_context:
                        self._log(f"   📊 Including Graph KB context ({len(kb_context)} chars)")
                        gpt4_prompt = f"{kb_context}\n\n---\n\nQuestion: {user_idea}"
                    else:
                        self._log(f"   No Graph KB context available")
                        gpt4_prompt = user_idea
                    
                    # System prompt
                    system_prompt = (
                        "You are a helpful neuroscience research assistant. "
                        "Answer questions based on the provided knowledge graph context and your expertise. "
                        "When the context provides semantic relations between concepts, use them to support your answer."
                    )
                    
                    # Call GPT-4 with enriched prompt
                    gpt4_response = gpt4_client.generate(
                        prompt=gpt4_prompt,
                        system_prompt=system_prompt,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    response = gpt4_response.content
                    references = []  # GPT-4 free-form output, no structured references
                    
                    self._log(f"   Response length: {len(response)} chars")
                    self._log(f"   Tokens: {gpt4_response.total_tokens} (prompt: {gpt4_response.prompt_tokens}, completion: {gpt4_response.completion_tokens})")
                    self._log(f"   GPT-4 latency: {gpt4_response.latency:.2f}s")
                    
                    # Populate based on task type
                    if task_type == "QA":
                        task_output = DummyGPT4Output(answer=response, references=references, confidence=0.9, raw_json={"mode": "single_model"})
                    elif task_type == "ER":
                        task_output = DummyGPT4Output(recommendation=response, references=references, confidence=0.9, raw_json={"mode": "single_model"})
                    elif task_type == "DA":
                        task_output = DummyGPT4Output(response=response, references=references, confidence=0.9, raw_json={"mode": "single_model"})
                
            except Exception as e:
                self._log(f"   ⚠️  GPT-4 backend failed: {str(e)}")
                self._log(f"   Falling back to local agents")
                use_gpt4_backend = False  # Trigger fallback below
        
        if not use_gpt4_backend:
            # Local Agent Path (original logic)
            self._log(f"\n🤖 Step 3: Task Agent ({task_type})")
            
            if task_type == "QA":
                task_output = self.qa_agent.answer(
                    question=user_idea,
                    kb_context=kb_context,
                    need_citations=need_citations
                )
                response = task_output.answer
                references = task_output.references
                # Ensure response is a string
                if not isinstance(response, str):
                    response = str(response)
                word_count = len(response.split())
                self._log(f"   Answer: {word_count} words")
                self._log(f"   References: {len(references)}")
                
            elif task_type == "ER":
                task_output = self.er_agent.recommend(
                    request=user_idea,
                    kb_context=kb_context,
                    need_citations=need_citations
                )
                response = task_output.recommendation
                references = task_output.references
                # Ensure response is a string
                if not isinstance(response, str):
                    response = str(response)
                word_count = len(response.split())
                self._log(f"   Recommendation: {word_count} words")
                self._log(f"   References: {len(references)}")
                
            elif task_type == "DA":
                if has_dataset:
                    task_output = self._run_da_execution(user_idea, dataset_info)
                else:
                    task_output = self.da_agent.analyze_or_guide(
                        request=user_idea,
                        has_dataset=has_dataset,
                        dataset_info=dataset_info
                    )
                response = task_output.response
                references = task_output.references
                # Ensure response is a string
                if not isinstance(response, str):
                    response = str(response)
                word_count = len(response.split())
                self._log(f"   Guidance: {word_count} words")
                self._log(f"   Needs Dataset: {getattr(task_output, 'needs_dataset', False)}")
            
            else:
                # Fallback
                task_output = None
                response = f"Unknown task type: {task_type}"
                references = []
        
        t_agent_end = time.perf_counter()
        timing['task_agent'] = t_agent_end - t_agent_start
        
        # Step 4: Controller - Validate
        t_controller_start = time.perf_counter()
        self._log(f"\n🎯 Step 4: Controller - Validation")
        
        # Create KB papers summary for controller
        kb_papers_summary = ""
        if kb_entries:
            kb_papers_summary = "; ".join([
                f"{entry.title[:40]}... ({entry.year}, {entry.paper_type})" 
                for entry in kb_entries[:3]
            ])
        
        # Check cache (optional - controller validation is important for quality)
        response_hash = self._hash_text(response[:500])  # First 500 chars
        controller_cache_key = (task_type, response_hash, need_citations)
        
        if self.enable_cache and controller_cache_key in self._controller_cache:
            controller_output = self._controller_cache[controller_cache_key]
            self._log("   ⚡ Cache hit (controller)")
        else:
            controller_output = self.controller.validate(
                task_type=task_type,
                user_request=user_idea,
                agent_output=response,
                references=references,
                need_citations=need_citations,
                has_dataset=has_dataset,
                kb_papers_summary=kb_papers_summary
            )
            # Only cache PASS results to be safe
            if self.enable_cache and controller_output.status == "PASS":
                self._controller_cache[controller_cache_key] = controller_output
        
        t_controller_end = time.perf_counter()
        timing['controller'] = t_controller_end - t_controller_start
        
        self._log(f"   Status: {controller_output.status}")
        if controller_output.issues:
            self._log(f"   Issues: {controller_output.issues}")
        self._log(f"   Next Action: {controller_output.next_action}")
        if hasattr(controller_output, 'rationale') and controller_output.rationale:
            self._log(f"   Rationale: {controller_output.rationale}")
        
        # Calculate total time
        t_end = time.perf_counter()
        timing['total'] = t_end - t_start
        
        # Package output
        print("\n✅ Pipeline Complete")
        
        # Print timing breakdown if developer mode
        if self.developer_mode:
            print("\n⏱️  Performance Breakdown:")
            print(f"   Router:      {timing['router']:.3f}s")
            print(f"   KB query:    {timing['kb_retrieval']:.3f}s")
            print(f"   Task agent:  {timing['task_agent']:.3f}s")
            print(f"   Controller:  {timing['controller']:.3f}s")
            print(f"   " + "-" * 30)
            print(f"   Total:       {timing['total']:.3f}s")
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)
        
        # Extract visualization paths and analysis result for DA tasks
        visualization_paths = None
        analysis_result = None
        if task_type == "DA" and hasattr(task_output, 'raw_json') and task_output.raw_json:
            visualization_paths = task_output.raw_json.get("visualization_paths", [])
            analysis_result_dict = task_output.raw_json.get("analysis_result")
            if analysis_result_dict:
                # Convert dict to AnalysisResult if needed
                try:
                    analysis_result = AnalysisResult(
                        summary=analysis_result_dict.get("summary", ""),
                        key_findings=analysis_result_dict.get("key_findings", []),
                        metrics_interpretation=analysis_result_dict.get("metrics_interpretation", {}),
                        visualizations=[],  # Already have paths
                        limitations=analysis_result_dict.get("limitations", []),
                        recommendations=analysis_result_dict.get("recommendations", []),
                        experiment_status=analysis_result_dict.get("experiment_status", "completed"),
                        total_attempts=analysis_result_dict.get("total_attempts", 1),
                        target_metric=analysis_result_dict.get("target_metric"),
                        target_threshold=analysis_result_dict.get("target_threshold"),
                        achieved_value=analysis_result_dict.get("achieved_value"),
                        target_met=analysis_result_dict.get("target_met", False),
                        raw_json=analysis_result_dict.get("raw_json", {}),
                    )
                except Exception:
                    pass
        
        # Save QA execution log (for QA tasks only)
        if task_type == "QA":
            try:
                self._save_qa_log(
                    user_idea=user_idea,
                    task_type=task_type,
                    domain=domain,
                    kb_context=kb_context,
                    response=response,
                    task_output=task_output,
                    timing=timing,
                    debate_used=debate_used,
                    debate_enabled=enable_debate_moe,
                )
            except Exception as e:
                self._log(f"   ⚠️ Failed to save QA log: {e}")
        
        return PipelineOutput(
            response=response,
            references=references,
            router_output=router_output,
            kb_entries=kb_entries,
            task_agent_output=task_output,
            controller_output=controller_output,
            task_type=task_type,
            domain=domain,
            status=controller_output.status,
            timing=timing,
            visualization_paths=visualization_paths,
            analysis_result=analysis_result
        )
    
    def clear_cache(self):
        """Clear all caches (useful for testing or when memory is a concern)"""
        self._router_cache.clear()
        self._kb_cache.clear()
        self._controller_cache.clear()
        if self.developer_mode:
            print("✅ All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'router_entries': len(self._router_cache),
            'kb_entries': len(self._kb_cache),
            'controller_entries': len(self._controller_cache)
        }

    @staticmethod
    def _run_codex_cli_baseline(prompt: str, dataset_path: str, audit_dir: str,
                                 primary_metric: Optional[str] = None,
                                 timeout: int = 600,
                                 additional_files: Optional[list] = None) -> Dict[str, Any]:
        """Run full Codex CLI agent as a fair baseline (no pipeline tools, no profiling).

        Codex explores the dataset on its own, generates code, executes it,
        and self-corrects errors autonomously.
        primary_metric is optional — when None, all metrics are collected but
        baseline_value is not extracted (caller should extract after Planner decides).
        """
        import subprocess, re, time as _time

        _codex_model = os.environ.get("CODEX_BASELINE_MODEL", "gpt-5.2-codex")
        _api_key = os.environ.get("OPENAI_API_KEY", "")

        # Run codex exec in an isolated temp directory to prevent cross-run contamination.
        # Symlink the dataset into the temp dir so Codex can access it.
        # --dangerously-bypass-approvals-and-sandbox is safe here because:
        # 1. Codex runs inside a Docker container with limited filesystem
        # 2. Datasets are mounted read-only or in isolated directories
        # 3. Output is parsed and validated before use in pipeline decisions
        import tempfile, shutil
        _temp_dir = tempfile.mkdtemp(prefix="codex_run_")

        if dataset_path and os.path.exists(dataset_path):
            if os.path.isdir(dataset_path):
                # Directory dataset: symlink the whole directory
                _dataset_link = os.path.join(_temp_dir, os.path.basename(dataset_path))
                os.symlink(os.path.abspath(dataset_path), _dataset_link)
            else:
                # Single file: symlink it + any additional files into the same dir
                os.symlink(os.path.abspath(dataset_path),
                           os.path.join(_temp_dir, os.path.basename(dataset_path)))
                for af in (additional_files or []):
                    if af and os.path.exists(af):
                        _link = os.path.join(_temp_dir, os.path.basename(af))
                        if not os.path.exists(_link):
                            os.symlink(os.path.abspath(af), _link)
                _dataset_link = _temp_dir  # point to directory containing all files

        _codex_workdir = _temp_dir
        _codex_dataset_ref = _dataset_link or dataset_path

        # Update prompt to use the symlinked path
        codex_prompt = (
            f"{prompt}\n\n"
            f"The dataset is located at: {_codex_dataset_ref}\n"
            f"Explore the dataset directory to understand its structure before writing code.\n"
            f"Do NOT write any files to the dataset directory.\n"
            f"You MUST print numeric values for ALL metrics requested in the task description above.\n"
            f"Format each metric on its own line as 'metric_name: value'."
        )

        cmd = [
            "codex", "exec",
            "-m", _codex_model,
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", _codex_workdir,
        ]
        cmd.append(codex_prompt)

        _start = _time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout,
                env={**os.environ, "OPENAI_API_KEY": _api_key},
            )
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = ""
            print(f"⚠️ Codex CLI timed out after {timeout}s", flush=True)
        except Exception as e:
            output = ""
            print(f"⚠️ Codex CLI failed: {e}", flush=True)
        _dur = _time.time() - _start

        # Save Codex-generated code files before cleanup
        _audit_path = Path(audit_dir)
        _audit_path.mkdir(parents=True, exist_ok=True)
        _codex_code_dir = _audit_path / "codex_generated_code"
        try:
            _codex_code_dir.mkdir(exist_ok=True)
            for _f in Path(_temp_dir).rglob("*.py"):
                if _f.is_file():
                    shutil.copy2(str(_f), str(_codex_code_dir / _f.name))
        except Exception:
            pass

        # Clean up temp directory
        try:
            shutil.rmtree(_temp_dir, ignore_errors=True)
        except Exception:
            pass

        # Save raw output to audit dir
        _audit_path = Path(audit_dir)
        _audit_path.mkdir(parents=True, exist_ok=True)
        (_audit_path / "codex_cli_output.txt").write_text(output, encoding="utf-8")

        # Parse metrics from free-text output using a generic key: value parser.
        # Scan all lines and collect every match; same key last-wins.
        # This avoids block detection which breaks on multi-line values
        # (e.g. confusion_matrix spanning multiple lines).
        metrics = {}
        _KV_RE = re.compile(
            r"^\s*([\w][\w\s\-]*?)\s*[:=]\s*`?([\d]+\.[\d]+(?:[eE][+-]?\d+)?|[\d]+(?:[eE][+-]?\d+)?)`?\s*$"
        )
        for line in output.splitlines():
            m = _KV_RE.match(line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                val = float(m.group(2))
                if val > 1.0:
                    val = val / 100.0
                metrics[key] = val

        baseline_value = metrics.get(primary_metric) if primary_metric else None

        # Save structured audit
        audit_data = {
            "codex_model": _codex_model,
            "prompt": codex_prompt,
            "dataset_path": dataset_path,
            "metrics": metrics,
            "primary_metric": primary_metric,
            "baseline_value": baseline_value,
            "duration_s": round(_dur, 2),
            "raw_output_length": len(output),
            "exit_code": getattr(result, 'returncode', None) if 'result' in dir() else None,
        }
        with open(_audit_path / "audit.json", "w") as f:
            json.dump(audit_data, f, indent=2, sort_keys=True, ensure_ascii=False)

        # Determine error reason if baseline failed
        _error = None
        if primary_metric and baseline_value is None:
            if not output.strip():
                _error = "Codex CLI produced no output (timeout or crash)"
            else:
                _error = f"Could not extract {primary_metric} from Codex output ({len(output)} chars)"
        elif not primary_metric and not output.strip():
            _error = "Codex CLI produced no output (timeout or crash)"

        return {
            "metrics": metrics,
            "baseline_value": baseline_value,
            "duration_s": _dur,
            "output": output,
            "error": _error,
        }

    def _run_da_execution(self, user_idea: str, dataset_info: Dict[str, Any]) -> DAOutput:
        return self._run_da_execution_inner(user_idea, dataset_info)

    def _run_da_execution_inner(self, user_idea: str, dataset_info: Dict[str, Any]) -> DAOutput:
        # Set global random seeds for reproducibility
        import random as _random
        import numpy as _np
        _GLOBAL_SEED = (dataset_info or {}).get("_run_seed", 42)
        _random.seed(_GLOBAL_SEED)
        _np.random.seed(_GLOBAL_SEED)
        try:
            import torch as _torch
            _torch.manual_seed(_GLOBAL_SEED)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(_GLOBAL_SEED)
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        dataset_path = (dataset_info or {}).get("dataset_path") or (dataset_info or {}).get("file_path") or (dataset_info or {}).get("path")
        additional_files = (dataset_info or {}).get("additional_files")
        preprocess_rules = (dataset_info or {}).get("preprocess_rules")
        retry_context = (dataset_info or {}).get("retry_context")

        if not dataset_path:
            return DAOutput(
                response="No data path provided. Cannot perform data analysis.",
                needs_dataset=True,
                analysis_type="execution",
                confidence=0.0,
            )

        print("🔧 DA execution: start", flush=True)
        state = NeuroGlobalState()

        dataset_root_dir: Optional[str] = None
        dataset_file_path: Optional[str] = None
        p = Path(dataset_path)
        # Supported dataset file extensions (both .mat and .nwb)
        DATASET_EXTENSIONS = (".nwb", ".mat", ".csv", ".set")
        
        if p.is_dir():
            # Directory dataset (e.g., BIDS format with per-subject .set files)
            dataset_root_dir = str(p)
            state.extra["dataset_root_dir"] = str(p)
            print(f"📂 DA execution: directory dataset detected: {p}", flush=True)
            # Find a representative dataset file for field catalog profiling
            import glob as _glob
            found_files = []
            for ext in DATASET_EXTENSIONS:
                found_files.extend(sorted(_glob.glob(str(p / "**" / f"*{ext}"), recursive=True)))
            if not found_files:
                return DAOutput(
                    response=f"No dataset files ({', '.join(DATASET_EXTENSIONS)}) found in directory {p}. Cannot perform analysis.",
                    needs_dataset=True,
                    analysis_type="execution",
                    confidence=0.0,
                )
            dataset_file_path = found_files[0]
            print(f"📂 DA execution: found {len(found_files)} dataset file(s), profiling: {os.path.basename(dataset_file_path)}", flush=True)
        elif p.name.endswith((".tar.gz", ".tgz")):
            dataset_root_dir = str(p)
            print(f"📦 DA execution: tar.gz detected: {p}", flush=True)
            from tarfile import open as tar_open
            with tar_open(p, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.lower().endswith(DATASET_EXTENSIONS)]
                if not members:
                    return DAOutput(
                        response="No dataset files (.nwb, .mat, .csv, or .set) found in archive. Cannot perform analysis.",
                        needs_dataset=True,
                        analysis_type="execution",
                        confidence=0.0,
                    )
                nwb_members = [m for m in members if m.name.lower().endswith(".nwb")]
                dataset_member = nwb_members[0] if nwb_members else members[0]
                dataset_name = dataset_member.name
                print(f"📦 DA execution: found dataset file in archive: {dataset_name}", flush=True)
            extract_root = p.parent / f"_extract_{p.stem}"
            dataset_candidate = extract_root / dataset_name
            if extract_root.exists() and dataset_candidate.exists():
                dataset_file_path = str(dataset_candidate)
                print(f"📦 DA execution: reuse extracted: {dataset_file_path}", flush=True)
            else:
                extract_root.mkdir(parents=True, exist_ok=True)
                print(f"📦 DA execution: extracting to {extract_root}", flush=True)
                with tar_open(p, "r:gz") as tar:
                    tar.extractall(extract_root)
                dataset_file_path = str(dataset_candidate)
                print(f"📦 DA execution: extracted dataset: {dataset_file_path}", flush=True)
            state.extra["dataset_root_dir"] = str(extract_root)
        elif p.suffix.lower() == ".zip":
            dataset_root_dir = str(p)
            print(f"📦 DA execution: zip detected: {p}", flush=True)
            from zipfile import ZipFile
            with ZipFile(p, "r") as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(DATASET_EXTENSIONS)]
                if not members:
                    return DAOutput(
                        response="No dataset files (.nwb, .mat, .csv, or .set) found in archive. Cannot perform analysis.",
                        needs_dataset=True,
                        analysis_type="execution",
                        confidence=0.0,
                    )
                nwb_members = [n for n in members if n.lower().endswith(".nwb")]
                dataset_name = nwb_members[0] if nwb_members else members[0]
                print(f"📦 DA execution: found dataset file in archive: {dataset_name}", flush=True)
            extract_root = p.parent / f"_extract_{p.stem}"
            dataset_candidate = extract_root / dataset_name
            if extract_root.exists() and dataset_candidate.exists():
                dataset_file_path = str(dataset_candidate)
                print(f"📦 DA execution: reuse extracted: {dataset_file_path}", flush=True)
            else:
                extract_root.mkdir(parents=True, exist_ok=True)
                print(f"📦 DA execution: extracting to {extract_root}", flush=True)
                with ZipFile(p, "r") as zf:
                    zf.extractall(extract_root)
                dataset_file_path = str(dataset_candidate)
                print(f"📦 DA execution: extracted dataset: {dataset_file_path}", flush=True)
            state.extra["dataset_root_dir"] = str(extract_root)
        else:
            dataset_file_path = str(p)
            state.extra["dataset_root_dir"] = str(p.parent)

        print(f"📦 DA execution: load dataset from {dataset_file_path}", flush=True)
        if additional_files:
            print(f"📦 DA execution: additional files: {additional_files}", flush=True)
        
        # Parse schema from user prompt if provided
        raw_schema = parse_schema_from_prompt(user_idea)
        if raw_schema:
            print(f"📦 DA execution: found schema section in prompt", flush=True)
        
        state = load_dataset_into_state(
            state, 
            dataset_file_path, 
            preprocess_rules=preprocess_rules,
            additional_paths=additional_files,
            schema=raw_schema,
        )
        print("📦 DA execution: dataset loaded", flush=True)
        
        # Use LLM to parse schema into structured form
        if raw_schema:
            print("🔍 DA execution: parsing schema with LLM...", flush=True)
            try:
                # Build data summary from loaded data
                neural_dataset = state.extra.get("neural_dataset")
                data_summary = None
                if neural_dataset and hasattr(neural_dataset, "raw"):
                    data_summary = build_data_summary_from_raw(neural_dataset.raw)
                
                # Parse schema
                parsed_schema = parse_schema(raw_schema, data_summary=data_summary)
                state.extra["parsed_schema"] = parsed_schema
                state.extra["schema_dict"] = parsed_schema.to_dict()
                print(f"📋 DA execution: schema parsed - data_type={parsed_schema.data_type}, "
                      f"files={list(parsed_schema.files.keys())}", flush=True)
                
                if parsed_schema.warnings:
                    for warn in parsed_schema.warnings:
                        print(f"  ⚠️ Schema warning: {warn}", flush=True)
            except Exception as e:
                print(f"⚠️ DA execution: schema parsing failed: {e}", flush=True)
                state.extra["schema_parse_error"] = str(e)

        print("🧾 DA execution: prepare audit log", flush=True)
        # Record pipeline start time for timing instrumentation
        pipeline_start_time = time.time()
        pipeline_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pipeline_start_time))
        
        # Prepare audit log with comprehensive information
        # Read controller mode from env for audit recording
        import os as _os_audit
        _audit_controller_mode = _os_audit.environ.get("CONTROLLER_MODE", "force").strip().lower()

        audit = {
            "user_prompt": user_idea,
            "dataset_path": dataset_path,
            "dataset_root_dir": state.extra.get("dataset_root_dir"),
            "controller_mode": _audit_controller_mode,
            "preprocess_rules": preprocess_rules,
            "schema_text": state.extra.get("schema_text"),  # Original schema from user
            "field_catalog_text": state.extra.get("field_catalog_text"),  # Inferred field info
            "data_summary": state.extra.get("data_summary_text"),  # Data statistics
            "timing": {
                "pipeline_start_time": pipeline_start_time_str,
                "pipeline_start_timestamp": pipeline_start_time,
            },
            "attempts": [],
        }
        # If running as auto-mode sub-run, use the parent's audit dir / model_name
        _auto_audit_dir = (dataset_info or {}).get("_auto_audit_dir")
        if _auto_audit_dir:
            audit_dir = Path(_auto_audit_dir)
            audit_run_id = audit_dir.name
        else:
            audit_run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            audit_dir = Path.cwd() / "outputs" / "da_run_logs" / audit_run_id
        audit_dir.mkdir(parents=True, exist_ok=True)

        class _NumpySafeEncoder(json.JSONEncoder):
            """JSON encoder that handles numpy types and NaN/Inf gracefully."""
            def default(self, obj):
                try:
                    import numpy as _np
                    if isinstance(obj, _np.integer):
                        return int(obj)
                    if isinstance(obj, _np.floating):
                        v = float(obj)
                        if _np.isnan(v) or _np.isinf(v):
                            return None
                        return v
                    if isinstance(obj, _np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, _np.bool_):
                        return bool(obj)
                except ImportError:
                    pass
                return super().default(obj)

            def encode(self, obj):
                """Override encode to sanitize NaN/Inf in plain Python floats."""
                return super().encode(self._sanitize(obj))

            def _sanitize(self, obj):
                import math
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                elif isinstance(obj, dict):
                    return {k: self._sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [self._sanitize(v) for v in obj]
                return obj

        def _save_audit(audit: Dict[str, Any]) -> None:
            """Save audit log with updated timing information."""
            # Calculate total run duration
            end_time = time.time()
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            start_timestamp = audit.get("timing", {}).get("pipeline_start_timestamp", end_time)
            total_duration_s = round(end_time - start_timestamp, 3)
            
            audit["timing"]["pipeline_end_time"] = end_time_str
            audit["timing"]["total_duration_s"] = total_duration_s
            
            (audit_dir / "audit.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True, cls=_NumpySafeEncoder), encoding="utf-8"
            )

        # -----------------------------------------------------------------
        # Metric-parsing helpers
        #   All alias/direction knowledge comes from config.METRIC_ALIASES.
        #   A fuzzy-match fallback handles metrics not listed there.
        # -----------------------------------------------------------------
        import re as _re
        from .config import METRIC_ALIASES as _METRIC_ALIASES

        def _enrich_error_with_type_hints(error_msg: str) -> str:
            """Append actionable hints when an error reveals a type mismatch.

            Detects patterns like "Got 'X' instead" where X is a numeric value
            passed as a string, and adds a clear instruction to convert it.
            This is fully generic — not tied to any specific library or parameter.
            """
            extra = ""
            # Detect "Got '<numeric>' instead" pattern (common in sklearn, etc.)
            for m in _re.finditer(r"Got '([^']+)' instead", error_msg):
                val = m.group(1)
                try:
                    float(val)
                    extra += (
                        f"\n\nTYPE FIX REQUIRED: The value '{val}' was passed "
                        f"as a string but a numeric type is expected. "
                        f"Remove the quotes: use {val} instead of '{val}'. "
                        f"Check ALL numeric parameter values in your code "
                        f"for the same quoting issue."
                    )
                    break  # one hint is enough
                except ValueError:
                    pass
            return error_msg + extra

        def _get_aliases(canonical: str) -> list:
            """Return alias list for a canonical metric name from config.

            Falls back to ``[canonical]`` if not registered.
            """
            entry = _METRIC_ALIASES.get(canonical)
            if entry:
                return entry["aliases"]
            return [canonical]

        def _get_direction(canonical: str) -> str:
            """Return optimization direction for a canonical metric.

            Falls back to ``"higher"`` (the common case) if not registered.
            """
            entry = _METRIC_ALIASES.get(canonical)
            if entry:
                return entry["direction"]
            return "higher"

        def _all_aliases_flat() -> dict:
            """Build a reverse map: alias → canonical metric name."""
            rev: dict = {}
            for canonical, entry in _METRIC_ALIASES.items():
                for alias in entry["aliases"]:
                    rev[alias.lower()] = canonical
            return rev

        # --- Derived constants from METRIC_ALIASES ---
        # Set of ALL known metric alias strings.  Used to detect whether a
        # dict contains real analysis metrics (vs filter stats like n_trials).
        _KNOWN_METRIC_ALIASES: set = set()
        for _ma_entry in _METRIC_ALIASES.values():
            _KNOWN_METRIC_ALIASES.update(_ma_entry["aliases"])

        # Ordered candidate list for selecting the primary comparison metric.
        # Priority: accuracy aliases first, then auroc aliases.
        # Derived from METRIC_ALIASES so new aliases are auto-included.
        _PRIMARY_METRIC_CANDIDATES: list = (
            _METRIC_ALIASES["accuracy"]["aliases"]
            + _METRIC_ALIASES["auroc"]["aliases"]
        )

        def _normalize_threshold_text(text: str) -> str:
            normalized = text.replace("\u2265", ">=").replace("\u2264", "<=")
            for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
                normalized = normalized.replace(ch, "")
            normalized = normalized.replace("\u00a0", " ")
            return normalized

        # Phrases that signal the user is specifying a performance target
        _TARGET_INTENT_WORDS = (
            r"(?:target|goal|threshold|require|need|want|achieve|reach|attain|"
            r"at\s+least|no\s+less\s+than|minimum|above|exceed|>=|>)"
        )

        def _parse_target_accuracy(text: str) -> Optional[float]:
            """Parse an explicit accuracy target from user text.

            Only matches when the user clearly states a target, e.g.:
              - "target accuracy >= 0.8"
              - "achieve accuracy 80%"
              - "accuracy should reach 0.75"
            Does NOT fire on incidental mentions like "Report: accuracy".
            """
            normalized = _normalize_threshold_text(text)
            # _CONNECTING allows "of", "=", ">=", ":", etc. between metric and number
            _CONNECTING = r"(?:\s+of\s+|\s*[<>=:]+\s*|\s+)"

            # Pattern 1: intent word ... accuracy ... number
            m = _re.search(
                rf"{_TARGET_INTENT_WORDS}\s+(?:.*?\s)?(?:accuracy|acc){_CONNECTING}([0-9]+\.?[0-9]*)\s*%?",
                normalized, _re.IGNORECASE,
            )
            # Pattern 2: accuracy ... intent word ... number
            if not m:
                m = _re.search(
                    rf"(?:accuracy|acc)\s+(?:should\s+)?{_TARGET_INTENT_WORDS}{_CONNECTING}([0-9]+\.?[0-9]*)\s*%?",
                    normalized, _re.IGNORECASE,
                )
            # Pattern 3: "accuracy of at least 0.7" (metric + of + intent + number)
            if not m:
                m = _re.search(
                    rf"(?:accuracy|acc)\s+of\s+{_TARGET_INTENT_WORDS}\s*([0-9]+\.?[0-9]*)\s*%?",
                    normalized, _re.IGNORECASE,
                )
            # Pattern 4: accuracy >= number (operator required, tight gap)
            if not m:
                m = _re.search(
                    r"\b(?:accuracy|acc)\s*(?:>=|<=|>|<)\s*([0-9]+\.?[0-9]*)\s*%?",
                    normalized, _re.IGNORECASE,
                )
            if not m:
                return None
            val = float(m.group(1))
            return val / 100.0 if val > 1 else val

        def _parse_target_metric(text: str) -> Optional[str]:
            """Detect a metric the user is setting a *target* for.

            Only matches when a target-intent phrase is present near the metric
            name, to avoid false positives on instructions like "Report accuracy".
            """
            lower = text.lower()
            # First check: is there any target-intent language at all?
            if not _re.search(_TARGET_INTENT_WORDS, lower, _re.IGNORECASE):
                return None
            for canonical, entry in _METRIC_ALIASES.items():
                for alias in entry["aliases"]:
                    # alias near a target-intent word (within ~60 chars)
                    pattern = (
                        rf"(?:{_TARGET_INTENT_WORDS})"
                        rf".{{0,40}}"
                        rf"\b{_re.escape(alias)}\b"
                    )
                    if _re.search(pattern, lower, _re.IGNORECASE):
                        return canonical
                    # or alias followed by target-intent word
                    pattern2 = (
                        rf"\b{_re.escape(alias)}\b"
                        rf".{{0,20}}"
                        rf"(?:{_TARGET_INTENT_WORDS})"
                    )
                    if _re.search(pattern2, lower, _re.IGNORECASE):
                        return canonical
            return None

        def _parse_target_threshold(text: str, metric: Optional[str]) -> Optional[float]:
            """Extract threshold value for a given metric from user text.

            Uses aliases from METRIC_ALIASES.  Only matches when the alias
            is *directly* followed by an operator/number (tight gap of ≤15 chars)
            or preceded by a target-intent phrase.
            """
            if not metric:
                return None
            normalized = _normalize_threshold_text(text)
            aliases = _get_aliases(metric)
            if not aliases:
                return None
            alias_pattern = "|".join(_re.escape(a) for a in aliases)

            _CONNECTING = r"(?:\s+of\s+|\s*[<>=:]+\s*|\s+)"

            # Pattern 1: "target/achieve/... <metric> <connecting> <number>"
            m = _re.search(
                rf"{_TARGET_INTENT_WORDS}\s*(?:.*?\s)?(?:{alias_pattern})"
                rf"{_CONNECTING}([0-9]+\.?[0-9]*)\s*%?",
                normalized, _re.IGNORECASE,
            )
            if m:
                val = float(m.group(1))
                return val / 100.0 if val > 1 else val

            # Pattern 2: "<metric> <op> <number>" (tight gap, operator required)
            for alias in aliases:
                m = _re.search(
                    rf"\b{_re.escape(alias)}\s{{0,5}}(?:>=|<=|>|<|=)\s*([0-9]+\.?[0-9]*)\s*%?",
                    normalized, _re.IGNORECASE,
                )
                if m:
                    val = float(m.group(1))
                    return val / 100.0 if val > 1 else val

            # Pattern 3: "<metric> : <number>" (colon separator, tight gap)
            for alias in aliases:
                m = _re.search(
                    rf"\b{_re.escape(alias)}\s{{0,3}}:\s*([0-9]+\.?[0-9]*)\s*%?",
                    normalized, _re.IGNORECASE,
                )
                if m:
                    val = float(m.group(1))
                    return val / 100.0 if val > 1 else val

            return None

        def _extract_metric_from_result(result_dict: Dict[str, Any], metric_name: str, direction: str) -> Optional[float]:
            """Extract metric value from code execution result dict.

            Uses ``_collect_metrics_recursive`` to flatten nested step results,
            then searches with aliases from METRIC_ALIASES plus a fuzzy fallback.
            """
            # Get registered aliases; falls back to [metric_name] for unknowns
            aliases = _get_aliases(metric_name)

            # Flatten the entire result tree so we never miss nested metrics.
            # Use _collect_metric_sets to handle multiple experiments (best wins).
            flat_metrics, _ = _collect_metric_sets(result_dict)

            best = None
            matched_keys: set = set()

            # Pass 1: exact alias match
            for alias in aliases:
                if alias in flat_metrics:
                    matched_keys.add(alias)
                    try:
                        val = float(flat_metrics[alias])
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(val):
                        continue
                    if best is None:
                        best = val
                    elif direction == "higher":
                        best = max(best, val)
                    else:
                        best = min(best, val)

            # Pass 2: fuzzy fallback — match any key containing the
            # canonical metric name as a substring (handles unknown prefixes
            # like "test_", "cv_", "train_" automatically)
            if best is None:
                canonical_lower = metric_name.lower().replace("_", "")
                for key, raw_val in flat_metrics.items():
                    if key in matched_keys:
                        continue
                    key_normalized = key.lower().replace("_", "")
                    if canonical_lower in key_normalized:
                        try:
                            val = float(raw_val)
                        except (TypeError, ValueError):
                            continue
                        if math.isnan(val):
                            continue
                        if best is None:
                            best = val
                        elif direction == "higher":
                            best = max(best, val)
                        else:
                            best = min(best, val)

            # Pass 3: token subset matching — handles word order differences
            # e.g. "macro_f1" tokens {macro,f1} ⊂ "cv_f1_macro_mean" tokens {cv,f1,macro,mean}
            if best is None:
                target_tokens = set(metric_name.lower().split("_"))
                candidates = []
                for key, raw_val in flat_metrics.items():
                    if key in matched_keys:
                        continue
                    key_tokens = set(key.lower().split("_"))
                    if target_tokens <= key_tokens:
                        try:
                            val = float(raw_val)
                        except (TypeError, ValueError):
                            continue
                        if math.isnan(val):
                            continue
                        # Prefer shortest superset (fewest extra tokens)
                        candidates.append((len(key_tokens) - len(target_tokens), val))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    best = candidates[0][1]

            return best

        target_accuracy = _parse_target_accuracy(user_idea)
        target_metric = _parse_target_metric(user_idea)
        target_metric_threshold = _parse_target_threshold(user_idea, target_metric)
        print(
            f"🧾 DA execution: target_metric={target_metric}, "
            f"target_metric_threshold={target_metric_threshold}, target_accuracy={target_accuracy}",
            flush=True,
        )
        # =================================================================
        # NEW DA EXECUTION FLOW
        #   Planner  →  Code Generator  →  CodeExecutor  →  Self-Correction
        # =================================================================

        # ----- helpers -----
        def _build_data_profile(raw_data_dict: Dict[str, Any]) -> str:
            """Generate a detailed per-column data profile for Code Generator.

            This gives the LLM concrete information about each DataFrame so
            it can make informed decisions (e.g. apply appropriate filtering,
            pick the right columns for feature extraction).

            The output is plain text, NOT dataset-specific — it is generated
            dynamically from whatever DataFrames are loaded.
            """
            import pandas as _pd
            import numpy as _np

            if not raw_data_dict:
                return ""

            # De-duplicate: multiple keys may point to the same DataFrame
            seen_ids: set = set()
            unique_items: list = []
            for key, df in raw_data_dict.items():
                if not isinstance(df, _pd.DataFrame):
                    continue
                obj_id = id(df)
                if obj_id in seen_ids:
                    continue
                seen_ids.add(obj_id)
                unique_items.append((key, df))

            if not unique_items:
                return ""

            lines = ["DATA PROFILE (auto-generated from loaded DataFrames)", ""]

            for key, df in unique_items:
                lines.append(f"### {key}  ({df.shape[0]} rows × {df.shape[1]} columns)")
                lines.append("")

                for col in df.columns:
                    series = df[col]
                    dtype_str = str(series.dtype)
                    n_null = int(series.isna().sum())

                    # Numeric column
                    if _pd.api.types.is_numeric_dtype(series):
                        valid = series.dropna()
                        if len(valid) == 0:
                            lines.append(f"  - {col}: {dtype_str}, all NaN")
                            continue
                        n_unique = int(valid.nunique())
                        vmin, vmax = float(valid.min()), float(valid.max())
                        vmean = float(valid.mean())
                        desc = f"  - {col}: {dtype_str}, range=[{vmin:.4g}, {vmax:.4g}], mean={vmean:.4g}"
                        if n_unique <= 10:
                            # Low-cardinality numeric — show unique values only (no counts to avoid label leakage)
                            unique_vals = sorted(valid.unique().tolist())
                            desc += f", values: {unique_vals}"
                        else:
                            desc += f", {n_unique} unique"
                        if n_null > 0:
                            desc += f", {n_null} NaN"
                        lines.append(desc)

                    # String / categorical column
                    else:
                        valid = series.dropna().astype(str)
                        n_unique = int(valid.nunique())
                        if n_unique == 0:
                            lines.append(f"  - {col}: {dtype_str}, all NaN")
                            continue
                        unique_vals = sorted(valid.unique().tolist())
                        if n_unique <= 20:
                            vals_str = ", ".join(f'"{v}"' for v in unique_vals)
                            desc = f"  - {col}: {dtype_str}, {n_unique} unique, values: {vals_str}"
                        else:
                            sample_str = ", ".join(f'"{v}"' for v in unique_vals[:5])
                            desc = f"  - {col}: {dtype_str}, {n_unique} unique, sample: {sample_str} ..."
                        if n_null > 0:
                            desc += f", {n_null} NaN"
                        lines.append(desc)

                lines.append("")  # blank line between tables

            return "\n".join(lines)

        def _build_data_summary_text() -> str:
            """Build a concise data-summary string from the loaded dataset.

            Combines the structural summary from the dataloader with a
            detailed per-column data profile generated from the actual
            loaded DataFrames.
            """
            parts: list = []

            # 1) Existing structural summary
            nd = state.extra.get("neural_dataset")
            if nd and hasattr(nd, "raw") and nd.raw.get("_data_summary"):
                existing = nd.raw["_data_summary"]
                if isinstance(existing, dict):
                    import json as _json
                    parts.append(_json.dumps(existing, indent=2, default=str))
                else:
                    parts.append(str(existing))
            elif state.extra.get("data_summary_text"):
                parts.append(state.extra["data_summary_text"])

            # 2) Detailed data profile (column-level statistics)
            profile = _build_data_profile(raw_data)
            if profile:
                parts.append(profile)

            # 3) EEG metadata (for .set files)
            _eeg_meta = state.extra.get("eeg_metadata")
            if _eeg_meta:
                import json as _json2
                parts.append(f"EEG Metadata (sample file): {_json2.dumps(_eeg_meta, default=str)}")

            # 4) Dataset root directory info (for multi-file datasets)
            _drd = state.extra.get("dataset_root_dir")
            if _drd:
                parts.append(f"Dataset root directory (DATASET_ROOT_DIR): {_drd}")

                # 4b) Directory listing (equivalent to `ls` — helps LLM discover files)
                try:
                    _top_entries = sorted(os.listdir(_drd))
                    # Truncate if too many entries
                    if len(_top_entries) > 30:
                        _shown = _top_entries[:15] + [f"... ({len(_top_entries) - 20} more entries)"] + _top_entries[-5:]
                    else:
                        _shown = _top_entries
                    parts.append(
                        "Files and directories at DATASET_ROOT_DIR:\n"
                        + "\n".join(f"  {e}" for e in _shown)
                    )
                except Exception:
                    pass

            # 5) Preview metadata files (.tsv, .csv, .json) found in dataset
            if _drd:
                import glob as _glob
                metadata_exts = ("*.tsv", "*.csv", "*.json")
                meta_previews = []
                for ext in metadata_exts:
                    for fpath in _glob.glob(
                        os.path.join(_drd, "**", ext), recursive=True
                    ):
                        # Skip hidden/system files
                        fname = os.path.basename(fpath)
                        if fname.startswith(".") or fname.startswith("_"):
                            continue
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="replace") as _mf:
                                lines = _mf.readlines()
                            if not lines:
                                continue
                            # Relative path for readability
                            rel = os.path.relpath(fpath, _drd)
                            # Show header + first 5 data rows
                            preview_lines = lines[:6]
                            preview = "".join(preview_lines).rstrip()

                            # Add value_counts for categorical columns
                            # (generic: works for any .tsv/.csv)
                            col_stats = ""
                            if fname.endswith((".tsv", ".csv")):
                                try:
                                    import pandas as _pd_meta
                                    sep = "\t" if fname.endswith(".tsv") else ","
                                    _meta_df = _pd_meta.read_csv(fpath, sep=sep, nrows=10000)
                                    stat_parts = []
                                    for col in _meta_df.columns:
                                        n_unique = _meta_df[col].nunique()
                                        # Categorical: few unique values relative to rows
                                        if 1 < n_unique <= 20:
                                            vc = _meta_df[col].value_counts().to_dict()
                                            stat_parts.append(
                                                f"Column '{col}': unique={list(vc.keys())}, counts={vc}"
                                            )
                                    if stat_parts:
                                        col_stats = "\n" + "\n".join(stat_parts)
                                except Exception:
                                    pass

                            meta_previews.append(
                                f"--- {rel} ({len(lines)} rows) ---\n{preview}{col_stats}"
                            )
                        except Exception:
                            continue
                        if len(meta_previews) >= 5:
                            break
                    if len(meta_previews) >= 5:
                        break
                if meta_previews:
                    parts.append(
                        "Metadata files found in dataset:\n\n"
                        + "\n\n".join(meta_previews)
                    )

            return "\n\n".join(parts) if parts else "No data summary available."

        def _is_dataframe(v: Any) -> bool:
            """Check if value is a pandas DataFrame (without hard import)."""
            return type(v).__name__ == "DataFrame" and hasattr(v, "to_dict")

        def _dataframe_to_dicts(df: Any) -> list:
            """Convert a DataFrame to a list of row dicts for metric extraction."""
            try:
                return df.to_dict("records")
            except Exception:
                return []

        def _to_json_safe(v: Any) -> Any:
            """Convert numpy/special types to JSON-serializable Python types."""
            try:
                import numpy as _np
                if isinstance(v, (_np.integer,)):
                    return int(v)
                if isinstance(v, (_np.floating,)):
                    return float(v)
                if isinstance(v, _np.ndarray):
                    return v.tolist()
                if isinstance(v, _np.bool_):
                    return bool(v)
            except ImportError:
                pass
            return v

        def _is_numeric(v: Any) -> bool:
            """Check if value is a scalar number (Python or numpy)."""
            if isinstance(v, (int, float)):
                return True
            try:
                import numpy as _np
                return isinstance(v, (_np.integer, _np.floating))
            except ImportError:
                return False

        def _collect_metrics_recursive(d: Any, depth: int = 0) -> Dict[str, Any]:
            """Recursively collect all scalar numeric values from a nested dict.

            The generated code typically stores results as::

                result = {
                    "step1": {...},
                    "step2": {...},
                    "step3": {"test_accuracy": 0.65, "test_f1_macro": 0.63, ...}
                }

            This helper walks the entire tree and lifts every numeric leaf
            into a flat dict so that metrics are never lost regardless of
            nesting structure.  If the same key appears in multiple sub-dicts
            the first encountered value wins (breadth-first).

            All values are converted to JSON-safe Python types.
            """
            if not isinstance(d, dict) or depth > 5:
                return {}
            collected: Dict[str, Any] = {}
            sub_dicts = []
            for k, v in d.items():
                if not isinstance(k, str):
                    continue
                if k.startswith("_"):
                    continue
                if _is_numeric(v):
                    if k not in collected:
                        collected[k] = _to_json_safe(v)
                elif isinstance(v, dict):
                    sub_dicts.append(v)
                elif _is_dataframe(v):
                    # Convert DataFrame rows to dicts for metric extraction
                    for row_dict in _dataframe_to_dicts(v):
                        sub_dicts.append(row_dict)
                elif isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, (list, int, float)) or _is_numeric(first):
                        # If this is a known metric key with a flat numeric list
                        # (e.g. per-fold accuracy values), aggregate to mean so
                        # downstream arithmetic (regression check, diagnosis) works.
                        if (
                            k.lower() in _KNOWN_METRIC_ALIASES
                            and all(isinstance(x, (int, float)) for x in v)
                            and not isinstance(first, list)
                        ):
                            if k not in collected:
                                collected[k] = sum(v) / len(v)
                        # Keep small lists (e.g. confusion_matrix, class_labels)
                        # but skip large index/id lists (> 50 elements)
                        elif len(v) <= 50 and k not in collected:
                            collected[k] = _to_json_safe(v)
                elif isinstance(v, str) and k not in collected:
                    # Generated code sometimes stores numeric metrics as
                    # formatted strings (e.g. "0.85").  Try float coercion
                    # first so downstream arithmetic works correctly.
                    try:
                        collected[k] = float(v)
                    except (TypeError, ValueError):
                        # Keep only short, metadata-like strings (e.g. paths, labels).
                        # Exclude long explanatory messages (status, reason, note, etc.)
                        # which are not actual metrics.
                        if "/" in v or "\\" in v:
                            # File paths — always keep
                            collected[k] = v
                        elif k == "error":
                            # Always keep error messages for debugging
                            collected[k] = v[:500]
                        elif len(v) < 50 and " " not in v:
                            # Short enum-like values (e.g. "logistic_regression")
                            collected[k] = v
                        # else: skip — likely a status message or explanation
            # Recurse into sub-dicts
            for sub in sub_dicts:
                for sk, sv in _collect_metrics_recursive(sub, depth + 1).items():
                    if sk not in collected:
                        collected[sk] = sv
            return collected

        def _has_metric_keys(d: dict, max_depth: int = 0) -> bool:
            """Check if *d* (or any sub-dict up to *max_depth* levels down)
            contains at least one key in ``_KNOWN_METRIC_ALIASES``."""
            if set(d.keys()) & _KNOWN_METRIC_ALIASES:
                return True
            if max_depth > 0:
                for v in d.values():
                    if isinstance(v, dict) and _has_metric_keys(v, max_depth - 1):
                        return True
            return False

        def _collect_metric_sets(d: Any) -> tuple:
            """Detect and collect multiple metric sets from a result dict.

            When generated code runs multiple experiments (e.g., pole-aligned
            vs cue-aligned), the result dict may contain named sub-dicts that
            each have analysis metrics (accuracy, auroc, etc.).  This function:
            1. Identifies such sub-dicts as distinct metric sets
            2. Selects the best-performing set (by accuracy then auroc)
            3. Returns both primary (best) metrics and all sets for audit

            Returns:
                (primary_metrics, all_metric_sets) where:
                - primary_metrics: flat dict with best metrics merged with
                  top-level scalars (same format as _collect_metrics_recursive)
                - all_metric_sets: list of {"label": str, "metrics": dict}
                  (empty list if only one or zero metric sets found)
            """
            if not isinstance(d, dict):
                return _collect_metrics_recursive(d), []

            # Scan top-level keys for metric-set sub-dicts.
            # Also look ONE level deeper: if a top-level sub-dict is a
            # container (e.g. "analysis_results") whose children are the
            # actual metric sets, expand those children into candidates.
            #
            # _has_metric_keys(cv, max_depth=1) handles 3-level nesting
            # where each experiment stores its metrics in a child dict
            # (e.g. result → analysis_results → config → test_metrics).
            # Expand any DataFrame values into row-dicts before scanning
            expanded_d = {}
            for k, v in d.items():
                if _is_dataframe(v):
                    rows = _dataframe_to_dicts(v)
                    # Use a name column if available, else index
                    for i, row in enumerate(rows):
                        label = row.get("config_name") or row.get("name") or f"{k}/row_{i}"
                        expanded_d[str(label)] = row
                else:
                    expanded_d[k] = v

            metric_set_candidates = []
            for k, v in expanded_d.items():
                if not isinstance(v, dict):
                    continue
                if set(v.keys()) & _KNOWN_METRIC_ALIASES:
                    # Direct metric-set sub-dict
                    metric_set_candidates.append((k, v))
                else:
                    # Check if this is a container whose children are
                    # metric sets (2- or 3-level nesting pattern)
                    child_sets = [
                        (f"{k}/{ck}", cv)
                        for ck, cv in v.items()
                        if isinstance(cv, dict)
                        and _has_metric_keys(cv, max_depth=1)
                    ]
                    if len(child_sets) >= 2:
                        metric_set_candidates.extend(child_sets)

            # If fewer than 2 metric-set sub-dicts, fall back to original
            if len(metric_set_candidates) < 2:
                return _collect_metrics_recursive(expanded_d), []

            # --- Multiple metric sets detected ---
            # Collect each set separately
            all_metric_sets = []
            for label, sub_dict in metric_set_candidates:
                flat = _collect_metrics_recursive(sub_dict)
                all_metric_sets.append({"label": label, "metrics": flat})

            # Select best set by primary metric (accuracy first, then auroc)
            def _primary_value(ms_metrics: dict) -> float:
                for key in _PRIMARY_METRIC_CANDIDATES:
                    if key in ms_metrics:
                        try:
                            return float(ms_metrics[key])
                        except (TypeError, ValueError):
                            continue
                return -1.0

            best_idx = max(range(len(all_metric_sets)),
                          key=lambda i: _primary_value(all_metric_sets[i]["metrics"]))

            # Build primary_metrics: top-level scalars + non-metric-set
            # sub-dicts (via original flatten) + best metric set values
            # Start with full flatten (which has first-wins issue)
            primary_metrics = _collect_metrics_recursive(expanded_d)
            # Override with best metric set values (this is the key fix)
            primary_metrics.update(all_metric_sets[best_idx]["metrics"])

            print(
                f"📊 Multiple metric sets detected: "
                f"{[ms['label'] for ms in all_metric_sets]}. "
                f"Selected '{all_metric_sets[best_idx]['label']}' as primary "
                f"(best accuracy/auroc).",
                flush=True,
            )

            return primary_metrics, all_metric_sets

        def _build_exec_report_from_result(
            exec_result: CodeExecutionResult,
            generated_code_obj: GeneratedCode,
            plan: Dict[str, Any],
            run_dir: Path,
        ) -> Dict[str, Any]:
            """
            Build an exec_report dict compatible with Analysis Agent and
            the rest of the pipeline (diagnosis, audit serialization, etc.).
            """
            result_dict = exec_result.result if isinstance(exec_result.result, dict) else {}

            # First try explicit "metrics" key — but only trust it if it
            # contains real analysis indicators (accuracy, auroc, etc.).
            # A "metrics" dict with only metadata (n_trials, hidden_dim, ...)
            # is not useful and we should fall through to full extraction.
            metrics = result_dict.get("metrics", {})
            all_metric_sets: list = []
            _has_real_metrics = (
                bool(metrics)
                and bool(_KNOWN_METRIC_ALIASES & set(metrics.keys()))
            )
            if not _has_real_metrics:
                # Recursively collect from the full result tree.
                # _collect_metric_sets detects multiple experiment results
                # (e.g., pole-aligned vs cue-aligned) and picks the best.
                collected, all_metric_sets = _collect_metric_sets(result_dict)
                # Merge: keep explicit metrics as base, overlay with collected
                if metrics:
                    collected.update({k: v for k, v in metrics.items()
                                      if k not in collected})
                metrics = collected

            # Inject executor-measured wall time as canonical efficiency metric
            metrics["execution_time_seconds"] = round(exec_result.execution_time, 4)

            # Persist step output to disk for Analysis Agent
            step_output = {
                "execution_method": "code_generation",
                "ok": exec_result.success,
                "metrics": metrics,
                "generated_code": generated_code_obj.code,
                "stdout": exec_result.stdout[:5000] if exec_result.stdout else "",
            }
            if all_metric_sets:
                step_output["all_metric_sets"] = all_metric_sets
            step_output_path = run_dir / "step_0_output.json"
            step_output_path.write_text(json.dumps(step_output, indent=2, ensure_ascii=False, cls=_NumpySafeEncoder), encoding="utf-8")

            # Also save generated code for reproducibility
            (run_dir / "generated_code.py").write_text(generated_code_obj.code, encoding="utf-8")

            return {
                "ok": exec_result.success,
                "run_id": run_dir.name,
                "metrics": metrics,
                "all_metric_sets": all_metric_sets,
                "steps": [
                    {
                        "step_index": 0,
                        "op_name": ", ".join(generated_code_obj.tool_calls) if generated_code_obj.tool_calls else "code_generation",
                        "ok": exec_result.success,
                        "output_path": str(step_output_path),
                        "metrics": metrics,
                        "error": {
                            "message": exec_result.error or "",
                            "type": exec_result.error_type or "",
                            "traceback": exec_result.traceback or "",
                        } if not exec_result.success else None,
                    }
                ],
            }

        def _run_final_analysis(
            exec_report: Dict[str, Any],
            audit_obj: Dict[str, Any],
            target_met: bool,
            achieved_value: Optional[float] = None,
        ) -> tuple:
            """Run Analysis Agent and return (formatted_summary, AnalysisResult | None)."""
            try:
                # Determine outputs directory from exec_report
                run_id = exec_report.get("run_id")
                outputs_dir = None
                if run_id:
                    possible_paths = [
                        Path(f"outputs/{run_id}"),
                        audit_dir.parent.parent / "outputs" / run_id,
                        Path(os.environ.get("OUTPUTS_DIR", "outputs")) / run_id,
                    ]
                    for p_path in possible_paths:
                        if p_path and p_path.exists():
                            outputs_dir = p_path
                            break

                analysis_config = AnalysisAgentConfig(model=ANALYSIS_MODEL)
                analysis_result = run_analysis_agent(
                    state,
                    user_idea=user_idea,
                    config=analysis_config,
                    audit=audit_obj,
                    exec_report=exec_report,
                    outputs_dir=outputs_dir,
                    target_metric=metric_name,
                    target_threshold=metric_threshold,
                    achieved_value=achieved_value,
                    target_met=target_met,
                )
                state.extra["analysis_result"] = analysis_result.to_dict()
                state.extra["visualization_paths"] = get_visualization_paths(analysis_result)
                formatted_summary = format_analysis_for_display(analysis_result)
                return formatted_summary, analysis_result
            except Exception as e:
                print(f"⚠️ Analysis Agent error: {e}", flush=True)
                return "Execution complete (analysis summary unavailable).", None

        # Metric bookkeeping
        metric_name: Optional[str] = None
        metric_threshold: Optional[float] = None
        direction = "higher"
        if target_metric is not None and target_metric_threshold is not None:
            metric_name = target_metric
            metric_threshold = target_metric_threshold
        elif target_accuracy is not None:
            metric_name = "accuracy"
            metric_threshold = target_accuracy
        if metric_name:
            direction = _get_direction(metric_name)

        # ----- Auto Model Selection Mode -----
        # Triggered when:
        #   1. CONTROLLER_MODE=auto (explicit), OR
        #   2. Prompt doesn't specify or forbid pretrained models (auto-detect)
        # In auto mode: Codex baseline → model selection → early stop → Controller retry
        _is_auto_sub_run = dataset_info.get("_auto_sub_run", False) if dataset_info else False
        _prompt_lower = user_idea.lower()
        _forbids_pretrained = "do not use any pretrained" in _prompt_lower or "do not use pretrained" in _prompt_lower
        _specifies_model = any(
            kw in _prompt_lower
            for kw in ["you must use the", "decode_with_reve", "decode_with_brainomni",
                        "decode_with_cbramod", "decode_with_eegpt", "decode_with_eegmamba",
                        "decode_with_labram", "decode_with_ndt3", "decode_with_mtm",
                        "decode_with_poyo"]
        )

        import os as _auto_os
        _explicit_controller_mode = _auto_os.environ.get("CONTROLLER_MODE", "").strip().lower()

        # Auto mode triggers when:
        # - User explicitly set CONTROLLER_MODE=auto, OR
        # - Prompt doesn't forbid/specify models (auto-detect)
        # Does NOT trigger for sub-runs (they use off/force internally)
        _should_auto = (
            not _is_auto_sub_run
            and (
                _explicit_controller_mode == "auto"
                or (not _forbids_pretrained and not _specifies_model)
            )
        )

        if _should_auto:
            _data_summary_text = state.extra.get("data_summary_text") or ""
            # All available pretrained model tools in the system
            _ALL_MODEL_TOOLS = (
                "Available pretrained foundation model tools:\n"
                "  - decode_with_reve: EEG foundation model (~900M params, 60,000h pretraining, 1536-dim features, any electrode montage)\n"
                "  - decode_with_brainomni: EEG foundation model (~60M params, 2,650h pretraining, 4096-dim features, standard electrodes)\n"
                "  - decode_with_cbramod: EEG foundation model (~4M params, 9,000h pretraining, 600-dim features, channel-agnostic)\n"
                "  - decode_with_eegpt: EEG foundation model (~750M params, 2048-dim features, standard 10-10/10-20 channels)\n"
                "  - decode_with_eegmamba: EEG foundation model (~3.3M params, 16,700h pretraining, 600-dim features, channel-agnostic)\n"
                "  - decode_with_labram: EEG foundation model (~180M params, 2,500h pretraining, 200-dim features, standard 10-20 channels)\n"
                "  - decode_with_ndt3: Spike foundation model (45M params, pretrained on 2,000h spike data, primate/multi-species)\n"
                "  - decode_with_mtm: Spike foundation model (mouse Neuropixels/silicon-probe data, NDT1 encoder)\n"
                "  - decode_with_poyo1: Spike foundation model (multi-species, variable neuron counts, PerceiverIO architecture)\n"
            )
            # --- Run Planner and Codex baseline in parallel ---
            # Planner decides primary_metric + model_order based on task & data only.
            # Codex runs independently and outputs all metrics.
            # After both finish, we extract the Planner-chosen metric from Codex results.
            # This ensures Planner cannot "cheat" by seeing Codex results, and saves time.
            import concurrent.futures

            _dataset_path = (dataset_info.get("dataset_root_dir")
                             or dataset_info.get("dataset_path")
                             or dataset_info.get("path")
                             or dataset_info.get("file_path")
                             or "")

            def _run_planner():
                from neuro_copilot.core.llm_client import generate_json as _auto_gen_json
                from neuro_copilot.core.config import PLANNER_MODEL
                resp = _auto_gen_json(
                    model=PLANNER_MODEL,
                    system_prompt=("You are a neuroscience data analysis planner. "
                            "Given the dataset summary, task description, and list of available pretrained model tools, decide:\n"
                            "1. primary_metric: which metric to use for comparing models "
                            "(e.g., 'balanced_accuracy' for imbalanced classification, "
                            "'r2' for regression, 'macro_f1' for multi-class). "
                            "Choose based ONLY on the task and data characteristics.\n"
                            "2. model_order: list of model names to try, ordered from most promising to least. "
                            "Always include 'vanilla' (traditional feature extraction without pretrained models). "
                            "Only include models whose data type matches the dataset.\n"
                            "3. model_constraints: a dict mapping each model name to a prompt constraint string. "
                            "For vanilla, the constraint should forbid pretrained models. "
                            "For each pretrained model, the constraint should require using that specific model.\n"
                            "4. reasoning: brief explanation of your choices.\n"
                            "Return JSON: {\"primary_metric\": \"...\", "
                            "\"model_order\": [...], \"model_constraints\": {\"model_name\": \"constraint string\", ...}, "
                            "\"reasoning\": \"...\"}"),
                    user_prompt=f"Task: {user_idea}\n\n{_ALL_MODEL_TOOLS}\n\nDataset summary:\n{_data_summary_text[:2000]}",
                    temperature=0.2,
                )
                if hasattr(resp, 'json_obj') and resp.json_obj:
                    return resp.json_obj
                elif hasattr(resp, 'raw_text'):
                    return json.loads(resp.raw_text)
                return resp

            def _run_codex():
                _codex_model_display = os.environ.get("CODEX_BASELINE_MODEL", "gpt-5.2-codex")
                print(f"\n{'─' * 50}", flush=True)
                print(f"🤖 Running Codex baseline via CLI ({_codex_model_display})", flush=True)
                print(f"{'─' * 50}", flush=True)
                return self._run_codex_cli_baseline(
                    prompt=user_idea,
                    dataset_path=_dataset_path,
                    audit_dir=str(audit_dir / "codex_baseline"),
                    primary_metric=None,  # Don't need metric yet — extract after Planner decides
                    timeout=600,
                    additional_files=dataset_info.get("additional_files"),
                )

            # Run Planner and Codex in parallel — both receive only prompt + dataset,
            # neither can see the other's output. This ensures fairness (Planner
            # cannot pick a metric that favors our pipeline over Codex).
            from neuro_copilot.core.progress_events import get_tracker as _get_tracker_early
            _get_tracker_early().emit(
                "codex_start", "🤖 Running Codex Baseline & Planner",
                "Codex baseline and Planner are running in parallel...",
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                planner_future = pool.submit(_run_planner)
                codex_future = pool.submit(_run_codex)

                try:
                    _auto_plan = planner_future.result()
                except Exception as e:
                    print(f"❌ Auto mode Planner failed: {e}", flush=True)
                    _tracker.emit("error", "❌ Planner Failed", f"Planner LLM call failed: {e}")
                    return DAOutput(
                        response=f"Planner failed: {e}. Please check your API key, network connection, and rate limits.",
                        needs_dataset=False,
                        analysis_type="execution",
                        confidence=0.0,
                    )

                _codex_result = codex_future.result()

            _primary_metric = _auto_plan.get("primary_metric")
            if not _primary_metric:
                print("❌ Planner did not return a primary_metric", flush=True)
                return DAOutput(
                    response="Planner did not determine a primary metric. Please try again.",
                    needs_dataset=False,
                    analysis_type="execution",
                    confidence=0.0,
                )
            _model_order = _auto_plan.get("model_order", [])
            _model_constraints = _auto_plan.get("model_constraints", {})
            _auto_reasoning = _auto_plan.get("reasoning", "")

            # Friendly display names for models
            _DISPLAY_NAMES = {
                "decode_with_ndt3": "NDT3",
                "decode_with_mtm": "MtM",
                "decode_with_poyo1": "POYO-1",
                "decode_with_reve": "REVE",
                "decode_with_labram": "LaBraM",
                "decode_with_cbramod": "CBraMod",
                "decode_with_eegpt": "EEGPT",
                "decode_with_eegmamba": "EEGMamba",
                "decode_with_brainomni": "BrainOmni",
                "vanilla": "Vanilla",
            }
            def _dname(m):
                return _DISPLAY_NAMES.get(m, m)

            print("=" * 60, flush=True)
            print("🔄 AUTO MODEL SELECTION MODE (with Codex baseline)", flush=True)
            print(f"📊 Primary metric: {_primary_metric}", flush=True)
            print(f"📋 Model order: {[_dname(m) for m in _model_order]}", flush=True)
            print(f"💭 Reasoning: {_auto_reasoning}", flush=True)
            print("=" * 60, flush=True)

            # Emit Planner progress event for web UI
            try:
                from neuro_copilot.core.progress_events import get_tracker
                _tracker = get_tracker()
                _tracker.emit(
                    "planner", "📋 Planner Decision",
                    f"**Primary Metric:** {_primary_metric}\n\n"
                    f"**Model Queue:** {', '.join(_dname(m) for m in _model_order)}\n\n"
                    f"**Reasoning:** {_auto_reasoning}",
                )
            except Exception:
                pass

            # Flexible metric lookup: find the best matching key in a metrics dict
            def _find_metric_value(metrics_dict, metric_name):
                """Search metrics dict for a key matching metric_name.
                Recursively flattens nested dicts, then uses token subset
                matching (shortest superset wins). Zero hardcoded patterns."""
                if not metrics_dict or not metric_name:
                    return None
                target_tokens = set(metric_name.lower().replace("-", "_").split("_"))
                # Recursively collect all leaf numeric values
                candidates = []
                def _collect(d, prefix=""):
                    for k, v in d.items():
                        path = f"{prefix}/{k}" if prefix else k
                        if isinstance(v, dict):
                            _collect(v, path)
                        elif isinstance(v, (int, float)) and v is not None:
                            candidates.append((path, k.lower(), v))
                _collect(metrics_dict)
                # Token subset match: target tokens must be subset of key tokens
                # Prefer shortest superset (fewest extra tokens = closest match)
                matches = []
                for path, leaf, v in candidates:
                    key_tokens = set(leaf.split("_"))
                    if target_tokens <= key_tokens:
                        matches.append((len(key_tokens) - len(target_tokens), path, v))
                if matches:
                    matches.sort(key=lambda x: x[0])
                    return matches[0][2]
                # No match found
                if candidates:
                    available = [p for p, _, _ in candidates]
                    print(f"⚠️ Metric '{metric_name}' not found in results. "
                          f"Available numeric keys: {available[:10]}", flush=True)
                return None

            # Ensure vanilla is in the list if not already
            if "vanilla" not in _model_order:
                _model_order.append("vanilla")
            if "vanilla" not in _model_constraints:
                _model_constraints["vanilla"] = "Do NOT use any pretrained foundation models."

            # Filter model_order to only those with constraints defined
            _model_order = [m for m in _model_order if m in _model_constraints]

            # Extract Codex baseline using Planner-chosen metric
            _codex_metrics = _codex_result.get("metrics", {})
            _codex_baseline_value = _find_metric_value(_codex_metrics, _primary_metric)
            _codex_dur = _codex_result.get("duration_s", 0)
            print(f"✅ Codex baseline: {_primary_metric}={_codex_baseline_value} ({_codex_dur:.0f}s)", flush=True)

            # --- Effective early-stopping threshold -----------------------------
            # The user may supply AUTO_TARGET_VALUE to replace the Codex-derived
            # baseline.  It applies to the metric the Planner already chose (which
            # is also the metric the Codex baseline is read on), so no metric
            # matching, no direction handling and no override is involved: only the
            # right-hand side of the existing comparison changes.  With the
            # variable unset, _stop_threshold IS _codex_baseline_value and every
            # comparison and message below is unchanged.
            _stop_threshold = _codex_baseline_value
            _stop_source = "Codex baseline"
            _stop_from_user = False
            _user_target_env = _auto_os.environ.get("AUTO_TARGET_VALUE", "")
            _user_target_raw = _user_target_env.strip()
            if _user_target_env and not _user_target_raw:
                print(
                    "⚠️ AUTO_TARGET_VALUE is whitespace only. "
                    "Using the Codex baseline instead.",
                    flush=True,
                )
            if _user_target_raw:
                try:
                    _user_target = float(_user_target_raw)
                except ValueError:
                    print(
                        f"⚠️ AUTO_TARGET_VALUE '{_user_target_raw}' is not a number. "
                        f"Using the Codex baseline instead.",
                        flush=True,
                    )
                else:
                    if math.isfinite(_user_target):
                        _stop_threshold = _user_target
                        _stop_source = "user target"
                        _stop_from_user = True
                        print(
                            f"🎯 Early-stopping threshold from user: "
                            f"{_primary_metric} > {_stop_threshold}",
                            flush=True,
                        )
                    else:
                        print(
                            f"⚠️ AUTO_TARGET_VALUE '{_user_target_raw}' is not finite. "
                            f"Using the Codex baseline instead.",
                            flush=True,
                        )

            # Emit Codex baseline progress event
            try:
                _tracker.emit(
                    "codex", "🤖 Codex Baseline Result",
                    f"**{_primary_metric}:** {_codex_baseline_value:.4f}\n\n"
                    f"**Duration:** {_codex_dur:.0f}s\n\n"
                    f"**Metrics:** {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in _codex_metrics.items() if isinstance(v, (int, float))}, indent=2)}" if _codex_baseline_value else "Codex baseline failed.",
                )
            except Exception:
                pass

            if _codex_baseline_value is None:
                _codex_fail_reason = _codex_result.get("error") or (
                    f"Could not extract '{_primary_metric}' from Codex metrics: "
                    f"{list(_codex_metrics.keys())[:10]}"
                )
                if _stop_threshold is None:
                    print(
                        f"⚠️ Codex baseline failed ({_codex_fail_reason}). "
                        f"Continuing without baseline — early stopping disabled.",
                        flush=True,
                    )
                    _tracker.emit(
                        "codex", "⚠️ Codex Baseline Failed",
                        f"Codex baseline could not produce {_primary_metric}. "
                        f"Continuing without baseline comparison — all models will be evaluated.",
                    )
                else:
                    print(
                        f"⚠️ Codex baseline failed ({_codex_fail_reason}). "
                        f"Early stopping still applies, using the {_stop_source}.",
                        flush=True,
                    )
                    _tracker.emit(
                        "codex", "⚠️ Codex Baseline Failed",
                        f"Codex baseline could not produce {_primary_metric}. "
                        f"Early stopping still applies against the {_stop_source}.",
                    )

            # --- Step 3: Run models sequentially with early stopping ---
            # Each round: run every model ONCE (no internal controller retries).
            # Retry rounds = controller optimizes each model's code, then re-run once.
            auto_results = {}  # model_name -> {value, metrics, output, code, plan}
            _best_model = None
            _best_value = None
            _best_metrics = {}
            _stopped_early = False

            MAX_RETRIES = int(os.environ.get("AUTO_MAX_RETRIES", "2"))  # Configurable via web UI

            for retry_round in range(MAX_RETRIES + 1):  # 0=initial, 1=retry1, 2=retry2
                round_name = "Initial" if retry_round == 0 else f"Retry {retry_round}"
                print(f"\n{'=' * 50}", flush=True)
                print(f"🔄 {round_name} round", flush=True)
                print(f"{'=' * 50}", flush=True)

                # Reorder models for retry rounds based on improvement potential
                _round_model_order = _model_order
                if retry_round > 0 and auto_results:
                    try:
                        _round_model_order = self.controller.reorder_models_for_retry(
                            model_results=auto_results,
                            primary_metric=_primary_metric,
                            codex_baseline_value=_codex_baseline_value,
                            model_order=_model_order,
                        )
                    except Exception as _reorder_err:
                        print(f"⚠️ Model reordering failed: {_reorder_err}", flush=True)
                        _round_model_order = _model_order
                    print(f"📋 Round model order: {[_dname(m) for m in _round_model_order]}", flush=True)

                for _model_idx, model_name in enumerate(_round_model_order):
                    model_constraint = _model_constraints.get(model_name, "")
                    print(f"\n{'─' * 40}", flush=True)
                    print(f"🧠 {round_name}: running {_dname(model_name)}", flush=True)
                    print(f"{'─' * 40}", flush=True)

                    # Emit progress: model starting
                    _model_pos = f"({_model_idx + 1}/{len(_round_model_order)})"
                    _tracker.emit(
                        "model_running", f"🧠 Running {_dname(model_name)} {_model_pos}",
                        f"**{round_name}** — {_dname(model_name)} is being executed...",
                    )

                    sub_prompt = user_idea.rstrip() + " " + model_constraint
                    sub_dataset_info = dict(dataset_info) if dataset_info else {}
                    sub_dataset_info["_auto_sub_run"] = True
                    # Each round gets its own audit subdirectory
                    sub_dataset_info["_auto_audit_dir"] = str(
                        audit_dir / model_name / round_name.lower().replace(" ", "_")
                    )
                    # No internal controller retries — each model runs once per round
                    sub_dataset_info["_no_retries"] = True

                    # For retry rounds, use Controller diagnosis for targeted feedback
                    if retry_round > 0 and model_name in auto_results:
                        prev = auto_results[model_name]
                        if prev and prev.get("metrics"):
                            # Run Controller diagnosis on previous result
                            _prev_diag = None
                            _prev_plan = prev.get("plan") or {}
                            try:
                                _prev_diag = self.controller.diagnose_execution_result(
                                    task_type=_prev_plan.get("task_type", "decoding") if isinstance(_prev_plan, dict) else "decoding",
                                    user_request=user_idea,
                                    metrics=prev.get("metrics", {}),
                                    plan=prev.get("plan") or {},
                                    generated_code=prev.get("generated_code", ""),
                                    force_improve=True,
                                )
                                if _prev_diag and _prev_diag.recommendations:
                                    print(f"🔍 Controller diagnosis for {_dname(model_name)}: "
                                          f"action={_prev_diag.suggested_action}, "
                                          f"recs={_prev_diag.recommendations[:2]}", flush=True)
                            except Exception as _diag_err:
                                print(f"⚠️ Controller diagnosis failed for {_dname(model_name)}: {_diag_err}",
                                      flush=True)

                            _retry_action = (_prev_diag.suggested_action if _prev_diag else None) or "replan"
                            _prev_code = prev.get("generated_code", "")

                            if _retry_action == "adjust_params" and _prev_code.strip():
                                # MODIFY MODE: skip Planner, modify existing code
                                print(f"🔧 Auto retry: modify mode for {_dname(model_name)}", flush=True)
                                _metrics_for_modify = {
                                    k: v for k, v in prev.get("metrics", {}).items()
                                    if isinstance(v, (int, float, str))
                                }
                                sub_dataset_info["_modify_mode"] = True
                                sub_dataset_info["_modify_original_code"] = _prev_code
                                sub_dataset_info["_modify_recommendations"] = (
                                    _prev_diag.recommendations[:5] if _prev_diag else []
                                )
                                sub_dataset_info["_modify_metrics_summary"] = json.dumps(
                                    _metrics_for_modify, indent=2, ensure_ascii=False
                                )
                                sub_dataset_info["_modify_previous_plan"] = prev.get("plan") or {}
                            else:
                                # REPLAN MODE: full Planner → Code Generator
                                print(f"📝 Auto retry: replan mode for {_dname(model_name)}", flush=True)
                                _diag_feedback = (
                                    f"Previous {_primary_metric}={prev.get('value')} "
                                    f"(Codex baseline={_codex_baseline_value}).\n"
                                )
                                if _prev_diag and _prev_diag.recommendations:
                                    _diag_feedback += (
                                        f"Controller diagnosis:\n"
                                        f"  Anomalies: {'; '.join(_prev_diag.anomalies) if _prev_diag.anomalies else 'None'}\n"
                                        f"  Recommendations: {'; '.join(_prev_diag.recommendations)}\n"
                                    )
                                else:
                                    _diag_feedback += "Improve to exceed the baseline."
                                sub_dataset_info["retry_context"] = {
                                    "previous_plan": prev.get("plan"),
                                    "metrics": prev.get("metrics", {}),
                                    "diagnosis_feedback": _diag_feedback,
                                }

                    try:
                        import time as _time_mod
                        _sub_start_time = _time_mod.time()
                        sub_output = self._run_da_execution(sub_prompt, sub_dataset_info)
                        _sub_wall_time = round(_time_mod.time() - _sub_start_time, 2)

                        # Extract metrics from the sub-run audit
                        _sub_audit_dir = sub_dataset_info["_auto_audit_dir"]
                        _sub_audit_path = Path(_sub_audit_dir) / "audit.json"
                        _sub_value = None
                        _sub_metrics = {}
                        _sub_plan = None
                        _sub_generated_code = ""
                        if _sub_audit_path.exists():
                            with open(_sub_audit_path) as _sf:
                                _sub_audit = json.load(_sf)
                            _sub_attempts = _sub_audit.get("attempts", [])
                            if _sub_attempts:
                                er = _sub_attempts[-1].get("execution_result")
                                if er:
                                    _sub_metrics = er.get("metrics", {})
                                    _sub_value = _find_metric_value(_sub_metrics, _primary_metric)
                                _sub_plan = _sub_attempts[-1].get("plan")
                                _sub_generated_code = (_sub_attempts[-1]
                                    .get("code_generation", {})
                                    .get("code", ""))

                        # Update this model's best result across all rounds
                        prev_best = auto_results.get(model_name, {}).get("value")
                        _sub_result_entry = {
                            "output": sub_output,
                            "value": _sub_value,
                            "metrics": _sub_metrics,
                            "plan": _sub_plan,
                            "generated_code": _sub_generated_code,
                            "round": round_name,
                            "wall_time_s": _sub_wall_time,
                            "audit_dir": str(_sub_audit_dir),
                            "exec_result": er,  # full execution_result from sub-run audit
                        }
                        if _sub_value is not None and (prev_best is None or _sub_value > prev_best):
                            auto_results[model_name] = _sub_result_entry
                        elif model_name not in auto_results:
                            auto_results[model_name] = _sub_result_entry

                        # Update global best
                        _cur_best = auto_results[model_name].get("value")
                        if _cur_best is not None:
                            if _best_value is None or _cur_best > _best_value:
                                _best_value = _cur_best
                                _best_model = model_name
                                _best_metrics = auto_results[model_name].get("metrics", {})

                            # Early stopping: exceed the effective threshold
                            if _stop_threshold is not None and _cur_best > _stop_threshold:
                                print(f"🎯 {_dname(model_name)} ({_primary_metric}={_cur_best:.4f}) "
                                      f"> {_stop_source} ({_stop_threshold:.4f}). "
                                      f"Early stopping!", flush=True)
                                _stopped_early = True

                                # Emit early stop + code events
                                try:
                                    _best_code = auto_results[model_name].get("generated_code", "")
                                    _best_round = auto_results[model_name].get("round", "")
                                    _tracker.emit(
                                        "code_generator", "💻 Generated Code (Winner)",
                                        f"**Model:** {_dname(model_name)} ({_best_round})\n\n"
                                        f"```python\n{_best_code[:3000]}\n```" if _best_code else "Code not available.",
                                    )
                                    _tracker.emit(
                                        "early_stop", "🎯 Early Stopping Triggered",
                                        f"**{_dname(model_name)}** ({_primary_metric}={_cur_best:.4f}) "
                                        f"exceeded {_stop_source} ({_stop_threshold:.4f}).\n\n"
                                        f"Search terminated successfully.",
                                    )
                                except Exception:
                                    pass
                                break

                        print(f"✅ {round_name}: {_dname(model_name)} "
                              f"({_primary_metric}={_sub_value if _sub_value else '—'})", flush=True)

                        # Emit progress: model completed but no early stop
                        _next_model = None
                        if _model_idx + 1 < len(_round_model_order):
                            _next_model = _round_model_order[_model_idx + 1]
                        if _sub_value is not None and _stop_threshold is not None:
                            _tracker.emit(
                                "model_done", f"✅ {_dname(model_name)} Completed",
                                f"**{round_name}** — {_primary_metric} = {_sub_value:.4f} "
                                f"({_stop_source}: {_stop_threshold:.4f}). "
                                f"Early stopping not triggered."
                                + (f" Now running **{_dname(_next_model)}**..." if _next_model else ""),
                            )
                        elif _sub_value is None:
                            _tracker.emit(
                                "model_done", f"⚠️ {_dname(model_name)} — No Metric",
                                f"**{round_name}** — Execution succeeded but no {_primary_metric} extracted."
                                + (f" Now running **{_dname(_next_model)}**..." if _next_model else ""),
                            )

                    except Exception as e:
                        print(f"❌ {round_name}: {_dname(model_name)} failed: {e}", flush=True)

                        # Emit progress: model failed
                        _next_model = None
                        if _model_idx + 1 < len(_round_model_order):
                            _next_model = _round_model_order[_model_idx + 1]
                        _tracker.emit(
                            "model_failed", f"❌ {_dname(model_name)} Failed",
                            f"**{round_name}** — {_dname(model_name)} encountered an error."
                            + (f" Now running **{_dname(_next_model)}**..." if _next_model else ""),
                        )
                        if model_name not in auto_results:
                            auto_results[model_name] = {"value": None, "metrics": {}, "round": round_name}

                if _stopped_early:
                    break

                # If all models ran and none beat baseline, continue to next retry round
                if retry_round < MAX_RETRIES:
                    print(f"\n⚠️ No model exceeded {_stop_source} in {round_name}. "
                          f"Proceeding to Retry {retry_round + 1}...", flush=True)

            # --- Step 4: Save results ---
            audit["auto_mode"] = True
            audit["auto_mode_version"] = "v2_codex_baseline"
            audit["primary_metric"] = _primary_metric
            audit["primary_metric_key"] = _primary_metric
            audit["model_order"] = _model_order
            audit["auto_reasoning"] = _auto_reasoning
            audit["codex_baseline"] = {
                "metrics": _codex_metrics,
                "primary_metric_value": _codex_baseline_value,
                "duration_s": _codex_dur if '_codex_dur' in dir() else None,
            }
            audit["stop_threshold"] = _stop_threshold
            audit["stop_threshold_source"] = _stop_source
            audit["best_model"] = _best_model
            audit["best_value"] = _best_value
            audit["best_metrics"] = _best_metrics
            audit["stopped_early"] = _stopped_early
            audit["auto_models"] = list(auto_results.keys())
            # Per-model detailed results (metrics + wall time)
            audit["auto_model_results"] = {}
            auto_summary_parts = []
            _reached_models_audit = set(auto_results.keys())
            for mn in _model_order:
                r = auto_results.get(mn)
                if r and isinstance(r, dict) and r.get("value") is not None:
                    auto_summary_parts.append(f"**{_dname(mn)}**: {_primary_metric}={r['value']:.4f} ({r.get('round', '?')})")
                    audit["auto_model_results"][mn] = {
                        "value": r["value"],
                        "metrics": r.get("metrics", {}),
                        "round": r.get("round"),
                        "wall_time_s": r.get("wall_time_s"),
                    }
                elif mn in _reached_models_audit:
                    auto_summary_parts.append(f"**{_dname(mn)}**: execution failed")
                    if r and isinstance(r, dict):
                        audit["auto_model_results"][mn] = {
                            "value": None,
                            "metrics": r.get("metrics", {}),
                            "round": r.get("round"),
                            "wall_time_s": r.get("wall_time_s"),
                        }
                else:
                    auto_summary_parts.append(f"**{_dname(mn)}**: skipped (early stopping)")
            audit["auto_summary"] = "; ".join(auto_summary_parts)

            # --- Step 5: Run Analysis Agent on the best model's result ---
            _analysis_summary = None
            if _best_model and _best_model in auto_results:
                _best_r = auto_results[_best_model]
                _best_output = _best_r.get("output")
                if _best_output:
                    try:
                        print(f"\n📊 Running Analysis Agent on best model: {_dname(_best_model)}", flush=True)
                        # Build a proper exec_report from sub-run data.
                        # _run_final_analysis / Analysis Agent expects:
                        #   ok, run_id, metrics, steps[{op_name, ok, metrics, ...}]
                        _sub_er = _best_r.get("exec_result") or {}
                        _sub_audit_dir_str = _best_r.get("audit_dir", "")
                        _auto_exec_report = {
                            "ok": _sub_er.get("ok", _best_value is not None),
                            "run_id": Path(_sub_audit_dir_str).name if _sub_audit_dir_str else "auto_best",
                            "metrics": _best_r.get("metrics", {}),
                            "steps": [{
                                "step_index": 0,
                                "op_name": _best_model,
                                "ok": _sub_er.get("ok", _best_value is not None),
                                "metrics": _best_r.get("metrics", {}),
                                "stdout": _sub_er.get("stdout", ""),
                                "error": {"message": _sub_er.get("error", "")} if _sub_er.get("error") else None,
                            }],
                        }
                        _analysis_summary, _ = _run_final_analysis(
                            exec_report=_auto_exec_report,
                            audit_obj=audit,
                            target_met=_stopped_early,
                            achieved_value=_best_value,
                        )
                        if _analysis_summary:
                            audit["analysis_summary"] = _analysis_summary[:500]
                            print(f"📊 Analysis Agent finished for {_dname(_best_model)}", flush=True)
                            # Emit Analysis Agent event
                            try:
                                _tracker.emit(
                                    "analysis", "📊 Analysis Agent Report",
                                    _analysis_summary,
                                )
                            except Exception:
                                pass
                    except Exception as _ae:
                        print(f"⚠️ Analysis Agent error: {_ae}", flush=True)

            _save_audit(audit)

            # --- Metric normalization via config.METRIC_ALIASES ---
            # Reuse the same alias→canonical mapping from config.py.
            # No hardcoded metric lists — new metrics added to config
            # are automatically picked up here.
            _auto_alias_to_canonical: dict = {}
            _auto_alias_to_priority: dict = {}
            for _canon, _entry in _METRIC_ALIASES.items():
                for _pri, _alias in enumerate(_entry.get("aliases", [_canon])):
                    _auto_alias_to_canonical[_alias] = _canon
                    _auto_alias_to_priority[_alias] = _pri

            def _normalize_to_canonical(raw_metrics: dict) -> dict:
                """Normalize raw metrics dict → {canonical_name: value}.

                Uses METRIC_ALIASES from config.py. Only returns metrics
                that have a registered canonical name (filters out all
                non-metric fields like seed, is_good, n_classes, etc.).
                When multiple raw keys map to the same canonical,
                the highest-priority alias wins.
                """
                out: dict = {}
                out_pri: dict = {}
                for raw_key, raw_val in raw_metrics.items():
                    if isinstance(raw_val, bool):
                        continue
                    try:
                        fval = round(float(raw_val), 4)
                    except (TypeError, ValueError):
                        continue
                    canon = _auto_alias_to_canonical.get(raw_key)
                    if canon is None:
                        continue  # Not a known metric → skip
                    pri = _auto_alias_to_priority.get(raw_key, 999)
                    if canon not in out or pri < out_pri[canon]:
                        out[canon] = fval
                        out_pri[canon] = pri
                return out

            # Did the best model actually beat the Codex baseline?
            _best_beat_baseline = (
                _best_value is not None
                and _stop_threshold is not None
                and _best_value > _stop_threshold
            )

            # Build response
            response_lines = [
                "## Auto Model Selection Results (with Codex Baseline)\n",
                f"**Primary metric**: {_primary_metric}",
                f"**Codex baseline**: {_codex_baseline_value}",
            ]
            if _stop_from_user:
                response_lines.append(
                    f"**Stopping criterion**: {_primary_metric} > {_stop_threshold} "
                    f"({_stop_source})"
                )
            if _best_beat_baseline:
                response_lines.append(f"**Best model**: 🏆 {_dname(_best_model)} ({_primary_metric}={_best_value:.4f})")
            elif _best_model and _best_value is not None:
                response_lines.append(f"**Best model**: {_dname(_best_model)} ({_primary_metric}={_best_value:.4f}, did not exceed {_stop_source})")
            else:
                response_lines.append("**Best model**: none (all models failed)")
            response_lines.append(f"**Early stopped**: {_stopped_early}\n")
            response_lines.append(f"**Model order** (LLM-decided): {[_dname(m) for m in _model_order]}")
            response_lines.append(f"**Reasoning**: {_auto_reasoning}\n")
            response_lines.append("### Per-model results:")

            _reached_models = set(auto_results.keys())
            for mn in _model_order:
                r = auto_results.get(mn)
                if r and isinstance(r, dict) and r.get("value") is not None:
                    if mn == _best_model and _best_beat_baseline:
                        response_lines.append(f"🏆 **{_dname(mn)}**: {_primary_metric}={r['value']:.4f} ({r.get('round', '?')})")
                    elif _stop_threshold is not None and r['value'] <= _stop_threshold:
                        response_lines.append(f"  **{_dname(mn)}**: {_primary_metric}={r['value']:.4f} ≤ {_stop_source} ({_stop_threshold:.4f})")
                    else:
                        response_lines.append(f"  **{_dname(mn)}**: {_primary_metric}={r['value']:.4f} ({r.get('round', '?')})")
                elif mn in _reached_models:
                    response_lines.append(f"  **{_dname(mn)}**: execution failed")
                else:
                    response_lines.append(f"  **{_dname(mn)}**: skipped (early stopping triggered)")

            # Comparison table: Codex baseline vs Best model
            # Uses _normalize_to_canonical to filter & align metrics
            _best_r = auto_results.get(_best_model, {})
            _best_all_metrics = _best_r.get("metrics", {}) if isinstance(_best_r, dict) else {}

            _codex_canonical = _normalize_to_canonical(_codex_metrics)
            _best_canonical = _normalize_to_canonical(_best_all_metrics)

            # Show canonical metrics where at least one side has a value
            _all_canonical = sorted(set(_codex_canonical) | set(_best_canonical))
            _table_rows = [(k, _codex_canonical.get(k), _best_canonical.get(k))
                           for k in _all_canonical]

            # The table compares against Codex specifically, so its trophy must mean
            # "beat Codex", not "met the stopping criterion" (which _best_beat_baseline
            # now means).  Identical to _best_beat_baseline when no user target is set.
            _best_beat_codex = (
                _best_value is not None
                and _codex_baseline_value is not None
                and _best_value > _codex_baseline_value
            )
            if _table_rows and _best_model:
                _best_label = ("🏆 " if _best_beat_codex else "") + _dname(_best_model)
                response_lines.append("\n### Comparison: Codex Baseline vs Best Model\n")
                response_lines.append(f"| Metric | Codex Baseline | {_best_label} |")
                response_lines.append("|--------|---------------|" + "-" * (len(_best_label) + 2) + "|")

                for k, codex_val, best_val in _table_rows:
                    codex_str = f"{codex_val:.4f}" if codex_val is not None else "—"
                    if best_val is not None:
                        if codex_val is not None and best_val > codex_val:
                            best_str = f"**{best_val:.4f}**"
                        else:
                            best_str = f"{best_val:.4f}"
                    else:
                        best_str = "—"
                    response_lines.append(f"| {k} | {codex_str} | {best_str} |")

            combined_response = "\n".join(response_lines)

            # Collect visualization paths from the best model's sub-run
            _viz_paths = []
            if _best_model:
                _best_audit_dir = auto_results.get(_best_model, {}).get("audit_dir", "")
                if _best_audit_dir:
                    import glob as _glob_viz
                    _viz_patterns = [
                        os.path.join(_best_audit_dir, "execution", "*.png"),
                        os.path.join(_best_audit_dir, "execution", "*.pdf"),
                    ]
                    for _pat in _viz_patterns:
                        _viz_paths.extend(_glob_viz.glob(_pat))

            _raw_json = {}
            if _viz_paths:
                _raw_json["visualization_paths"] = _viz_paths

            return DAOutput(
                response=combined_response,
                needs_dataset=False,
                analysis_type="execution",
                confidence=0.8,
                raw_json=_raw_json,
            )

        # ----- Check for modify mode (auto mode retry with Controller diagnosis) -----
        _modify_mode = (dataset_info or {}).get("_modify_mode", False)
        _modify_original_code = (dataset_info or {}).get("_modify_original_code", "")
        _modify_recommendations = (dataset_info or {}).get("_modify_recommendations", [])
        _modify_metrics_summary = (dataset_info or {}).get("_modify_metrics_summary", "")

        if _modify_mode and _modify_original_code:
            # Skip Planner — directly modify existing code
            print("🔧 DA execution: MODIFY MODE — skipping Planner, modifying existing code", flush=True)
            planner_duration = 0.0
            validation_feedback = None
            plan = (dataset_info or {}).get("_modify_previous_plan", {"task_type": "decoding", "steps": []})
        else:
            # ----- Phase 1: Planner -----
            print("🤖 DA execution: call Planner agent", flush=True)
            planner_start = time.time()

            validation_feedback: Optional[str] = None
            previous_plan = None
            if retry_context:
                print("🔄 DA execution: RETRY MODE", flush=True)
                previous_plan = retry_context.get("previous_plan")
                diagnosis_feedback = retry_context.get("diagnosis_feedback", "")
                validation_feedback = (
                    f"RETRY MODE: Previous analysis produced poor results.\n"
                    f"PREVIOUS PLAN:\n{json.dumps(previous_plan, indent=2, ensure_ascii=False) if previous_plan else 'N/A'}\n"
                    f"METRICS FROM PREVIOUS RUN:\n{json.dumps(retry_context.get('metrics', {}), indent=2, ensure_ascii=False)}\n"
                    f"DIAGNOSIS:\n{diagnosis_feedback}\n"
                    f"Generate an IMPROVED plan that fixes these issues."
                )

            state = run_planner_agent(
                state,
                user_idea=user_idea,
                tools_whitelist_text=None,  # No longer needed
                validation_feedback=validation_feedback,
                previous_plan=previous_plan,
            )
            plan = state.extra.get("analysis_plan_json")

            planner_end = time.time()
            planner_duration = round(planner_end - planner_start, 3)
            print(f"📋 DA execution: Planner finished ({planner_duration}s)", flush=True)

            if plan is None:
                _save_audit(audit)
                return DAOutput(
                    response="Failed to generate a valid analysis plan.",
                    needs_dataset=False,
                    analysis_type="execution",
                    confidence=0.0,
                )

        state.extra["validated_plan_json"] = plan  # backward compat

        # ----- Validate plan: check tool names against known tools -----
        if plan and not _modify_mode:
            from neuro_copilot.core.tools.generic_tools import GENERIC_TOOLS
            _known_tools = set(GENERIC_TOOLS.keys()) if isinstance(GENERIC_TOOLS, dict) else set()
            _plan_steps = plan.get("steps", []) if isinstance(plan, dict) else []
            for _step in _plan_steps:
                _tool = (_step or {}).get("tool") or (_step or {}).get("op_name", "")
                if _tool and _known_tools and _tool not in _known_tools:
                    print(f"⚠️ Plan validation: unknown tool '{_tool}' in plan. "
                          f"Known tools: {sorted(_known_tools)[:15]}", flush=True)

        # ----- Prepare raw_data (needed by both Code Generator and Executor) -----
        # Build raw_data dict mapping friendly names -> DataFrames.
        # For multi-CSV datasets the DataFrames live inside
        #   nd.raw["_files"][<long_filename>]["_dataframe"]
        # We expose them under *both* the original long filename AND a
        # short schema-friendly alias (e.g. "trials.csv") so that
        # LLM-generated code can reference either.
        import pandas as _pd

        parsed_schema = state.extra.get("parsed_schema")
        nd = state.extra.get("neural_dataset")
        raw_data: Dict[str, Any] = {}

        if nd and hasattr(nd, "raw"):
            _raw = nd.raw

            # --- Multi-CSV path ---
            if isinstance(_raw.get("_files"), dict):
                schema_files = set()
                if parsed_schema and hasattr(parsed_schema, "files"):
                    schema_files = set(parsed_schema.files.keys())

                for long_name, file_data in _raw["_files"].items():
                    df = file_data.get("_dataframe") if isinstance(file_data, dict) else None
                    if df is None or not isinstance(df, _pd.DataFrame):
                        continue
                    # Always store under original filename
                    raw_data[long_name] = df

                    # Compute a short alias: strip common prefixes so
                    # "nwb_anm369962_20170310_trials.csv" matches "trials.csv"
                    for schema_name in schema_files:
                        if schema_name == long_name:
                            continue  # already stored
                        bare = schema_name  # e.g. "trials.csv"
                        stem = bare.rsplit(".", 1)[0]  # e.g. "trials"
                        if long_name.endswith(bare) or long_name.endswith(f"_{bare}"):
                            raw_data[schema_name] = df
                        elif f"_{stem}." in long_name or long_name.startswith(stem):
                            raw_data[schema_name] = df

                    # Also store under stem without extension as convenience
                    stem_no_ext = long_name.rsplit(".", 1)[0]
                    if stem_no_ext not in raw_data:
                        raw_data[stem_no_ext] = df

            # --- Single-file / .mat / .nwb path ---
            if not raw_data:
                for key, val in _raw.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(val, _pd.DataFrame):
                        raw_data[key] = val
                # Single-CSV: also stored under "_dataframe"
                if "_dataframe" in _raw and isinstance(_raw["_dataframe"], _pd.DataFrame):
                    df = _raw["_dataframe"]
                    src = _raw.get("_source_file", "data.csv")
                    if src not in raw_data:
                        raw_data[src] = df
                    stem_no_ext = src.rsplit(".", 1)[0]
                    if stem_no_ext not in raw_data:
                        raw_data[stem_no_ext] = df

        # --- EEG dataset: surface metadata for code generator ---
        if nd and hasattr(nd, "raw") and nd.raw.get("_eeg_format"):
            _eeg = nd.raw
            eeg_info = {
                "format": "EEG (.set)",
                "loader": _eeg.get("_loader", "unknown"),
                "source_file": _eeg.get("_source_file", ""),
                "nbchan": _eeg.get("nbchan"),
                "pnts": _eeg.get("pnts"),
                "srate": _eeg.get("srate"),
                "ch_names": _eeg.get("ch_names"),
                "duration_sec": _eeg.get("duration_sec"),
            }
            state.extra["eeg_metadata"] = eeg_info
            print(f"📦 DA execution: EEG dataset detected — {eeg_info['nbchan']} channels, "
                  f"{eeg_info['srate']}Hz, {eeg_info.get('duration_sec', '?')}s", flush=True)

        if raw_data:
            print(f"📦 DA execution: raw_data keys = {list(raw_data.keys())}", flush=True)

        # ----- Phase 2: Code Generation -----
        data_summary_text = _build_data_summary_text()
        code_generator = CodeGenerator()
        code_gen_start = time.time()

        if _modify_mode and _modify_original_code:
            print("🔧 DA execution: modifying existing code with Controller recommendations", flush=True)
            generated = code_generator.modify_existing_code(
                original_code=_modify_original_code,
                recommendations=_modify_recommendations,
                metrics_summary=_modify_metrics_summary,
                parsed_schema=parsed_schema,
                data_summary=data_summary_text,
            )
        else:
            print("💻 DA execution: generate code from plan", flush=True)
            generated = code_generator.generate_from_plan(
                plan=plan,
                parsed_schema=parsed_schema,
                data_summary=data_summary_text,
                available_data_keys=list(raw_data.keys()) if raw_data else None,
                user_instructions=user_idea,
            )

        # Build task description from plan for self-correction context
        _plan_steps = plan.get("steps", [])
        _task_desc_for_regen = "; ".join(
            s.get("description", s.get("op_name", ""))
            for s in _plan_steps
        ) if _plan_steps else ""
        _tool_name_for_regen = (
            _plan_steps[0].get("tool", _plan_steps[0].get("op_name"))
            if len(_plan_steps) == 1 else None
        )

        code_gen_end = time.time()
        code_gen_duration = round(code_gen_end - code_gen_start, 3)
        print(f"💻 DA execution: Code Generator finished ({code_gen_duration}s)", flush=True)

        if not generated or not generated.code.strip():
            _save_audit(audit)
            return DAOutput(
                response="Code Generator failed to produce executable code.",
                needs_dataset=False,
                analysis_type="execution",
                confidence=0.0,
            )

        # ----- Phase 3: Code Execution with Self-Correction -----
        MAX_CODE_RETRIES = 3
        run_dir = audit_dir / "execution"
        run_dir.mkdir(parents=True, exist_ok=True)

        executor = CodeExecutor(output_dir=str(run_dir))
        # Inject per-run seed for reproducibility
        executor._random_seed = dataset_info.get("_run_seed", 42)
        exec_result: Optional[CodeExecutionResult] = None
        current_code = generated

        for code_attempt in range(MAX_CODE_RETRIES):
            print(f"▶️ DA execution: execute code (attempt {code_attempt})", flush=True)
            exec_start = time.time()
            exec_result = executor.execute(
                code=current_code.code,
                raw_data=raw_data,
                extra_context={
                    "parsed_schema": parsed_schema,  # Needed for NDT3 and schema-driven tools
                    "DATASET_ROOT_DIR": state.extra.get("dataset_root_dir", ""),
                },
            )
            exec_end = time.time()
            exec_duration = round(exec_end - exec_start, 3)

            if exec_result.success:
                print(f"✅ DA execution: code succeeded ({exec_duration}s)", flush=True)
                break

            # Self-correction: feed error back to Code Generator
            print(
                f"⚠️ DA execution: code failed (attempt {code_attempt}): "
                f"{exec_result.error_type}: {(exec_result.error or '')[:200]}",
                flush=True,
            )
            if code_attempt < MAX_CODE_RETRIES - 1:
                print("🔄 DA execution: regenerating code with error feedback", flush=True)
                enriched_error = exec_result.error or "Unknown error"

                # Always attach data context so the LLM can self-correct
                # regardless of error type — no keyword matching needed.
                enriched_error += (
                    f"\n\nDATA CONTEXT (use this to fix your code):"
                    f"\n  raw_data keys: {list(raw_data.keys())}"
                )
                for dk, dv in raw_data.items():
                    if isinstance(dv, _pd.DataFrame):
                        enriched_error += f"\n\n  --- {dk} (shape {dv.shape}) ---"
                        enriched_error += f"\n  dtypes: {dict(dv.dtypes)}"
                        # Report NaN counts so the LLM knows to handle missing values
                        _nan_cols = {c: int(dv[c].isna().sum()) for c in dv.columns if dv[c].isna().any()}
                        if _nan_cols:
                            enriched_error += f"\n  ⚠ NaN counts: {_nan_cols}"
                        for col in dv.columns:
                            nunique = dv[col].nunique()
                            if nunique <= 20:
                                vals = dv[col].dropna().unique().tolist()[:20]
                                enriched_error += f"\n  {col}: unique={vals}"
                            else:
                                sample = dv[col].dropna().unique()[:5].tolist()
                                enriched_error += f"\n  {col}: {nunique} unique, sample={sample}"

                enriched_error = _enrich_error_with_type_hints(enriched_error)

                current_code = code_generator.regenerate_with_error(
                    original_code=current_code.code,
                    error_message=enriched_error,
                    error_traceback=exec_result.traceback or "",
                    parsed_schema=parsed_schema,
                    data_summary=data_summary_text,
                    tool_name=_tool_name_for_regen,
                    task_description=_task_desc_for_regen,
                )
                if not current_code or not current_code.code.strip():
                    print("❌ DA execution: Code Generator returned empty code on retry", flush=True)
                    break

        # ----- Phase 4: Build exec report & audit -----
        exec_report = _build_exec_report_from_result(
            exec_result=exec_result,
            generated_code_obj=current_code,
            plan=plan,
            run_dir=run_dir,
        )
        state.extra["execution_result_json"] = exec_report

        initial_phase_end = time.time()
        initial_phase_total = round(initial_phase_end - pipeline_start_time, 3)
        execution_duration = round(initial_phase_end - code_gen_end, 3)

        attempt_record = {
            "attempt_index": 0,
            "timing": {
                "planner_duration_s": planner_duration,
                "code_gen_duration_s": code_gen_duration,
                "execution_duration_s": execution_duration,
                "phase_total_s": initial_phase_total,
            },
            "plan": plan,
            "planner_input": {
                "user_idea": user_idea,
                "prompt": state.extra.get("planner_prompt"),
                "system_prompt": state.extra.get("planner_system_prompt"),
                "llm_meta": state.extra.get("analysis_plan_llm_meta"),
                "validation_feedback": validation_feedback,
            },
            "code_generation": {
                "code": current_code.code[:5000],
                "explanation": current_code.explanation[:2000] if current_code.explanation else "",
                "tool_calls": current_code.tool_calls,
                "confidence": current_code.confidence,
                "prompt": (current_code.prompt or "")[:5000],
                "system_prompt": (current_code.system_prompt or "")[:5000],
            },
            "execution_result": {
                "ok": exec_result.success,
                "error": exec_result.error,
                "stdout": (exec_result.stdout or "")[:3000],
                "metrics": exec_report.get("metrics", {}),
                "all_metric_sets": exec_report.get("all_metric_sets", []),
                "result_keys": list((exec_result.result or {}).keys()) if isinstance(exec_result.result, dict) else [],
            },
        }
        audit["attempts"].append(attempt_record)

        # =================================================================
        # Phase 5-6: Controller evaluation + metric improvement loop
        #
        # CONTROLLER_MODE (env var):
        #   off   — skip Controller entirely, go straight to Analysis Agent
        #   auto  — Auto Model Selection Mode (handled by outer pipeline,
        #           sub-runs arrive here with _no_retries=True)
        #   force — Controller always retries FORCE_IMPROVEMENT_COUNT times
        #
        # When CONTROLLER_MODE=auto, individual sub-runs use off (no internal
        # retries); the auto mode orchestration layer handles retry rounds
        # with Controller diagnosis externally.
        # =================================================================
        import os as _os
        controller_mode = _os.environ.get("CONTROLLER_MODE", "force").strip().lower()
        if controller_mode not in ("off", "auto", "force"):
            controller_mode = "force"
        # Auto mode sub-runs and explicit auto mode use "off" here;
        # retry logic is handled by the auto mode orchestration layer
        if controller_mode == "auto":
            controller_mode = "off"
        _force_improvement_count = int(_os.environ.get("FORCE_IMPROVEMENT_COUNT", "2"))

        # Codex baseline runs get no retries
        if dataset_info and dataset_info.get("_no_retries"):
            controller_mode = "off"

        if not exec_result.success:
            if controller_mode == "off":
                # Controller is disabled — no recovery possible, return error
                _save_audit(audit)
                error_msg = (
                    f"Code execution failed after {MAX_CODE_RETRIES} attempts.\n\n"
                    f"Error: {exec_result.error_type}: {exec_result.error}\n\n"
                    f"Detailed log: outputs/da_run_logs/{audit_run_id}/audit.json"
                )
                return DAOutput(
                    response=error_msg,
                    needs_dataset=False,
                    analysis_type="execution",
                    confidence=0.0,
                )
            else:
                # Controller is enabled — let the improvement loop attempt recovery
                print(
                    f"⚠️ Initial execution failed after {MAX_CODE_RETRIES} code-level retries. "
                    f"Controller ({controller_mode} mode) will attempt recovery.",
                    flush=True,
                )

        force_mode = (controller_mode == "force")
        MAX_IMPROVEMENT_RETRIES = _force_improvement_count

        if controller_mode == "off":
            print("⏭️ Controller mode: OFF — skipping improvement loop", flush=True)
        elif force_mode:
            print(
                f"🔁 Controller mode: FORCE — will run {MAX_IMPROVEMENT_RETRIES} "
                f"mandatory improvement iterations",
                flush=True,
            )
        else:
            print(
                f"🔍 Controller mode: AUTO — Controller will decide "
                f"(max {MAX_IMPROVEMENT_RETRIES} retries)",
                flush=True,
            )

        result_dict = exec_result.result if isinstance(exec_result.result, dict) else {}
        actual_metric: Optional[float] = None
        target_met = False
        diagnosis = None
        prev_primary_metric: Optional[float] = None  # for regression detection
        prev_metric_name: Optional[str] = None  # track which metric was used

        # ----- Rollback mechanism: always retry from the best code so far -----
        _best_code = current_code          # best GeneratedCode so far
        _best_primary_metric: Optional[float] = None   # best primary metric value
        _best_plan = plan                  # plan associated with best code
        _best_exec_result = exec_result    # exec_result associated with best code
        _best_result_dict = result_dict    # result_dict associated with best code

        # When Controller is OFF, skip the entire improvement loop
        _loop_range = range(MAX_IMPROVEMENT_RETRIES + 1) if controller_mode != "off" else range(0)

        for improvement_attempt in _loop_range:
            # ----- Phase 5: Metric evaluation -----
            result_dict = exec_result.result if isinstance(exec_result.result, dict) else {}
            if metric_name and metric_threshold is not None:
                actual_metric = _extract_metric_from_result(result_dict, metric_name, direction)
                if actual_metric is not None:
                    target_met = (
                        actual_metric >= metric_threshold
                        if direction == "higher"
                        else actual_metric <= metric_threshold
                    )
                    print(
                        f"📊 DA execution: {metric_name}={actual_metric:.4f} vs target={metric_threshold} → "
                        f"{'MET' if target_met else 'NOT MET'}",
                        flush=True,
                    )

            # ----- Phase 6: Controller diagnosis -----
            controller_phase_start = time.time()  # start timing for this retry phase
            print(
                f"🔍 DA execution: Controller diagnosis "
                f"(iteration {improvement_attempt}/{MAX_IMPROVEMENT_RETRIES}"
                f"{', force_improve' if force_mode else ''})",
                flush=True,
            )
            controller_diag_start = time.time()
            try:
                all_metrics, _diag_metric_sets = _collect_metric_sets(result_dict)

                # If execution failed, surface the error so Controller can
                # diagnose it (follows existing _tool_diagnostics convention).
                if not exec_result.success and exec_result.error:
                    all_metrics["_execution_error"] = (
                        f"{exec_result.error_type or 'Error'}: {exec_result.error}"
                    )

                if all_metrics or force_mode:
                    diag_data_summary = None
                    if nd and hasattr(nd, "raw"):
                        diag_data_summary = nd.raw.get("_data_summary")
                    diagnosis = self.controller.diagnose_execution_result(
                        task_type=plan.get("task_type", "unknown"),
                        user_request=user_idea,
                        metrics=all_metrics or {},
                        plan=plan,
                        dataset_info=f"Dataset: {state.extra.get('dataset_root_dir', 'N/A')}",
                        trial_columns=state.extra.get("field_catalog", {}),
                        data_summary=diag_data_summary,
                        data_summary_text=data_summary_text,
                        generated_code=current_code.code[:5000],
                        force_improve=(force_mode and improvement_attempt < MAX_IMPROVEMENT_RETRIES),
                    )
                else:
                    diagnosis = None
            except Exception as e:
                print(f"⚠️ Execution diagnosis error: {e}", flush=True)
                diagnosis = None
            controller_diag_duration = round(time.time() - controller_diag_start, 3)

            # --- Decide: accept or retry ---
            should_retry = False

            if force_mode and improvement_attempt < MAX_IMPROVEMENT_RETRIES:
                # Force mode: always retry for the configured number of iterations
                should_retry = True
                if diagnosis:
                    print(
                        f"🔁 Force-improve: iteration {improvement_attempt + 1}/{MAX_IMPROVEMENT_RETRIES}, "
                        f"recommendations={diagnosis.recommendations[:2]}",
                        flush=True,
                    )
            elif diagnosis and diagnosis.needs_review:
                action = diagnosis.suggested_action or "accept"
                print(
                    f"🔍 Controller suggested_action={action}, "
                    f"anomalies={diagnosis.anomalies[:2]}",
                    flush=True,
                )
                if action in ("replan", "adjust_params") and improvement_attempt < MAX_IMPROVEMENT_RETRIES:
                    should_retry = True

            # Guard: detect metric regression (stop if getting worse, even in force mode)
            # IMPORTANT: We must compare the SAME metric across iterations to avoid
            # false regression detection (e.g., test_accuracy vs cv_accuracy_mean).
            if should_retry:
                cur_primary = None
                cur_metric_name = None
                flat, _ = _collect_metric_sets(result_dict)
                for candidate in _PRIMARY_METRIC_CANDIDATES:
                    if candidate in flat:
                        try:
                            cur_primary = float(flat[candidate])
                            cur_metric_name = candidate
                        except (TypeError, ValueError):
                            pass
                        break

                # --- Degenerate prediction detection ---
                # If any class has 0% recall, the model has collapsed to
                # predicting a single class.  Log a warning but do NOT break
                # the loop — the controller may have already diagnosed the
                # root cause and the next retry can fix it.
                _degenerate = False
                _bal_acc = flat.get("balanced_accuracy")
                if _bal_acc is not None:
                    try:
                        _bal_acc = float(_bal_acc)
                    except (TypeError, ValueError):
                        _bal_acc = None

                # Check per-class recall: any class with recall == 0 is degenerate
                _cm = flat.get("confusion_matrix")
                if isinstance(_cm, list) and all(isinstance(r, list) for r in _cm):
                    for row_idx, row in enumerate(_cm):
                        row_sum = sum(row)
                        if row_sum > 0 and row[row_idx] == 0:
                            _degenerate = True
                            break

                # Also check balanced_accuracy <= chance level (1/n_classes)
                if _bal_acc is not None and not _degenerate:
                    _n_classes = flat.get("n_classes")
                    if _n_classes is not None:
                        try:
                            _chance = 1.0 / int(_n_classes)
                        except (TypeError, ValueError, ZeroDivisionError):
                            _chance = 0.0
                        from neuro_copilot.core.controller.controller_agent import MIN_CHANCE_TOLERANCE
                        if _bal_acc <= _chance + MIN_CHANCE_TOLERANCE:
                            _degenerate = True

                if _degenerate:
                    print(
                        f"⚠️ Degenerate prediction detected "
                        f"(balanced_accuracy={_bal_acc}, confusion_matrix={_cm}), "
                        f"controller will attempt recovery in next retry",
                        flush=True,
                    )

                # In force mode, skip regression check — always run all
                # configured retries so the controller can recover from
                # fundamental approach changes (e.g., raw PCA → bandpower).
                if not force_mode:
                    if (
                        prev_primary_metric is not None
                        and cur_primary is not None
                        and cur_metric_name == prev_metric_name
                    ):
                        from neuro_copilot.core.controller.controller_agent import METRIC_REGRESSION_TOLERANCE
                        if cur_primary < prev_primary_metric - METRIC_REGRESSION_TOLERANCE:
                            print(
                                f"⚠️ Metric regression detected: {cur_metric_name}="
                                f"{cur_primary:.4f} < {prev_primary_metric:.4f} (prev), "
                                f"stopping improvement loop",
                                flush=True,
                            )
                            break
                    elif prev_primary_metric is not None and cur_metric_name != prev_metric_name:
                        print(
                            f"ℹ️ Metric key changed across iterations "
                            f"({prev_metric_name}→{cur_metric_name}), "
                            f"skipping regression check",
                            flush=True,
                        )
                prev_primary_metric = cur_primary
                prev_metric_name = cur_metric_name

                # ----- Rollback: track best code, metric, and execution state -----
                if cur_primary is not None:
                    if _best_primary_metric is None or cur_primary > _best_primary_metric:
                        _best_primary_metric = cur_primary
                        _best_code = current_code
                        _best_plan = plan
                        _best_exec_result = exec_result
                        _best_result_dict = result_dict
                        print(
                            f"📈 New best: {cur_metric_name}={cur_primary:.4f}",
                            flush=True,
                        )
                    else:
                        # Rollback: revert to best code AND metrics for next retry
                        print(
                            f"📉 No improvement ({cur_metric_name}={cur_primary:.4f} "
                            f"vs best={_best_primary_metric:.4f}), "
                            f"rolling back to best code and metrics",
                            flush=True,
                        )
                        current_code = _best_code
                        plan = _best_plan
                        exec_result = _best_exec_result
                        result_dict = _best_result_dict

            if not should_retry:
                break

            # If Controller diagnosis failed (e.g. exception caught above),
            # we have no recommendations to act on — stop the loop gracefully.
            if diagnosis is None:
                print(
                    "⚠️ Controller diagnosis unavailable, "
                    "stopping improvement loop",
                    flush=True,
                )
                break

            # ----- Phase 6b: Improvement retry -----
            # Release GPU memory from previous attempt before starting new one
            try:
                import gc
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    gc.collect()
                    print("🧹 GPU cache cleared before retry", flush=True)
            except Exception:
                pass  # non-critical

            print(
                f"🔄 DA execution: improvement retry {improvement_attempt + 1}/{MAX_IMPROVEMENT_RETRIES} "
                f"(action={diagnosis.suggested_action})",
                flush=True,
            )

            # Log diagnosis in audit
            audit.setdefault("improvement_retries", []).append({
                "attempt": improvement_attempt,
                "diagnosis": {
                    "suggested_action": diagnosis.suggested_action,
                    "anomalies": diagnosis.anomalies,
                    "recommendations": diagnosis.recommendations,
                    "summary": diagnosis.summary_for_user,
                },
                "previous_metrics": {k: v for k, v in all_metrics.items()
                                     if isinstance(v, (int, float, str))},
            })

            # 6b: Choose between modify mode and full replan mode
            _retry_action = diagnosis.suggested_action or "replan"
            refined_prompt = ""  # default for modify mode (no prompt refinement)

            if _retry_action == "adjust_params" and current_code and current_code.code.strip():
                # ----- MODIFY MODE: targeted code modification -----
                # Skip planner, directly modify existing working code
                print(
                    f"🔧 DA execution: modify mode — adjusting parameters in existing code",
                    flush=True,
                )
                prompt_refine_duration = 0.0
                retry_planner_duration = 0.0

                # Build metrics summary for context
                _metrics_for_modify = {
                    k: v for k, v in all_metrics.items()
                    if isinstance(v, (int, float, str))
                }
                _metrics_summary = json.dumps(
                    _metrics_for_modify, indent=2, ensure_ascii=False, cls=_NumpySafeEncoder
                )

                retry_codegen_start = time.time()
                new_generated = code_generator.modify_existing_code(
                    original_code=current_code.code,
                    recommendations=diagnosis.recommendations[:5],
                    metrics_summary=_metrics_summary,
                    parsed_schema=parsed_schema,
                    data_summary=data_summary_text,
                )
                retry_codegen_duration = round(time.time() - retry_codegen_start, 3)

                if not new_generated or not new_generated.code.strip():
                    # Fallback to full replan if modify mode fails
                    print(
                        "⚠️ Modify mode returned empty code, falling back to full replan",
                        flush=True,
                    )
                    _retry_action = "replan"

                else:
                    current_code = new_generated

            if _retry_action == "replan":
                # ----- REPLAN MODE: full rewrite (original behavior) -----
                print(
                    f"📝 DA execution: replan mode — regenerating plan and code",
                    flush=True,
                )

                # 6b-i: Controller refines the prompt
                prompt_refine_start = time.time()
                refined_prompt = self.controller.refine_prompt_with_diagnosis(
                    original_prompt=user_idea,
                    recommendations=diagnosis.recommendations,
                    suggested_action=diagnosis.suggested_action,
                    dataset_info={
                        "trial_columns": list(
                            (state.extra.get("field_catalog") or {}).keys()
                        ),
                    },
                )
                prompt_refine_duration = round(time.time() - prompt_refine_start, 3)
                if not refined_prompt or not refined_prompt.strip():
                    print("⚠️ Prompt refinement failed, stopping improvement loop", flush=True)
                    break

                # 6b-ii: Re-run Planner with feedback
                retry_feedback = (
                    f"IMPROVEMENT RETRY: Previous analysis produced suboptimal results.\n"
                    f"PREVIOUS PLAN:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n"
                    f"METRICS FROM PREVIOUS RUN:\n{json.dumps(all_metrics, indent=2, ensure_ascii=False, cls=_NumpySafeEncoder)}\n"
                    f"DIAGNOSIS:\n{diagnosis.summary_for_user}\n"
                    f"RECOMMENDATIONS:\n" + "\n".join(f"- {r}" for r in diagnosis.recommendations[:5]) + "\n"
                    f"Generate an IMPROVED plan that addresses these issues."
                )
                retry_planner_start = time.time()
                state = run_planner_agent(
                    state,
                    user_idea=refined_prompt,
                    tools_whitelist_text=None,
                    validation_feedback=retry_feedback,
                    previous_plan=plan,
                )
                new_plan = state.extra.get("analysis_plan_json")
                retry_planner_duration = round(time.time() - retry_planner_start, 3)

                if new_plan is None:
                    print("⚠️ Planner returned no plan on retry, stopping improvement loop", flush=True)
                    break

                plan = new_plan

                # 6b-iii: Re-run Code Generator
                retry_codegen_start = time.time()
                new_generated = code_generator.generate_from_plan(
                    plan=plan,
                    parsed_schema=parsed_schema,
                    data_summary=data_summary_text,
                    available_data_keys=list(raw_data.keys()) if raw_data else None,
                    previous_code=current_code.code,
                    user_instructions=user_idea,
                )
                retry_codegen_duration = round(time.time() - retry_codegen_start, 3)

                if not new_generated or not new_generated.code.strip():
                    print("⚠️ Code Generator returned empty code on retry, stopping", flush=True)
                    break

                current_code = new_generated

            # Update task description from the (possibly new) plan
            _retry_steps = plan.get("steps", [])
            _task_desc_for_regen = "; ".join(
                s.get("description", s.get("op_name", ""))
                for s in _retry_steps
            ) if _retry_steps else ""
            _tool_name_for_regen = (
                _retry_steps[0].get("tool", _retry_steps[0].get("op_name"))
                if len(_retry_steps) == 1 else None
            )

            # 6b-iv: Re-execute with self-correction
            retry_exec_start = time.time()
            for retry_code_attempt in range(MAX_CODE_RETRIES):
                print(
                    f"▶️ DA retry {improvement_attempt + 1}: execute code "
                    f"(self-correction attempt {retry_code_attempt})",
                    flush=True,
                )
                exec_result = executor.execute(
                    code=current_code.code,
                    raw_data=raw_data,
                    extra_context={
                        "parsed_schema": parsed_schema,
                        "DATASET_ROOT_DIR": state.extra.get("dataset_root_dir", ""),
                    },
                )
                if exec_result.success:
                    print(
                        f"✅ DA retry {improvement_attempt + 1}: code succeeded",
                        flush=True,
                    )
                    break
                print(
                    f"⚠️ DA retry {improvement_attempt + 1}: code failed "
                    f"(attempt {retry_code_attempt}): "
                    f"{exec_result.error_type}: {(exec_result.error or '')[:300]}",
                    flush=True,
                )
                if retry_code_attempt < MAX_CODE_RETRIES - 1:
                    print(
                        f"🔄 DA retry {improvement_attempt + 1}: regenerating code "
                        f"with error feedback",
                        flush=True,
                    )
                    enriched_error = exec_result.error or "Unknown error"
                    enriched_error += (
                        f"\n\nDATA CONTEXT (use this to fix your code):"
                        f"\n  raw_data keys: {list(raw_data.keys())}"
                    )
                    for dk, dv in raw_data.items():
                        if isinstance(dv, _pd.DataFrame):
                            enriched_error += f"\n\n  --- {dk} (shape {dv.shape}) ---"
                            enriched_error += f"\n  dtypes: {dict(dv.dtypes)}"
                            _nan_cols = {c: int(dv[c].isna().sum()) for c in dv.columns if dv[c].isna().any()}
                            if _nan_cols:
                                enriched_error += f"\n  ⚠ NaN counts: {_nan_cols}"
                            for col in dv.columns:
                                nunique = dv[col].nunique()
                                if nunique <= 20:
                                    vals = dv[col].dropna().unique().tolist()[:20]
                                    enriched_error += f"\n  {col}: unique={vals}"
                                else:
                                    sample = dv[col].dropna().unique()[:5].tolist()
                                    enriched_error += f"\n  {col}: {nunique} unique, sample={sample}"

                    enriched_error = _enrich_error_with_type_hints(enriched_error)

                    current_code = code_generator.regenerate_with_error(
                        original_code=current_code.code,
                        error_message=enriched_error,
                        error_traceback=exec_result.traceback or "",
                        parsed_schema=parsed_schema,
                        data_summary=data_summary_text,
                        tool_name=_tool_name_for_regen,
                        task_description=_task_desc_for_regen,
                    )
                    if not current_code or not current_code.code.strip():
                        break
            retry_exec_duration = round(time.time() - retry_exec_start, 3)

            if not exec_result.success:
                print(
                    f"⚠️ Code execution failed on improvement retry "
                    f"{improvement_attempt + 1}. "
                    f"Error: {exec_result.error_type}: {(exec_result.error or '')[:200]}",
                    flush=True,
                )
                # Save failed retry info to audit for debugging
                failed_retry_record = {
                    "attempt_index": improvement_attempt + 1,
                    "attempt_type": "improvement_retry_failed",
                    "improvement_trigger": diagnosis.summary_for_user[:500] if diagnosis else "",
                    "timing": {
                        "controller_diagnosis_s": controller_diag_duration,
                        "prompt_refine_s": prompt_refine_duration,
                        "planner_duration_s": retry_planner_duration,
                        "code_gen_duration_s": retry_codegen_duration,
                        "execution_duration_s": round(time.time() - retry_exec_start, 3),
                        "phase_total_s": round(time.time() - controller_phase_start, 3),
                    },
                    "error": f"{exec_result.error_type}: {exec_result.error}"[:1000],
                    "failed_code_snippet": (current_code.code[:3000] if current_code else "N/A"),
                }
                audit["attempts"].append(failed_retry_record)
                _save_audit(audit)
                # Don't break — let the loop continue so the Controller can
                # diagnose the failure and try a different approach next iteration.
                print(
                    f"🔁 Will continue to next iteration; Controller will see "
                    f"the execution error and adapt.",
                    flush=True,
                )
                continue

            # 6b-v: Rebuild exec report and log this attempt
            run_dir_retry = audit_dir / "execution" / f"retry_{improvement_attempt + 1}"
            run_dir_retry.mkdir(parents=True, exist_ok=True)
            exec_report = _build_exec_report_from_result(
                exec_result=exec_result,
                generated_code_obj=current_code,
                plan=plan,
                run_dir=run_dir_retry,
            )
            state.extra["execution_result_json"] = exec_report

            retry_phase_end = time.time()
            retry_phase_total = round(retry_phase_end - controller_phase_start, 3)

            retry_attempt_record = {
                "attempt_index": improvement_attempt + 1,
                "attempt_type": "improvement_retry",
                "improvement_trigger": diagnosis.summary_for_user[:500],
                "refined_prompt": refined_prompt[:2000],
                "timing": {
                    "controller_diagnosis_s": controller_diag_duration,
                    "prompt_refine_s": prompt_refine_duration,
                    "planner_duration_s": retry_planner_duration,
                    "code_gen_duration_s": retry_codegen_duration,
                    "execution_duration_s": retry_exec_duration,
                    "phase_total_s": retry_phase_total,
                },
                "plan": plan,
                "code_generation": {
                    "code": current_code.code[:5000],
                    "explanation": current_code.explanation[:2000] if current_code.explanation else "",
                    "tool_calls": current_code.tool_calls,
                    "confidence": current_code.confidence,
                },
                "execution_result": {
                    "ok": exec_result.success,
                    "error": exec_result.error,
                    "stdout": (exec_result.stdout or "")[:3000],
                    "metrics": exec_report.get("metrics", {}),
                    "all_metric_sets": exec_report.get("all_metric_sets", []),
                    "result_keys": list((exec_result.result or {}).keys()) if isinstance(exec_result.result, dict) else [],
                },
            }
            audit["attempts"].append(retry_attempt_record)
            _save_audit(audit)
            print(
                f"✅ Improvement retry {improvement_attempt + 1} complete "
                f"(controller={controller_diag_duration}s, refine={prompt_refine_duration}s, "
                f"planner={retry_planner_duration}s, codegen={retry_codegen_duration}s, "
                f"exec={retry_exec_duration}s, phase_total={retry_phase_total}s)",
                flush=True,
            )
            # Loop continues: next iteration re-evaluates metrics via Controller

        # Store final diagnosis in audit if available
        if diagnosis and diagnosis.needs_review:
            audit["diagnosis"] = {
                "needs_review": diagnosis.needs_review,
                "anomalies": diagnosis.anomalies,
                "probable_causes": diagnosis.probable_causes,
                "recommendations": diagnosis.recommendations,
                "suggested_action": diagnosis.suggested_action,
            }

        # ----- Check if ALL attempts failed (including Controller retries) -----
        _any_success = any(
            att.get("execution_result", {}).get("ok")
            for att in audit.get("attempts", [])
        )
        if not _any_success:
            _save_audit(audit)
            _total_att = len(audit.get("attempts", []))
            error_msg = (
                f"Code execution failed after {_total_att} attempt(s) "
                f"(including Controller recovery).\n\n"
                f"Error: {exec_result.error_type}: {exec_result.error}\n\n"
                f"Detailed log: outputs/da_run_logs/{audit_run_id}/audit.json"
            )
            return DAOutput(
                response=error_msg,
                needs_dataset=False,
                analysis_type="execution",
                confidence=0.0,
            )

        # ----- Phase 7: Analysis Agent -----
        # Skip Analysis Agent for auto-mode sub-runs to save API calls.
        # The auto-mode orchestrator will call it once for the final winner.
        _is_sub_run = (dataset_info or {}).get("_auto_sub_run", False)
        if _is_sub_run:
            print("⏭️ DA execution: skipping Analysis Agent (auto-mode sub-run)", flush=True)
            summary = None
            analysis_result = None
            analysis_agent_duration = 0.0
        else:
            print("📊 DA execution: running Analysis Agent", flush=True)
            analysis_agent_start = time.time()
            summary, analysis_result = _run_final_analysis(
                exec_report=exec_report,
                audit_obj=audit,
                target_met=target_met,
                achieved_value=actual_metric,
            )
            analysis_agent_duration = round(time.time() - analysis_agent_start, 3)
            print(f"📊 DA execution: Analysis Agent finished ({analysis_agent_duration}s)", flush=True)

        audit["analysis_summary"] = summary[:500] if summary else None
        audit["total_attempts"] = len(audit.get("attempts", []))

        # Persist collected metrics at top level for easy access.
        # Strategy: select the BEST-performing attempt (not just the last one)
        # across all successful attempts, using accuracy/auroc as the primary
        # comparison metric.
        def _has_analysis_metrics(m: dict) -> bool:
            return bool(m) and bool(_KNOWN_METRIC_ALIASES & set(m.keys()))

        def _best_primary(m: dict) -> float:
            """Extract the best primary metric value for comparison."""
            for key in _PRIMARY_METRIC_CANDIDATES:
                if key in m:
                    try:
                        return float(m[key])
                    except (TypeError, ValueError):
                        continue
            return -1.0

        # Collect candidate metrics from ALL successful attempts
        best_metrics = {}
        best_primary_val = -1.0
        best_attempt_idx = None

        for att in audit.get("attempts", []):
            if not att.get("execution_result", {}).get("ok"):
                continue
            candidate = att.get("execution_result", {}).get("metrics", {})
            if not _has_analysis_metrics(candidate):
                continue
            pval = _best_primary(candidate)
            if pval > best_primary_val:
                best_primary_val = pval
                best_metrics = candidate
                best_attempt_idx = att.get("attempt_index")

        # Fallback: if no attempt had analysis metrics, use last execution
        if not best_metrics:
            result_dict = exec_result.result if isinstance(exec_result.result, dict) else {}
            best_metrics, _ = _collect_metric_sets(result_dict)

        if best_attempt_idx is not None:
            last_idx = audit["attempts"][-1].get("attempt_index") if audit.get("attempts") else None
            if best_attempt_idx != last_idx:
                print(
                    f"📊 Best metrics from attempt {best_attempt_idx} "
                    f"(primary={best_primary_val:.4f}), not the latest attempt {last_idx}.",
                    flush=True,
                )

        audit["metrics"] = best_metrics
        if best_attempt_idx is not None:
            audit["best_attempt_index"] = best_attempt_idx

        # ----- Build top-level phase_timing summary -----
        # This gives a clear, at-a-glance breakdown of where time was spent.
        phase_timing_list = []
        for att in audit.get("attempts", []):
            att_timing = att.get("timing", {})
            phase_total = att_timing.get("phase_total_s")
            att_type = att.get("attempt_type", "")
            if att_type in ("improvement_retry", "improvement_retry_failed"):
                entry = {
                    "phase": f"retry_{att['attempt_index']}",
                    "phase_total_s": phase_total,
                    "controller_diagnosis_s": att_timing.get("controller_diagnosis_s"),
                    "prompt_refine_s": att_timing.get("prompt_refine_s"),
                    "planner_s": att_timing.get("planner_duration_s"),
                    "code_gen_s": att_timing.get("code_gen_duration_s"),
                    "execution_s": att_timing.get("execution_duration_s"),
                }
                if att_type == "improvement_retry_failed":
                    entry["status"] = "failed"
                    entry["error"] = att.get("error", "")[:200]
                phase_timing_list.append(entry)
            else:
                phase_timing_list.append({
                    "phase": "initial_run",
                    "phase_total_s": phase_total,
                    "planner_s": att_timing.get("planner_duration_s"),
                    "code_gen_s": att_timing.get("code_gen_duration_s"),
                    "execution_s": att_timing.get("execution_duration_s"),
                })
        phase_timing_list.append({
            "phase": "analysis_agent",
            "phase_total_s": analysis_agent_duration,
        })
        audit["phase_timing"] = phase_timing_list

        # ----- Build per-phase metrics summary -----
        # Dynamically extract ALL numeric metrics from each attempt.
        # Uses METRIC_ALIASES (already imported as _METRIC_ALIASES) to
        # normalize key names (e.g. "test_accuracy" -> "accuracy").

        # Build reverse lookup: any alias -> (canonical, priority)
        # Priority = position in the aliases list (lower = higher priority).
        # e.g. ["accuracy", "acc", "test_accuracy", "cv_accuracy_mean"]
        #         pri=0      pri=1     pri=2             pri=3
        _alias_to_canonical: dict = {}
        _alias_to_priority: dict = {}
        for canonical, entry_cfg in _METRIC_ALIASES.items():
            for pri, alias in enumerate(entry_cfg.get("aliases", [canonical])):
                _alias_to_canonical[alias] = canonical
                _alias_to_priority[alias] = pri

        # Skip non-metric keys (metadata, arrays, paths, etc.)
        _NON_METRIC_KEYS = {"confusion_matrix", "figure_path", "save_path"}

        def _normalize_metrics_entry(raw_metrics: dict) -> tuple:
            """Convert raw metrics dict to normalized (entry_dict, keys_set).

            When multiple raw keys map to the same canonical name the key with
            the **highest priority** wins (= lowest index in the aliases list
            defined in METRIC_ALIASES) instead of the old first-wins behaviour.
            """
            entry: dict = {}
            _entry_pri: dict = {}        # canonical -> current winner priority
            keys: set = set()
            for raw_key, raw_val in raw_metrics.items():
                if raw_key in _NON_METRIC_KEYS:
                    continue
                if isinstance(raw_val, bool):
                    continue
                if any(op in raw_key for op in ("==", "!=", ">=", "<=", ">", "<", " in ", " not ")):
                    continue
                try:
                    float_val = round(float(raw_val), 4)
                except (TypeError, ValueError):
                    continue
                canonical = _alias_to_canonical.get(raw_key, raw_key)
                pri = _alias_to_priority.get(raw_key, 999)
                if canonical not in entry or pri < _entry_pri[canonical]:
                    entry[canonical] = float_val
                    _entry_pri[canonical] = pri
                    keys.add(canonical)
            return entry, keys

        phase_metrics_list = []
        all_metric_keys = set()  # Collect all metric keys across phases
        for att in audit.get("attempts", []):
            att_type = att.get("attempt_type", "")
            if att_type in ("improvement_retry", "improvement_retry_failed"):
                phase_name = f"retry_{att['attempt_index']}"
            else:
                phase_name = "initial_run"

            exec_metrics = att.get("execution_result", {}).get("metrics", {})
            att_metric_sets = att.get("execution_result", {}).get("all_metric_sets", [])

            # Primary entry: uses best metrics (already selected in exec_metrics)
            entry, keys = _normalize_metrics_entry(exec_metrics)
            entry["phase"] = phase_name
            all_metric_keys |= keys

            # If multiple metric sets, annotate and add sub-entries
            if att_metric_sets and len(att_metric_sets) > 1:
                entry["note"] = f"best of {len(att_metric_sets)} experiments"
                # Add individual metric set entries for comparison
                for ms in att_metric_sets:
                    sub_entry, sub_keys = _normalize_metrics_entry(ms.get("metrics", {}))
                    sub_entry["phase"] = f"{phase_name}/{ms.get('label', '?')}"
                    all_metric_keys |= sub_keys
                    phase_metrics_list.append(sub_entry)

            phase_metrics_list.append(entry)
        audit["phase_metrics"] = phase_metrics_list

        _save_audit(audit)

        # ----- Build Markdown comparison table from phase_metrics -----
        # Only include phases with at least one metric.
        _pm_list = [
            pm for pm in phase_metrics_list
            if any(k != "phase" and v is not None for k, v in pm.items())
        ]
        if _pm_list and all_metric_keys:
            # Sort metric keys for consistent display
            _sorted_keys = sorted(all_metric_keys)
            _table_lines = [
                "",
                "---",
                "",
                "### Performance Comparison across Phases",
                "",
                "| Metric | " + " | ".join(pm["phase"] for pm in _pm_list) + " |",
                "| --- | " + " | ".join("---" for _ in _pm_list) + " |",
            ]
            for mk in _sorted_keys:
                display_name = mk.replace("_", " ").title()
                row = f"| {display_name} |"
                for pm in _pm_list:
                    v = pm.get(mk)
                    row += f" {v:.4f} |" if v is not None else " — |"
                _table_lines.append(row)
            summary = (summary or "") + "\n".join(_table_lines)

        da_output = DAOutput(
            response=summary,
            needs_dataset=False,
            analysis_type="execution",
            confidence=0.8 if target_met else 0.7,
        )
        da_output.raw_json = da_output.raw_json or {}
        da_output.raw_json["analysis_result"] = analysis_result.to_dict() if analysis_result else None
        da_output.raw_json["visualization_paths"] = get_visualization_paths(analysis_result) if analysis_result else []
        return da_output


def test_pipeline():
    """Test complete pipeline with performance monitoring"""
    print("="*80)
    print("Testing Complete Pipeline (with performance monitoring)")
    print("="*80)
    
    # Enable developer mode for timing breakdown
    pipeline = NeuroCopilotPipeline(enable_cache=True, developer_mode=True)
    
    # Test 1: QA without dataset
    print("\n" + "="*80)
    print("TEST 1: QA - Question Answering (no dataset)")
    print("="*80)
    result = pipeline.process(
        user_idea="What brain regions are involved in working memory?",
        has_dataset=False
    )
    print(f"\n📄 Final Response:")
    print(f"   {result.response[:200]}...")
    print(f"\n📚 References: {len(result.references)}")
    print(f"✅ Status: {result.status}")
    
    # Test cache hit
    print("\n" + "="*80)
    print("TEST 1b: Same query (should use cache)")
    print("="*80)
    result = pipeline.process(
        user_idea="What brain regions are involved in working memory?",
        has_dataset=False
    )
    print(f"✅ Status: {result.status}")
    
    # Test 2: ER without dataset
    print("\n" + "="*80)
    print("TEST 2: ER - Experiment Recommendation (no dataset)")
    print("="*80)
    result = pipeline.process(
        user_idea="How do I design an experiment to test working memory capacity in mice?",
        has_dataset=False
    )
    print(f"\n📄 Final Response:")
    print(f"   {result.response[:200]}...")
    print(f"✅ Status: {result.status}")
    
    # Test 3: DA without dataset
    print("\n" + "="*80)
    print("TEST 3: DA - Data Analysis (no dataset)")
    print("="*80)
    result = pipeline.process(
        user_idea="I want to decode cue location from neural activity",
        has_dataset=False
    )
    print(f"\n📄 Final Response:")
    print(f"   {result.response[:200]}...")
    print(f"✅ Status: {result.status}")
    
    # Print cache statistics
    print("\n" + "="*80)
    print("Cache Statistics")
    print("="*80)
    stats = pipeline.get_cache_stats()
    print(f"Router cache entries: {stats['router_entries']}")
    print(f"KB cache entries: {stats['kb_entries']}")
    print(f"Controller cache entries: {stats['controller_entries']}")


if __name__ == "__main__":
    test_pipeline()

