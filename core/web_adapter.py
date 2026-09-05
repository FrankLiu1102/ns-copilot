"""
Web Adapter - Bridge between new pipeline and existing web_app.py

This adapter allows web_app.py to use the new unified pipeline
while maintaining backwards compatibility.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from neuro_copilot.core.pipeline import NeuroCopilotPipeline, PipelineOutput


@dataclass
class WebAppResponse:
    """Response format for web app"""
    # User-facing output
    main_output: str
    references: list[str]
    status_message: str
    
    # Debug info (for developer mode)
    debug_info: Dict[str, Any]
    
    # Action flags
    needs_data: bool = False
    needs_more_retrieval: bool = False
    
    # Visualization paths (for DA tasks with visualizations)
    visualization_paths: list[str] = None
    
    # NEW: Execution diagnosis for user review
    needs_user_review: bool = False
    diagnosis_summary: str = ""
    diagnosis_recommendations: list[str] = None
    suggested_action: str = ""  # "replan" | "adjust_params" | "accept"
    diagnosis_raw: Dict[str, Any] = None  # Full diagnosis data for retry
    
    # NEW: Dataset metadata for retry context
    dataset_metadata: Dict[str, Any] = None  # trial_columns, unit_columns, etc.

    # Progress events from pipeline (for step-by-step display in chat)
    progress_events: list = None
    
    def __post_init__(self):
        if self.visualization_paths is None:
            self.visualization_paths = []
        if self.diagnosis_recommendations is None:
            self.diagnosis_recommendations = []
        if self.diagnosis_raw is None:
            self.diagnosis_raw = {}
        if self.dataset_metadata is None:
            self.dataset_metadata = {}
        if self.progress_events is None:
            self.progress_events = []


class WebAdapter:
    """Adapter to use new pipeline in existing web app with performance optimizations"""
    
    def __init__(self, developer_mode: bool = False, enable_cache: bool = True):
        """
        Initialize web adapter
        
        Args:
            developer_mode: Enable verbose logging and timing breakdown
            enable_cache: Enable in-memory caching for better performance
        """
        self.pipeline = NeuroCopilotPipeline(
            enable_cache=enable_cache,
            developer_mode=developer_mode
        )
        self.developer_mode = developer_mode
    
    def process_request(
        self,
        user_idea: str,
        dataset_file: Optional[str] = None,
        additional_files: Optional[list[str]] = None,
        preprocess_rules: Optional[Dict[str, Any]] = None,
        task_type_override: Optional[str] = None,
        retry_context: Optional[Dict[str, Any]] = None,
    ) -> WebAppResponse:
        """
        Process user request and format for web app
        
        Args:
            user_idea: User's question/request
            dataset_file: Path to primary uploaded dataset (optional)
            additional_files: List of additional file paths (for multi-CSV support)
            preprocess_rules: Preprocessing rules (optional)
            task_type_override: Override task type detection (optional)
            retry_context: Context from previous failed attempt (optional)
                - previous_plan: The plan that was executed
                - diagnosis_feedback: Feedback from diagnosis (why it failed)
                - metrics: Execution metrics from previous run
                - attempt_number: Which retry attempt this is
            
        Returns:
            WebAppResponse with formatted output
        """
        has_dataset = dataset_file is not None

        # Reset progress tracker for this run
        try:
            from neuro_copilot.core.progress_events import reset_tracker
            reset_tracker()
        except Exception:
            pass

        # Build dataset_info with retry context
        dataset_info = None
        if has_dataset:
            dataset_info = {
                "file_path": dataset_file, 
                "preprocess_rules": preprocess_rules
            }
            # Add additional files for multi-CSV support
            if additional_files:
                dataset_info["additional_files"] = additional_files
            if retry_context:
                dataset_info["retry_context"] = retry_context
        
        # Run pipeline
        result: PipelineOutput = self.pipeline.process(
            user_idea=user_idea,
            has_dataset=has_dataset,
            dataset_info=dataset_info,
            task_type_override=task_type_override
        )
        
        # Extract visualization paths from result
        visualization_paths = result.visualization_paths or []
        
        # Format main output (including visualization info)
        main_output = self._format_main_output(result, visualization_paths)
        
        # Format status message
        status_msg = self._format_status(result)
        
        # Collect debug info
        debug_info = self._collect_debug_info(result)
        
        # Determine action flags
        needs_data = result.status == "NEED_DATA"
        needs_more_retrieval = result.status == "RETRIEVE_MORE"
        
        # Collect progress events from pipeline
        progress_events = []
        try:
            from neuro_copilot.core.progress_events import get_tracker
            progress_events = get_tracker().get_events()
        except Exception:
            pass

        # Check for execution diagnosis requiring user review
        needs_user_review = False
        diagnosis_summary = ""
        diagnosis_recommendations = []
        suggested_action = ""
        diagnosis_raw = {}
        
        # Extract diagnosis info from DA task agent output
        dataset_metadata = {}
        if result.task_type == "DA" and result.task_agent_output:
            da_output = result.task_agent_output
            if hasattr(da_output, 'needs_user_review') and da_output.needs_user_review:
                needs_user_review = True
                diagnosis_summary = getattr(da_output, 'diagnosis_summary', '')
                diagnosis_recommendations = getattr(da_output, 'diagnosis_recommendations', []) or []
                suggested_action = getattr(da_output, 'suggested_action', '')
                # Store full diagnosis for retry
                if hasattr(da_output, 'raw_json') and da_output.raw_json:
                    diagnosis_raw = da_output.raw_json.get('diagnosis', {})
            
            # Extract dataset metadata for context in retries
            if hasattr(da_output, 'raw_json') and da_output.raw_json:
                raw = da_output.raw_json
                dataset_metadata = {
                    "trial_columns": raw.get("trial_columns", []),
                    "unit_columns": raw.get("unit_columns", []),
                    "n_trials": raw.get("n_trials"),
                    "n_units": raw.get("n_units"),
                }
        
        return WebAppResponse(
            main_output=main_output,
            references=result.references,
            status_message=status_msg,
            debug_info=debug_info,
            needs_data=needs_data,
            needs_more_retrieval=needs_more_retrieval,
            visualization_paths=visualization_paths,
            needs_user_review=needs_user_review,
            diagnosis_summary=diagnosis_summary,
            diagnosis_recommendations=diagnosis_recommendations,
            suggested_action=suggested_action,
            diagnosis_raw=diagnosis_raw,
            dataset_metadata=dataset_metadata,
            progress_events=progress_events,
        )
    
    def _format_main_output(self, result: PipelineOutput, visualization_paths: list[str] = None) -> str:
        """Format main output for user"""
        from pathlib import Path
        
        output_parts = []
        
        # Add response
        output_parts.append(result.response)
        
        # Add visualization section if paths exist
        if visualization_paths:
            output_parts.append("\n\n### 📊 Generated Visualizations\n")
            for path in visualization_paths:
                filename = Path(path).name
                # Use markdown image syntax for display in Gradio
                # Note: This requires the path to be accessible from the web server
                output_parts.append(f"- **{filename}**")
                output_parts.append(f"  - Path: `{path}`")
        
        # Add references if present
        if result.references:
            output_parts.append("\n\n## References\n")
            for i, ref in enumerate(result.references, 1):
                output_parts.append(f"[{i}] {ref}")
        
        # Add status-specific messages
        if result.status == "NEED_DATA":
            output_parts.append("\n\n⚠️ **Note**: Please upload a dataset to proceed with analysis execution.")
        elif result.status == "RETRIEVE_MORE":
            output_parts.append("\n\n⚠️ **Note**: Insufficient references. More literature retrieval recommended.")
        elif result.status == "REWRITE_REQUIRED":
            output_parts.append("\n\n⚠️ **Note**: Some claims may not be fully supported by retrieved literature. Please verify or refine your query.")
        
        return "\n".join(output_parts)
    
    def _format_status(self, result: PipelineOutput) -> str:
        """Format status message"""
        task_type_names = {
            "QA": "Question Answering",
            "DA": "Data Analysis",
            "ER": "Experiment Recommendation"
        }
        
        task_name = task_type_names.get(result.task_type, result.task_type)
        
        if result.status == "PASS":
            return f"✅ {task_name} complete"
        elif result.status == "NEED_DATA":
            return f"⚠️ Dataset required for {task_name}"
        elif result.status == "RETRIEVE_MORE":
            return f"⚠️ {task_name} may need more references"
        elif result.status == "REWRITE_REQUIRED":
            return f"⚠️ {task_name} requires revision (claims not supported by literature)"
        else:
            return f"ℹ️ {task_name} processed"
    
    def _collect_debug_info(self, result: PipelineOutput) -> Dict[str, Any]:
        """Collect debug information"""
        from core.config import ROUTER_MODEL, TASK_AGENT_MODEL, CONTROLLER_MODEL
        
        return {
            "llm_config": {
                "router_model": ROUTER_MODEL,
                "task_agent_model": TASK_AGENT_MODEL,
                "controller_model": CONTROLLER_MODEL
            },
            "task_type": result.task_type,
            "router_output": {
                "task_type": result.router_output.task_type,
                "keywords": result.router_output.keywords,
                "constraints": result.router_output.constraints,
                "confidence": result.router_output.confidence,
                "raw_json": result.router_output.raw_json
            },
            "kb_retrieval": {
                "num_papers": len(result.kb_entries),
                "papers": [
                    {
                        "title": entry.title,
                        "authors": entry.authors,
                        "year": entry.year,
                        "type": entry.paper_type
                    }
                    for entry in result.kb_entries[:5]
                ]
            },
            "controller_output": {
                "status": result.controller_output.status,
                "issues": result.controller_output.issues,
                "next_action": result.controller_output.next_action,
                "rationale": getattr(result.controller_output, 'rationale', ''),
                "raw_json": result.controller_output.raw_json
            },
            "final_status": result.status
        }
    
    def format_debug_display(self, debug_info: Dict[str, Any]) -> str:
        """Format debug info for display in developer mode"""
        lines = []
        
        lines.append("# 🔍 Developer Mode - Pipeline Debug Info\n")
        
        # LLM Configuration
        lines.append("## 🤖 LLM Configuration")
        llm_config = debug_info.get("llm_config", {})
        lines.append(f"- **Router Model**: {llm_config.get('router_model', 'N/A')}")
        lines.append(f"- **Task Agent Model**: {llm_config.get('task_agent_model', 'N/A')} (Llama 3.1 8B)")
        lines.append(f"- **Controller Model**: {llm_config.get('controller_model', 'N/A')}")
        
        # Router
        lines.append("\n## 📍 Router Output")
        router = debug_info["router_output"]
        lines.append(f"- **Task Type**: {router['task_type']}")
        lines.append(f"- **Keywords**: {', '.join(router['keywords'])}")
        lines.append(f"- **Need Citations**: {router['constraints'].get('need_citations', False)}")
        lines.append(f"- **Confidence**: {router['confidence']:.2f}")
        
        # KB Retrieval
        lines.append("\n## 📚 KB Retrieval")
        kb = debug_info["kb_retrieval"]
        lines.append(f"- **Papers Retrieved**: {kb['num_papers']}")
        if kb['papers']:
            lines.append("- **Top Papers**:")
            for paper in kb['papers']:
                lines.append(f"  - {paper['title'][:60]}... ({paper['year']}, {paper['type']})")
        
        # Controller
        lines.append("\n## 🎯 Controller Validation")
        ctrl = debug_info["controller_output"]
        lines.append(f"- **Status**: {ctrl['status']}")
        if ctrl['issues']:
            lines.append(f"- **Issues**: {', '.join(ctrl['issues'])}")
        lines.append(f"- **Next Action**: {ctrl['next_action']}")
        if ctrl.get('rationale'):
            lines.append(f"- **Rationale**: {ctrl['rationale']}")
        
        # Final Status
        lines.append(f"\n## ✅ Final Status: {debug_info['final_status']}")
        
        return "\n".join(lines)


def test_adapter():
    """Test web adapter"""
    print("="*80)
    print("Testing Web Adapter")
    print("="*80)
    
    adapter = WebAdapter(developer_mode=True)
    
    # Test QA
    print("\n" + "="*80)
    print("TEST: QA Request")
    print("="*80)
    response = adapter.process_request(
        user_idea="What brain regions are involved in working memory?",
        dataset_file=None
    )
    
    print(f"\n📄 Main Output:")
    print(response.main_output[:300] + "...")
    print(f"\n📊 Status: {response.status_message}")
    print(f"\n🔍 Debug Info (formatted):")
    print(adapter.format_debug_display(response.debug_info))


if __name__ == "__main__":
    test_adapter()

