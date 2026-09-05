# neuro_copilot/analysis_agent.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from neuro_copilot.core.state import NeuroGlobalState
from neuro_copilot.core.llm_client import generate_json, LLMResponse


class AnalysisAgentError(Exception):
    pass


@dataclass
class AnalysisAgentConfig:
    model: str = os.getenv("ANALYSIS_MODEL", "llama3.1:8b")
    base_url: str = "http://localhost:11434"
    timeout_s: float = 180.0
    temperature: float = float(os.getenv("ANALYSIS_TEMPERATURE", "0.3"))
    seed: Optional[int] = int(os.getenv("DA_SEED", "0"))


@dataclass
class AnalysisResult:
    """Structured result returned by the Analysis Agent."""
    summary: str = ""
    key_findings: list = field(default_factory=list)
    metrics_interpretation: dict = field(default_factory=dict)
    visualizations: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    experiment_status: str = "completed"
    total_attempts: int = 1
    target_metric: Optional[str] = None
    target_threshold: Optional[float] = None
    achieved_value: Optional[float] = None
    target_met: bool = False
    raw_json: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_findings": self.key_findings,
            "metrics_interpretation": self.metrics_interpretation,
            "visualizations": self.visualizations,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "experiment_status": self.experiment_status,
            "total_attempts": self.total_attempts,
            "target_metric": self.target_metric,
            "target_threshold": self.target_threshold,
            "achieved_value": self.achieved_value,
            "target_met": self.target_met,
            "raw_json": self.raw_json,
        }


def format_analysis_for_display(result: AnalysisResult) -> str:
    """Format an AnalysisResult as user-facing markdown."""
    if result is None:
        return "Execution complete (analysis summary unavailable)."

    lines: list = []

    # Summary
    status_icon = "✓" if result.target_met or result.experiment_status == "completed" else ""
    lines.append(f"## Analysis Summary {status_icon}")
    lines.append("")
    if result.summary:
        lines.append(result.summary)
        lines.append("")

    # Key findings
    if result.key_findings:
        lines.append("### Key Findings")
        for i, f in enumerate(result.key_findings, 1):
            if isinstance(f, dict):
                lines.append(f"{i}. **{f.get('finding', '')}**")
                if f.get("evidence"):
                    lines.append(f"   - Evidence: {f['evidence']}")
                if f.get("significance"):
                    lines.append(f"   - Significance: {f['significance']}")
            else:
                lines.append(f"{i}. {f}")
        lines.append("")

    # Metrics interpretation
    if result.metrics_interpretation:
        lines.append("### Metrics Interpretation")
        for k, v in result.metrics_interpretation.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    # Limitations
    if result.limitations:
        lines.append("### Limitations")
        for lim in result.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    # Recommendations
    if result.recommendations:
        lines.append("### Recommendations for Next Steps")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Target info
    if result.target_metric and result.achieved_value is not None:
        if result.target_met:
            lines.append(
                f"_Target Met: {result.target_metric} = {result.achieved_value:.4f} "
                f"(target: ≥ {result.target_threshold})_"
            )
        else:
            lines.append(
                f"_Target Not Met: {result.target_metric} = {result.achieved_value:.4f} "
                f"(target: ≥ {result.target_threshold})_"
            )
        lines.append("")

    lines.append(
        f"_Status: {'✅' if result.experiment_status == 'completed' else '⚠️'} "
        f"Data Analysis complete_"
    )
    return "\n".join(lines)


def get_visualization_paths(result: AnalysisResult) -> list:
    """Extract visualization file paths from an AnalysisResult."""
    if result is None:
        return []
    paths: list = []
    for viz in (result.visualizations or []):
        if isinstance(viz, dict):
            p = viz.get("figure_path") or viz.get("path")
            if p:
                paths.append(str(p))
        elif isinstance(viz, str):
            paths.append(viz)
    return paths


ANALYSIS_OUTPUT_SCHEMA_HINT = """
{
  "summary": "string (brief overall summary of the analysis)",
  "key_findings": [
    {
      "finding": "string (what was discovered)",
      "evidence": "string (supporting metric or observation)",
      "significance": "string (why this matters)"
    }
  ],
  "metrics_interpretation": {
    "metric_name": "string (interpretation of what this metric means)"
  },
  "visualizations": [
    {
      "figure_path": "string",
      "description": "string (what this visualization shows)",
      "insights": "string (key insights from this figure)"
    }
  ],
  "limitations": ["string (any limitations or caveats)"],
  "recommendations": ["string (suggestions for further analysis)"]
}
""".strip()


SYSTEM_PROMPT = """
You are the LLM-based Analysis Agent for a neuroscience data analysis system.
You receive execution results from analysis tools (decoding, encoding, visualization, etc.)
and your job is to interpret these results, summarize key findings, and provide insights
in a clear, scientific manner.

Focus on:
1. Interpreting metrics (accuracy, confusion matrices, tuning curves, etc.)
2. Identifying patterns and significant findings
3. Explaining what visualizations reveal
4. Highlighting limitations and suggesting improvements

Be concise but thorough. Use scientific terminology appropriately.
""".strip()


def _extract_visualizations_from_steps(
    steps: List[Dict[str, Any]],
    outputs_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Extract all visualization information from execution steps.
    Returns a list of visualization dictionaries with metadata.
    """
    visualizations = []
    
    for i, step in enumerate(steps):
        if not step.get("ok", False):
            continue
        
        op_name = step.get("op_name", "")
        output_path_str = step.get("output_path")
        
        # Check if this is a visualization tool
        is_viz_tool = op_name.startswith("plot_")
        
        if output_path_str:
            try:
                output_path = Path(output_path_str)
                if not output_path.is_absolute():
                    if outputs_root:
                        output_path = outputs_root / output_path.name
                    else:
                        output_path = Path.cwd() / output_path
                
                if output_path.exists():
                    output_data = json.loads(output_path.read_text(encoding='utf-8'))
                    
                    # Extract figure_path if present (check both top-level and artifacts)
                    figure_path = output_data.get("figure_path") or output_data.get("artifacts", {}).get("figure_path")
                    if figure_path:
                        viz_info = {
                            "step_index": i,
                            "step_name": op_name,
                            "figure_path": figure_path,
                            "figure_exists": Path(figure_path).exists() if figure_path else False,
                        }
                        
                        # Add tool-specific metadata
                        if op_name == "plot_confusion_matrix":
                            viz_info["type"] = "confusion_matrix"
                            viz_info["confusion_matrix_shape"] = output_data.get("confusion_matrix_shape")
                            viz_info["normalized"] = output_data.get("normalized", False)
                            viz_info["class_labels"] = output_data.get("class_labels")
                            viz_info["n_classes"] = output_data.get("metadata", {}).get("n_classes")
                            viz_info["total_samples"] = output_data.get("metadata", {}).get("total_samples")
                        
                        elif op_name == "plot_tuning_curve":
                            viz_info["type"] = "tuning_curve"
                            viz_info["n_conditions"] = output_data.get("n_conditions")
                            viz_info["n_neurons"] = output_data.get("n_neurons")
                            viz_info["style"] = output_data.get("style", "line")
                            viz_info["error_bars"] = output_data.get("error_bars")
                            viz_info["conditions"] = output_data.get("conditions")
                            viz_info["response_range"] = output_data.get("metadata", {}).get("response_range")
                            viz_info["response_mean"] = output_data.get("metadata", {}).get("response_mean")
                        
                        elif op_name.startswith("decode_"):
                            # Decoding tools that auto-generated visualization
                            metrics = output_data.get("metrics", {})
                            viz_info["type"] = "confusion_matrix"
                            viz_info["confusion_matrix_shape"] = (metrics.get("n_classes"), metrics.get("n_classes"))
                            viz_info["normalized"] = True
                            viz_info["class_labels"] = metrics.get("class_labels")
                            viz_info["n_classes"] = metrics.get("n_classes")
                            viz_info["total_samples"] = metrics.get("n_samples_total")
                        
                        else:
                            # Generic visualization
                            viz_info["type"] = "unknown"
                        
                        visualizations.append(viz_info)
                    
                    # Also check for confusion_matrix in decoding outputs (even if not plotted)
                    # Check both in metrics and at top level
                    metrics = output_data.get("metrics", {})
                    has_cm = "confusion_matrix" in metrics or "confusion_matrix" in output_data
                    
                    if op_name.startswith("decode_") and has_cm:
                        # This is a decoding result that has confusion matrix data
                        # but might not have been visualized yet
                        cm_data = metrics.get("confusion_matrix") or output_data.get("confusion_matrix")
                        if cm_data and not any(v.get("step_name") == op_name and v.get("type") == "confusion_matrix_data" for v in visualizations):
                            # Determine shape
                            if isinstance(cm_data, list):
                                if len(cm_data) > 0:
                                    cm_shape = (len(cm_data), len(cm_data[0]) if isinstance(cm_data[0], list) else 1)
                                else:
                                    cm_shape = None
                            else:
                                cm_shape = None
                            
                            viz_info = {
                                "step_index": i,
                                "step_name": op_name,
                                "type": "confusion_matrix_data",
                                "figure_path": None,  # Not yet visualized
                                "figure_exists": False,
                                "confusion_matrix_shape": metrics.get("confusion_matrix_shape") or cm_shape,
                                "class_labels": metrics.get("class_labels") or output_data.get("class_labels"),
                                "n_classes": metrics.get("n_classes") or output_data.get("n_classes"),
                            }
                            visualizations.append(viz_info)
            
            except Exception as e:
                # Skip this step if we can't load it
                continue
    
    return visualizations


def _format_execution_result_for_llm(
    exec_result: Dict[str, Any],
    outputs_root: Optional[Path] = None,
) -> str:
    """
    Format execution result into a text summary that LLM can understand.
    Includes metrics, artifacts, and visualization paths.
    """
    lines = []
    
    # Overall status
    ok = exec_result.get("ok", False)
    run_id = exec_result.get("run_id", "unknown")
    lines.append(f"EXECUTION STATUS: {'SUCCESS' if ok else 'FAILED'}")
    lines.append(f"Run ID: {run_id}")
    
    summary = exec_result.get("summary", {})
    if summary:
        total_duration = summary.get("total_duration_ms", 0)
        lines.append(f"Total duration: {total_duration} ms ({total_duration/1000:.2f} seconds)")
        if "seed" in summary:
            lines.append(f"Random seed: {summary.get('seed')}")
    
    lines.append("")
    
    # Step-by-step results
    steps = exec_result.get("steps", [])
    if not steps:
        lines.append("No steps were executed.")
        return "\n".join(lines)
    
    lines.append(f"EXECUTED {len(steps)} STEP(S):")
    lines.append("")
    
    # Extract all visualizations first
    visualizations = _extract_visualizations_from_steps(steps, outputs_root=outputs_root)
    
    # Add visualization summary at the beginning if any exist
    if visualizations:
        lines.append("VISUALIZATIONS GENERATED:")
        for viz in visualizations:
            viz_type = viz.get("type", "unknown")
            step_name = viz.get("step_name", "unknown")
            figure_path = viz.get("figure_path")
            
            if figure_path and viz.get("figure_exists"):
                lines.append(f"  - {step_name} ({viz_type}): {figure_path}")
                if viz_type == "confusion_matrix":
                    shape = viz.get("confusion_matrix_shape")
                    n_classes = viz.get("n_classes")
                    if shape:
                        lines.append(f"    Shape: {shape}, Classes: {n_classes}")
                elif viz_type == "tuning_curve":
                    n_cond = viz.get("n_conditions")
                    n_neurons = viz.get("n_neurons")
                    if n_cond:
                        lines.append(f"    Conditions: {n_cond}, Neurons/Trials: {n_neurons}")
            elif viz_type == "confusion_matrix_data":
                lines.append(f"  - {step_name} (confusion_matrix_data): Available but not yet visualized")
        lines.append("")
    
    for i, step in enumerate(steps):
        step_num = i + 1
        op_name = step.get("op_name", "unknown")
        step_ok = step.get("ok", False)
        duration = step.get("duration_ms", 0)
        
        lines.append(f"--- Step {step_num}: {op_name} ---")
        lines.append(f"Status: {'✓ SUCCESS' if step_ok else '✗ FAILED'}")
        lines.append(f"Duration: {duration} ms")
        
        if step_ok:
            # Try to load output data — prefer file, fallback to inline metrics
            output_data = None
            output_path_str = step.get("output_path")
            if output_path_str:
                try:
                    output_path = Path(output_path_str)
                    if not output_path.is_absolute():
                        if outputs_root:
                            output_path = outputs_root / output_path.name
                        else:
                            output_path = Path.cwd() / output_path

                    if output_path.exists():
                        output_data = json.loads(output_path.read_text(encoding='utf-8'))
                except Exception:
                    pass

            # Fallback: use metrics from the step dict itself
            if output_data is None and step.get("metrics"):
                output_data = {"metrics": step["metrics"]}

            if output_data is not None:
                try:
                    if "metrics" in output_data:
                        lines.append("\nMetrics:")
                        metrics = output_data["metrics"]
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                if isinstance(value, float):
                                    lines.append(f"  {key}: {value:.4f}")
                                else:
                                    lines.append(f"  {key}: {value}")
                            elif isinstance(value, list):
                                if len(value) <= 10:
                                    lines.append(f"  {key}: {value}")
                                else:
                                    lines.append(f"  {key}: [list with {len(value)} elements]")
                            elif isinstance(value, dict):
                                lines.append(f"  {key}: {json.dumps(value, indent=4)}")
                            else:
                                lines.append(f"  {key}: {value}")

                    if "artifacts" in output_data:
                        lines.append("\nArtifacts:")
                        artifacts = output_data["artifacts"]
                        for key, value in artifacts.items():
                            if isinstance(value, (str, int, float, bool)):
                                lines.append(f"  {key}: {value}")
                            elif isinstance(value, list) and len(value) <= 20:
                                lines.append(f"  {key}: {value}")
                            else:
                                lines.append(f"  {key}: (complex data)")

                    if "figure_path" in output_data:
                        figure_path = output_data["figure_path"]
                        figure_exists = Path(figure_path).exists() if figure_path else False
                        lines.append(f"\nVisualization Generated:")
                        lines.append(f"  Figure path: {figure_path}")
                        lines.append(f"  File exists: {figure_exists}")

                        if op_name == "plot_confusion_matrix":
                            lines.append(f"  Type: Confusion Matrix")
                            if "confusion_matrix_shape" in output_data:
                                lines.append(f"  Shape: {output_data['confusion_matrix_shape']}")
                            if "normalized" in output_data:
                                lines.append(f"  Normalized: {output_data['normalized']}")
                            if "class_labels" in output_data:
                                labels = output_data["class_labels"]
                                if isinstance(labels, list) and len(labels) <= 20:
                                    lines.append(f"  Class labels: {labels}")
                        elif op_name == "plot_tuning_curve":
                            lines.append(f"  Type: Tuning Curve")
                            if "n_conditions" in output_data:
                                lines.append(f"  Conditions: {output_data['n_conditions']}")
                            if "n_neurons" in output_data:
                                lines.append(f"  Neurons/Trials: {output_data['n_neurons']}")
                            if "style" in output_data:
                                lines.append(f"  Style: {output_data['style']}")
                            if "error_bars" in output_data and output_data["error_bars"]:
                                lines.append(f"  Error bars: {output_data['error_bars']}")

                    if op_name.startswith("decode_") and "confusion_matrix" in output_data.get("metrics", {}):
                        lines.append(f"\nConfusion Matrix Data Available:")
                        cm = output_data["metrics"].get("confusion_matrix")
                        if cm:
                            if isinstance(cm, list) and len(cm) > 0:
                                lines.append(f"  Shape: {len(cm)}x{len(cm[0]) if isinstance(cm[0], list) else '?'}")
                            class_labels = output_data["metrics"].get("class_labels")
                            if class_labels:
                                lines.append(f"  Class labels: {class_labels}")
                            lines.append(f"  Note: This confusion matrix can be visualized using plot_confusion_matrix tool")

                    if "metadata" in output_data:
                        lines.append("\nMetadata:")
                        metadata = output_data["metadata"]
                        for key, value in metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                lines.append(f"  {key}: {value}")
                            elif isinstance(value, list) and len(value) <= 10:
                                lines.append(f"  {key}: {value}")

                except Exception as e:
                    lines.append(f"\n(Error loading output: {e})")
            
            # Also include output directly if available
            output = step.get("output")
            if output and isinstance(output, dict):
                if "metrics" in output:
                    lines.append("\nDirect Output Metrics:")
                    for key, value in output["metrics"].items():
                        if isinstance(value, (int, float)):
                            lines.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        else:
            # Error information
            error = step.get("error", {})
            error_code = error.get("code", "unknown")
            error_message = error.get("message", "Unknown error")
            lines.append(f"\nError: [{error_code}] {error_message}")
        
        lines.append("")
    
    return "\n".join(lines)


def build_analysis_prompt(
    execution_result_text: str,
    user_idea: Optional[str] = None,
    visualizations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build prompt for Analysis Agent.
    """
    prompt_parts = []
    
    if user_idea:
        prompt_parts.append(f"USER'S ORIGINAL QUESTION/IDEA:\n{user_idea}\n")
    
    prompt_parts.append("EXECUTION RESULTS:")
    prompt_parts.append(execution_result_text)
    
    # Add visualization summary if available
    if visualizations:
        prompt_parts.append("")
        prompt_parts.append("VISUALIZATIONS SUMMARY:")
        for viz in visualizations:
            viz_type = viz.get("type", "unknown")
            step_name = viz.get("step_name", "unknown")
            figure_path = viz.get("figure_path")
            
            if figure_path and viz.get("figure_exists"):
                prompt_parts.append(f"- {step_name} ({viz_type}): {figure_path}")
                if viz_type == "confusion_matrix":
                    shape = viz.get("confusion_matrix_shape")
                    n_classes = viz.get("n_classes")
                    normalized = viz.get("normalized", False)
                    prompt_parts.append(f"  Type: Confusion Matrix, Shape: {shape}, Classes: {n_classes}, Normalized: {normalized}")
                elif viz_type == "tuning_curve":
                    n_cond = viz.get("n_conditions")
                    n_neurons = viz.get("n_neurons")
                    style = viz.get("style", "line")
                    prompt_parts.append(f"  Type: Tuning Curve, Conditions: {n_cond}, Neurons/Trials: {n_neurons}, Style: {style}")
            elif viz_type == "confusion_matrix_data":
                prompt_parts.append(f"- {step_name}: Confusion matrix data available but not yet visualized")
                shape = viz.get("confusion_matrix_shape")
                if shape:
                    prompt_parts.append(f"  Shape: {shape}, Can be visualized using plot_confusion_matrix tool")
    
    prompt_parts.append("")
    prompt_parts.append(
        "Please analyze these execution results and provide:\n"
        "1. A brief overall summary\n"
        "2. Key findings with evidence\n"
        "3. Interpretation of metrics\n"
        "4. Insights from visualizations (if any) - describe what each visualization shows and what insights can be drawn\n"
        "5. Limitations and recommendations"
    )
    
    return "\n".join(prompt_parts)


def run_analysis_agent(
    state: NeuroGlobalState,
    *,
    user_idea: Optional[str] = None,
    config: Optional[AnalysisAgentConfig] = None,
    execution_result_key: str = "execution_result_json",
    audit: Optional[Dict[str, Any]] = None,
    exec_report: Optional[Dict[str, Any]] = None,
    outputs_dir: Optional[Path] = None,
    target_metric: Optional[str] = None,
    target_threshold: Optional[float] = None,
    achieved_value: Optional[float] = None,
    target_met: bool = False,
) -> AnalysisResult:
    """
    Run Analysis Agent to summarize and interpret execution results.

    Args:
        state: NeuroGlobalState containing execution results
        user_idea: Optional original user question/idea for context
        config: AnalysisAgentConfig (optional)
        execution_result_key: Key in state.extra for execution_result_json
        audit: Optional audit dict (unused currently, reserved)
        exec_report: Optional execution report dict (used when provided)
        outputs_dir: Optional path to outputs directory for loading step outputs
        target_metric: Optional metric name the user cares about
        target_threshold: Optional threshold for that metric
        achieved_value: Optional actual value of the metric
        target_met: Whether the target was met

    Returns:
        AnalysisResult with structured summary, findings, and recommendations
    """
    if config is None:
        config = AnalysisAgentConfig()

    # Determine exec_result and outputs_root
    exec_result = exec_report or state.extra.get(execution_result_key)
    if exec_result is None:
        raise AnalysisAgentError(
            f"Execution result not found in state.extra['{execution_result_key}']. "
            "Run execution agent first."
        )

    outputs_root = outputs_dir
    if outputs_root is None:
        if "execution_result" in state.extra:
            exec_result_obj = state.extra["execution_result"]
            if hasattr(exec_result_obj, "run_id"):
                artifacts_root = Path(exec_result_obj.artifacts_root)
                outputs_root = artifacts_root.parent.parent / "outputs" / exec_result_obj.run_id
        elif "execution_result_json" in state.extra:
            run_id = exec_result.get("run_id")
            artifacts_root_str = exec_result.get("artifacts_root")
            if run_id and artifacts_root_str:
                artifacts_root = Path(artifacts_root_str)
                outputs_root = artifacts_root.parent.parent / "outputs" / run_id

    # Format execution result for LLM
    exec_result_text = _format_execution_result_for_llm(
        exec_result,
        outputs_root=outputs_root,
    )

    if user_idea is None:
        user_idea = state.extra.get("user_idea")

    prompt = build_analysis_prompt(exec_result_text, user_idea=user_idea)

    # Call LLM
    resp: LLMResponse = generate_json(
        model=config.model,
        timeout_s=config.timeout_s,
        temperature=config.temperature,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_hint=ANALYSIS_OUTPUT_SCHEMA_HINT,
    )

    # Store in state for backward compatibility
    state.extra["analysis_output_raw"] = resp.raw_text
    state.extra["analysis_output_json"] = resp.json_obj
    state.extra["analysis_output_llm_meta"] = {
        "model": resp.model,
        "prompt_tokens": resp.prompt_tokens,
        "eval_tokens": resp.eval_tokens,
        "total_duration_ms": resp.total_duration_ms,
    }

    # Build AnalysisResult from LLM response
    obj = resp.json_obj or {}
    return AnalysisResult(
        summary=obj.get("summary", ""),
        key_findings=obj.get("key_findings", []),
        metrics_interpretation=obj.get("metrics_interpretation", {}),
        visualizations=obj.get("visualizations", []),
        limitations=obj.get("limitations", []),
        recommendations=obj.get("recommendations", []),
        experiment_status="completed",
        total_attempts=audit.get("total_attempts", 1) if audit else 1,
        target_metric=target_metric,
        target_threshold=target_threshold,
        achieved_value=achieved_value,
        target_met=target_met,
        raw_json=obj,
    )

