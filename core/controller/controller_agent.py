"""
Controller Agent - Quality Control and Validation

Responsibilities:
1. Validate task agent outputs
2. Check if citations are present when required
3. Determine if more retrieval is needed
4. For DA: check if dataset is needed but not provided
5. Decide next action (PASS / RETRIEVE_MORE / NEED_DATA)
6. Validate execution results for statistical anomalies (NEW)
7. Generate diagnostic insights and actionable recommendations (NEW)
"""

from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass, field

from neuro_copilot.core.llm_client import generate_json, LLMClientError
from neuro_copilot.core.config import CONTROLLER_MODEL, CONTROLLER_DIAGNOSIS_MODEL, CONTROLLER_PROMPT_REFINE_MODEL


@dataclass
class ControllerOutput:
    """Controller decision output"""
    status: str  # PASS | RETRIEVE_MORE | REWRITE_REQUIRED | NEED_DATA
    issues: list[str]
    next_action: str
    rationale: str  # 1-2 line explanation
    recommended_action: str  # Internal: retrieve_more / downgrade_claims / pass
    raw_json: Dict[str, Any]


@dataclass
class ExecutionDiagnosis:
    """Diagnosis output for execution result validation"""
    needs_review: bool  # Whether user review is recommended
    anomalies: List[str]  # List of detected statistical anomalies
    probable_causes: List[str]  # LLM-inferred probable causes
    recommendations: List[str]  # Actionable recommendations
    suggested_action: str  # "replan" | "adjust_params" | "accept"
    suggested_param_updates: Optional[List[Dict[str, Any]]] = None  # For adjust_params
    summary_for_user: str = ""  # User-friendly summary
    raw_json: Dict[str, Any] = field(default_factory=dict)


CONTROLLER_SYSTEM_PROMPT = """You are a quality control agent for a neuroscience research assistant.

Your job: Validate QA/ER/DA outputs with SEMANTIC CITATION-CLAIM ALIGNMENT checks.

VALIDATION FRAMEWORK (for QA / ER only):

1. **Concept-Evidence Alignment**
   Check if each key concept in the output appears in retrieved papers:
   - Brain regions (e.g., "prefrontal cortex", "basal ganglia")
   - Cell types (e.g., "dopaminergic neurons", "parvalbumin interneurons")
   - Mechanisms (e.g., "synaptic plasticity", "persistent activity")
   - Methods (e.g., "6-OHDA lesions", "optogenetics")
   
   → If output mentions a concept NOT present in paper titles/abstracts → REWRITE_REQUIRED

2. **Causal Language Gating**
   Check if causal claims are supported by intervention evidence:
   - Causal words: "causes", "necessary for", "drives", "controls", "regulates"
   - Requires: Papers describing interventions (lesions, optogenetics, pharmacology, TMS, knockouts)
   - Review papers or correlational fMRI do NOT support causal claims
   
   → If causal language used without intervention evidence → REWRITE_REQUIRED (downgrade to "is associated with", "correlates with")

3. **Method-Species Consistency (ER only)**
   - Cell-type–specific manipulations require cellular-resolution evidence
   - Human fMRI/review papers do NOT support cell-type causal claims
   - Mouse/rat models can support cellular mechanisms
   
   → If species mismatch → REWRITE_REQUIRED

4. **Citation Adequacy**
   - QA/ER with citations required: ≥2 independent sources
   - ER causal manipulations: ≥2 papers demonstrating the manipulation
   - ER aging/disease: ≥2 supporting references
   
   → If insufficient citations → RETRIEVE_MORE

5. **Topical Alignment**
   Check if cited papers are relevant to the claim:
   - Sleep/rTMS enhancement papers should NOT be used to support cellular causality
   - Review papers are acceptable for background but not for specific causal mechanisms
   
   → If citations are topically mismatched → REWRITE_REQUIRED

STATUS DECISIONS:

**PASS** (all must be true):
  ✓ Every key concept appears in retrieved papers
  ✓ Causal language only used with intervention evidence
  ✓ ≥2 independent citations (if required)
  ✓ Method-species consistency maintained
  ✓ No topical mismatches

**RETRIEVE_MORE** (one or more):
  • <2 citations when required
  • Causal manipulation proposed but no papers demonstrating it
  • Topic coverage too narrow

**REWRITE_REQUIRED** (one or more):
  • Concepts mentioned not present in papers
  • Causal language used without intervention evidence
  • Species-method mismatch
  • Citations topically mismatched

**NEED_DATA** (DA only):
  • User requests execution but no dataset provided

For DA: Use existing simple rules (no semantic checks needed).
"""

# ============================================================================
# NEW: Execution Result Diagnosis Prompts
# ============================================================================

EXECUTION_DIAGNOSIS_SYSTEM_PROMPT = """You are an expert data analyst reviewing execution results.

Your job: Determine whether the execution results meet the user's expectations
and, if not, provide actionable recommendations for improvement.

You will receive:
1. USER REQUEST: The full user prompt including any self-correction hints
2. EXECUTION METRICS: The results from running the analysis (accuracy, etc.)
3. DATA SUMMARY: Factual statistics about the dataset
4. PLAN STEPS: What analysis was planned
5. GENERATED CODE: The actual Python code that was executed

Your task:
- Read the user's full request carefully, including any Self-Correction Hints
- Compare the execution metrics against what the user expects
- Examine the generated code to see what parameters were actually used
  (e.g., time windows, features, model hyperparameters)
- Cross-reference the DATA SUMMARY to diagnose any issues
- Provide specific, actionable recommendations if improvement is needed

OUTPUT FORMAT:
Return ONLY this JSON:
{
  "needs_review": true|false,
  "anomalies": ["Specific anomaly 1", ...],
  "probable_causes": ["Most likely cause 1", ...],
  "recommendations": ["Actionable fix 1", ...],
  "suggested_action": "replan|adjust_params|accept",
  "suggested_param_updates": [{"step_index": 0, "params": {...}}] or null,
  "summary_for_user": "2-3 sentence user-friendly explanation"
}

DECISION RULES:
- needs_review = true if results are below user expectations or any significant anomaly detected
- suggested_action = "replan" if the analysis approach needs fundamental changes
  (e.g., different feature extraction, different data filtering)
- suggested_action = "adjust_params" if small parameter tweaks might help
  (e.g., regularization strength, number of folds)
- suggested_action = "accept" if results look reasonable given the data and task
- When recommending changes, be SPECIFIC: reference actual column names and values
  from the DATA SUMMARY provided (not from prior knowledge)
- BEFORE recommending any filter condition (e.g. "column == value"), you MUST
  verify from the DATA SUMMARY that the value actually exists in that column.
  Check the column's min/max, unique values, or value counts. Do NOT recommend
  filtering by values that are absent or impossible in the data.
- If a foundation model (NDT3, MtM, POYO, etc.) is used, its internal architecture
  is fixed — do not recommend changing model-internal parameters.
- Be mindful of GPU memory constraints when recommending changes to data dimensions.
- Do not recommend dropping or replacing a tool the user explicitly required.
- When a foundation model is used for feature extraction, the classifier input
  should be the model's output features only (not augmented with external metadata).
- Be mindful of the feature-to-sample ratio when recommending improvements.
"""

# ---------------------------------------------------------------------------
# Force-improve mode: Controller ALWAYS suggests improvements.
# Used when CONTROLLER_MODE=force in config.
#
# DESIGN PRINCIPLE: This prompt is intentionally generic — no dataset-specific
# column names, values, or domain hints.  The Controller must derive all
# recommendations purely from the code, metrics, and data profile it receives.
# ---------------------------------------------------------------------------
FORCE_IMPROVE_SYSTEM_PROMPT = """You are an expert data analyst reviewing analysis code.

You receive:
1. The executed Python code
2. Execution results and metrics
3. A detailed data profile (column types, value distributions, ranges)
4. The original user request

YOUR TASK: Based solely on the code, results, and data profile provided,
identify concrete improvements that could enhance the analysis quality
and performance. Do NOT rely on prior assumptions — ground every
recommendation in what you actually observe in the code and data.

For each recommendation, explain:
- WHAT to change (specific code-level modification)
- WHY (evidence from the data profile or metrics)
- EXPECTED IMPACT on the primary metric

OUTPUT FORMAT:
Return ONLY this JSON:
{
  "needs_review": true,
  "anomalies": ["Specific issue observed in the code or results"],
  "probable_causes": ["Root cause grounded in the data profile"],
  "recommendations": ["Concrete, actionable code change with specific details from the data"],
  "suggested_action": "replan or adjust_params",
  "suggested_param_updates": null,
  "summary_for_user": "2-3 sentence explanation of key improvements"
}

RULES:
- ALWAYS set "needs_review": true
- PREFER "suggested_action": "adjust_params" when the overall approach is sound and only the downstream pipeline needs modification. This preserves working data loading and feature extraction code.
- ONLY use "suggested_action": "replan" when the fundamental approach itself needs to change.
- Every recommendation MUST reference specific evidence from the DATA PROFILE
  or EXECUTION METRICS — no generic advice
- Prioritize changes most likely to improve the primary metric
- BEFORE recommending any filter condition (e.g. "column == value"), you MUST
  verify from the DATA PROFILE that the value actually exists in that column.
  Check the column's min/max, unique values, or value counts. Do NOT recommend
  filtering by values that are absent or impossible in the data.
- If a foundation model is used, its internal architecture is fixed — do not
  recommend changing model-internal parameters.
- Be mindful of GPU memory constraints when recommending changes to data dimensions.
- Do not recommend dropping or replacing a tool the user explicitly required.
- Ensure recommendations preserve the user's metric requirements.
- When a foundation model is used for feature extraction, the classifier input
  should be the model's output features only (not augmented with external metadata).
- Be mindful of the feature-to-sample ratio when recommending improvements.
"""

EXECUTION_DIAGNOSIS_USER_PROMPT_TEMPLATE = """Review this execution result:

TASK TYPE: {task_type}

USER REQUEST (read carefully, including any self-correction hints):
{user_request}

DATASET INFO:
{dataset_info}

DATA SUMMARY (factual statistics):
{data_summary}

EXECUTION ERROR (empty if execution succeeded):
{execution_error}

EXECUTION METRICS:
{metrics_json}

PLAN STEPS:
{plan_steps}

GENERATED CODE (what was actually executed):
{generated_code}

AVAILABLE TRIAL COLUMNS:
{trial_columns}

Analyze the above information:
1. If execution failed, identify the root cause from the error and code, and recommend a fix.
2. Do the execution metrics meet the user's expectations (check their hints)?
3. Look at the GENERATED CODE — what parameters / filters / features were actually used?
4. Cross-reference the DATA SUMMARY — could different column choices or parameters improve results?
5. What specific, actionable fixes would you recommend? Ground each in evidence from the data.

Output ONLY the JSON as specified.
"""

CONTROLLER_USER_PROMPT_TEMPLATE = """Validate this task output:

TASK TYPE: {task_type}
CITATIONS REQUIRED: {need_citations}
DATASET PROVIDED: {has_dataset}
KB PAPERS RETRIEVED: {kb_papers_summary}

USER REQUEST:
{user_request}

AGENT OUTPUT (first 1000 chars):
{agent_output}

{references_info}

FOR QA / ER TASKS, PERFORM SEMANTIC ALIGNMENT CHECKS:

1. Concept-Evidence Check:
   - List key concepts in agent output (brain regions, methods, mechanisms)
   - Check if each appears in KB paper titles/abstracts
   - Flag concepts NOT present in KB

2. Causal Language Check:
   - Identify causal words ("causes", "drives", "necessary", "controls")
   - Check if KB papers describe interventions (not just correlations/reviews)
   - Flag causal language without intervention evidence

3. Citation Adequacy:
   - Count independent citations
   - For ER causal designs: check if ≥2 papers demonstrate the proposed manipulation
   - Flag if insufficient

FOR DA TASKS:
   - Use simple rules (dataset check only)

Output ONLY this JSON:
{{
  "status": "PASS|RETRIEVE_MORE|REWRITE_REQUIRED|NEED_DATA",
  "issues": ["Specific issue 1", "Specific issue 2", ...],
  "next_action": "Brief user-facing action description",
  "rationale": "1-2 line explanation of decision",
  "recommended_action": "retrieve_more|downgrade_claims|pass"
}}

DECISION TREE (QA/ER):
1. Concepts not in KB? → REWRITE_REQUIRED, recommended_action: downgrade_claims
2. Causal language without intervention? → REWRITE_REQUIRED, recommended_action: downgrade_claims
3. <2 citations when required? → RETRIEVE_MORE, recommended_action: retrieve_more
4. Species mismatch? → REWRITE_REQUIRED, recommended_action: downgrade_claims
5. All checks pass? → PASS, recommended_action: pass

Be STRICT. PASS means: concepts grounded + causal language justified + citations adequate.
"""

DA_RETRY_SYSTEM_PROMPT = """You are a controller for data-analysis execution.
You must decide whether to:
- adjust parameters and rerun the current plan, or
- replan via the planner, or
- stop.

Return ONLY a JSON object:
{
  "action": "adjust_params|replan|stop",
  "reason": "short justification",
  "updates": [{"step_index": 0, "params": {"param_name": value}}]
}

Rules:
- Only use "adjust_params" if a small, safe tweak could improve the target metric.
- If the gap is large or the plan seems mismatched, choose "replan".
- If max attempts reached or no safe tweak, choose "stop".
- updates is required only when action="adjust_params".
"""

DA_RETRY_USER_PROMPT_TEMPLATE = """You are deciding how to improve an execution that did not meet target.

USER REQUEST:
{user_request}

TARGET METRIC:
- name: {metric_name}
- threshold: {target_threshold}
- direction: {direction} (higher or lower is better)

ACTUAL METRIC:
{actual_metric}

PLAN (JSON):
{plan_json}

EXECUTION SUMMARY:
{exec_summary}

Return ONLY the JSON object as specified.
"""


# ---------------------------------------------------------------------------
# Chance-level detection thresholds for anomaly detection.
# These define "how close to random" a metric must be to trigger a review.
#
# AUROC_CHANCE_TOLERANCE: AUROC chance level is always 0.5; any value within
#   this tolerance of 0.5 is flagged. Set conservatively to avoid false alarms.
#
# MIN_CHANCE_TOLERANCE: floor for accuracy/balanced_accuracy tolerance to
#   avoid overly tight thresholds on many-class problems.
#
# CHANCE_TOLERANCE_SCALE: tolerance = max(MIN, SCALE / n_classes). For binary
#   (n=2) this gives 0.25 (wide), for 10-class gives 0.05 (tight). The idea
#   is that small deviations from chance are less meaningful when chance is low.
# ---------------------------------------------------------------------------
AUROC_CHANCE_TOLERANCE = 0.02
MIN_CHANCE_TOLERANCE = 0.02
CHANCE_TOLERANCE_SCALE = 0.5

# Metric regression tolerance: if the primary metric drops by more than this
# amount between retries, stop the improvement loop to prevent degradation.
METRIC_REGRESSION_TOLERANCE = 0.02

# Gap threshold for choosing replan vs adjust_params: if the metric gap
# between current result and target is larger than this, prefer a full replan
# over incremental parameter adjustment.
REPLAN_GAP_THRESHOLD = 0.05

# Parameter schedules for rule-based adjust_params in force mode.
# SVM: try increasing C (stronger regularization → weaker regularization)
# GLM/population: try decreasing alpha (looser regularization)
SVM_C_CANDIDATES = [1.0, 2.0, 5.0, 10.0]
REGULARIZATION_ALPHA_CANDIDATES = [1.0, 0.5, 0.1, 0.01]


class ControllerAgent:
    """Controller Agent for quality control"""
    
    def __init__(
        self,
        model: str = CONTROLLER_MODEL,
        diagnosis_model: str | None = None,
        prompt_refine_model: str | None = None,
    ):
        self.model = model
        self.diagnosis_model = diagnosis_model or CONTROLLER_DIAGNOSIS_MODEL
        self.prompt_refine_model = prompt_refine_model or CONTROLLER_PROMPT_REFINE_MODEL
    
    def _rule_based_validation(
        self,
        task_type: str,
        user_request: str,
        agent_output: str,
        has_references: bool,
        need_citations: bool,
        has_dataset: bool,
        num_references: int = 0
    ) -> ControllerOutput:
        """Rule-based validation as fallback"""
        issues = []
        status = "PASS"
        rationale = ""
        
        # Check citations for QA/ER
        if task_type in ["QA", "ER"]:
            if need_citations and num_references < 2:
                if num_references == 0:
                    issues.append("Citations required but missing")
                else:
                    issues.append("Insufficient independent references (<2)")
                status = "RETRIEVE_MORE"
                rationale = f"Only {num_references} reference(s) provided when ≥2 required for scientific rigor"
        
        # Check dataset for DA
        if task_type == "DA":
            request_lower = user_request.lower()
            wants_execution = any(phrase in request_lower for phrase in [
                "run", "execute", "analyze my", "perform the", "do the analysis"
            ])
            
            if wants_execution and not has_dataset:
                issues.append("Dataset required for execution")
                status = "NEED_DATA"
                rationale = "User requests execution but no dataset provided"
        
        # Determine next action, rationale, and recommended_action
        if status == "PASS":
            next_action = "Output is ready for user"
            rationale = "Claims are grounded, citations sufficient" if need_citations else "Output addresses request adequately"
            recommended_action = "pass"
        elif status == "RETRIEVE_MORE":
            next_action = "Retrieve more papers and regenerate with citations"
            recommended_action = "retrieve_more"
        elif status == "NEED_DATA":
            next_action = "Request dataset from user"
            recommended_action = "pass"  # Not applicable for citation control
        else:
            next_action = "Unknown"
            recommended_action = "pass"
        
        return ControllerOutput(
            status=status,
            issues=issues,
            next_action=next_action,
            rationale=rationale,
            recommended_action=recommended_action,
            raw_json={
                "status": status,
                "issues": issues,
                "next_action": next_action,
                "rationale": rationale,
                "recommended_action": recommended_action,
                "fallback": True
            }
        )
    
    def validate(
        self,
        task_type: str,
        user_request: str,
        agent_output: str,
        references: list[str],
        need_citations: bool = False,
        has_dataset: bool = False,
        kb_papers_summary: str = ""
    ) -> ControllerOutput:
        """
        Validate task agent output
        
        Args:
            task_type: QA / DA / ER
            user_request: Original user request
            agent_output: Output from task agent
            references: List of references (if any)
            need_citations: Whether citations were required
            has_dataset: Whether dataset was provided
            kb_papers_summary: Summary of retrieved KB papers
            
        Returns:
            ControllerOutput with validation decision
        """
        num_references = len(references)
        has_references = num_references > 0
        
        # Format references info
        if references:
            refs_info = f"REFERENCES PROVIDED ({num_references}):\n" + "\n".join(f"- {ref}" for ref in references)
        else:
            refs_info = "REFERENCES PROVIDED: None"
        
        # Ensure agent_output is a string
        if not isinstance(agent_output, str):
            agent_output = str(agent_output)
        
        # Strategy: For DA, always use rule-based validation
        if task_type == "DA":
            print("⚠️  Controller: DA task, using rule-based validation (skip Ollama)")
            return self._rule_based_validation(
                task_type, user_request, agent_output,
                has_references, need_citations, has_dataset,
                num_references
            )

        # Strategy: When no dataset, skip Ollama and use rule-based validation
        # This avoids connection errors and is sufficient for basic validation
        if not has_dataset:
            print("⚠️  Controller: No dataset, using rule-based validation (skip Ollama)")
            return self._rule_based_validation(
                task_type, user_request, agent_output,
                has_references, need_citations, has_dataset,
                num_references
            )
        
        user_prompt = CONTROLLER_USER_PROMPT_TEMPLATE.format(
            task_type=task_type,
            need_citations="YES" if need_citations else "NO",
            has_dataset="YES" if has_dataset else "NO",
            kb_papers_summary=kb_papers_summary or "None retrieved",
            user_request=user_request,
            agent_output=agent_output[:1000],  # Limit length
            references_info=refs_info
        )
        
        try:
            # Only use Ollama when has_dataset=True
            response = generate_json(
                model=self.model,
                system_prompt=CONTROLLER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_s=30.0,
                temperature=0.1,  # Low temperature for consistent validation
                retries=1
            )
            
            if response.json_obj:
                status = response.json_obj.get("status", "PASS").upper()
                if status not in ["PASS", "RETRIEVE_MORE", "REWRITE_REQUIRED", "NEED_DATA"]:
                    status = "PASS"
                
                issues = response.json_obj.get("issues", [])
                if not isinstance(issues, list):
                    issues = []
                
                next_action = response.json_obj.get("next_action", "")
                rationale = response.json_obj.get("rationale", "")
                recommended_action = response.json_obj.get("recommended_action", "pass")
                
                # Normalize recommended_action
                if recommended_action not in ["retrieve_more", "downgrade_claims", "pass"]:
                    recommended_action = "pass"
                
                return ControllerOutput(
                    status=status,
                    issues=issues,
                    next_action=next_action,
                    rationale=rationale,
                    recommended_action=recommended_action,
                    raw_json=response.json_obj
                )
            else:
                # JSON parsing failed, use rule-based fallback
                print("⚠️  Controller: LLM parsing failed, using rule-based validation")
                return self._rule_based_validation(
                    task_type, user_request, agent_output,
                    has_references, need_citations, has_dataset,
                    num_references
                )
                
        except (LLMClientError, Exception) as e:
            print(f"⚠️  Controller: LLM error ({e}), using rule-based validation")
            return self._rule_based_validation(
                task_type, user_request, agent_output,
                has_references, need_citations, has_dataset,
                num_references
            )

    def _rule_based_decide_da_retry(
        self,
        *,
        metric_name: str,
        target_threshold: float,
        actual_metric: float,
        direction: str,
        plan: Dict[str, Any],
        attempt_idx: int,
        max_attempts: int,
    ) -> Dict[str, Any]:
        """
        Decide whether to retry DA execution by adjusting params or re-planning.
        Rule-based to avoid LLM dependency in container.
        """
        if direction == "higher":
            met = actual_metric >= target_threshold
            gap = target_threshold - actual_metric
        else:
            met = actual_metric <= target_threshold
            gap = actual_metric - target_threshold

        if met:
            return {"action": "pass", "reason": "target met"}

        if attempt_idx >= max_attempts - 1:
            return {"action": "stop", "reason": "max attempts reached"}

        # Find first decoding/encoding step (if any)
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        for i, step in enumerate(steps):
            op = (step or {}).get("op_name", "")
            params = (step or {}).get("params", {})
            if op == "decode_svm":
                # Simple C schedule based on attempt
                c_candidates = SVM_C_CANDIDATES
                current_c = params.get("C", None)
                try:
                    current_c = float(current_c) if current_c is not None else None
                except Exception:
                    current_c = None
                # If C not set, assume default 1.0 (smallest)
                if current_c is None:
                    current_c = 1.0
                next_c = None
                for c in c_candidates:
                    if c > current_c:
                        next_c = c
                        break
                # If no larger value found, we've exhausted options
                if next_c is None:
                    return {"action": "replan", "reason": f"{metric_name} gap {gap:.3f}, C exhausted, consider new plan"}
                return {
                    "action": "adjust_params",
                    "reason": f"{metric_name} gap {gap:.3f}, try higher C ({current_c} -> {next_c})",
                    "updates": [{"step_index": i, "params": {"C": next_c}}],
                }

            if op in ("fit_glm_poisson", "fit_population_model"):
                alpha_candidates = REGULARIZATION_ALPHA_CANDIDATES
                current_alpha = params.get("regularization", None)
                try:
                    current_alpha = float(current_alpha) if current_alpha is not None else None
                except Exception:
                    current_alpha = None
                # If regularization not set, assume default 1.0 (most conservative)
                if current_alpha is None:
                    current_alpha = 1.0
                next_alpha = None
                for a in alpha_candidates:
                    if a < current_alpha:
                        next_alpha = a
                        break
                # If no smaller value found, we've exhausted options
                if next_alpha is None:
                    return {"action": "replan", "reason": f"{metric_name} gap {gap:.3f}, regularization exhausted, consider new plan"}
                return {
                    "action": "adjust_params",
                    "reason": f"{metric_name} gap {gap:.3f}, try lower regularization ({current_alpha} -> {next_alpha})",
                    "updates": [{"step_index": i, "params": {"regularization": next_alpha}}],
                }

        # If no safe param tweak is available, prefer replan when gap is large
        if gap >= REPLAN_GAP_THRESHOLD:
            return {"action": "replan", "reason": "gap large, consider new plan"}

        return {"action": "stop", "reason": "no safe param tweak available"}

    def decide_da_retry(
        self,
        *,
        user_request: str,
        metric_name: str,
        target_threshold: float,
        actual_metric: float,
        direction: str,
        plan: Dict[str, Any],
        attempt_idx: int,
        max_attempts: int,
        exec_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Decide whether to retry DA execution by adjusting params or re-planning.
        Uses LLM with rule-based fallback for robustness.
        
        Returns:
            Dict with keys:
                - action: "adjust_params" | "replan" | "stop"
                - reason: Explanation for the decision
                - updates: (optional) Parameter updates for adjust_params
                - _controller_input: Full input context for audit logging
        """
        if attempt_idx >= max_attempts - 1:
            return {"action": "stop", "reason": "max attempts reached"}

        user_prompt = DA_RETRY_USER_PROMPT_TEMPLATE.format(
            user_request=user_request,
            metric_name=metric_name,
            target_threshold=target_threshold,
            direction=direction,
            actual_metric=actual_metric,
            plan_json=json.dumps(plan, indent=2, ensure_ascii=False),
            exec_summary=exec_summary or "N/A",
        )
        
        # Build controller input context for audit logging
        controller_input = {
            "user_request": user_request,
            "metric_name": metric_name,
            "target_threshold": target_threshold,
            "actual_metric": actual_metric,
            "direction": direction,
            "plan": plan,
            "attempt_idx": attempt_idx,
            "max_attempts": max_attempts,
            "exec_summary": exec_summary,
            "system_prompt": DA_RETRY_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        }

        try:
            response = generate_json(
                model=self.model,
                system_prompt=DA_RETRY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_s=30.0,
                temperature=0.1,
                retries=1,
            )
            if response.json_obj:
                action = response.json_obj.get("action")
                if action in ["adjust_params", "replan", "stop"]:
                    if action == "adjust_params":
                        updates = response.json_obj.get("updates", [])
                        if isinstance(updates, list) and updates:
                            return {
                                "action": "adjust_params",
                                "reason": response.json_obj.get("reason", ""),
                                "updates": updates,
                                "_controller_input": controller_input,
                                "_controller_response": response.json_obj,
                            }
                        return {"action": "stop", "reason": "invalid updates from controller", "_controller_input": controller_input}
                    return {
                        "action": action,
                        "reason": response.json_obj.get("reason", ""),
                        "_controller_input": controller_input,
                        "_controller_response": response.json_obj,
                    }
        except (LLMClientError, Exception):
            pass

        result = self._rule_based_decide_da_retry(
            metric_name=metric_name,
            target_threshold=target_threshold,
            actual_metric=actual_metric,
            direction=direction,
            plan=plan,
            attempt_idx=attempt_idx,
            max_attempts=max_attempts,
        )
        result["_controller_input"] = controller_input
        result["_rule_based"] = True
        return result

    # ========================================================================
    # NEW: Execution Result Diagnosis Methods
    # ========================================================================

    def _rule_based_diagnose_execution(
        self,
        task_type: str,
        metrics: Dict[str, Any],
        plan: Dict[str, Any],
        trial_columns: Optional[Dict[str, Any]] = None,
    ) -> ExecutionDiagnosis:
        """
        Minimal rule-based diagnosis for execution results.
        
        Only detects STATISTICAL ANOMALIES (chance-level accuracy, etc.)
        Does NOT provide interpretations or recommendations - that's for the LLM.
        """
        anomalies = []
        needs_review = False
        suggested_action = "accept"
        
        # Extract tool diagnostics data_info (factual data only, no warnings/recommendations)
        # Use .get() instead of .pop() to preserve the data for LLM diagnosis
        tool_diagnostics = metrics.get("_tool_diagnostics", None)
        
        # Extract key metrics
        accuracy = metrics.get("test_accuracy") or metrics.get("accuracy")
        balanced_acc = metrics.get("test_balanced_accuracy") or metrics.get("balanced_accuracy")
        auroc = metrics.get("test_auroc") or metrics.get("auroc")
        r2 = metrics.get("r2_score") or metrics.get("r2")
        n_classes = metrics.get("n_classes", 2)
        
        # Determine task type from plan
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        op_names = [s.get("op_name", "") for s in steps if isinstance(s, dict)]
        is_decoding = any("decode" in op.lower() for op in op_names)
        is_encoding = any(op in ["fit_glm_poisson", "fit_tuning_function", "fit_population_model"] for op in op_names)
        
        # ===== DETECT STATISTICAL ANOMALIES ONLY =====
        # No interpretations or recommendations here - just flag anomalies.
        # See module-level constants for threshold definitions and rationale.

        if is_decoding or task_type == "decoding":
            random_baseline = 1.0 / n_classes if n_classes > 0 else 0.5
            chance_tolerance = (
                max(MIN_CHANCE_TOLERANCE, CHANCE_TOLERANCE_SCALE / n_classes)
                if n_classes > 0 else MIN_CHANCE_TOLERANCE
            )

            # Check if accuracy is at chance level
            if accuracy is not None and abs(accuracy - random_baseline) < chance_tolerance:
                anomalies.append(f"Accuracy ({accuracy:.3f}) is at chance level ({random_baseline:.3f})")
                needs_review = True

            # Check balanced accuracy (chance = 1/n_classes, same as accuracy)
            if balanced_acc is not None and abs(balanced_acc - random_baseline) < chance_tolerance:
                anomalies.append(f"Balanced accuracy ({balanced_acc:.3f}) indicates no class discrimination")
                needs_review = True

            # Check AUROC (chance = 0.5 for any n_classes)
            if auroc is not None and abs(auroc - 0.5) < AUROC_CHANCE_TOLERANCE:
                anomalies.append(f"AUROC ({auroc:.3f}) indicates no discriminative power")
                needs_review = True
        
        if is_encoding or task_type == "encoding":
            if r2 is not None and r2 <= 0:
                anomalies.append(f"R² ({r2:.4f}) is at or below zero")
                needs_review = True
        
        # Build minimal user summary - no interpretations
        if needs_review:
            summary = "Statistical anomalies detected in the results. LLM analysis recommended."
        else:
            summary = "Results appear statistically reasonable."
        
        return ExecutionDiagnosis(
            needs_review=needs_review,
            anomalies=anomalies,
            probable_causes=[],  # Empty - LLM will fill this
            recommendations=[],  # Empty - LLM will fill this
            suggested_action="accept" if not needs_review else "replan",
            suggested_param_updates=None,
            summary_for_user=summary,
            raw_json={"rule_based": True, "metrics": metrics, "tool_diagnostics": tool_diagnostics}
        )

    def diagnose_execution_result(
        self,
        *,
        task_type: str,
        user_request: str,
        metrics: Dict[str, Any],
        plan: Dict[str, Any],
        dataset_info: str = "",
        trial_columns: Optional[Dict[str, Any]] = None,
        data_summary: Optional[Dict[str, Any]] = None,
        data_summary_text: str = "",
        generated_code: str = "",
        force_improve: bool = False,
    ) -> ExecutionDiagnosis:
        """
        Diagnose execution results for statistical anomalies.
        Uses LLM for intelligent diagnosis with rule-based fallback.
        
        Args:
            task_type: "decoding" | "encoding" | etc.
            user_request: Original user request (including self-correction hints)
            metrics: Execution metrics dict
            plan: The executed plan
            dataset_info: Optional dataset description
            trial_columns: Optional dict of trial column info
            data_summary: Optional data summary dict from dataloader (legacy)
            data_summary_text: Detailed data profile string (preferred)
            generated_code: The actual Python code that was executed
            force_improve: If True, always call LLM and always suggest improvements
            
        Returns:
            ExecutionDiagnosis with anomalies, causes, and recommendations
        """
        # Extract tool diagnostics
        tool_diagnostics = metrics.get("_tool_diagnostics", {})
        
        rule_diagnosis = self._rule_based_diagnose_execution(
            task_type=task_type,
            metrics=metrics,
            plan=plan,
            trial_columns=trial_columns,
        )
        
        # In force_improve mode: ALWAYS call LLM, skip early-return
        if not force_improve and not rule_diagnosis.needs_review:
            return rule_diagnosis
        
        # ----- Build LLM prompt -----
        plan_steps_str = json.dumps(plan.get("steps", []), indent=2, ensure_ascii=False)
        trial_cols_str = json.dumps(trial_columns, indent=2, ensure_ascii=False) if trial_columns else "N/A"
        
        # Extract execution error (if any) and filter internal keys
        execution_error = metrics.get("_execution_error", "")
        metrics_for_llm = {k: v for k, v in metrics.items() if not k.startswith("_")}
        if tool_diagnostics:
            metrics_for_llm["_execution_context"] = tool_diagnostics.get("data_info", {})
        metrics_str = json.dumps(metrics_for_llm, indent=2, ensure_ascii=False)

        # Prefer the detailed data_summary_text (from data profile);
        # fall back to legacy dict-based data_summary.
        if data_summary_text and data_summary_text.strip():
            data_summary_str = data_summary_text
        elif data_summary:
            data_summary_str = json.dumps(data_summary, indent=2, ensure_ascii=False)
        else:
            data_summary_str = "N/A"

        code_str = (generated_code or "N/A")[:5000]

        user_prompt = EXECUTION_DIAGNOSIS_USER_PROMPT_TEMPLATE.format(
            task_type=task_type,
            user_request=user_request,
            dataset_info=dataset_info or "N/A",
            data_summary=data_summary_str,
            execution_error=execution_error or "N/A",
            metrics_json=metrics_str,
            plan_steps=plan_steps_str,
            generated_code=code_str,
            trial_columns=trial_cols_str,
        )
        
        # Choose system prompt based on mode
        system_prompt = (
            FORCE_IMPROVE_SYSTEM_PROMPT if force_improve
            else EXECUTION_DIAGNOSIS_SYSTEM_PROMPT
        )

        print(
            f"🔍 Controller diagnosis: force_improve={force_improve}, "
            f"data_summary_len={len(data_summary_str)}, code_len={len(code_str)}",
            flush=True,
        )

        try:
            response = generate_json(
                model=self.diagnosis_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_s=60.0,
                temperature=0.3 if force_improve else 0.2,
                retries=1,
            )
            
            if response.json_obj:
                obj = response.json_obj
                diag = ExecutionDiagnosis(
                    needs_review=obj.get("needs_review", True),
                    anomalies=obj.get("anomalies", rule_diagnosis.anomalies),
                    probable_causes=obj.get("probable_causes", rule_diagnosis.probable_causes),
                    recommendations=obj.get("recommendations", rule_diagnosis.recommendations),
                    suggested_action=obj.get("suggested_action", rule_diagnosis.suggested_action),
                    suggested_param_updates=obj.get("suggested_param_updates"),
                    summary_for_user=obj.get("summary_for_user", rule_diagnosis.summary_for_user),
                    raw_json=obj,
                )
                # In force_improve mode, override "accept" but preserve "adjust_params"
                if force_improve:
                    diag.needs_review = True
                    if diag.suggested_action == "accept":
                        diag.suggested_action = "replan"
                    # Keep "adjust_params" as-is — modify mode is preferred over full replan
                return diag
        except (LLMClientError, Exception) as e:
            print(f"⚠️ Controller diagnosis LLM error ({e}), using rule-based fallback")
        
        # Fallback: in force mode, ensure we still return "replan"
        if force_improve:
            rule_diagnosis.needs_review = True
            rule_diagnosis.suggested_action = "replan"
            if not rule_diagnosis.recommendations:
                rule_diagnosis.recommendations = [
                    "Review the data profile for columns that could be used for quality filtering",
                    "Compare the current model choice against alternatives suited to the data characteristics",
                    "Examine whether the feature extraction parameters are optimal for the data distribution",
                ]
        return rule_diagnosis

    # =========================================================================
    # =========================================================================
    # Model Reordering for Auto Mode Retry Rounds
    # =========================================================================

    def reorder_models_for_retry(
        self,
        model_results: Dict[str, Dict[str, Any]],
        primary_metric: str,
        codex_baseline_value: float,
        model_order: List[str],
    ) -> List[str]:
        """
        Reorder models for retry round based on improvement potential.

        Analyzes each model's metrics and generated code to determine which
        has the highest potential to exceed the baseline after optimization.

        Args:
            model_results: {model_name: {"value": float, "metrics": dict, "generated_code": str}}
            primary_metric: e.g. "balanced_accuracy"
            codex_baseline_value: the target to beat
            model_order: original model order from Planner

        Returns:
            Reordered list of model names (most promising first).
        """
        # Build summary of each model's results for LLM
        model_summaries = []
        for mn in model_order:
            r = model_results.get(mn, {})
            value = r.get("value")
            code_snippet = (r.get("generated_code", "") or "")[:2000]
            metrics = r.get("metrics", {})
            metrics_str = json.dumps(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                indent=2, ensure_ascii=False
            )[:500]
            model_summaries.append(
                f"Model: {mn}\n"
                f"  {primary_metric}: {value}\n"
                f"  Metrics: {metrics_str}\n"
                f"  Code (truncated): {code_snippet[:800]}\n"
            )

        prompt = (
            f"You are evaluating foundation models for a neuroscience analysis task.\n\n"
            f"Codex baseline {primary_metric} = {codex_baseline_value}\n"
            f"All models failed to exceed the baseline in the previous round.\n\n"
            f"{'=' * 40}\n"
            f"{''.join(model_summaries)}\n"
            f"{'=' * 40}\n\n"
            f"Rank these models by their POTENTIAL to exceed the baseline after optimization.\n"
            f"Consider:\n"
            f"- How close the model is to the baseline\n"
            f"- Whether the code has obvious fixable issues that could yield large improvements\n"
            f"- Whether the model's approach is fundamentally sound vs. flawed\n"
            f"- Models that failed to run should be ranked last\n\n"
            f"Return JSON: {{\"ordered_models\": [\"model1\", \"model2\", ...], \"reasoning\": \"...\"}}"
        )

        try:
            response = generate_json(
                model=self.diagnosis_model,
                user_prompt=prompt,
                temperature=0.1,
            )
            if response.ok and response.parsed:
                new_order = response.parsed.get("ordered_models", [])
                reasoning = response.parsed.get("reasoning", "")
                # Validate: must contain exactly the same models
                if set(new_order) == set(model_order):
                    print(f"🔀 Controller reordered models: {new_order}", flush=True)
                    if reasoning:
                        print(f"   Reasoning: {reasoning[:200]}", flush=True)
                    return new_order
                else:
                    print(f"⚠️ Controller returned invalid model list "
                          f"(got {new_order}, expected {model_order}), keeping original order", flush=True)
            else:
                print(f"⚠️ Controller reordering returned empty/invalid response, keeping original order", flush=True)
        except Exception as e:
            print(f"⚠️ Model reordering failed: {e}, keeping original order", flush=True)

        print(f"   Original order preserved: {model_order}", flush=True)
        return model_order

    # NEW: Prompt Refinement with Diagnosis
    # =========================================================================

    def refine_prompt_with_diagnosis(
        self,
        original_prompt: str,
        recommendations: List[str],
        suggested_action: str,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Refine user prompt by incorporating diagnosis recommendations.
        
        Uses GPT-4o to intelligently merge the original user request with
        actionable recommendations from the diagnosis.
        
        Args:
            original_prompt: The original user request
            recommendations: List of actionable recommendations from diagnosis
            suggested_action: "replan" | "adjust_params" | "accept"
            dataset_info: Optional dataset metadata for context
            
        Returns:
            Refined prompt that incorporates the recommendations
        """
        print("\n" + "="*60)
        print("🔧 Controller: Refining prompt with diagnosis")
        print("="*60)
        print(f"Original prompt: {original_prompt[:100]}...")
        print(f"Recommendations: {recommendations[:3]}")
        print(f"Suggested action: {suggested_action}")
        
        # Build system prompt for refinement
        refine_system_prompt = """You are a data analysis assistant helping to refine user requests.

Your job: Take the user's original analysis request and incorporate specific technical recommendations to produce an improved, more precise request.

GUIDELINES:
1. Keep the user's original intent intact
2. Add specific technical parameters from recommendations
3. Make the request more actionable and precise
4. Use natural language, but PRESERVE any JSON-like parameter formats exactly as given
5. Include relevant column/field names if mentioned in recommendations
6. If recommendations reference specific columns or values from the data profile, include them verbatim
7. Be concise but complete
8. PRESERVE all mandatory tool requirements from the original prompt (e.g. "You MUST use decode_with_ndt3")
9. PRESERVE all metric requirements from the original prompt (e.g. "report accuracy, balanced accuracy, AUROC...")
10. FOUNDATION MODEL FEATURE PURITY: If a foundation model (REVE, LaBraM, NDT3, MtM, POYO) is used,
   DISCARD any recommendation that adds metadata columns (clinical scores, age, sex, MMSE, etc.) to
   the feature matrix. The refined prompt must NOT instruct augmenting foundation model features with
   external metadata. Only include improvements to ML methodology (classifiers, CV strategy, PCA, etc.)

OUTPUT FORMAT:
Return ONLY a JSON with:
{
  "refined_prompt": "The improved user request as a single string"
}
"""
        
        # Build user prompt
        rec_text = "\n".join([f"- {r}" for r in recommendations[:5]])
        dataset_context = ""
        if dataset_info:
            if dataset_info.get("trial_columns"):
                dataset_context += f"\nAvailable trial columns: {', '.join(dataset_info['trial_columns'][:10])}"
            if dataset_info.get("unit_columns"):
                dataset_context += f"\nAvailable unit columns: {', '.join(dataset_info['unit_columns'][:10])}"
        
        refine_user_prompt = f"""ORIGINAL USER REQUEST:
{original_prompt}

DIAGNOSIS RECOMMENDATIONS:
{rec_text}

SUGGESTED ACTION: {suggested_action}
{dataset_context}

Please produce a refined version of the user's request that incorporates these recommendations.
The refined prompt should be a complete, standalone request that a data analysis agent can execute.
"""
        
        try:
            result = generate_json(
                system_prompt=refine_system_prompt,
                user_prompt=refine_user_prompt,
                model=self.prompt_refine_model,
            )
            
            # result is LLMResponse object, use json_obj to get the parsed dict
            refined = result.json_obj.get("refined_prompt", "") if result.json_obj else ""
            if refined:
                print(f"✅ Refined prompt: {refined[:150]}...")
                return refined
            else:
                print("⚠️ LLM returned empty refined prompt, using fallback")
        except Exception as e:
            print(f"⚠️ Prompt refinement LLM error: {e}, using fallback")
        
        # Fallback: Simple concatenation
        fallback_refined = original_prompt
        if recommendations:
            fallback_refined += "\n\nAdditional requirements:\n"
            for rec in recommendations[:3]:
                fallback_refined += f"- {rec}\n"
        
        print(f"📝 Fallback refined prompt: {fallback_refined[:150]}...")
        return fallback_refined

    # =========================================================================
    # NEW: Code Generation Error Handling
    # =========================================================================
    
    def diagnose_code_generation_error(
        self,
        error_message: str,
        error_traceback: str,
        generated_code: str,
        op_name: str,
        params: Dict[str, Any],
        schema_summary: str = "",
        data_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Diagnose code generation execution errors and suggest fixes.
        
        When LLM-generated code fails to execute, this method analyzes
        the error and provides actionable guidance for self-correction.
        
        Args:
            error_message: The error message
            error_traceback: Full traceback
            generated_code: The code that failed
            op_name: Tool/operation name
            params: Parameters passed
            schema_summary: Summary of data schema
            data_summary: Summary of raw data
            
        Returns:
            Dict with:
                - error_type: Classification of error
                - probable_cause: LLM-inferred cause
                - suggested_fix: How to fix the code
                - should_retry: Whether to attempt code regeneration
                - retry_hints: Hints for code regenerator
        """
        print("\n" + "="*60)
        print("🔍 Controller: Diagnosing code generation error")
        print("="*60)
        print(f"Operation: {op_name}")
        print(f"Error: {error_message[:200]}...")
        
        # Rule-based classification first
        error_type = "UNKNOWN"
        probable_cause = ""
        should_retry = True
        retry_hints = []
        
        error_lower = error_message.lower()
        
        if "keyerror" in error_lower or "column" in error_lower:
            error_type = "COLUMN_NOT_FOUND"
            probable_cause = "Code references a column name that doesn't exist in the DataFrame"
            retry_hints = [
                "Check actual column names in data_summary",
                "Use exact column names from schema",
                "Handle case sensitivity",
            ]
        elif "shape" in error_lower or "dimension" in error_lower:
            error_type = "SHAPE_MISMATCH"
            probable_cause = "Array dimensions don't match expected shape for the operation"
            retry_hints = [
                "Verify X has shape [n_samples, n_features]",
                "Verify y has shape [n_samples]",
                "Check for NaN values that might change shape",
            ]
        elif "nan" in error_lower or "missing" in error_lower:
            error_type = "MISSING_DATA"
            probable_cause = "Data contains NaN or missing values"
            retry_hints = [
                "Use dropna() to remove rows with missing values",
                "Use fillna() with appropriate default",
                "Check for empty arrays",
            ]
        elif "type" in error_lower or "dtype" in error_lower:
            error_type = "TYPE_ERROR"
            probable_cause = "Data type mismatch (e.g., string where number expected)"
            retry_hints = [
                "Convert labels with pd.factorize() or LabelEncoder",
                "Ensure features are numeric (astype(float))",
                "Handle categorical columns properly",
            ]
        elif "import" in error_lower or "module" in error_lower:
            error_type = "IMPORT_ERROR"
            probable_cause = "Required module not available in execution environment"
            should_retry = False  # Can't fix import errors by regenerating
            retry_hints = []
        elif "memory" in error_lower or "oom" in error_lower:
            error_type = "MEMORY_ERROR"
            probable_cause = "Dataset too large for available memory"
            should_retry = True
            retry_hints = [
                "Use subset of data for analysis",
                "Reduce number of features",
                "Use incremental algorithms",
            ]
        elif "syntax" in error_lower:
            error_type = "SYNTAX_ERROR"
            probable_cause = "Generated code has Python syntax errors"
            retry_hints = [
                "Check for proper indentation",
                "Ensure balanced parentheses/brackets",
                "Verify string formatting",
            ]
        else:
            error_type = "RUNTIME_ERROR"
            probable_cause = "General execution error in generated code"
            retry_hints = [
                "Review the traceback for specific line numbers",
                "Check data access patterns",
                "Verify variable names",
            ]
        
        # Try LLM for more sophisticated diagnosis
        diagnosis_prompt = f"""Analyze this code execution error and provide diagnosis.

## Failed Code
```python
{generated_code[:2000]}
```

## Error Message
{error_message}

## Traceback
{error_traceback[:1000]}

## Operation
{op_name}

## Data Schema
{schema_summary[:500]}

## Data Summary
{data_summary[:500]}

Provide a JSON response:
{{
  "error_type": "COLUMN_NOT_FOUND | SHAPE_MISMATCH | MISSING_DATA | TYPE_ERROR | OTHER",
  "probable_cause": "One sentence explanation",
  "suggested_fix": "Specific code fix suggestion",
  "should_retry": true/false,
  "retry_hints": ["hint1", "hint2"]
}}
"""
        
        try:
            result = generate_json(
                system_prompt="You are a Python debugging expert. Analyze errors and suggest fixes.",
                user_prompt=diagnosis_prompt,
                model=self.model,
                temperature=0.2,
            )
            
            if result.json_obj:
                obj = result.json_obj
                return {
                    "error_type": obj.get("error_type", error_type),
                    "probable_cause": obj.get("probable_cause", probable_cause),
                    "suggested_fix": obj.get("suggested_fix", ""),
                    "should_retry": obj.get("should_retry", should_retry),
                    "retry_hints": obj.get("retry_hints", retry_hints),
                    "llm_diagnosis": True,
                }
        except Exception as e:
            print(f"⚠️ LLM diagnosis failed: {e}, using rule-based")
        
        # Return rule-based diagnosis
        return {
            "error_type": error_type,
            "probable_cause": probable_cause,
            "suggested_fix": "",
            "should_retry": should_retry,
            "retry_hints": retry_hints,
            "llm_diagnosis": False,
        }


def test_controller():
    """Test controller agent"""
    print("="*80)
    print("Testing Controller Agent")
    print("="*80)
    
    controller = ControllerAgent()
    
    # Test 1: QA with citations (PASS)
    print("\n📝 Test 1: QA with citations required and provided")
    result = controller.validate(
        task_type="QA",
        user_request="What brain regions are involved in working memory?",
        agent_output="The prefrontal cortex is critical (Smith et al., 2023).",
        references=["Smith et al., 2023"],
        need_citations=True,
        has_dataset=False
    )
    print(f"   Status: {result.status}")
    print(f"   Issues: {result.issues}")
    print(f"   Next Action: {result.next_action}")
    
    # Test 4: Prompt refinement with diagnosis
    print("\n📝 Test 4: Refine prompt with diagnosis recommendations")
    refined = controller.refine_prompt_with_diagnosis(
        original_prompt="decode trial_type from spike times",
        recommendations=[
            "Use trial-aligned firing rates with window relative to a key event field",
            "Bin spike times into 100-200ms bins",
            "Filter by valid trials only"
        ],
        suggested_action="replan"
    )
    print(f"   Refined prompt: {refined[:200]}...")
    
    # Test 2: QA without citations when required (RETRIEVE_MORE)
    print("\n📝 Test 2: QA with citations required but missing")
    result = controller.validate(
        task_type="QA",
        user_request="What brain regions are involved in working memory?",
        agent_output="The prefrontal cortex is critical.",
        references=[],
        need_citations=True,
        has_dataset=False
    )
    print(f"   Status: {result.status}")
    print(f"   Issues: {result.issues}")
    
    # Test 3: DA without dataset when needed (NEED_DATA)
    print("\n📝 Test 3: DA wanting execution but no dataset")
    result = controller.validate(
        task_type="DA",
        user_request="Please run the decoding analysis on my data",
        agent_output="I need your dataset to proceed.",
        references=[],
        need_citations=False,
        has_dataset=False
    )
    print(f"   Status: {result.status}")
    print(f"   Issues: {result.issues}")


if __name__ == "__main__":
    test_controller()

