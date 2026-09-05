# neuro_copilot/execution_agent.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_copilot.core.state import NeuroGlobalState
from neuro_copilot.core.DA.tools_registry import ToolRegistry, ToolSpec
from neuro_copilot.core.DA.code_generator import CodeGenerator, GeneratedCode
from neuro_copilot.core.DA.code_executor import CodeExecutor, ExecutionResult as CodeExecutionResult
from neuro_copilot.core.DA.schema_parser import ParsedSchema


# -----------------------------
# Exceptions
# -----------------------------

class ExecutionAgentError(Exception):
    pass


class ToolDispatchError(ExecutionAgentError):
    pass


class ToolParamError(ExecutionAgentError):
    pass


class ToolRuntimeError(ExecutionAgentError):
    pass


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ExecutionAgentConfig:
    """
    Execution layer config (no semantics).
    """
    artifacts_dir: str = "artifacts"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    run_name: Optional[str] = None

    # Reproducibility
    seed: Optional[int] = int(os.getenv("DA_SEED", "0"))

    # Validation behavior
    strict_extra_params: bool = True

    # Safety: maximum steps to execute
    max_steps: int = int(os.getenv("DA_MAX_STEPS", "50"))


@dataclass
class StepResult:
    step_index: int
    op_name: str
    params: Dict[str, Any]
    ok: bool
    started_at_ms: int
    finished_at_ms: int
    duration_ms: int
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    ok: bool
    run_id: str
    artifacts_root: str
    steps: List[StepResult] = field(default_factory=list)

    # Summary for Analysis Agent
    summary: Dict[str, Any] = field(default_factory=dict)

    # "feed back to Reasoning Agent" structured feedback
    plan_fix_feedback: Optional[Dict[str, Any]] = None
    
    # NEW: Support for recoverable errors that need user clarification
    needs_user_clarification: bool = False
    pending_questions: Optional[List[Dict[str, str]]] = None
    recoverable_error_context: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionContext:
    """
    Internal execution context.
    """
    run_id: str
    artifacts_root: Path
    outputs_root: Path
    logs_root: Path

    # cached state pointers
    field_catalog_text: Optional[str] = None
    field_catalog: Optional[Dict[str, Any]] = None

    # random generator
    rng: Optional[np.random.Generator] = None


# -----------------------------
# Utilities: filesystem + IDs
# -----------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_run_id(run_name: Optional[str] = None) -> str:
    # compact, stable, filesystem-friendly, collision-resistant
    import uuid
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    short_uuid = uuid.uuid4().hex[:6]
    if run_name:
        return f"{ts}_{short_uuid}__{run_name}"
    return f"{ts}_{short_uuid}"


def _ensure_dirs(base_dir: Path, run_id: str, cfg: ExecutionAgentConfig) -> ExecutionContext:
    artifacts_root = base_dir / cfg.artifacts_dir / run_id
    outputs_root = base_dir / cfg.outputs_dir / run_id
    logs_root = base_dir / cfg.logs_dir / run_id

    artifacts_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    return ExecutionContext(
        run_id=run_id,
        artifacts_root=artifacts_root,
        outputs_root=outputs_root,
        logs_root=logs_root,
    )


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# Utilities: minimal type checks
# -----------------------------

def _check_param_types(params: Dict[str, Any], types_spec: Dict[str, str]) -> List[str]:
    """
    types_spec examples:
      {"cv_folds": "int", "feature": "string", "predictors": "list[string]"}
    Returns list of error strings (empty if ok).
    """
    errs: List[str] = []

    def _is_str(x): return isinstance(x, str)
    def _is_int(x): return isinstance(x, int) and not isinstance(x, bool)
    def _is_float(x): return isinstance(x, (float, int)) and not isinstance(x, bool)
    def _is_bool(x): return isinstance(x, bool)

    for k, t in (types_spec or {}).items():
        if k not in params:
            continue
        v = params[k]

        if t in ("string", "str"):
            if not _is_str(v):
                errs.append(f"param '{k}' expected string, got {type(v).__name__}")
        elif t in ("int",):
            if not _is_int(v):
                errs.append(f"param '{k}' expected int, got {type(v).__name__}")
        elif t in ("float", "number"):
            if not _is_float(v):
                errs.append(f"param '{k}' expected float/number, got {type(v).__name__}")
        elif t in ("bool",):
            if not _is_bool(v):
                errs.append(f"param '{k}' expected bool, got {type(v).__name__}")
        elif t == "list[string]":
            if not (isinstance(v, list) and all(_is_str(x) for x in v)):
                errs.append(f"param '{k}' expected list[string], got {type(v).__name__}")
        else:
            # unknown spec: do not fail hard (extensible)
            pass

    return errs


def _validate_params_against_spec(
    op_name: str,
    params: Dict[str, Any],
    spec: Dict[str, Any],
    strict_extra_params: bool,
) -> None:
    required = spec.get("required", []) or []
    optional = spec.get("optional", []) or []
    allowed = set(required) | set(optional)

    missing = [k for k in required if k not in params]
    if missing:
        raise ToolParamError(
            f"[{op_name}] missing required params: {missing}"
        )

    if strict_extra_params:
        extra = [k for k in params.keys() if k not in allowed]
        if extra:
            raise ToolParamError(
                f"[{op_name}] extra params not allowed (strict mode): {extra}"
            )

    # optional type checks
    type_errs = _check_param_types(params, spec.get("types", {}) or {})
    if type_errs:
        raise ToolParamError(f"[{op_name}] param type errors: {type_errs}")


# -----------------------------
# Tools: built-in minimal implementations (dataset-agnostic)
# -----------------------------

def _tool_describe_dataset(state: NeuroGlobalState, ctx: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses loader-provided summaries. Does NOT inspect full raw data.
    """
    top_k_fields = params.get("top_k_fields", 30)

    fc = ctx.field_catalog or state.extra.get("field_catalog") or {}
    fct = ctx.field_catalog_text or state.extra.get("field_catalog_text") or ""

    # Extract top-level
    top_level = fc.get("top_level", [])
    record_stats = fc.get("record_field_stats", {})
    common90 = record_stats.get("common_keys_ratio_ge_0.90", [])[:top_k_fields]
    common50 = record_stats.get("common_keys_ratio_ge_0.50", [])[:top_k_fields]

    summary_text = fct if fct else "(field_catalog_text not available)"

    return {
        "summary_text": summary_text,
        "high_level_stats": {
            "top_level_vars": top_level,
            "common_keys_ge_0.90": common90,
            "common_keys_ge_0.50": common50,
            "n_records_sampled": record_stats.get("n_records_sampled"),
        }
    }


def _tool_list_fields(state: NeuroGlobalState, ctx: ExecutionContext, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search over field_catalog fields (not raw).
    """
    query = (params.get("query") or "").strip().lower()
    top_k = int(params.get("top_k", 20))

    fc = ctx.field_catalog or state.extra.get("field_catalog") or {}
    fields = ((fc.get("record_field_stats") or {}).get("fields") or {})

    # Rank by presence ratio + simple name match
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for name, meta in fields.items():
        ratio = float(((meta.get("presence") or {}).get("ratio")) or 0.0)
        bonus = 0.0
        if query:
            if query in name.lower():
                bonus += 0.2
            # also match kind names
            kinds = meta.get("kinds", {})
            if any(query in k.lower() for k in kinds.keys()):
                bonus += 0.05
        scored.append((ratio + bonus, name, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, name, meta in scored[:top_k]:
        out.append({
            "field": name,
            "presence_ratio": (meta.get("presence") or {}).get("ratio"),
            "kinds": meta.get("kinds"),
            "mixed_kinds": meta.get("mixed_kinds"),
        })

    return {"candidates": out}


def _tool_not_implemented(state: NeuroGlobalState, ctx: ExecutionContext, params: Dict[str, Any], op_name: str) -> Dict[str, Any]:
    return {
        "status": "not_implemented",
        "op_name": op_name,
        "message": "Tool callable not implemented yet. Fill ToolSpec.callable in registry and retry."
    }


# -----------------------------
# Code Generation Execution
# -----------------------------

def _get_raw_data_from_state(state: NeuroGlobalState) -> Dict[str, pd.DataFrame]:
    """
    Extract raw data from state for code execution.
    
    Handles multiple data formats:
    - Single CSV: neural_dataset.raw["_dataframe"]
    - Multi-CSV: neural_dataset.raw["_files"][filename]["_dataframe"]
    - MAT files: neural_dataset.raw with numpy arrays
    - NWB files: neural_dataset.raw with structured data
    
    Returns:
        Dict mapping filename/varname -> DataFrame or array
    """
    raw_data = {}
    
    # Get NeuralDataset from state
    neural_dataset = state.extra.get("neural_dataset")
    if neural_dataset is not None:
        nd_raw = neural_dataset.raw
        
        # Check for multi-CSV format: _files dict with multiple CSVs
        if "_files" in nd_raw and isinstance(nd_raw["_files"], dict):
            for filename, file_data in nd_raw["_files"].items():
                if isinstance(file_data, dict) and "_dataframe" in file_data:
                    raw_data[filename] = file_data["_dataframe"]
                    
        # Check for single CSV format: _dataframe directly in raw
        elif "_dataframe" in nd_raw:
            source_file = nd_raw.get("_source_file", "data")
            raw_data[source_file] = nd_raw["_dataframe"]
            
        # Fallback: iterate through all items in raw
        else:
            for var_name, var_value in nd_raw.items():
                # Skip internal keys
                if var_name.startswith("_"):
                    continue
                if isinstance(var_value, pd.DataFrame):
                    raw_data[var_name] = var_value
                elif isinstance(var_value, np.ndarray):
                    # Convert numpy arrays to DataFrame if 2D
                    if var_value.ndim == 2:
                        raw_data[var_name] = pd.DataFrame(var_value)
                    else:
                        raw_data[var_name] = var_value  # Keep as numpy for 1D/3D+
    
    # Also check for directly loaded DataFrames in state.extra
    if "loaded_dataframes" in state.extra:
        raw_data.update(state.extra["loaded_dataframes"])
    
    # Debug output
    if raw_data:
        print(f"📦 _get_raw_data_from_state: Found {len(raw_data)} data sources: {list(raw_data.keys())}")
    else:
        print(f"⚠️ _get_raw_data_from_state: No raw data found!")
        if neural_dataset is not None:
            print(f"   neural_dataset.raw keys: {list(neural_dataset.raw.keys())[:20]}")
    
    return raw_data


def _get_data_summary(raw_data: Dict[str, Any]) -> str:
    """
    Generate a summary of the raw data for code generation prompts.
    """
    lines = []
    for name, data in raw_data.items():
        if isinstance(data, pd.DataFrame):
            lines.append(f"- {name}: DataFrame with shape {data.shape}")
            lines.append(f"  Columns: {list(data.columns[:20])}" + ("..." if len(data.columns) > 20 else ""))
            lines.append(f"  Types: {dict(list(data.dtypes.items())[:10])}" + ("..." if len(data.dtypes) > 10 else ""))
        elif isinstance(data, np.ndarray):
            lines.append(f"- {name}: numpy array with shape {data.shape}, dtype={data.dtype}")
        else:
            lines.append(f"- {name}: {type(data).__name__}")
    return "\n".join(lines) if lines else "No data loaded"


def _execute_with_code_generation(
    state: NeuroGlobalState,
    ctx: ExecutionContext,
    op_name: str,
    params: Dict[str, Any],
    parsed_schema: Optional[ParsedSchema],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Execute a tool using LLM-generated code.
    
    Flow:
    1. Get raw data and schema from state
    2. Generate data extraction + tool invocation code via LLM
    3. Execute code in sandboxed environment
    4. If error, regenerate code with error feedback (self-correction)
    5. Return result
    
    Args:
        state: Global state with raw data
        ctx: Execution context with paths
        op_name: Name of tool to call (e.g., "decode_logistic_regression")
        params: Parameters for the tool
        parsed_schema: Parsed schema describing the data (can be None)
        max_retries: Maximum number of code regeneration attempts on error
        
    Returns:
        Tool execution result
        
    Raises:
        ToolRuntimeError: If code execution fails after all retries
    """
    # Get raw data
    raw_data = _get_raw_data_from_state(state)
    if not raw_data:
        raise ToolRuntimeError(f"No raw data found in state for code generation execution")
    
    # Get data summary for prompt
    data_summary = _get_data_summary(raw_data)
    
    # If no parsed_schema, try to get schema info from state or build minimal schema
    if parsed_schema is None:
        # Try to get schema from state
        parsed_schema = state.extra.get("parsed_schema")
    
    # If still no schema, use field_catalog as fallback schema info
    if parsed_schema is None:
        field_catalog_text = state.extra.get("field_catalog_text", "")
        if field_catalog_text:
            # Add field catalog info to data_summary
            data_summary = f"{data_summary}\n\nField Catalog:\n{field_catalog_text}"
    
    # Build task description from op_name and params
    task_description = f"Perform {op_name} analysis"
    if "description" in params:
        task_description = params.pop("description")
    elif "task" in params:
        task_description = params.pop("task")
    
    # Initialize code generator and executor
    code_generator = CodeGenerator()
    code_executor = CodeExecutor(
        output_dir=str(ctx.outputs_root),
        allow_file_write=True,
    )
    
    # Track attempts for self-correction
    attempts = []
    last_error = None
    
    for attempt in range(max_retries + 1):
        # Generate code
        try:
            if attempt == 0:
                # First attempt: generate fresh code
                generated = code_generator.generate_analysis_code(
                    task_description=task_description,
                    parsed_schema=parsed_schema,
                    data_summary=data_summary,
                    tool_name=op_name,
                    tool_parameters=params,
                )
            else:
                # Retry with error feedback
                generated = code_generator.regenerate_with_error(
                    original_code=attempts[-1]["code"],
                    error_message=str(last_error),
                    error_traceback=attempts[-1].get("traceback", ""),
                    parsed_schema=parsed_schema,
                    data_summary=data_summary,
                    tool_name=op_name,
                )
        except Exception as gen_err:
            raise ToolRuntimeError(f"Code generation failed: {gen_err}")
        
        # Execute the generated code
        exec_result = code_executor.execute(
            code=generated.code,
            raw_data=raw_data,
            extra_context={
                "params": params,
                "output_dir": str(ctx.outputs_root),
                "parsed_schema": parsed_schema,  # Needed for NDT3 and schema-driven tools
            },
        )
        
        # Track this attempt with full details
        attempts.append({
            "attempt": attempt + 1,
            "code": generated.code,
            "prompt": generated.prompt,
            "system_prompt": generated.system_prompt,
            "success": exec_result.success,
            "result": exec_result.result if exec_result.success else None,
            "error": exec_result.error,
            "traceback": exec_result.traceback,
            "stdout": exec_result.stdout,
            "execution_time": exec_result.execution_time,
        })
        
        if exec_result.success:
            # Success! Return the result with full attempt history
            return {
                "status": "success",
                "op_name": op_name,
                "result": exec_result.result,
                "execution_method": "code_generation",
                "attempts_count": len(attempts),
                "attempts_detail": attempts,  # Full history of all attempts
                "generated_code": generated.code,
                "prompt": generated.prompt,
                "system_prompt": generated.system_prompt,
                "stdout": exec_result.stdout,
                "execution_time": exec_result.execution_time,
            }
        else:
            last_error = exec_result.error
            # Log the error for debugging
            if hasattr(ctx, 'logs_root') and ctx.logs_root:
                error_log = ctx.logs_root / f"code_gen_attempt_{attempt+1}.json"
                _json_dump(error_log, {
                    "attempt": attempt + 1,
                    "code": generated.code,
                    "error": exec_result.error,
                    "traceback": exec_result.traceback,
                })
    
    # All attempts failed
    raise ToolRuntimeError(
        f"Code execution failed after {max_retries + 1} attempts. "
        f"Last error: {last_error}\n"
        f"Attempts: {json.dumps(attempts, indent=2, default=str)}"
    )


# -----------------------------
# Dispatch
# -----------------------------

# Tools that should use code generation (generic tools with X, y interface)
# These tools have "generic_" prefix and work with any dataset via code generation
CODE_GENERATION_TOOLS = {
    "generic_decode_logistic",
    "generic_decode_svm",
    "generic_decode_holdout",
    "generic_compute_firing_rates",
    "generic_bin_spike_times",
    "generic_extract_trial_features",
    "generic_pca",
    "generic_plot_confusion_matrix",
    "generic_plot_raster",
}

# Mapping from legacy tools to generic equivalents
# These legacy tools will be automatically redirected to use code generation
LEGACY_TO_GENERIC_MAP = {
    "decode_logistic_regression": "generic_decode_logistic",
    "decode_svm": "generic_decode_svm",
    "decode_with_holdout": "generic_decode_holdout",
    "decode_multiclass": "generic_decode_logistic",  # Multiclass uses logistic
}

# Tools that should always use direct callable (metadata/utility tools)
DIRECT_CALLABLE_TOOLS = {
    "describe_dataset",
    "list_fields",
}


def _dispatch_tool(
    state: NeuroGlobalState,
    ctx: ExecutionContext,
    reg: ToolRegistry,
    op_name: str,
    params: Dict[str, Any],
    strict_extra_params: bool,
) -> Dict[str, Any]:
    if not reg.has(op_name):
        raise ToolDispatchError(f"op_name not in registry (not whitelisted): {op_name}")

    tool: ToolSpec = reg.get(op_name)

    # Validate params against spec (dataset-agnostic)
    _validate_params_against_spec(
        op_name=op_name,
        params=params,
        spec=tool.input_spec or {},
        strict_extra_params=strict_extra_params,
    )

    # Built-in minimal implementations for early stage
    if op_name == "describe_dataset":
        return _tool_describe_dataset(state, ctx, params)
    if op_name == "list_fields":
        return _tool_list_fields(state, ctx, params)

    # Check if this is a legacy tool that should be redirected to generic version
    # This allows code generation even when Planner selects old tool names
    effective_op_name = op_name
    if op_name in LEGACY_TO_GENERIC_MAP:
        effective_op_name = LEGACY_TO_GENERIC_MAP[op_name]
        print(f"📌 Redirecting legacy tool '{op_name}' -> '{effective_op_name}' (code generation)")

    # Use code generation for generic tools or redirected legacy tools
    use_code_generation = (
        effective_op_name in CODE_GENERATION_TOOLS
        and effective_op_name not in DIRECT_CALLABLE_TOOLS
    )
    
    if use_code_generation:
        # Build description from legacy params if not provided
        if "description" not in params:
            # Convert legacy params to description for code generator
            desc_parts = []
            if "target" in params:
                desc_parts.append(f"decode {params['target']}")
            if "feature" in params:
                desc_parts.append(f"from {params['feature']}")
            if "feature_type" in params:
                desc_parts.append(f"using {params['feature_type']}")
            if "time_window_fields" in params:
                desc_parts.append(f"with time windows {params['time_window_fields']}")
            if desc_parts:
                params["description"] = " ".join(desc_parts)
        
        # Use LLM-generated code for data extraction + tool invocation
        return _execute_with_code_generation(
            state=state,
            ctx=ctx,
            op_name=effective_op_name,
            params=params,
            parsed_schema=state.extra.get("parsed_schema"),  # Can be None
            max_retries=2,
        )

    # Fallback: If registry has callable, run it directly
    if tool.callable is not None:
        # Add op_name to params for tools that need it (e.g., for unique filename generation)
        params_with_op = dict(params)
        params_with_op["_op_name"] = op_name
        return tool.callable(state=state, ctx=ctx, params=params_with_op)

    # Otherwise stub -> treat as hard failure
    out = _tool_not_implemented(state, ctx, params, op_name=op_name)
    raise ToolDispatchError(f"[{op_name}] not implemented: {out.get('message', '')}")


# -----------------------------
# Plan execution
# -----------------------------

def _build_plan_fix_feedback(
    plan: Dict[str, Any],
    step_index: int,
    op_name: str,
    err_code: str,
    message: str,
    suggestions: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Structured feedback that can be fed back to Reasoning Agent to revise plan.
    """
    return {
    "type": "execution_plan_fix_feedback",
    "status": "failed",
    "failed_step": {
        "step_index": step_index,
        "op_name": op_name,
    },
    "error": {
        "code": err_code,
        "message": message,
        "details": details or {},
    },
    "suggestions": suggestions or [],

    # NEW: Give Reasoning Agent a structured plan modification instruction (can be directly copied)
    "suggested_plan_change": {
        "action": "replace_or_remove_step",
        "step_index": step_index,
        "failed_op_name": op_name,
        "hint": (
            "Replace this step with a whitelisted tool that has an implementation, "
            "or remove it for now. If the issue is unknown field names, use field_catalog "
            "fields (e.g., RT/TS) or add questions_for_user."
        ),
    },

    "plan_snapshot": plan,
    "instructions_for_reasoning_agent": (
        "Revise the plan JSON to fix the failure. "
        "Use ONLY whitelisted op_name and satisfy tool input_spec. "
        "If a field path is ambiguous or missing, add a question_for_user."
    ),
}


def run_execution_agent(
    state: NeuroGlobalState,
    *,
    registry: ToolRegistry,
    config: Optional[ExecutionAgentConfig] = None,
    plan_key: str = "validated_plan_json",
    project_root: Optional[Union[str, Path]] = None,
) -> NeuroGlobalState:
    """
    Execute a validated plan with strict tool dispatch.

    Inputs:
      - state.extra[plan_key]: validated plan JSON
      - registry: ToolRegistry

    Outputs:
      - state.extra["execution_result"]: ExecutionResult (dict-like via dataclasses in memory)
      - state.extra["execution_result_json"]: JSON-serializable report
      - artifacts written under artifacts/<run_id>/
    """
    if config is None:
        config = ExecutionAgentConfig()

    plan = state.extra.get(plan_key)
    if plan is None:
        raise ExecutionAgentError(f"Plan not found in state.extra['{plan_key}'].")

    # project root (where artifacts/outputs/logs live)
    base_dir = Path(project_root) if project_root is not None else Path.cwd()

    run_id = _make_run_id(config.run_name)
    ctx = _ensure_dirs(base_dir, run_id, config)

    # Cache loader summaries
    ctx.field_catalog_text = state.extra.get("field_catalog_text")
    ctx.field_catalog = state.extra.get("field_catalog")

    # Reproducibility
    if config.seed is not None:
        ctx.rng = np.random.default_rng(int(config.seed))

    # Persist plan snapshot
    _json_dump(ctx.artifacts_root / "validated_plan.json", plan)

    started = _now_ms()
    exec_result = ExecutionResult(
        ok=False,
        run_id=run_id,
        artifacts_root=str(ctx.artifacts_root),
        steps=[],
        summary={
            "started_at_ms": started,
            "seed": config.seed,
            "strict_extra_params": config.strict_extra_params,
        },
    )

    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        raise ExecutionAgentError("Plan.steps must be a list.")

    if len(steps) > config.max_steps:
        raise ExecutionAgentError(f"Plan has too many steps ({len(steps)}), max_steps={config.max_steps}.")

    for i, step in enumerate(steps):
        step_started = _now_ms()

        op_name = (step or {}).get("op_name")
        params = (step or {}).get("params") or {}

        step_rec = StepResult(
            step_index=i,
            op_name=str(op_name),
            params=dict(params) if isinstance(params, dict) else {"_raw": params},
            ok=False,
            started_at_ms=step_started,
            finished_at_ms=step_started,
            duration_ms=0,
            output=None,
            error=None,
        )

        try:
            if not isinstance(op_name, str) or not op_name.strip():
                raise ToolDispatchError("Missing or invalid op_name in step.")

            if not isinstance(params, dict):
                raise ToolParamError("params must be a JSON object (dict).")

            out = _dispatch_tool(
                state=state,
                ctx=ctx,
                reg=registry,
                op_name=op_name,
                params=params,
                strict_extra_params=config.strict_extra_params,
            )

            step_finished = _now_ms()
            step_rec.ok = True
            step_rec.finished_at_ms = step_finished
            step_rec.duration_ms = step_finished - step_started
            step_rec.output = out

            # Save each step output as artifact JSON
            # For code generation results, ensure we wrap the result in expected format
            save_output = out
            if isinstance(out, dict) and out.get("execution_method") == "code_generation":
                # Extract the actual result from code generation output
                # and wrap it in the format expected by pipeline (with "metrics" key)
                code_gen_result = out.get("result", {})
                if isinstance(code_gen_result, dict):
                    # The result is the metrics dict directly
                    # Include full code generation details for audit trail
                    save_output = {
                        "metrics": code_gen_result,
                        "execution_method": "code_generation",
                        "generated_code": out.get("generated_code"),
                        "prompt": out.get("prompt"),
                        "system_prompt": out.get("system_prompt"),
                        "attempts_count": out.get("attempts_count"),
                        "attempts_detail": out.get("attempts_detail"),
                        "stdout": out.get("stdout"),
                        "execution_time": out.get("execution_time"),
                    }
            _json_dump(ctx.outputs_root / f"step_{i:02d}__{op_name}.json", save_output)

        except (ToolDispatchError, ToolParamError) as e:
            step_finished = _now_ms()
            step_rec.ok = False
            step_rec.finished_at_ms = step_finished
            step_rec.duration_ms = step_finished - step_started
            step_rec.error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=3),
            }
            exec_result.steps.append(step_rec)

            # Build plan-fix feedback for Reasoning Agent
            exec_result.plan_fix_feedback = _build_plan_fix_feedback(
                plan=plan,
                step_index=i,
                op_name=str(op_name),
                err_code="DISPATCH_OR_PARAM_ERROR",
                message=str(e),
                suggestions=[
                    "Use only op_name that exists in registry whitelist.",
                    "Ensure params contain all required keys and no extra keys in strict mode.",
                    "Match param types specified by tool input_spec.types when present.",
                ],
                details={"step": step, "strict_extra_params": config.strict_extra_params},
            )
            break

        except Exception as e:
            step_finished = _now_ms()
            step_rec.ok = False
            step_rec.finished_at_ms = step_finished
            step_rec.duration_ms = step_finished - step_started
            step_rec.error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=8),
            }
            exec_result.steps.append(step_rec)
            
            # NEW: Try to recover from error with user clarification
            from core.error_recovery import analyze_error_for_recovery
            
            # Get already answered fields from state
            answered_fields = state.extra.get("answered_fields", set())
            
            recoverable_error = analyze_error_for_recovery(
                error=e,
                error_message=str(e),
                step_params=params,
                field_catalog=ctx.field_catalog,
                answered_fields=answered_fields,
            )
            
            if recoverable_error:
                # This error can be recovered through user clarification
                exec_result.needs_user_clarification = True
                exec_result.pending_questions = recoverable_error.suggested_questions
                exec_result.recoverable_error_context = {
                    "failed_step_index": i,
                    "failed_op_name": str(op_name),
                    "error_type": recoverable_error.error_type,
                    "error_context": recoverable_error.context,
                }
                
                # Build special feedback for recoverable errors
                exec_result.plan_fix_feedback = {
                    "type": "recoverable_error_with_questions",
                    "status": "needs_clarification",
                    "failed_step": {
                        "step_index": i,
                        "op_name": str(op_name),
                    },
                    "error": {
                        "code": "RECOVERABLE_ERROR",
                        "type": recoverable_error.error_type,
                        "message": recoverable_error.error_message,
                        "details": recoverable_error.context,
                    },
                    "questions_for_user": recoverable_error.suggested_questions,
                    "recovery_instructions": (
                        "This error can be recovered by providing clarifications. "
                        "Please answer the questions above, and the system will update "
                        "the analysis plan and retry from this point."
                    ),
                    "plan_snapshot": plan,
                }
                
                # Don't break - we'll return to let the web app handle the questions
                break
            else:
                # Not recoverable - use original error handling
                exec_result.plan_fix_feedback = _build_plan_fix_feedback(
                    plan=plan,
                    step_index=i,
                    op_name=str(op_name),
                    err_code="TOOL_RUNTIME_ERROR",
                    message=str(e),
                    suggestions=[
                        "If failure is due to missing/shape mismatch fields, revise plan params (paths/time_window) or ask user clarification.",
                        "Keep steps minimal; validate intermediate outputs with utility ops before heavy modeling.",
                    ],
                    details={"step": step},
                )
                break

        exec_result.steps.append(step_rec)

    finished = _now_ms()
    exec_result.summary["finished_at_ms"] = finished
    exec_result.summary["total_duration_ms"] = finished - started

    exec_result.ok = all(s.ok for s in exec_result.steps) if exec_result.steps else False

    # Persist final report JSON
    report = {
        "ok": exec_result.ok,
        "run_id": exec_result.run_id,
        "artifacts_root": exec_result.artifacts_root,
        "summary": exec_result.summary,
        "steps": [
            {
                "step_index": s.step_index,
                "op_name": s.op_name,
                "params": s.params,
                "ok": s.ok,
                "started_at_ms": s.started_at_ms,
                "finished_at_ms": s.finished_at_ms,
                "duration_ms": s.duration_ms,
                "output_path": str((ctx.outputs_root / f"step_{s.step_index:02d}__{s.op_name}.json")),
                "error": s.error,
            }
            for s in exec_result.steps
        ],
        "plan_fix_feedback": exec_result.plan_fix_feedback,
        # NEW: Include recoverable error info
        "needs_user_clarification": exec_result.needs_user_clarification,
        "pending_questions": exec_result.pending_questions,
        "recoverable_error_context": exec_result.recoverable_error_context,
    }
    _json_dump(ctx.artifacts_root / "execution_report.json", report)

    # Store into state for downstream Analysis Agent
    state.extra["execution_result_json"] = report
    state.extra["execution_result"] = exec_result  # keep rich object too

    return state