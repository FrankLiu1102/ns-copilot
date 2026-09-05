# neuro_copilot/plan_validator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tools_registry import ToolRegistry

class PlanValidationError(Exception):
    pass


@dataclass
class ValidationMessage:
    level: str  # "error" | "warning"
    path: str   # e.g., "steps[0].params.cv_folds"
    message: str


@dataclass
class ValidatedPlan:
    ok: bool
    plan: Optional[Dict[str, Any]] = None
    errors: List[ValidationMessage] = field(default_factory=list)
    warnings: List[ValidationMessage] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            msgs = "\n".join([f"[{m.level}] {m.path}: {m.message}" for m in self.errors])
            raise PlanValidationError(f"Plan validation failed:\n{msgs}")


# -----------------------------
# Type checking helpers (v0)
# -----------------------------

def _type_matches(val: Any, type_spec: str) -> bool:
    """
    Very lightweight runtime checks.
    Supported:
    - "string", "int", "float", "bool"
    - "list[string]", "list[int]", "list[float]"
    - "dict"
    - "any"
    """
    if type_spec in (None, "", "any"):
        return True

    ts = type_spec.strip().lower()

    if ts == "string":
        return isinstance(val, str)
    if ts == "int":
        # allow bool? no (bool is subclass of int)
        return isinstance(val, int) and not isinstance(val, bool)
    if ts == "float":
        return isinstance(val, (float, int)) and not isinstance(val, bool)
    if ts == "bool":
        return isinstance(val, bool)
    if ts == "dict":
        return isinstance(val, dict)

    if ts.startswith("list[") and ts.endswith("]"):
        if not isinstance(val, list):
            return False
        inner = ts[len("list["):-1].strip()
        return all(_type_matches(x, inner) for x in val)

    # unknown type spec -> don't hard-fail, treat as pass
    return True


def _validate_params_against_spec(
    params: Dict[str, Any],
    input_spec: Dict[str, Any],
    *,
    path_prefix: str,
    strict_extra_params: bool,
) -> Tuple[List[ValidationMessage], List[ValidationMessage]]:
    errors: List[ValidationMessage] = []
    warnings: List[ValidationMessage] = []

    required = input_spec.get("required", []) or []
    optional = input_spec.get("optional", []) or []
    allowed = set(required) | set(optional)

    # required keys
    for k in required:
        if k not in params:
            errors.append(ValidationMessage(
                level="error",
                path=f"{path_prefix}.{k}",
                message=f"Missing required param: '{k}'"
            ))

    # extra keys
    extra = [k for k in params.keys() if k not in allowed]
    if extra:
        msg = f"Unexpected params not declared in input_spec: {extra}"
        if strict_extra_params:
            errors.append(ValidationMessage(level="error", path=path_prefix, message=msg))
        else:
            warnings.append(ValidationMessage(level="warning", path=path_prefix, message=msg))

    # type checks (if provided)
    type_map: Dict[str, str] = input_spec.get("types", {}) or {}
    for k, ts in type_map.items():
        if k in params:
            if not _type_matches(params[k], ts):
                errors.append(ValidationMessage(
                    level="error",
                    path=f"{path_prefix}.{k}",
                    message=f"Type mismatch for '{k}': expected {ts}, got {type(params[k]).__name__}"
                ))

    return errors, warnings


# -----------------------------
# Plan validation (v0)
# -----------------------------

def validate_plan_json(
    plan: Dict[str, Any],
    registry: ToolRegistry,
    *,
    strict_extra_params: bool = True,
    field_catalog: Optional[Dict[str, Any]] = None,
) -> ValidatedPlan:
    errors: List[ValidationMessage] = []
    warnings: List[ValidationMessage] = []

    if not isinstance(plan, dict):
        return ValidatedPlan(
            ok=False,
            plan=None,
            errors=[ValidationMessage("error", "plan", "Plan must be a JSON object (dict).")],
            warnings=[],
        )

    # ---- NEW: allowed field refs (optional) ----
    if field_catalog:
        fc_fields = set(
            (((field_catalog.get("record_field_stats") or {}).get("fields")) or {}).keys()
        )
    else:
        fc_fields = set()

    plan_inputs = set()
    plan_targets = set()
    for x in (plan.get("inputs") or []):
        if isinstance(x, dict) and isinstance(x.get("name"), str):
            plan_inputs.add(x["name"])
    for x in (plan.get("targets") or []):
        if isinstance(x, dict) and isinstance(x.get("name"), str):
            plan_targets.add(x["name"])

    allowed_field_refs = fc_fields | plan_inputs | plan_targets
    # -------------------------------------------

    # minimal top-level sanity
    if "steps" not in plan:
        errors.append(ValidationMessage("error", "plan.steps", "Missing 'steps' in plan."))
        return ValidatedPlan(ok=False, plan=plan, errors=errors, warnings=warnings)

    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        errors.append(ValidationMessage("error", "plan.steps", "'steps' must be a non-empty list."))
        return ValidatedPlan(ok=False, plan=plan, errors=errors, warnings=warnings)

    # validate each step
    for i, step in enumerate(plan["steps"]):
        step_path = f"steps[{i}]"

        if not isinstance(step, dict):
            errors.append(ValidationMessage("error", step_path, "Each step must be an object/dict."))
            continue

        op_name = step.get("op_name")
        params = step.get("params", {})

        if not isinstance(op_name, str) or not op_name:
            errors.append(ValidationMessage("error", f"{step_path}.op_name", "op_name must be a non-empty string."))
            continue

        if not registry.has(op_name):
            errors.append(ValidationMessage(
                "error",
                f"{step_path}.op_name",
                f"op_name '{op_name}' not in registry (not whitelisted)."
            ))
            continue

        if not isinstance(params, dict):
            errors.append(ValidationMessage("error", f"{step_path}.params", "params must be a dict/object."))
            continue

        spec = registry.get(op_name)
        step_errors, step_warnings = _validate_params_against_spec(
            params,
            spec.input_spec or {},
            path_prefix=f"{step_path}.params",
            strict_extra_params=strict_extra_params,
        )
        errors.extend(step_errors)
        warnings.extend(step_warnings)

        # ---- NEW: field reference validation (dataset-agnostic) ----
        if allowed_field_refs:
            ref_keys = []
            if "feature" in params: ref_keys.append(("feature", params.get("feature")))
            if "target" in params: ref_keys.append(("target", params.get("target")))
            if "response" in params: ref_keys.append(("response", params.get("response")))
            if "predictors" in params: ref_keys.append(("predictors", params.get("predictors")))

            for rk, rv in ref_keys:
                if rv is None:
                    continue
                
                # Handle list values (e.g., feature can be a list of field names)
                if isinstance(rv, list):
                    bad = [x for x in rv if isinstance(x, str) and x not in allowed_field_refs]
                    if bad:
                        # Check if these look like time window names rather than field names
                        common_time_windows = {"fixation", "cue", "delay", "baseline", "stimulus", "response"}
                        time_window_matches = [x for x in bad if x.lower() in common_time_windows]
                        
                        if time_window_matches and rk == "predictors":
                            # This looks like a mistake - using time window names as predictors
                            # Suggest using decode_multiclass with firing_rates instead
                            errors.append(ValidationMessage(
                                "error",
                                f"{step_path}.params.{rk}",
                                f"'{time_window_matches}' appear to be time window names, not field names. "
                                f"For decoding with firing rates, use decode_multiclass or decode_svm with: "
                                f"feature='TS', feature_type='firing_rates', and time_window_fields mapping. "
                                f"If you have precomputed rate fields, use their actual field names from the dataset."
                            ))
                        else:
                            errors.append(ValidationMessage(
                                "error",
                                f"{step_path}.params.{rk}",
                                f"Unknown field references in {rk}: {bad}. Use fields from field_catalog or declare them in plan inputs/targets."
                            ))
                    continue
                
                # Handle string values
                if isinstance(rv, str):
                    # For 'feature' parameter in decoding tools, be more lenient
                    # because it might be resolved via plan aliases or field name resolution
                    # Only warn if it's clearly not a valid field reference
                    if rk == "feature":
                        # Check if it's a known feature_type value (these are not field names)
                        known_feature_types = {"firing_rates", "fixed_features", "raw", "precomputed_rates", "existing_rates", "time_series", "direct"}
                        if rv.lower() in known_feature_types:
                            # This is likely a mistake - feature_type should be a separate parameter
                            errors.append(ValidationMessage(
                                "error",
                                f"{step_path}.params.{rk}",
                                f"'{rv}' appears to be a feature_type value, not a field name. Use 'feature' for field name and 'feature_type' parameter for processing type."
                            ))
                        elif rv not in allowed_field_refs:
                            # Not in allowed refs, but might be resolved at runtime
                            # Only warn, don't error, as tools can resolve field names dynamically
                            warnings.append(ValidationMessage(
                                "warning",
                                f"{step_path}.params.{rk}",
                                f"Field reference '{rv}' not found in field_catalog or plan inputs/targets. It may be resolved at runtime or may cause an error."
                            ))
                    else:
                        # For other parameters (target, response, predictors)
                        # For encoding tools, be more lenient as field names might be resolved at runtime
                        # or declared in plan inputs/targets
                        if rv not in allowed_field_refs:
                            # Check if this is an encoding tool (more lenient validation)
                            encoding_tools = {"fit_glm_poisson", "fit_tuning_function", "fit_population_model", 
                                            "compute_rate_prediction_metrics"}
                            is_encoding_tool = op_name in encoding_tools
                            
                            if is_encoding_tool and rk in ("response", "predictors", "condition_field", 
                                                          "actual_field", "predicted_field", "stimulus_field",
                                                          "neuron_response_fields"):
                                # For encoding tools, these fields might be resolved at runtime or declared in plan
                                # Issue a warning instead of error
                                warnings.append(ValidationMessage(
                                    "warning",
                                    f"{step_path}.params.{rk}",
                                    f"Field reference '{rv}' not found in field_catalog or plan inputs/targets. "
                                    f"It may be resolved at runtime, or you may need to declare it in plan inputs/targets. "
                                    f"If the field doesn't exist, the tool will fail at execution time."
                                ))
                            else:
                                # For decoding tools and other cases, be stricter
                                errors.append(ValidationMessage(
                                    "error",
                                    f"{step_path}.params.{rk}",
                                    f"Unknown field reference '{rv}'. Use a field from field_catalog (e.g., RT/TS) or declare it in plan inputs/targets."
                                ))

    ok = (len(errors) == 0)
    return ValidatedPlan(ok=ok, plan=plan if ok else plan, errors=errors, warnings=warnings)