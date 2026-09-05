# core/tools/decoding.py

from __future__ import annotations

# Module-level seed, injected by CodeExecutor before each run.
_RANDOM_SEED: int = 42

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import math

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix, 
    classification_report,
    f1_score,
    roc_auc_score,
)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

from ..state import NeuroGlobalState
from .preprocessing import apply_preprocess_rules

if TYPE_CHECKING:
    # only for type hints; avoids circular import at runtime
    from ..execution_agent import ExecutionContext


# -----------------------------
# Helpers: data extraction
# -----------------------------

def _is_list_like(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int)) and not isinstance(x, bool):
            return float(x)
        # numpy scalar
        if hasattr(x, "item"):
            v = x.item()
            if isinstance(v, (float, int)) and not isinstance(v, bool):
                return float(v)
    except Exception:
        return None
    return None


def _ts_to_fixed_features(ts: Any) -> Optional[np.ndarray]:
    """
    Convert variable-length TS (1D array) -> fixed-length feature vector.
    v0: [mean, std, min, max, length]
    """
    if ts is None:
        return None

    # Some TS may be scalar in your field_catalog
    if isinstance(ts, (float, int)) and not isinstance(ts, bool):
        x = float(ts)
        return np.array([x, 0.0, x, x, 1.0], dtype=np.float32)

    arr = None
    if isinstance(ts, np.ndarray):
        arr = ts
    elif isinstance(ts, list):
        arr = np.array(ts)
    else:
        # unknown
        return None

    if arr.ndim != 1 or arr.size == 0:
        return None

    arr = arr.astype(np.float32, copy=False)
    m = float(np.mean(arr))
    s = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    ln = float(arr.size)

    # guard nan
    if any([math.isnan(m), math.isnan(s), math.isnan(mn), math.isnan(mx)]):
        return None

    return np.array([m, s, mn, mx, ln], dtype=np.float32)


def _compute_firing_rates_from_ts(
    ts: Any,
    time_windows: List[Tuple[str, Optional[float], Optional[float]]],
) -> Optional[Dict[str, float]]:
    """
    Compute firing rates for different time windows from TS (spike times).
    
    Args:
        ts: Spike times array (1D, in seconds relative to trial start)
        time_windows: List of (window_name, start_time, end_time) tuples.
                     start_time/end_time can be None to use trial start/end.
    
    Returns:
        Dict with keys: {window_name}_rate for each window (spikes per second)
        Returns None if insufficient data.
    
    Example:
        time_windows = [
            ("fixation", 0.0, 1.0),
            ("cue", 1.0, 2.0),
            ("delay", 2.0, 3.0)
        ]
    """
    if ts is None:
        return None
    
    # Convert TS to array
    arr = None
    if isinstance(ts, np.ndarray):
        arr = ts
    elif isinstance(ts, list):
        arr = np.array(ts)
    else:
        return None
    
    if arr.ndim != 1 or arr.size == 0:
        return None
    
    # TS contains spike times in seconds
    # Filter out any invalid values
    arr = arr[arr >= 0]  # Only positive times
    if arr.size == 0:
        return None
    
    rates = {}
    
    for window_name, start_t, end_t in time_windows:
        # Determine actual window boundaries
        actual_start = start_t if start_t is not None else 0.0
        actual_end = end_t if end_t is not None else (float(np.max(arr)) if arr.size > 0 else 0.0)
        
        if actual_end <= actual_start:
            rates[f"{window_name}_rate"] = 0.0
            continue
        
        # Count spikes in this window
        window_spikes = np.sum((arr >= actual_start) & (arr < actual_end))
        window_duration = actual_end - actual_start
        
        rate = float(window_spikes / window_duration) if window_duration > 0 else 0.0
        
        # Guard against NaN/Inf
        if math.isnan(rate) or math.isinf(rate):
            rate = 0.0
        
        rates[f"{window_name}_rate"] = rate
    
    return rates


def _extract_trials_generic(
    nd_raw: Dict[str, Any],
    field_catalog: Optional[Dict[str, Any]] = None,
    target_field: Optional[str] = None,
    target_from_column_index: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generic trial extractor that works with different dataset structures.
    
    Strategy:
    1. Look for record-like dicts in the dataset (from field_catalog if available)
    2. If target_field is specified and target_from_column_index=True, infer target from structure
    3. Fallback: traverse common structures (list of lists, nested dicts)
    
    Args:
        nd_raw: Raw dataset dictionary
        field_catalog: Optional field catalog from dataloader
        target_field: Optional target field name to add (e.g., "cue_location")
        target_from_column_index: If True, infer target from column index in matrix-like structures
    
    Returns:
        List of trial dictionaries
    """
    trials: List[Dict[str, Any]] = []

    # [LEGACY] Special-case: "result" matrix format (MATLAB cell array with header row).
    # Not used by current benchmark datasets (AD/PD/WM). Retained for .mat file compatibility.
    if "result" in nd_raw:
        result_obj = nd_raw.get("result")
        result_trials = _extract_trials_from_result_matrix(result_obj)
        if result_trials:
            return result_trials
    
    # Strategy 1: Use field_catalog to find record-like structures
    if field_catalog:
        record_samples = field_catalog.get("record_dict_samples", {})
        example_paths = record_samples.get("example_paths", [])
        
        if example_paths:
            # Try to extract from the most common path pattern
            # Example: "top_level_key[row][col][item]" -> extract from top_level_key
            first_path = example_paths[0] if example_paths else ""
            path_parts = first_path.split("[")[0] if "[" in first_path else first_path
            
            if path_parts in nd_raw:
                obj = nd_raw[path_parts]
                # If we need to infer from column index, use specialized extraction
                if target_field and target_from_column_index:
                    # Try to extract trials with column index from paths
                    trials = _extract_trials_with_column_index(nd_raw, path_parts, example_paths, target_field)
                    if trials:
                        return trials
                    # If specialized extraction failed, fall back to regular traversal
                    # but we'll need to add target_field later if possible
                    trials = _traverse_and_extract_trials(obj, target_field, target_from_column_index)
                    if trials:
                        return trials
                else:
                    trials = _traverse_and_extract_trials(obj, target_field, target_from_column_index)
                    if trials:
                        return trials
    
    # Strategy 2: Try common top-level keys that might contain trials
    common_keys = ["data", "trials", "records", "sessions", "experiments"]
    for key in common_keys:
        if key in nd_raw:
            obj = nd_raw[key]
            trials = _traverse_and_extract_trials(obj, target_field, target_from_column_index)
            if trials:
                return trials
    
    # Strategy 3: Traverse all top-level structures
    for key, value in nd_raw.items():
        if isinstance(value, (list, dict)):
            trials = _traverse_and_extract_trials(value, target_field, target_from_column_index)
            if trials:
                return trials
    
    return trials


def _extract_trials_from_result_matrix(result_obj: Any) -> List[Dict[str, Any]]:
    """
    [LEGACY] Extract trial dicts from MATLAB-style 'result' matrix with header row.
    Not used by current benchmark datasets. Retained for .mat file compatibility.
    """
    try:
        import numpy as np
    except Exception:
        np = None

    if result_obj is None:
        return []

    # Normalize to 2D list-like
    if np is not None and isinstance(result_obj, np.ndarray):
        if result_obj.ndim != 2 or result_obj.shape[0] < 2:
            return []
        mat = result_obj
        n_rows, n_cols = mat.shape

        def _to_str(x: Any) -> str:
            if isinstance(x, str):
                return x
            if np is not None and isinstance(x, np.ndarray):
                if x.size == 0:
                    return ""
                try:
                    return str(np.array(x).squeeze().tolist())
                except Exception:
                    return str(x)
            if isinstance(x, (list, tuple)) and len(x) > 0:
                return str(x[0])
            return str(x)

        headers = [_to_str(mat[0, i]) for i in range(n_cols)]
        trials: List[Dict[str, Any]] = []
        for r in range(1, n_rows):
            rec = {}
            for c, name in enumerate(headers):
                if not name:
                    continue
                rec[name] = mat[r, c]
            if rec:
                trials.append(rec)
        return trials

    # list-of-lists fallback
    if isinstance(result_obj, list) and len(result_obj) >= 2:
        header_row = result_obj[0]
        if not isinstance(header_row, list):
            return []
        headers = [str(h) for h in header_row]
        trials = []
        for row in result_obj[1:]:
            if not isinstance(row, list):
                continue
            rec = {}
            for name, val in zip(headers, row):
                if name:
                    rec[name] = val
            if rec:
                trials.append(rec)
        return trials

    return []


def _extract_trials_with_column_index(
    nd_raw: Dict[str, Any],
    top_key: str,
    example_paths: List[str],
    target_field: str,
) -> List[Dict[str, Any]]:
    """
    Extract trials from matrix-like structure (e.g., data_group[row][col][item])
    and add target_field based on column index.
    
    Args:
        nd_raw: Raw dataset dictionary
        top_key: Top-level key (e.g., "data_group")
        example_paths: List of example paths like "data_group[0][0][0]"
        target_field: Field name to add (e.g., "cue_location")
    
    Returns:
        List of trial dictionaries with target_field added
    """
    trials: List[Dict[str, Any]] = []
    
    if top_key not in nd_raw:
        return trials
    
    obj = nd_raw[top_key]
    
    # Try to traverse matrix structure: obj[row][col][item]
    if isinstance(obj, list):
        for row_idx, row in enumerate(obj):
            if isinstance(row, list):
                for col_idx, col in enumerate(row):
                    if isinstance(col, list):
                        # Column contains list of items
                        for item in col:
                            if isinstance(item, dict) and _looks_like_trial(item):
                                trial = dict(item)
                                trial[target_field] = col_idx + 1  # 1-indexed
                                trials.append(trial)
                    elif isinstance(col, dict) and _looks_like_trial(col):
                        # Column is directly a trial dict
                        trial = dict(col)
                        trial[target_field] = col_idx + 1  # 1-indexed
                        trials.append(trial)
    
    # If we didn't find trials in matrix structure, try parsing paths directly
    if not trials and example_paths:
        # Parse paths to extract column indices
        # Example: "data_group[0][2][5]" -> col_idx = 2
        path_to_trial_map: Dict[str, Dict[str, Any]] = {}
        for path in example_paths:
            try:
                # Parse path like "data_group[0][2][5]"
                parts = path.split("[")
                if len(parts) >= 3:
                    col_part = parts[2]  # "[2]"
                    col_idx = int(col_part.rstrip("]"))
                    # Navigate to the actual trial
                    trial = _navigate_path(nd_raw, path)
                    if trial and isinstance(trial, dict) and _looks_like_trial(trial):
                        trial_copy = dict(trial)
                        trial_copy[target_field] = col_idx + 1  # 1-indexed
                        # Use a unique key to avoid duplicates
                        trial_key = id(trial)  # Use object id as key
                        if trial_key not in path_to_trial_map:
                            path_to_trial_map[trial_key] = trial_copy
            except (ValueError, IndexError, KeyError):
                continue
        
        if path_to_trial_map:
            trials = list(path_to_trial_map.values())
    
    return trials


def _navigate_path(obj: Any, path: str) -> Any:
    """
    Navigate to an object using a path string like "key[0][1][2]".
    Returns None if path is invalid.
    """
    try:
        parts = path.split("[")
        current = obj
        # First part is the key
        if parts:
            key = parts[0]
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        # Remaining parts are indices
        for part in parts[1:]:
            idx_str = part.rstrip("]")
            idx = int(idx_str)
            if isinstance(current, (list, tuple)) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        
        return current
    except (ValueError, IndexError, KeyError, TypeError):
        return None


def _traverse_and_extract_trials(
    obj: Any,
    target_field: Optional[str] = None,
    target_from_column_index: bool = False,
    max_depth: int = 5,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """
    Recursively traverse structure to extract trial-like dictionaries.
    """
    if depth > max_depth:
        return []

    trials: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        # Check if this dict itself looks like a trial (has common trial fields)
        if _looks_like_trial(obj):
            trial = dict(obj)  # Make a copy
            trials.append(trial)
        else:
            # Recurse into nested structures
            for v in obj.values():
                trials.extend(_traverse_and_extract_trials(v, target_field, target_from_column_index, max_depth, depth + 1))
    
    elif isinstance(obj, list):
        # Check if this is a matrix-like structure (list of lists)
        if len(obj) > 0 and isinstance(obj[0], list):
            # Matrix structure: rows x columns
            # If target_from_column_index, each column might represent a different target value
            for row_idx, row in enumerate(obj):
                if isinstance(row, list):
                    for col_idx, col in enumerate(row):
                        if isinstance(col, list):
                            # Column contains a list of items
                            for item in col:
                                if isinstance(item, dict) and _looks_like_trial(item):
                                    trial = dict(item)
                                    if target_field and target_from_column_index:
                                        trial[target_field] = col_idx + 1  # 1-indexed
                                    trials.append(trial)
                        elif isinstance(col, dict) and _looks_like_trial(col):
                            trial = dict(col)
                            if target_field and target_from_column_index:
                                trial[target_field] = col_idx + 1
                            trials.append(trial)
        else:
            # Simple list: recurse into items
            for item in obj:
                trials.extend(_traverse_and_extract_trials(item, target_field, target_from_column_index, max_depth, depth + 1))

    return trials


def _looks_like_trial(d: Dict[str, Any]) -> bool:
    """
    Heuristic: check if a dict looks like a trial record.
    Common indicators: has numeric/array fields, not too nested, has common trial field names.
    """
    if not isinstance(d, dict) or len(d) == 0:
        return False
    
    # Check for common trial field patterns
    common_patterns = ["trial", "time", "rate", "spike", "response", "stimulus", "event"]
    keys_lower = [k.lower() for k in d.keys()]
    
    # If has fields matching common patterns, likely a trial
    if any(any(pattern in key for pattern in common_patterns) for key in keys_lower):
        return True
    
    # If has multiple numeric/array fields, might be a trial
    numeric_count = sum(1 for v in d.values() if isinstance(v, (int, float, np.ndarray, list)))
    if numeric_count >= 3:
        return True
    
    return False


def _resolve_field_name_via_plan_alias(
    state: NeuroGlobalState,
    alias: str,
    *,
    plan_key: str = "validated_plan_json",
) -> str:
    """
    If the plan uses alias names (e.g., "neural_features"), try mapping it to a real field name.
    v0 strategy:
      - if alias is already a real field in field_catalog -> return it
      - else if alias matches a plan input/target name and its path_hint contains a final field key
        (e.g., "...[0]" not helpful; "...RT" not available in your hints)
      - otherwise return alias unchanged
    """
    fc = state.extra.get("field_catalog") or {}
    fc_fields = set((((fc.get("record_field_stats") or {}).get("fields")) or {}).keys())
    if alias in fc_fields:
        return alias

    plan = state.extra.get(plan_key) or {}
    for side in ("inputs", "targets"):
        for item in (plan.get(side) or []):
            if isinstance(item, dict) and item.get("name") == alias:
                # we don't have a reliable parser from path_hint -> key here,
                # so just return alias unchanged for now.
                return alias

    return alias


def _maybe_apply_preprocess(trials: List[Dict[str, Any]], state: NeuroGlobalState) -> List[Dict[str, Any]]:
    rules = state.extra.get("preprocess_rules")
    if not rules:
        return trials
    filtered, summary = apply_preprocess_rules(trials, rules)
    state.extra["preprocess_summary"] = summary
    return filtered


# -----------------------------
# Tool implementation
# -----------------------------

def decode_logistic_regression_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",   # <- Note: this is a string for forward reference
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Binary classification decoding using logistic regression.
    
    Args:
        feature: Field name for neural features (time series or array)
        target: Field name for target variable (will be binarized by median)
        cv_folds: Number of cross-validation folds
        seed: Random seed
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError("neural_dataset not found in state.extra; run dataloader first.")

    feature_name = str(params.get("feature", "")).strip()
    target_name = str(params.get("target", "")).strip()
    cv_folds = int(params.get("cv_folds", 5))
    seed = _RANDOM_SEED  # Framework-managed seed; ignores LLM-provided value

    if not feature_name:
        raise RuntimeError("feature parameter is required")
    if not target_name:
        raise RuntimeError("target parameter is required")

    # allow alias names used by the Reasoning Agent
    feature_name = _resolve_field_name_via_plan_alias(state, feature_name)
    target_name = _resolve_field_name_via_plan_alias(state, target_name)

    # Use generic trial extractor
    field_catalog = state.extra.get("field_catalog")
    trials = _extract_trials_generic(nd.raw, field_catalog=field_catalog)
    trials = _maybe_apply_preprocess(trials, state)
    if len(trials) == 0:
        raise RuntimeError("No trial dicts extracted. Check dataset structure or field names.")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for rec in trials:
        feature_data = rec.get(feature_name, None)
        target_data = rec.get(target_name, None)

        x = _ts_to_fixed_features(feature_data)
        y = _safe_float(target_data)

        if x is None or y is None:
            continue

        X_list.append(x)
        y_list.append(float(y))

    if len(X_list) < max(20, cv_folds * 4):
        raise RuntimeError(f"Too few usable samples after filtering: n={len(X_list)}")

    X = np.stack(X_list, axis=0)  # (n, 5)
    y_cont = np.array(y_list, dtype=np.float32)

    # binarize target by median (high vs low)
    thr = float(np.median(y_cont))
    y = (y_cont > thr).astype(np.int64)

    # guard: need both classes
    if len(np.unique(y)) < 2:
        raise RuntimeError("Target after binarization has <2 classes; cannot run logistic regression.")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_acc: List[float] = []
    fold_sizes: List[int] = []

    # simple, robust defaults
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=None,
        random_state=seed,
    )

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, pred))

        fold_acc.append(acc)
        fold_sizes.append(int(len(test_idx)))

    metrics = {
        "cv_folds": cv_folds,
        "n_samples_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_binarize_threshold_median": thr,
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "accuracy_by_fold": fold_acc,
        "fold_sizes": fold_sizes,
    }

    artifacts = {
        "feature": feature_name,
        "target": target_name,
        "feature_engineering": f"{feature_name} -> [mean,std,min,max,len]",
        "target_processing": f"{target_name} binarized by median threshold",
    }

    return {"metrics": metrics, "artifacts": artifacts}


def decode_multiclass_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic multi-class decoding tool that works with any dataset structure.
    This is a fully dataset-agnostic implementation with no hardcoded field names.
    
    Args:
        feature: Field name(s) for neural features (string or list of strings).
                Examples: "spike_times", ["rate_fixation", "rate_cue", "rate_delay"]
        target: Field name for target variable (must be categorical/integer).
                Examples: "stimulus_location", "choice", "condition"
        feature_type: How to process features:
            - "raw": Use field directly (must be numeric array)
            - "fixed_features": Convert variable-length to fixed features (for time series)
            - "firing_rates": Compute firing rates from time series (requires time_window_fields)
            - "precomputed_rates": Use existing rate fields directly (feature must be a list)
        time_window_fields: Optional dict mapping window names to (start_field, end_field) tuples
                           for firing_rates feature_type. Example:
                           {"window1": ("start_time", "end_time"), "window2": ("start2", "end2")}
        target_from_structure: If True, try to infer target from dataset structure (e.g., column index)
        cv_folds: Number of cross-validation folds (default: 5)
        seed: Random seed for reproducibility
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError("neural_dataset not found in state.extra; run dataloader first.")

    feature = params.get("feature")  # Can be string or list
    target = params.get("target")
    feature_type = str(params.get("feature_type", "fixed_features")).strip().lower()
    time_window_fields = params.get("time_window_fields", {})  # e.g., {"fixation": ("Cue_onT", None), ...}
    target_from_structure = params.get("target_from_structure", False)
    cv_folds = int(params.get("cv_folds", 5))
    seed = _RANDOM_SEED  # Framework-managed seed; ignores LLM-provided value

    # Use auto-detected fields as defaults if not provided
    from ..dataset_detector import get_field_mapping
    field_mapping = get_field_mapping(state)
    
    # If feature not provided, use detected primary neural activity field
    if not feature and field_mapping:
        feature = field_mapping.primary_neural_activity
    
    # If target not provided, use detected primary target field
    if not target and field_mapping and field_mapping.primary_target:
        target = field_mapping.primary_target

    # Resolve field names via plan aliases
    if isinstance(feature, str):
        feature = _resolve_field_name_via_plan_alias(state, feature)
    if target:
        target = _resolve_field_name_via_plan_alias(state, target)

    # Extract trials
    field_catalog = state.extra.get("field_catalog")
    
    # Auto-detect if target needs structure inference using field mapping
    target_needs_inference = False
    if target:
        target_lower = target.lower()
        # Check if target is in detected target fields (might need inference)
        if field_mapping:
            # If target is in detected target fields but not in raw data, needs inference
            if target in field_mapping.target_fields:
                target_needs_inference = True
    
    # If target_from_structure is explicitly set, use it; otherwise auto-detect
    if target_from_structure or target_needs_inference:
        # Use target name as the field name for inferred value
        target_field_for_extraction = target if target else params.get("default_target_field", "_inferred_target")
        if not target:
            # If no target name but target_from_structure=True, use default
            target_field_for_extraction = params.get("default_target_field", "_inferred_target")
    else:
        target_field_for_extraction = None
    
    trials = _extract_trials_generic(
        nd.raw,
        field_catalog=field_catalog,
        target_field=target_field_for_extraction if (target_from_structure or target_needs_inference) else None,
        target_from_column_index=(target_from_structure or target_needs_inference),
    )
    trials = _maybe_apply_preprocess(trials, state)
    
    if len(trials) == 0:
        raise RuntimeError("No trial dicts extracted. Check dataset structure or field names.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # Track filtering reasons for debugging
    missing_target_count = 0
    invalid_target_count = 0
    missing_feature_count = 0
    invalid_feature_count = 0
    total_trials = len(trials)

    # Extract features and targets
    for rec in trials:
        # Extract target
        if target_from_structure or target_needs_inference:
            # Target was inferred from structure and added to trial dict
            # Try target name first, then common inferred field names
            if target:
                y_val = rec.get(target) or rec.get("_inferred_target") or rec.get("target")
            else:
                default_target_field = params.get("default_target_field", "_inferred_target")
                y_val = rec.get(default_target_field) or rec.get("_inferred_target") or rec.get("target")
        elif target:
            y_val = rec.get(target)
        else:
            # Try to find any common target field names
            for common_name in ["target", "label", "class", "category", "condition"]:
                if common_name in rec:
                    y_val = rec[common_name]
                    break
            else:
                y_val = None
        
        if y_val is None:
            missing_target_count += 1
            continue
        
        # Convert target to integer class label
        try:
            y_int = int(y_val)
            if y_int < 0:  # Skip negative labels
                invalid_target_count += 1
                continue
        except (ValueError, TypeError):
            # Try to convert string to int
            try:
                y_int = int(float(str(y_val)))
            except (ValueError, TypeError):
                invalid_target_count += 1
                continue

        # Extract features
        x = None
        
        if feature_type == "fixed_features" or feature_type == "time_series":
            # Convert variable-length time series to fixed features
            if isinstance(feature, str):
                ts = rec.get(feature)
                if ts is None:
                    missing_feature_count += 1
                    continue
                x = _ts_to_fixed_features(ts)
            elif isinstance(feature, list):
                # Multiple features: concatenate
                feat_list = []
                for fname in feature:
                    ts = rec.get(fname)
                    if ts is None:
                        continue
                    feat = _ts_to_fixed_features(ts)
                    if feat is not None:
                        feat_list.append(feat)
                if feat_list:
                    x = np.concatenate(feat_list)
                else:
                    missing_feature_count += 1
                    continue
        
        elif feature_type == "firing_rates":
            # Compute firing rates from time series using time windows
            if not isinstance(feature, str):
                raise RuntimeError("firing_rates feature_type requires single feature field name")
            if not time_window_fields:
                raise RuntimeError("firing_rates requires time_window_fields parameter")
            
            ts = rec.get(feature)
            if ts is None:
                missing_feature_count += 1
                continue
            
            # Build time windows from field values
            time_windows = []
            for window_name, (start_field, end_field) in time_window_fields.items():
                start_t = _safe_float(rec.get(start_field)) if start_field else None
                end_t = _safe_float(rec.get(end_field)) if end_field else None
                time_windows.append((window_name, start_t, end_t))
            
            rates_dict = _compute_firing_rates_from_ts(ts, time_windows)
            if rates_dict:
                # Extract rates in order
                rate_values = [rates_dict.get(f"{wn}_rate", 0.0) for wn, _, _ in time_windows]
                x = np.array(rate_values, dtype=np.float32)
            else:
                invalid_feature_count += 1
                continue
        
        elif feature_type == "raw" or feature_type == "direct":
            # Use field directly (must be numeric array)
            if isinstance(feature, str):
                feat_data = rec.get(feature)
                if feat_data is None:
                    missing_feature_count += 1
                    continue
                if isinstance(feat_data, np.ndarray):
                    if feat_data.ndim == 1:
                        x = feat_data.astype(np.float32)
                    else:
                        x = feat_data.flatten().astype(np.float32)
                elif isinstance(feat_data, list):
                    x = np.array(feat_data, dtype=np.float32)
                elif isinstance(feat_data, (int, float)):
                    x = np.array([float(feat_data)], dtype=np.float32)
            elif isinstance(feature, list):
                # Multiple features: concatenate
                feat_list = []
                for fname in feature:
                    feat_data = rec.get(fname)
                    if feat_data is None:
                        continue
                    if isinstance(feat_data, np.ndarray):
                        feat_list.append(feat_data.flatten().astype(np.float32))
                    elif isinstance(feat_data, (int, float)):
                        feat_list.append(np.array([float(feat_data)], dtype=np.float32))
                if feat_list:
                    x = np.concatenate(feat_list)
                else:
                    missing_feature_count += 1
                    continue
        
        elif feature_type == "precomputed_rates" or feature_type == "existing_rates":
            # Use existing rate fields directly
            if isinstance(feature, list):
                rate_values = []
                for fname in feature:
                    val = _safe_float(rec.get(fname, 0.0)) or 0.0
                    rate_values.append(val)
                x = np.array(rate_values, dtype=np.float32)
            else:
                raise RuntimeError("precomputed_rates requires feature to be a list of field names")
        
        else:
            raise RuntimeError(f"Unknown feature_type: {feature_type}. Use 'fixed_features', 'firing_rates', 'raw', or 'precomputed_rates'.")

        if x is None or x.size == 0:
            continue

        X_list.append(x)
        y_list.append(y_int)

    if len(X_list) < max(20, cv_folds * 4):
        # Provide detailed error message
        error_parts = [
            f"Too few usable samples after filtering: n={len(X_list)} (out of {total_trials} trials)"
        ]
        
        if missing_target_count > 0:
            error_parts.append(f"Missing target field '{target or '(inferred)'}' in {missing_target_count} trials")
            if target_from_structure:
                error_parts.append(f"  (Note: target_from_structure=True, but target was not inferred)")
            elif not target:
                error_parts.append(f"  (Note: target parameter not provided, and no common target field found)")
        
        if invalid_target_count > 0:
            error_parts.append(f"Invalid target values (cannot convert to int or negative) in {invalid_target_count} trials")
        
        if missing_feature_count > 0:
            error_parts.append(f"Missing feature field '{feature}' in {missing_feature_count} trials")
        
        if invalid_feature_count > 0:
            error_parts.append(f"Invalid feature values (empty or cannot process) in {invalid_feature_count} trials")
        
        # Show available fields from first trial
        if len(trials) > 0:
            first_trial = trials[0]
            available_fields = list(first_trial.keys())
            if available_fields:
                error_parts.append(f"Available fields in first trial: {available_fields[:20]}")
        
        raise RuntimeError("\n".join(error_parts))

    X = np.stack(X_list, axis=0)  # (n, features)
    y = np.array(y_list, dtype=np.int64)  # (n,)

    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        raise RuntimeError(f"Need at least 2 classes. Found: {unique_classes.tolist()}")

    # Use StratifiedKFold for multi-class
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_acc: List[float] = []
    fold_sizes: List[int] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    # Multi-class logistic regression
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        # multi_class="multinomial" is default in newer sklearn versions
        n_jobs=None,
        random_state=seed,
    )

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, pred))

        fold_acc.append(acc)
        fold_sizes.append(int(len(test_idx)))
        all_y_true.extend(yte.tolist())
        all_y_pred.extend(pred.tolist())

    # Compute confusion matrix
    class_labels = sorted(unique_classes.tolist())
    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)  # Normalized by row

    # Per-class accuracy
    per_class_acc = {}
    for class_label in class_labels:
        class_mask = np.array(all_y_true) == class_label
        if np.sum(class_mask) > 0:
            class_acc = float(np.mean(np.array(all_y_pred)[class_mask] == class_label))
            per_class_acc[f"class_{class_label}"] = class_acc

    metrics = {
        "cv_folds": cv_folds,
        "n_samples_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_labels": class_labels,
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "accuracy_by_fold": fold_acc,
        "fold_sizes": fold_sizes,
        "per_class_accuracy": per_class_acc,
        "chance_level": 1.0 / n_classes,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
    }

    artifacts = {
        "feature": feature,
        "target": target,
        "feature_type": feature_type,
        "classifier": "LogisticRegression (multinomial)",
    }

    # Automatically generate confusion matrix visualization
    figure_path = None
    if HAS_PLOTTING and ctx:
        try:
            # Generate unique figure path using feature and target info
            feature_str = str(feature).replace("/", "_").replace("\\", "_")[:20] if feature else "unknown"
            # Use inferred target name if target is None (from target_from_structure)
            if target is None or target == "":
                # Try to infer a meaningful name from the context
                if target_from_structure:
                    target_str = "inferred_target"
                else:
                    target_str = "unknown"
            else:
                target_str = str(target).replace("/", "_").replace("\\", "_")[:20]
            # Add step identifier to ensure uniqueness when multiple decode steps exist
            # Use op_name from params if available, or generate from feature_type
            op_identifier = params.get("_op_name", "decode_multiclass")
            figure_filename = f"confusion_matrix_{op_identifier}_{feature_str}_{target_str}.png"
            figure_path = ctx.outputs_root / figure_filename
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create normalized confusion matrix for visualization
            cm_normalized_viz = cm_normalized
            
            # Create figure
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized_viz,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar_kws={'label': 'Normalized'},
                xticklabels=[f"Class {l}" for l in class_labels],
                yticklabels=[f"Class {l}" for l in class_labels],
            )
            plt.title(f"Confusion Matrix (Accuracy: {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f})")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            # If plotting fails, continue without visualization
            print(f"Warning: Failed to generate confusion matrix visualization: {e}")

    return {"metrics": metrics, "artifacts": artifacts}


def decode_svm_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic multi-class decoding tool using SVM (Support Vector Machine).
    Works with any dataset structure, similar to decode_multiclass_tool but uses SVM.
    This is a fully dataset-agnostic implementation with no hardcoded field names.
    
    Args:
        feature: Field name(s) for neural features (string or list of strings).
                Examples: "spike_times", ["rate_fixation", "rate_cue", "rate_delay"]
        target: Field name for target variable (must be categorical/integer).
                Examples: "stimulus_location", "choice", "condition"
        feature_type: How to process features:
            - "raw": Use field directly (must be numeric array)
            - "fixed_features": Convert variable-length to fixed features (for time series)
            - "firing_rates": Compute firing rates from time series (requires time_window_fields)
            - "precomputed_rates": Use existing rate fields directly (feature must be a list)
        time_window_fields: Optional dict mapping window names to (start_field, end_field) tuples
                           for firing_rates feature_type. Example:
                           {"window1": ("start_time", "end_time"), "window2": ("start2", "end2")}
        target_from_structure: If True, try to infer target from dataset structure (e.g., column index)
        cv_folds: Number of cross-validation folds (default: 5)
        kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid'), default 'rbf'
        C: SVM regularization parameter, default 1.0
        standardize: Whether to standardize features before training, default True
        seed: Random seed for reproducibility
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError("neural_dataset not found in state.extra; run dataloader first.")

    feature = params.get("feature")  # Can be string or list
    target = params.get("target")
    feature_type = str(params.get("feature_type", "fixed_features")).strip().lower()
    time_window_fields = params.get("time_window_fields", {})
    target_from_structure = params.get("target_from_structure", False)
    cv_folds = int(params.get("cv_folds", 5))
    kernel = str(params.get("kernel", "rbf")).strip().lower()
    C = float(params.get("C", 1.0))
    standardize = params.get("standardize", True) if params.get("standardize") is not None else True
    seed = _RANDOM_SEED  # Framework-managed seed; ignores LLM-provided value

    # Resolve field names via plan aliases
    if isinstance(feature, str):
        feature = _resolve_field_name_via_plan_alias(state, feature)
    if target:
        target = _resolve_field_name_via_plan_alias(state, target)

    # Extract trials (reuse logic from decode_multiclass_tool)
    field_catalog = state.extra.get("field_catalog")
    target_field_for_extraction = target
    if target_from_structure and not target:
        default_target_field = params.get("default_target_field", "_inferred_target")
        target_field_for_extraction = default_target_field
    
    trials = _extract_trials_generic(
        nd.raw,
        field_catalog=field_catalog,
        target_field=target_field_for_extraction if target_from_structure else None,
        target_from_column_index=target_from_structure,
    )
    trials = _maybe_apply_preprocess(trials, state)
    
    if len(trials) == 0:
        raise RuntimeError("No trial dicts extracted. Check dataset structure or field names.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    # Extract features and targets (reuse logic from decode_multiclass_tool)
    for rec in trials:
        # Extract target
        if target_from_structure and not target:
            default_target_field = params.get("default_target_field", "_inferred_target")
            y_val = rec.get(default_target_field) or rec.get("_inferred_target") or rec.get("target")
        elif target:
            y_val = rec.get(target)
        else:
            for common_name in ["target", "label", "class", "category", "condition"]:
                if common_name in rec:
                    y_val = rec[common_name]
                    break
            else:
                y_val = None
        
        if y_val is None:
            continue
        
        try:
            y_int = int(y_val)
            if y_int < 0:
                continue
        except (ValueError, TypeError):
            try:
                y_int = int(float(str(y_val)))
            except (ValueError, TypeError):
                continue

        # Extract features (reuse logic from decode_multiclass_tool)
        x = None
        
        if feature_type == "fixed_features" or feature_type == "time_series":
            if isinstance(feature, str):
                ts = rec.get(feature)
                x = _ts_to_fixed_features(ts)
            elif isinstance(feature, list):
                feat_list = []
                for fname in feature:
                    ts = rec.get(fname)
                    feat = _ts_to_fixed_features(ts)
                    if feat is not None:
                        feat_list.append(feat)
                if feat_list:
                    x = np.concatenate(feat_list)
        
        elif feature_type == "firing_rates":
            if not isinstance(feature, str):
                raise RuntimeError("firing_rates feature_type requires single feature field name")
            if not time_window_fields:
                raise RuntimeError("firing_rates requires time_window_fields parameter")
            
            ts = rec.get(feature)
            if ts is None:
                continue
            
            time_windows = []
            for window_name, (start_field, end_field) in time_window_fields.items():
                start_t = _safe_float(rec.get(start_field)) if start_field else None
                end_t = _safe_float(rec.get(end_field)) if end_field else None
                time_windows.append((window_name, start_t, end_t))
            
            rates_dict = _compute_firing_rates_from_ts(ts, time_windows)
            if rates_dict:
                rate_values = [rates_dict.get(f"{wn}_rate", 0.0) for wn, _, _ in time_windows]
                x = np.array(rate_values, dtype=np.float32)
        
        elif feature_type == "raw" or feature_type == "direct":
            if isinstance(feature, str):
                feat_data = rec.get(feature)
                if isinstance(feat_data, np.ndarray):
                    if feat_data.ndim == 1:
                        x = feat_data.astype(np.float32)
                    else:
                        x = feat_data.flatten().astype(np.float32)
                elif isinstance(feat_data, list):
                    x = np.array(feat_data, dtype=np.float32)
                elif isinstance(feat_data, (int, float)):
                    x = np.array([float(feat_data)], dtype=np.float32)
            elif isinstance(feature, list):
                feat_list = []
                for fname in feature:
                    feat_data = rec.get(fname)
                    if isinstance(feat_data, np.ndarray):
                        feat_list.append(feat_data.flatten().astype(np.float32))
                    elif isinstance(feat_data, (int, float)):
                        feat_list.append(np.array([float(feat_data)], dtype=np.float32))
                if feat_list:
                    x = np.concatenate(feat_list)
        
        elif feature_type == "precomputed_rates" or feature_type == "existing_rates":
            if isinstance(feature, list):
                rate_values = []
                for fname in feature:
                    val = _safe_float(rec.get(fname, 0.0)) or 0.0
                    rate_values.append(val)
                x = np.array(rate_values, dtype=np.float32)
            else:
                raise RuntimeError("precomputed_rates requires feature to be a list of field names")
        
        else:
            raise RuntimeError(f"Unknown feature_type: {feature_type}. Use 'fixed_features', 'firing_rates', 'raw', or 'precomputed_rates'.")

        if x is None or x.size == 0:
            continue

        X_list.append(x)
        y_list.append(y_int)

    if len(X_list) < max(20, cv_folds * 4):
        raise RuntimeError(f"Too few usable samples after filtering: n={len(X_list)}")

    X = np.stack(X_list, axis=0)  # (n, features)
    y = np.array(y_list, dtype=np.int64)  # (n,)

    # Get unique classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    if n_classes < 2:
        raise RuntimeError(f"Need at least 2 classes. Found: {unique_classes.tolist()}")

    # Use StratifiedKFold for multi-class
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_acc: List[float] = []
    fold_sizes: List[int] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    # SVM classifier
    clf = SVC(
        kernel=kernel,
        C=C,
        random_state=seed,
        probability=False,  # Faster, we only need predictions
    )

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Standardize per fold: fit on train only to prevent leakage
        if standardize:
            fold_scaler = StandardScaler()
            Xtr = fold_scaler.fit_transform(Xtr)
            Xte = fold_scaler.transform(Xte)

        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float(accuracy_score(yte, pred))

        fold_acc.append(acc)
        fold_sizes.append(int(len(test_idx)))
        all_y_true.extend(yte.tolist())
        all_y_pred.extend(pred.tolist())

    # Compute confusion matrix
    class_labels = sorted(unique_classes.tolist())
    cm = confusion_matrix(all_y_true, all_y_pred, labels=class_labels)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Per-class accuracy
    per_class_acc = {}
    for class_label in class_labels:
        class_mask = np.array(all_y_true) == class_label
        if np.sum(class_mask) > 0:
            class_acc = float(np.mean(np.array(all_y_pred)[class_mask] == class_label))
            per_class_acc[f"class_{class_label}"] = class_acc

    metrics = {
        "cv_folds": cv_folds,
        "n_samples_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_labels": class_labels,
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "accuracy_by_fold": fold_acc,
        "fold_sizes": fold_sizes,
        "per_class_accuracy": per_class_acc,
        "chance_level": 1.0 / n_classes,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
    }

    artifacts = {
        "feature": feature,
        "target": target,
        "feature_type": feature_type,
        "classifier": f"SVM (kernel={kernel}, C={C})",
        "standardized": standardize,
    }

    # Automatically generate confusion matrix visualization
    figure_path = None
    if HAS_PLOTTING and ctx:
        try:
            # Generate unique figure path using feature and target info
            feature_str = str(feature).replace("/", "_").replace("\\", "_")[:20] if feature else "unknown"
            # Use inferred target name if target is None (from target_from_structure)
            if target is None or target == "":
                # Try to infer a meaningful name from the context
                if target_from_structure:
                    target_str = "inferred_target"
                else:
                    target_str = "unknown"
            else:
                target_str = str(target).replace("/", "_").replace("\\", "_")[:20]
            # Add step identifier to ensure uniqueness when multiple decode steps exist
            # Use op_name from params if available, or generate from feature_type
            op_identifier = params.get("_op_name", "decode_svm")
            figure_filename = f"confusion_matrix_{op_identifier}_{feature_str}_{target_str}.png"
            figure_path = ctx.outputs_root / figure_filename
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create normalized confusion matrix for visualization
            cm_normalized_viz = cm_normalized
            
            # Create figure
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized_viz,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar_kws={'label': 'Normalized'},
                xticklabels=[f"Class {l}" for l in class_labels],
                yticklabels=[f"Class {l}" for l in class_labels],
            )
            plt.title(f"Confusion Matrix (Accuracy: {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f})")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            # If plotting fails, continue without visualization
            print(f"Warning: Failed to generate confusion matrix visualization: {e}")

    return {"metrics": metrics, "artifacts": artifacts}


def decode_odr_cue_location_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience wrapper for location decoding tasks (e.g., ODR cue location).
    This wrapper uses auto-detected field mappings instead of hardcoded field names.
    Falls back to generic decode_multiclass_tool with auto-detected defaults.
    """
    # Get auto-detected field mapping
    from ..dataset_detector import get_field_mapping
    field_mapping = get_field_mapping(state)
    
    feature_type = str(params.get("feature_type", "TS")).strip().lower()
    
    # Use auto-detected fields, with fallbacks
    neural_field = field_mapping.primary_neural_activity if field_mapping else "TS"
    target_field = field_mapping.primary_target if field_mapping else None
    
    # Map feature types to generic parameters
    if feature_type == "ts" or feature_type == "time_series":
        generic_params = {
            "feature": neural_field,
            "target": target_field,  # Use detected target or None (will be inferred)
            "feature_type": "fixed_features",
            "target_from_structure": target_field is None,  # Infer if not detected
            "cv_folds": params.get("cv_folds", 5),
            "seed": _RANDOM_SEED,
        }
    elif feature_type == "firing_rates" or feature_type == "computed_rates":
        # Build time_window_fields from detected time events
        time_window_fields = {}
        if field_mapping and field_mapping.time_event_fields:
            events = field_mapping.time_event_fields
            # Try to build common time windows from detected events
            cue_on = events.get("cue_on", [None])[0] if events.get("cue_on") else None
            fix_off = events.get("fix_off", [None])[0] if events.get("fix_off") else None
            saccade_on = events.get("saccade_on", [None])[0] if events.get("saccade_on") else None
            
            if cue_on and fix_off:
                time_window_fields["fixation"] = (None, cue_on)
                time_window_fields["cue"] = (cue_on, fix_off)
                if saccade_on:
                    time_window_fields["delay"] = (fix_off, saccade_on)
            elif cue_on:
                # Fallback: just use cue_on if available
                time_window_fields["pre_cue"] = (None, cue_on)
                time_window_fields["post_cue"] = (cue_on, None)
        
        # If no time windows detected, raise error
        if not time_window_fields:
            raise RuntimeError(
                "Could not auto-detect time event fields for computing firing rates. "
                "Please provide time_window_fields manually or use precomputed_rates."
            )
        
        generic_params = {
            "feature": neural_field,
            "target": target_field,
            "feature_type": "firing_rates",
            "time_window_fields": time_window_fields,
            "target_from_structure": target_field is None,
            "cv_folds": params.get("cv_folds", 5),
            "seed": _RANDOM_SEED,
        }
    elif feature_type == "precomputed_rates" or feature_type == "existing_rates":
        # Use detected firing rate fields
        if field_mapping and field_mapping.firing_rate_fields:
            # Collect all detected firing rate fields
            rate_fields = []
            for period_fields in field_mapping.firing_rate_fields.values():
                rate_fields.extend(period_fields)
            
            if not rate_fields:
                raise RuntimeError(
                    "Could not auto-detect precomputed firing rate fields. "
                    "Please specify feature field names manually."
                )
        else:
            # Fallback to common names (for backward compatibility)
            # Fallback: try to find any rate-like fields
            rate_fields = []
            # Get field catalog to search for rate fields
            field_catalog = state.extra.get("field_catalog", {})
            record_stats = field_catalog.get("record_field_stats", {})
            catalog_fields = record_stats.get("fields", {})
            
            for field_name in catalog_fields.keys():
                if "rate" in field_name.lower() or "rate" in _normalize_field_name(field_name):
                    rate_fields.append(field_name)
            if not rate_fields:
                raise RuntimeError(
                    "Could not auto-detect precomputed firing rate fields. "
                    "Please specify feature field names manually or use firing_rates feature_type."
                )
        
        generic_params = {
            "feature": rate_fields,
            "target": target_field,
            "feature_type": "precomputed_rates",
            "target_from_structure": target_field is None,
            "cv_folds": params.get("cv_folds", 5),
            "seed": _RANDOM_SEED,
        }
    else:
        raise RuntimeError(f"Unknown feature_type: {feature_type}. Use 'TS', 'firing_rates', or 'precomputed_rates'.")
    
    # Add op_name identifier to params for unique filename generation
    generic_params["_op_name"] = "decode_odr_cue_location"
    
    # Call generic tool
    return decode_multiclass_tool(state=state, ctx=ctx, params=generic_params)


# -----------------------------
# describe_trial_columns - List trial columns and unique values
# -----------------------------

def _to_python_native(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    return val

def describe_trial_columns_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    List trial/interval columns with their unique values and counts.
    Useful for identifying label columns (stimulus side, response side, etc.)
    
    Args:
        columns: Optional list of specific columns to describe (default: all)
        max_unique: Maximum number of unique values to show per column (default: 20)
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError("neural_dataset not found in state.extra; run dataloader first.")
    
    columns = params.get("columns")  # Optional filter
    max_unique = int(params.get("max_unique", 20))
    
    raw = nd.raw
    
    # Collect all trial-related fields
    trial_columns = {}
    
    # Common trial column patterns
    trial_patterns = [
        "trial_", "type", "response", "stim", "cue", "choice", "correct",
        "side", "left", "right", "condition", "label", "category"
    ]
    
    for key, value in raw.items():
        # Check if it looks like a trial column (list/array of values, one per trial)
        if isinstance(value, (list, np.ndarray)):
            try:
                # Convert to numpy array, handling inhomogeneous shapes
                if isinstance(value, list):
                    # Check if it's a simple 1D list (not nested with different lengths)
                    if len(value) > 0 and isinstance(value[0], (list, np.ndarray)):
                        # Skip nested structures with inhomogeneous shapes
                        continue
                    arr = np.array(value, dtype=object)  # Use object dtype to avoid shape issues
                else:
                    arr = value
                
                # Check if it's 1D and has reasonable length for trials
                if arr.ndim == 1 and 1 < len(arr) < 100000:
                    # Check if values are categorical-ish (strings, integers, or few unique values)
                    try:
                        unique_vals = np.unique(arr)
                        n_unique = len(unique_vals)
                        
                        # Include if:
                        # 1. Explicitly requested
                        # 2. Has string values
                        # 3. Has few unique values (likely categorical)
                        # 4. Matches trial patterns
                        is_string = arr.dtype.kind in ('U', 'S', 'O')
                        is_categorical = n_unique <= 50
                        matches_pattern = any(p in key.lower() for p in trial_patterns)
                        
                        if columns:
                            include = key in columns or key.lower() in [c.lower() for c in columns]
                        else:
                            include = is_string or is_categorical or matches_pattern
                        
                        if include:
                            # Get value counts
                            unique, counts = np.unique(arr, return_counts=True)
                            
                            # Convert to readable format with native Python types
                            if len(unique) <= max_unique:
                                value_counts = {str(_to_python_native(v)): int(c) for v, c in zip(unique, counts)}
                            else:
                                # Show top values only
                                top_idx = np.argsort(counts)[::-1][:max_unique]
                                value_counts = {str(_to_python_native(unique[i])): int(counts[i]) for i in top_idx}
                                value_counts["..."] = f"({len(unique) - max_unique} more unique values)"
                            
                            # Convert unique values to Python native types
                            unique_list = [_to_python_native(v) for v in unique[:max_unique]] if len(unique) <= max_unique else [_to_python_native(v) for v in unique[:10]] + ["..."]
                            
                            trial_columns[key] = {
                                "n_values": int(len(arr)),
                                "n_unique": int(n_unique),
                                "dtype": str(arr.dtype),
                                "unique_values": unique_list,
                                "value_counts": value_counts,
                            }
                    except Exception:
                        pass
            except Exception:
                # Skip any problematic fields
                pass
    
    # Build summary text — pure data description, no heuristic classification
    summary_lines = [
        f"Found {len(trial_columns)} trial-related columns:",
        ""
    ]

    for col, info in sorted(trial_columns.items()):
        summary_lines.append(f"📊 {col}:")
        summary_lines.append(f"   - N values: {info['n_values']}, N unique: {info['n_unique']}")
        summary_lines.append(f"   - dtype: {info['dtype']}")
        summary_lines.append(f"   - Values: {info['value_counts']}")
        summary_lines.append("")

    metrics = {
        "n_columns": len(trial_columns),
        "columns": trial_columns,
    }

    artifacts = {
        "summary_text": "\n".join(summary_lines),
    }
    
    return {"metrics": metrics, "artifacts": artifacts}


# -----------------------------
# Decoding Diagnostics for Controller
# -----------------------------

def _generate_decoding_diagnostics(
    *,
    raw: Dict[str, Any],
    time_window: Any,
    test_accuracy: float,
    test_balanced_accuracy: float,
    n_classes: int,
) -> Dict[str, Any]:
    """
    Generate factual diagnostic information for Controller.
    
    This function provides RAW DATA ONLY - no interpretations, warnings, or recommendations.
    The LLM must independently analyze this data to identify any issues.
    
    Returns:
        Dict with factual data info only
    """
    diagnostics = {
        "data_info": {},
    }
    
    # Get data_summary if available
    data_summary = raw.get("_data_summary", {})
    spike_info = data_summary.get("spike_times_info", {})
    time_fields = data_summary.get("time_fields", {})
    
    # Store factual data info for Controller - NO interpretations
    if spike_info:
        diagnostics["data_info"]["spike_times_range"] = spike_info.get("range", [])
        diagnostics["data_info"]["total_spikes"] = spike_info.get("total_spikes", 0)
    
    # Store the time window that was used
    if isinstance(time_window, (tuple, list)) and len(time_window) >= 2:
        diagnostics["data_info"]["used_time_window"] = [time_window[0], time_window[1]]
    elif isinstance(time_window, dict):
        diagnostics["data_info"]["used_time_window"] = time_window
    
    # Store available time fields - factual info only
    if time_fields:
        diagnostics["data_info"]["available_time_fields"] = {
            k: {"range": v.get("range"), "absolute_range": v.get("absolute_range")}
            for k, v in time_fields.items()
        }
    
    # Store accuracy and chance level as facts
    chance_level = 1.0 / n_classes if n_classes > 0 else 0.5
    diagnostics["data_info"]["n_classes"] = n_classes
    diagnostics["data_info"]["chance_level"] = chance_level
    diagnostics["data_info"]["test_accuracy"] = test_accuracy
    diagnostics["data_info"]["test_balanced_accuracy"] = test_balanced_accuracy
    
    return diagnostics


# -----------------------------
# decode_with_holdout - Train/val/test split with full metrics
# -----------------------------

def decode_with_holdout_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decode with train/val/test split and comprehensive metrics.
    
    Features:
    - Stratified train/val/test split (default 70/15/15)
    - Reports: accuracy, balanced accuracy, macro/micro F1, AUROC
    - Confusion matrix
    - 5-fold cross-validation with mean±std
    - Model type and hyperparameters
    
    Args:
        feature: Field name for spike times or neural data
        target: Field name for labels (e.g., "trial_type")
        target_values: Dict mapping target values to binary labels (e.g., {"class_A": 0, "class_B": 1})
        filter_field: Optional field to filter trials (e.g., "trial_quality")
        filter_values: Values to keep for filter_field (e.g., ["valid", "included"])
        time_window: Tuple (start, end) in seconds for spike count window
        train_ratio: Training set ratio (default 0.70)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        cv_folds: Number of cross-validation folds (default 5)
        seed: Random seed (default 42)
        classifier: "logistic_regression" or "svm" (default: logistic_regression)
        C: Regularization parameter (default 1.0)
        penalty: Regularization type for logistic regression (default "l2")
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError("neural_dataset not found in state.extra; run dataloader first.")
    
    # Extract parameters - feature and target will be auto-inferred if not provided
    feature = params.get("feature")  # Auto-infer if None
    target = params.get("target")  # Auto-infer if None
    target_values = params.get("target_values", None)  # Auto-infer if None
    filter_field = params.get("filter_field", None)  # No filter by default
    filter_values = params.get("filter_values", None)  # No filter by default
    time_window = params.get("time_window", (0.0, 1.0))  # Default 1s window
    trial_filter = params.get("trial_filter", {})  # Optional trial filter dict
    unit_filter = params.get("unit_filter", {})  # Optional unit filter dict
    train_ratio = float(params.get("train_ratio", 0.70))
    val_ratio = float(params.get("val_ratio", 0.15))
    test_ratio = float(params.get("test_ratio", 0.15))
    cv_folds = int(params.get("cv_folds", 5))
    seed = _RANDOM_SEED  # Framework-managed seed; ignores LLM-provided value
    classifier_type = str(params.get("classifier", "logistic_regression")).lower()
    C = float(params.get("C", 1.0))
    penalty = str(params.get("penalty", "l2"))
    
    raw = nd.raw
    
    # Auto-infer feature field if not provided
    if feature is None:
        # Try common spike data field names
        for candidate in ["spike_times_by_unit", "spike_times", "spikes", "TS", "neural_data"]:
            if candidate in raw:
                feature = candidate
                print(f"Auto-inferred feature field: {feature}")
                break
        if feature is None:
            available_fields = [k for k in raw.keys() if "spike" in k.lower() or "neural" in k.lower() or "ts" in k.lower()]
            raise RuntimeError(f"Could not auto-infer feature field. Available candidates: {available_fields[:10]}")
    
    # Auto-infer target field if not provided
    if target is None:
        # Try common target field names
        for candidate in ["trial_type", "condition", "stimulus", "choice", "label", "target", "category"]:
            if candidate in raw:
                target = candidate
                print(f"Auto-inferred target field: {target}")
                break
        if target is None:
            available_fields = [k for k in raw.keys() if any(x in k.lower() for x in ["type", "condition", "stim", "choice", "label"])]
            raise RuntimeError(f"Could not auto-infer target field. Available candidates: {available_fields[:10]}")
    
    # Get trial data
    target_data = raw.get(target)
    if target_data is None:
        raise RuntimeError(f"Target field '{target}' not found in dataset")
    target_arr = np.array(target_data)
    n_trials = len(target_arr)
    
    # Auto-infer target_values if not provided (for binary classification)
    if target_values is None:
        unique_targets = np.unique(target_arr)
        # Remove None/nan values
        unique_targets = [v for v in unique_targets if v is not None and str(v) != 'nan']
        if len(unique_targets) == 2:
            # Binary classification: auto-assign 0 and 1
            target_values = {str(unique_targets[0]): 0, str(unique_targets[1]): 1}
            print(f"Auto-inferred target_values: {target_values}")
        elif len(unique_targets) > 2:
            # Multi-class: assign 0, 1, 2, ...
            target_values = {str(v): i for i, v in enumerate(sorted(unique_targets))}
            print(f"Auto-inferred target_values (multi-class): {target_values}")
        else:
            raise RuntimeError(f"Cannot infer target_values: found {len(unique_targets)} unique values in '{target}'")
    
    # Get filter data if specified
    if filter_field and filter_values:
        filter_data = raw.get(filter_field)
        if filter_data is None:
            print(f"Warning: Filter field '{filter_field}' not found, skipping filter")
            valid_mask = np.ones(n_trials, dtype=bool)
        else:
            filter_arr = np.array(filter_data)
            valid_mask = np.isin(filter_arr, filter_values)
    else:
        valid_mask = np.ones(n_trials, dtype=bool)
    
    # Apply additional trial_filter (e.g., {"trial_is_good": 1})
    if trial_filter and isinstance(trial_filter, dict):
        for field_name, expected_value in trial_filter.items():
            field_data = raw.get(field_name)
            if field_data is not None:
                field_arr = np.array(field_data)
                if isinstance(expected_value, list):
                    field_mask = np.isin(field_arr, expected_value)
                else:
                    field_mask = (field_arr == expected_value)
                valid_mask = valid_mask & field_mask
                print(f"Applied trial_filter: {field_name} = {expected_value}, {np.sum(field_mask)} trials match")
            else:
                print(f"Warning: trial_filter field '{field_name}' not found in dataset")
    
    # Filter by target values
    target_mask = np.isin(target_arr, list(target_values.keys()))
    valid_mask = valid_mask & target_mask
    
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 20:
        raise RuntimeError(f"Too few valid trials: {len(valid_indices)} (need at least 20)")
    
    # Extract spike counts for each trial
    # Handle different spike data formats
    spike_data = raw.get(feature)
    if spike_data is None:
        # Try alternative names
        for alt_name in ["spike_times", "spike_times_by_unit", "spikes", "TS"]:
            if alt_name in raw:
                spike_data = raw[alt_name]
                feature = alt_name
                break
    
    if spike_data is None:
        raise RuntimeError(f"Could not find spike data field. Tried: {feature}")
    
    # Handle time_window - can be tuple (start, end) or dict {"start": field, "end": field/expr}
    trial_aligned = False
    align_field = None
    window_offset_start = 0.0
    window_offset_end = 1.0
    align_times = None  # Initialize here
    
    
    if isinstance(time_window, dict):
        # Trial-aligned window: {"start": "<event_col>", "end": "<event_col> + 1"}
        trial_aligned = True
        start_spec = time_window.get("start", "0")
        end_spec = time_window.get("end", "1")
        if isinstance(start_spec, str):
            # Check if it's a field name or expression
            if "+" in start_spec:
                parts = start_spec.split("+")
                align_field = parts[0].strip()
                window_offset_start = float(parts[1].strip())
            elif "-" in start_spec and not start_spec.startswith("-"):
                parts = start_spec.split("-")
                align_field = parts[0].strip()
                window_offset_start = -float(parts[1].strip())
            else:
                # Just a field name, offset is 0
                align_field = start_spec
                window_offset_start = 0.0
        else:
            window_offset_start = float(start_spec)
        
        # Parse end specification
        if isinstance(end_spec, str):
            if "+" in end_spec:
                parts = end_spec.split("+")
                # align_field already set from start
                window_offset_end = float(parts[1].strip())
            elif "-" in end_spec and not end_spec.startswith("-"):
                parts = end_spec.split("-")
                window_offset_end = -float(parts[1].strip())
            else:
                # Just a field name - treat as offset 0 from end field
                window_offset_end = 0.0
        else:
            window_offset_end = float(end_spec)
        
        # Get alignment times from dataset
        if align_field:
            align_times = raw.get(align_field)
            if align_times is None:
                # Try alternative field names - include trial_ prefix variants
                # Dynamically find event/time columns as fallback
                # Look for any column containing time-related keywords
                time_keywords = ["_time", "_onset", "_start", "_offset", "_end"]
                alternatives = [k for k in raw.keys() if any(tw in k.lower() for tw in time_keywords)]
                # Also include common generic names
                alternatives.extend(["trial_start_time", "start_time", "event_time"])
                for alt in alternatives:
                    if alt in raw:
                        align_times = raw[alt]
                        align_field = alt
                        break
            if align_times is None:
                # List available time-related fields for error message
                available_time_fields = [k for k in raw.keys() if any(
                    x in k.lower() for x in ["time", "onset", "offset", "start", "end", "cue", "pole"]
                )]
                raise RuntimeError(
                    f"Alignment field '{align_field}' not found in dataset. "
                    f"Available time-related fields: {available_time_fields[:20]}. "
                    f"Please specify a valid time field in time_window."
                )
            else:
                # Convert to float array
                align_times = np.array([float(t) if t is not None else 0.0 for t in align_times])
    else:
        # Simple tuple format
        window_offset_start, window_offset_end = time_window
    
    window_duration = window_offset_end - window_offset_start
    
    # Build unit filter mask if unit_filter specified
    unit_valid_mask = None
    n_units = len(spike_data) if isinstance(spike_data, list) else 0
    if unit_filter and isinstance(unit_filter, dict) and n_units > 0:
        unit_valid_mask = np.ones(n_units, dtype=bool)
        for field_name, expected_values in unit_filter.items():
            # Common field name variations
            possible_names = [
                field_name,
                f"unit_{field_name}",
                f"unit_{field_name}s",
                field_name.replace("quality", "unit_quality"),
            ]
            field_data = None
            for pn in possible_names:
                if pn in raw:
                    field_data = raw[pn]
                    print(f"Found unit filter field: {pn}")
                    break
            
            if field_data is not None:
                field_arr = np.array(field_data)
                if len(field_arr) == n_units:
                    if isinstance(expected_values, list):
                        field_mask = np.isin(field_arr, expected_values)
                    else:
                        field_mask = (field_arr == expected_values)
                    unit_valid_mask = unit_valid_mask & field_mask
                    print(f"Applied unit_filter: {field_name} in {expected_values}, {np.sum(field_mask)}/{n_units} units pass")
                else:
                    print(f"Warning: unit_filter field '{field_name}' length {len(field_arr)} != n_units {n_units}")
            else:
                print(f"Warning: unit_filter field '{field_name}' not found in dataset")
        
        if unit_valid_mask is not None:
            n_valid_units = np.sum(unit_valid_mask)
            print(f"Unit filtering: {n_valid_units}/{n_units} units pass all filters")
    
    X_list = []
    y_list = []
    
    # Get trial start times for computing absolute windows
    trial_start_times = raw.get("trial_start_time")
    if trial_start_times is None:
        trial_start_times = raw.get("start_time")
    if trial_start_times is not None:
        trial_start_times = np.array([float(t) if t is not None else 0.0 for t in trial_start_times])
    
    # Check if align_times are relative (small values) or absolute (large values)
    # If align_times max < 100, they're probably relative to trial start
    if trial_aligned and align_times is not None and trial_start_times is not None:
        if np.max(align_times) < 100:
            print(f"Trial alignment: times are relative (max={np.max(align_times):.3f}); adding trial_start_times")
            # align_times are relative to trial start, convert to absolute
            absolute_align_times = trial_start_times + align_times
        else:
            print(f"Trial alignment: times are absolute (max={np.max(align_times):.3f})")
            absolute_align_times = align_times
    elif trial_aligned and align_times is not None:
        # No trial_start_times, assume align_times are already absolute
        absolute_align_times = align_times
    else:
        absolute_align_times = None
    
    for trial_idx in valid_indices:
        # Get label
        label_str = str(target_arr[trial_idx])
        if label_str not in target_values:
            continue
        y_val = target_values[label_str]
        
        # Compute trial-specific absolute time window
        if trial_aligned and absolute_align_times is not None:
            trial_align_time = absolute_align_times[trial_idx]
            window_start = trial_align_time + window_offset_start
            window_end = trial_align_time + window_offset_end
        else:
            window_start = window_offset_start
            window_end = window_offset_end
        
        
        # Get spike counts - spike_data is list of [unit_spike_times, ...]
        # Each unit_spikes is an array of ABSOLUTE spike times for that unit
        if isinstance(spike_data, list) and len(spike_data) > 0:
            spike_counts = []
            for unit_idx, unit_spikes in enumerate(spike_data):
                # Skip filtered units
                if unit_valid_mask is not None and not unit_valid_mask[unit_idx]:
                    continue
                
                # unit_spikes is the array of ALL spike times for this unit (absolute)
                if isinstance(unit_spikes, (list, np.ndarray)) and len(unit_spikes) > 0:
                    unit_spikes_arr = np.array(unit_spikes, dtype=np.float64)
                    # Count spikes in the absolute time window
                    count = np.sum((unit_spikes_arr >= window_start) & (unit_spikes_arr < window_end))
                    spike_counts.append(count)
                else:
                    spike_counts.append(0)
            
            if spike_counts:
                X_list.append(np.array(spike_counts, dtype=np.float32))
                y_list.append(y_val)
                
        else:
            # Flat array format - skip for now
            print(f"Warning: Unsupported spike data format for trial {trial_idx}")
            continue
    
    if len(X_list) < 20:
        raise RuntimeError(f"Too few valid samples with spike data: {len(X_list)}")
    
    X = np.stack(X_list, axis=0)  # (n_samples, n_units)
    y = np.array(y_list, dtype=np.int64)
    
    # Class counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_count_dict = {int(c): int(cnt) for c, cnt in zip(unique_classes, class_counts)}
    n_classes = len(unique_classes)

    # Create class labels for display
    class_labels = []
    for val, label in sorted(target_values.items(), key=lambda x: x[1]):
        class_labels.append(val)

    # Stratified train/val/test split BEFORE standardizing to prevent data leakage
    # First split: train+val vs test
    train_val_ratio = train_ratio + val_ratio
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / train_val_ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adjusted, stratify=y_trainval, random_state=seed
    )

    # Standardize: fit on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Split class counts
    split_counts = {
        "train": {int(c): int((y_train == c).sum()) for c in unique_classes},
        "val": {int(c): int((y_val == c).sum()) for c in unique_classes},
        "test": {int(c): int((y_test == c).sum()) for c in unique_classes},
    }
    
    # Create classifier
    if classifier_type in ["logistic_regression", "logreg", "lr"]:
        clf = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="lbfgs" if penalty == "l2" else "saga",
            max_iter=1000,
            random_state=seed,
        )
        model_info = {
            "type": "LogisticRegression",
            "C": C,
            "penalty": penalty,
            "solver": clf.solver,
        }
    else:  # SVM
        clf = SVC(
            C=C,
            kernel="rbf",
            probability=True,
            random_state=seed,
        )
        model_info = {
            "type": "SVC",
            "C": C,
            "kernel": "rbf",
        }
    
    # Train on training set
    clf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    # Probabilities for AUROC (if available)
    try:
        y_test_proba = clf.predict_proba(X_test)
        has_proba = True
    except Exception:
        y_test_proba = None
        has_proba = False
    
    # Compute metrics on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_micro = f1_score(y_test, y_test_pred, average="micro")
    
    # AUROC
    if has_proba and n_classes == 2:
        test_auroc = roc_auc_score(y_test, y_test_proba[:, 1])
    elif has_proba and n_classes > 2:
        try:
            test_auroc = roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="macro")
        except Exception:
            test_auroc = None
    else:
        test_auroc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(n_classes)))
    
    # Cross-validation on training set only (not full dataset, to prevent leakage)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_accuracies = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy")
    cv_f1_macro = cross_val_score(clf, X_train, y_train, cv=skf, scoring="f1_macro")
    
    # Build metrics dict
    metrics = {
        "n_samples_total": len(y),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "class_labels": class_labels,
        "class_counts_overall": class_count_dict,
        "split_sizes": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        },
        "split_class_counts": split_counts,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_balanced_accuracy),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_micro": float(test_f1_micro),
        "test_auroc": float(test_auroc) if test_auroc is not None else None,
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std": float(np.std(cv_accuracies)),
        "cv_f1_macro_mean": float(np.mean(cv_f1_macro)),
        "cv_f1_macro_std": float(np.std(cv_f1_macro)),
        "cv_folds": cv_folds,
    }
    
    artifacts = {
        "model": model_info,
        "feature_field": feature,
        "target_field": target,
        "target_mapping": target_values,
        "filter_field": filter_field,
        "filter_values": filter_values,
        "time_window": time_window if isinstance(time_window, dict) else list(time_window),
        "seed": seed,
    }
    
    # Generate confusion matrix figure
    if HAS_PLOTTING and ctx:
        try:
            figure_path = ctx.outputs_root / f"confusion_matrix_holdout_{target}.png"
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
            )
            plt.title(f"Test Set Confusion Matrix\nAccuracy: {test_accuracy:.3f}, Balanced Acc: {test_balanced_accuracy:.3f}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            print(f"Warning: Failed to generate confusion matrix plot: {e}")
    
    # Add diagnostic information for Controller to analyze
    diagnostics = _generate_decoding_diagnostics(
        raw=raw,
        time_window=time_window,
        test_accuracy=test_accuracy,
        test_balanced_accuracy=test_balanced_accuracy,
        n_classes=n_classes,
    )
    
    return {"metrics": metrics, "artifacts": artifacts, "diagnostics": diagnostics}