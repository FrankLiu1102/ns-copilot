from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, r2_score
from scipy.stats import pearsonr

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
from .decoding import (
    _extract_trials_generic,
    _resolve_field_name_via_plan_alias,
    _safe_float,
    _maybe_apply_preprocess,
)

if TYPE_CHECKING:
    # only for type hints; avoids circular import at runtime
    from ..execution_agent import ExecutionContext


def fit_glm_poisson_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generic Poisson GLM encoding tool.

    Predicts a (non‑negative) response variable (e.g., firing rate) from a set of
    task / stimulus predictors using Poisson regression.

    This implementation is **dataset‑agnostic**:
    - It does **not** assume any particular field names.
    - All field names are provided via `predictors` and `response`.

    Args:
        predictors: list of field names to use as predictors (X).
        response: field name for response variable (y), e.g. firing rate.
        regularization: non‑negative float, L2 regularization strength (alpha).
        time_window: (reserved) optional spec for computing response from TS;
                     currently ignored – response field must be numeric.
        seed: random seed (not used directly by PoissonRegressor but kept for API
              compatibility / future extensions).

    Returns:
        Dict with:
            - metrics: model fit metrics (deviance, R², etc.)
            - model_path_optional: always None for now (no serialization yet)
            - artifacts: predictors/response names and model coefficients
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError(
            "neural_dataset not found in state.extra; run dataloader first."
        )

    predictors_raw = params.get("predictors")
    response_raw = params.get("response")
    response_from_spikes = params.get("response_from_spikes")

    if not predictors_raw:
        raise RuntimeError("predictors parameter is required and must be non‑empty")
    if not response_raw:
        raise RuntimeError("response parameter is required")

    # Normalize predictors to list[str]
    if isinstance(predictors_raw, str):
        predictors_list: List[str] = [predictors_raw]
    else:
        predictors_list = [str(p).strip() for p in predictors_raw]

    # Resolve aliases via plan for better robustness
    predictors: List[str] = [
        _resolve_field_name_via_plan_alias(state, name) for name in predictors_list
    ]
    response: str = _resolve_field_name_via_plan_alias(state, str(response_raw).strip())

    alpha = float(params.get("regularization", 1.0))

    # time_window is reserved for future extensions (e.g., computing rates from TS)
    _ = params.get("time_window", None)

    # Extract trials generically
    field_catalog = state.extra.get("field_catalog")
    
    # Check if any predictor needs to be inferred from structure using field mapping
    from ..dataset_detector import get_field_mapping
    field_mapping = get_field_mapping(state)
    
    predictors_to_infer = []
    for p_name in predictors:
        # Check if this predictor might need structure inference
        p_lower = p_name.lower()
        needs_inference = False
        
        if field_mapping:
            # Check if predictor is in detected target fields (likely needs inference)
            if p_name in field_mapping.target_fields:
                needs_inference = True
        else:
            # Fallback: use common patterns
            # if p_lower in ("location", "condition", "target"):
            #     needs_inference = True
            pass
        
        if needs_inference:
            predictors_to_infer.append(p_name)
    
    # Extract trials with structure inference if needed
    # For each predictor that needs inference, we'll extract it separately
    # But _extract_trials_generic only supports one target_field, so we use the first one
    target_field_for_inference = None
    if predictors_to_infer:
        # Use the first predictor name that needs inference as target_field
        # This will be extracted from column index and added to trial dict with this name
        target_field_for_inference = predictors_to_infer[0]
    
    trials = _extract_trials_generic(
        nd.raw,
        field_catalog=field_catalog,
        target_field=target_field_for_inference,  # This name will be used in trial dict
        target_from_column_index=(target_field_for_inference is not None),
    )
    trials = _maybe_apply_preprocess(trials, state)
    
    if len(trials) == 0:
        raise RuntimeError(
            "No trial dicts extracted. Check dataset structure or field names."
        )

    X_rows: List[List[float]] = []
    y_list: List[float] = []
    
    # Track filtering reasons for debugging
    missing_response_count = 0
    negative_response_count = 0
    missing_predictor_counts: Dict[str, int] = {}
    total_trials = len(trials)

    def _compute_response_from_spikes(rec: Dict[str, Any]) -> Optional[float]:
        if not isinstance(response_from_spikes, dict):
            return None
        ts_field = response_from_spikes.get("ts_field", "spikes")
        ts_field = _resolve_field_name_via_plan_alias(state, str(ts_field))
        start_field = response_from_spikes.get("start_field")
        end_field = response_from_spikes.get("end_field")
        start_offset_ms = float(response_from_spikes.get("start_offset_ms", 0.0))
        end_offset_ms = float(response_from_spikes.get("end_offset_ms", 0.0))
        aggregate = str(response_from_spikes.get("aggregate", "mean")).strip().lower()

        ts = rec.get(ts_field)
        if ts is None:
            return None

        start_base = _safe_float(rec.get(start_field)) if start_field else 0.0
        end_base = _safe_float(rec.get(end_field)) if end_field else None
        if start_base is None:
            return None
        if end_base is None:
            end_base = start_base

        t_start = start_base + start_offset_ms
        t_end = end_base + end_offset_ms
        if t_end <= t_start:
            return None

        # spikes may be list of neurons (arrays)
        spike_lists: List[np.ndarray] = []
        if isinstance(ts, list):
            for item in ts:
                if isinstance(item, np.ndarray):
                    spike_lists.append(item)
        elif isinstance(ts, np.ndarray) and ts.dtype == object:
            for item in ts:
                if isinstance(item, np.ndarray):
                    spike_lists.append(item)
        elif isinstance(ts, np.ndarray):
            spike_lists.append(ts)

        if not spike_lists:
            return None

        total_spikes = 0
        for sp in spike_lists:
            if sp.size == 0:
                continue
            arr = sp.squeeze().astype(float)
            total_spikes += int(((arr >= t_start) & (arr <= t_end)).sum())

        window_sec = (t_end - t_start) / 1000.0
        if window_sec <= 0:
            return None

        if aggregate == "sum":
            return float(total_spikes / window_sec)

        # default: mean across neurons
        return float((total_spikes / max(len(spike_lists), 1)) / window_sec)

    for rec in trials:
        # Response
        y_val = _safe_float(rec.get(response, None))
        if y_val is None and response_from_spikes:
            y_val = _compute_response_from_spikes(rec)
        if y_val is None:
            missing_response_count += 1
            continue
        if y_val < 0:
            # Poisson requires non‑negative response
            negative_response_count += 1
            continue

        row: List[float] = []
        valid = True
        missing_predictor = None
        
        # Extract predictors
        for p_name in predictors:
            # Check if this predictor was inferred from structure
            if p_name in predictors_to_infer:
                # The field should have been added to trial dict with the predictor name
                # (because we passed it as target_field to _extract_trials_generic)
                # But also check common fallback names
                x_val = (
                    rec.get(p_name) or  # First try the exact predictor name
                    rec.get("_inferred_target") or  # Fallback: common inferred name
                    rec.get("target")  # Fallback: generic target name
                )
            else:
                x_val = rec.get(p_name, None)
            
            x_val_float = _safe_float(x_val)
            if x_val_float is None:
                valid = False
                missing_predictor = p_name
                if p_name not in missing_predictor_counts:
                    missing_predictor_counts[p_name] = 0
                missing_predictor_counts[p_name] += 1
                break
            row.append(float(x_val_float))

        if not valid:
            continue

        X_rows.append(row)
        y_list.append(float(y_val))

    if len(X_rows) < 20:
        # Provide detailed error message
        error_parts = [
            f"Too few usable samples after filtering for GLM: n={len(X_rows)} (out of {total_trials} trials)"
        ]
        
        if missing_response_count > 0:
            error_parts.append(f"Missing response field '{response}' in {missing_response_count} trials")
        
        if negative_response_count > 0:
            error_parts.append(f"Negative response values in {negative_response_count} trials (Poisson requires non-negative)")
        
        if missing_predictor_counts:
            error_parts.append(f"Missing predictor fields:")
            for p_name, count in missing_predictor_counts.items():
                error_parts.append(f"  - '{p_name}': missing in {count} trials")
                if p_name in predictors_to_infer:
                    error_parts.append(f"    (Note: '{p_name}' should be inferred from structure, but was not found)")
        
        # Show available fields from first trial (if any)
        if len(trials) > 0:
            first_trial = trials[0]
            available_fields = [k for k in first_trial.keys() if isinstance(first_trial.get(k), (int, float, list, np.ndarray))]
            if available_fields:
                error_parts.append(f"Available numeric/array fields in first trial: {available_fields[:15]}")
        
        raise RuntimeError("\n".join(error_parts))

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)

    if X.ndim != 2 or y.ndim != 1:
        raise RuntimeError(
            f"Design matrix / response have wrong shape: X={X.shape}, y={y.shape}"
        )

    # Fit Poisson GLM with L2 regularization
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X, y)

    y_pred = model.predict(X)

    # Metrics
    try:
        dev = float(mean_poisson_deviance(y, y_pred))
    except Exception:
        dev = float("nan")

    try:
        r2 = float(r2_score(y, y_pred))
    except Exception:
        r2 = float("nan")

    metrics: Dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_predictors": int(X.shape[1]),
        "mean_response": float(np.mean(y)),
        "var_response": float(np.var(y)),
        "mean_poisson_deviance": dev,
        "r2_score": r2,
        "regularization_alpha": alpha,
    }

    # Artifacts – lightweight, JSON‑serializable
    coef = model.coef_.ravel().tolist()
    intercept = float(model.intercept_[0] if np.ndim(model.intercept_) > 0 else model.intercept_)

    artifacts: Dict[str, Any] = {
        "predictors": predictors,
        "response": response,
        "coef": coef,
        "intercept": intercept,
        "model_type": "PoissonRegressor",
        "link": "log",
        "predicted_values": y_pred.tolist(),  # For use by compute_rate_prediction_metrics
        "actual_values": y.tolist(),  # For reference
    }

    # Automatically generate visualization: predicted vs actual scatter plot and coefficients
    figure_path = None
    if HAS_PLOTTING and ctx:
        try:
            op_name = params.get("_op_name", "fit_glm_poisson")
            response_str = str(response).replace("/", "_").replace("\\", "_")[:20]
            predictors_str = "_".join([str(p).replace("/", "_")[:10] for p in predictors[:3]])
            figure_filename = f"glm_predicted_vs_actual_{op_name}_{response_str}_{predictors_str}.png"
            figure_path = ctx.outputs_root / figure_filename
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Predicted vs Actual scatter plot
            ax1 = axes[0]
            ax1.scatter(y, y_pred, alpha=0.5, s=20)
            # Add diagonal line (perfect prediction)
            min_val = min(np.min(y), np.min(y_pred))
            max_val = max(np.max(y), np.max(y_pred))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            ax1.set_xlabel(f'Actual {response}')
            ax1.set_ylabel(f'Predicted {response}')
            ax1.set_title(f'Predicted vs Actual\nR² = {r2:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Coefficient bar plot
            ax2 = axes[1]
            predictor_names = predictors if len(predictors) <= 10 else predictors[:10]
            coef_to_plot = coef[:len(predictor_names)]
            colors = ['green' if c > 0 else 'red' for c in coef_to_plot]
            bars = ax2.barh(range(len(predictor_names)), coef_to_plot, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(predictor_names)))
            ax2.set_yticklabels([str(p)[:15] for p in predictor_names])
            ax2.set_xlabel('Coefficient')
            ax2.set_title('Model Coefficients')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            # If plotting fails, continue without visualization
            print(f"Warning: Failed to generate GLM visualization: {e}")

    return {
        "metrics": metrics,
        "model_path_optional": None,
        "artifacts": artifacts,
    }


# -----------------------------
# Tuning function fitting
# -----------------------------

def _gaussian_tuning_function(x: np.ndarray, baseline: float, peak: float, preferred: float, width: float) -> np.ndarray:
    """
    Gaussian tuning function: baseline + peak * exp(-0.5 * ((x - preferred) / width)^2)
    
    Args:
        x: Condition values (e.g., directions in degrees, positions)
        baseline: Baseline firing rate
        peak: Peak firing rate above baseline
        preferred: Preferred condition value
        width: Tuning width (standard deviation)
    
    Returns:
        Predicted firing rates
    """
    return baseline + peak * np.exp(-0.5 * ((x - preferred) / width) ** 2)


def fit_tuning_function_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fit a tuning function (e.g., Gaussian) to neuronal responses across conditions.
    
    This tool fits a Gaussian tuning curve to predict firing rates from condition values
    (e.g., stimulus direction, position, contrast).
    
    Args:
        condition_field: Field name for condition values (e.g., "condition", "direction")
        response_field: Field name for response variable (firing rate)
        function_type: Type of tuning function - "gaussian" (default) or "von_mises" (for circular variables)
        initial_params: Optional dict with initial parameter guesses:
            - baseline: Initial baseline rate guess
            - peak: Initial peak rate guess
            - preferred: Initial preferred condition guess
            - width: Initial tuning width guess
    
    Returns:
        Dict with:
            - metrics: Fit quality metrics (R², RMSE, etc.)
            - artifacts: Fitted parameters (baseline, peak, preferred, width) and predicted responses
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError(
            "neural_dataset not found in state.extra; run dataloader first."
        )
    
    condition_field_raw = params.get("condition_field")
    response_field_raw = params.get("response_field")
    
    if not condition_field_raw:
        raise RuntimeError("condition_field parameter is required")
    if not response_field_raw:
        raise RuntimeError("response_field parameter is required")
    
    condition_field: str = _resolve_field_name_via_plan_alias(state, str(condition_field_raw).strip())
    response_field: str = _resolve_field_name_via_plan_alias(state, str(response_field_raw).strip())
    
    function_type = params.get("function_type", "gaussian")
    initial_params = params.get("initial_params", {})
    
    # Check if condition_field needs structure inference using field mapping
    from ..dataset_detector import get_field_mapping
    field_mapping = get_field_mapping(state)
    
    condition_needs_inference = False
    if field_mapping:
        # Check if condition_field is in detected target fields
        if condition_field in field_mapping.target_fields:
            condition_needs_inference = True
    else:
        # Fallback: use common patterns
        # condition_needs_inference = condition_field.lower() in ("location", "condition", "target")
        condition_needs_inference = False
    
    # Extract trials with structure inference if needed
    field_catalog = state.extra.get("field_catalog")
    target_field_for_inference = None
    if condition_needs_inference:
        target_field_for_inference = condition_field
    
    trials = _extract_trials_generic(
        nd.raw,
        field_catalog=field_catalog,
        target_field=target_field_for_inference,
        target_from_column_index=(target_field_for_inference is not None),
    )
    trials = _maybe_apply_preprocess(trials, state)
    
    if len(trials) == 0:
        raise RuntimeError(
            "No trial dicts extracted. Check dataset structure or field names."
        )
    
    conditions: List[float] = []
    responses: List[float] = []
    
    # Track filtering reasons for debugging
    missing_condition_count = 0
    missing_response_count = 0
    negative_response_count = 0
    total_trials = len(trials)
    
    for rec in trials:
        # Extract condition
        if condition_needs_inference:
            # Try common inferred field names
            cond_val = (
                rec.get(condition_field) or
                rec.get("_inferred_target") or
                rec.get("target")
            )
        else:
            cond_val = rec.get(condition_field, None)
        
        cond_val_float = _safe_float(cond_val)
        if cond_val_float is None:
            missing_condition_count += 1
            continue
        
        # Extract response
        resp_val = _safe_float(rec.get(response_field, None))
        if resp_val is None:
            missing_response_count += 1
            continue
        if resp_val < 0:
            negative_response_count += 1
            continue  # Firing rates should be non-negative
        
        conditions.append(float(cond_val_float))
        responses.append(float(resp_val))
    
    if len(conditions) < 10:
        # Provide detailed error message
        error_parts = [
            f"Too few usable samples for tuning function fit: n={len(conditions)} (out of {total_trials} trials)"
        ]
        
        if missing_condition_count > 0:
            error_parts.append(f"Missing condition field '{condition_field}' in {missing_condition_count} trials")
            if condition_needs_inference:
                error_parts.append(f"  (Note: '{condition_field}' should be inferred from structure, but was not found)")
        
        if missing_response_count > 0:
            error_parts.append(f"Missing response field '{response_field}' in {missing_response_count} trials")
        
        if negative_response_count > 0:
            error_parts.append(f"Negative response values in {negative_response_count} trials")
        
        # Show available fields from first trial
        if len(trials) > 0:
            first_trial = trials[0]
            available_fields = [k for k in first_trial.keys() if isinstance(first_trial.get(k), (int, float, list, np.ndarray))]
            if available_fields:
                error_parts.append(f"Available numeric/array fields in first trial: {available_fields[:15]}")
        
        raise RuntimeError("\n".join(error_parts))
    
    conditions_arr = np.array(conditions, dtype=np.float64)
    responses_arr = np.array(responses, dtype=np.float64)
    
    # Fit Gaussian tuning function
    if function_type == "gaussian":
        # Initial parameter guesses
        baseline_init = initial_params.get("baseline", float(np.min(responses_arr)))
        peak_init = initial_params.get("peak", float(np.max(responses_arr) - baseline_init))
        preferred_init = initial_params.get("preferred", float(np.mean(conditions_arr)))
        width_init = initial_params.get("width", float(np.std(conditions_arr)))
        
        # Bounds: baseline >= 0, peak >= 0, width > 0
        p0 = [baseline_init, peak_init, preferred_init, width_init]
        bounds = ([0, 0, -np.inf, 1e-6], [np.inf, np.inf, np.inf, np.inf])
        
        try:
            popt, pcov = curve_fit(
                _gaussian_tuning_function,
                conditions_arr,
                responses_arr,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
            baseline_fit, peak_fit, preferred_fit, width_fit = popt
            
            # Predictions
            responses_pred = _gaussian_tuning_function(conditions_arr, *popt)
            
        except Exception as e:
            raise RuntimeError(f"Tuning function fitting failed: {e}")
    
    elif function_type == "von_mises":
        # For circular variables (e.g., direction in degrees)
        # Convert to radians and use von Mises distribution
        # Simplified: use Gaussian approximation with periodic boundary
        raise NotImplementedError("von_mises tuning function not yet implemented")
    
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    # Metrics
    try:
        r2 = float(r2_score(responses_arr, responses_pred))
    except Exception:
        r2 = float("nan")
    
    rmse = float(np.sqrt(np.mean((responses_arr - responses_pred) ** 2)))
    mae = float(np.mean(np.abs(responses_arr - responses_pred)))
    
    try:
        corr, corr_p = pearsonr(responses_arr, responses_pred)
        correlation = float(corr)
    except Exception:
        correlation = float("nan")
    
    metrics: Dict[str, Any] = {
        "n_samples": int(len(conditions)),
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "mean_response": float(np.mean(responses_arr)),
        "var_response": float(np.var(responses_arr)),
    }
    
    artifacts: Dict[str, Any] = {
        "function_type": function_type,
        "condition_field": condition_field,
        "response_field": response_field,
        "baseline": float(baseline_fit),
        "peak": float(peak_fit),
        "preferred": float(preferred_fit),
        "width": float(width_fit),
        "predicted_responses": responses_pred.tolist(),
        "conditions": conditions_arr.tolist(),
        "observed_responses": responses_arr.tolist(),
    }

    # Automatically generate visualization: tuning curve with fitted function
    figure_path = None
    if HAS_PLOTTING and ctx:
        try:
            op_name = params.get("_op_name", "fit_tuning_function")
            condition_str = str(condition_field).replace("/", "_").replace("\\", "_")[:20]
            response_str = str(response_field).replace("/", "_").replace("\\", "_")[:20]
            figure_filename = f"tuning_curve_{op_name}_{condition_str}_{response_str}.png"
            figure_path = ctx.outputs_root / figure_filename
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Sort conditions for smooth curve plotting
            sort_idx = np.argsort(conditions_arr)
            conditions_sorted = conditions_arr[sort_idx]
            responses_sorted = responses_arr[sort_idx]
            responses_pred_sorted = responses_pred[sort_idx]
            
            # Plot observed data points
            plt.scatter(conditions_sorted, responses_sorted, alpha=0.5, s=30, label='Observed', color='blue')
            
            # Plot fitted curve (smooth line)
            condition_range = np.linspace(np.min(conditions_arr), np.max(conditions_arr), 200)
            curve_pred = _gaussian_tuning_function(condition_range, baseline_fit, peak_fit, preferred_fit, width_fit)
            plt.plot(condition_range, curve_pred, 'r-', lw=2, label='Fitted curve', alpha=0.8)
            
            # Mark preferred condition
            preferred_response = baseline_fit + peak_fit
            plt.axvline(x=preferred_fit, color='green', linestyle='--', alpha=0.5, label=f'Preferred: {preferred_fit:.2f}')
            plt.plot(preferred_fit, preferred_response, 'go', markersize=10, label='Peak response')
            
            plt.xlabel(f'{condition_field}')
            plt.ylabel(f'{response_field}')
            plt.title(f'Tuning Curve (Gaussian Fit)\nBaseline={baseline_fit:.2f}, Peak={peak_fit:.2f}, Preferred={preferred_fit:.2f}, Width={width_fit:.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            # If plotting fails, continue without visualization
            print(f"Warning: Failed to generate tuning curve visualization: {e}")
    
    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }


# -----------------------------
# Rate prediction metrics
# -----------------------------

def compute_rate_prediction_metrics_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute metrics for rate prediction quality (R², log-likelihood, correlation, etc.).
    
    This is a generic tool that can evaluate predictions from any encoding model.
    
    Args:
        predicted_field: Field name for predicted firing rates (from raw data), OR
                        use predicted_from_step to read from previous step output
        actual_field: Field name for actual/observed firing rates
        predicted_from_step: Optional step index (int) to read predicted values from previous step output.
                            If provided, reads from step output JSON file and extracts predicted values.
                            Path format: step_{step_index:02d}__{op_name}.json
                            Looks for predicted values in: output['artifacts']['predicted_responses'] or
                            output['artifacts']['predicted_values'] or output['metrics']['predicted_mean']
        metric_type: Which metrics to compute - "all" (default), "r2", "log_likelihood", "correlation", "mse"
    
    Returns:
        Dict with:
            - metrics: All computed metrics (R², log-likelihood, correlation, MSE, MAE, etc.)
            - artifacts: Summary statistics and prediction/actual arrays
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError(
            "neural_dataset not found in state.extra; run dataloader first."
        )
    
    predicted_field_raw = params.get("predicted_field")
    actual_field_raw = params.get("actual_field")
    predicted_from_step = params.get("predicted_from_step")
    
    if not predicted_field_raw and predicted_from_step is None:
        raise RuntimeError("Either predicted_field or predicted_from_step parameter is required")
    if not actual_field_raw:
        raise RuntimeError("actual_field parameter is required")
    
    metric_type = params.get("metric_type", "all")
    
    # Get predicted values: either from previous step output or from raw data
    predicted: List[float] = []
    
    if predicted_from_step is not None:
        # Read from previous step output
        import json
        from pathlib import Path
        
        step_index = int(predicted_from_step)
        # Find the step output file
        step_files = list(ctx.outputs_root.glob(f"step_{step_index:02d}__*.json"))
        if not step_files:
            raise RuntimeError(
                f"Step output file not found for step {step_index}. "
                f"Expected file pattern: step_{step_index:02d}__*.json in {ctx.outputs_root}"
            )
        
        step_file = step_files[0]  # Use first matching file
        with open(step_file, 'r') as f:
            step_output = json.load(f)
        
        # Try to extract predicted values from various possible locations
        predicted_data = None
        if isinstance(step_output, dict):
            # Try artifacts first (most common location)
            if "artifacts" in step_output:
                artifacts = step_output["artifacts"]
                # Check predicted_values first (from fit_glm_poisson)
                if "predicted_values" in artifacts:
                    predicted_data = artifacts["predicted_values"]
                # Then predicted_responses (from fit_tuning_function)
                elif "predicted_responses" in artifacts:
                    predicted_data = artifacts["predicted_responses"]
                # predicted_stimulus is for population models (stimulus prediction, not rates)
                # Skip for rate prediction metrics
        
        if predicted_data is None:
            raise RuntimeError(
                f"Could not find predicted values in step {step_index} output. "
                f"Expected in: artifacts.predicted_values (from fit_glm_poisson) or "
                f"artifacts.predicted_responses (from fit_tuning_function). "
                f"Available keys in step output: {list(step_output.keys()) if isinstance(step_output, dict) else 'N/A'}"
            )
        
        if not isinstance(predicted_data, list):
            raise RuntimeError(
                f"Predicted values from step {step_index} must be a list, got {type(predicted_data)}"
            )
        
        # Convert to float list
        for val in predicted_data:
            fval = _safe_float(val)
            if fval is not None and fval >= 0:
                predicted.append(float(fval))
    
    else:
        # Read from raw data field
        predicted_field: str = _resolve_field_name_via_plan_alias(state, str(predicted_field_raw).strip())
        
        # Extract trials
        field_catalog = state.extra.get("field_catalog")
        trials = _extract_trials_generic(nd.raw, field_catalog=field_catalog)
        trials = _maybe_apply_preprocess(trials, state)
        if len(trials) == 0:
            raise RuntimeError(
                "No trial dicts extracted. Check dataset structure or field names."
            )
        
        for rec in trials:
            pred_val = _safe_float(rec.get(predicted_field, None))
            if pred_val is not None and pred_val >= 0:
                predicted.append(float(pred_val))
    
    # Get actual values from raw data
    actual_field: str = _resolve_field_name_via_plan_alias(state, str(actual_field_raw).strip())
    
    # Extract trials
    field_catalog = state.extra.get("field_catalog")
    trials = _extract_trials_generic(nd.raw, field_catalog=field_catalog)
    trials = _maybe_apply_preprocess(trials, state)
    if len(trials) == 0:
        raise RuntimeError(
            "No trial dicts extracted. Check dataset structure or field names."
        )
    
    actual: List[float] = []
    for rec in trials:
        actual_val = _safe_float(rec.get(actual_field, None))
        if actual_val is not None and actual_val >= 0:
            actual.append(float(actual_val))
    
    # Ensure same length
    min_len = min(len(predicted), len(actual))
    if min_len < 5:
        raise RuntimeError(
            f"Too few usable samples for metrics computation: "
            f"predicted n={len(predicted)}, actual n={len(actual)}, min={min_len}"
        )
    
    # Truncate to same length (take first min_len elements)
    predicted = predicted[:min_len]
    actual = actual[:min_len]
    
    predicted_arr = np.array(predicted, dtype=np.float64)
    actual_arr = np.array(actual, dtype=np.float64)
    
    # Compute metrics
    metrics: Dict[str, Any] = {}
    
    if metric_type in ("all", "r2"):
        try:
            r2 = float(r2_score(actual_arr, predicted_arr))
            metrics["r2_score"] = r2
        except Exception:
            metrics["r2_score"] = float("nan")
    
    if metric_type in ("all", "log_likelihood"):
        # Poisson log-likelihood: sum(actual * log(predicted) - predicted - log(actual!))
        # For numerical stability, use: sum(actual * log(predicted + eps) - predicted)
        eps = 1e-10
        predicted_safe = np.maximum(predicted_arr, eps)
        log_likelihood = np.sum(actual_arr * np.log(predicted_safe) - predicted_arr)
        # Subtract log factorial term (using Stirling's approximation for large values)
        # For simplicity, we'll use the simpler form: sum(actual * log(predicted) - predicted)
        metrics["log_likelihood"] = float(log_likelihood)
        metrics["mean_log_likelihood"] = float(log_likelihood / len(predicted_arr))
    
    if metric_type in ("all", "correlation"):
        try:
            corr, corr_p = pearsonr(actual_arr, predicted_arr)
            metrics["correlation"] = float(corr)
            metrics["correlation_p_value"] = float(corr_p)
        except Exception:
            metrics["correlation"] = float("nan")
            metrics["correlation_p_value"] = float("nan")
    
    if metric_type in ("all", "mse"):
        mse = float(np.mean((actual_arr - predicted_arr) ** 2))
        metrics["mse"] = mse
        metrics["rmse"] = float(np.sqrt(mse))
    
    if metric_type in ("all", "mae"):
        mae = float(np.mean(np.abs(actual_arr - predicted_arr)))
        metrics["mae"] = mae
    
    if metric_type in ("all", "poisson_deviance"):
        try:
            dev = float(mean_poisson_deviance(actual_arr, predicted_arr))
            metrics["mean_poisson_deviance"] = dev
        except Exception:
            metrics["mean_poisson_deviance"] = float("nan")
    
    # Summary statistics
    metrics["n_samples"] = int(len(predicted_arr))
    metrics["mean_actual"] = float(np.mean(actual_arr))
    metrics["mean_predicted"] = float(np.mean(predicted_arr))
    metrics["std_actual"] = float(np.std(actual_arr))
    metrics["std_predicted"] = float(np.std(predicted_arr))
    
    # Artifacts
    artifacts: Dict[str, Any] = {
        "predicted_values": predicted_arr.tolist(),
        "actual_values": actual_arr.tolist(),
        "actual_field": actual_field,
        "predicted_source": f"step_{predicted_from_step}" if predicted_from_step is not None else predicted_field_raw,
    }
    
    # Automatically generate visualization: predicted vs actual scatter plot
    figure_path = None
    if HAS_PLOTTING and ctx:
        try:
            op_name = params.get("_op_name", "compute_rate_prediction_metrics")
            actual_str = str(actual_field).replace("/", "_").replace("\\", "_")[:20]
            predicted_str = str(artifacts["predicted_source"]).replace("/", "_").replace("\\", "_")[:20]
            figure_filename = f"rate_prediction_{op_name}_{actual_str}_{predicted_str}.png"
            figure_path = ctx.outputs_root / figure_filename
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(actual_arr, predicted_arr, alpha=0.5, s=20)
            
            # Add diagonal line (perfect prediction)
            min_val = min(np.min(actual_arr), np.min(predicted_arr))
            max_val = max(np.max(actual_arr), np.max(predicted_arr))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            # Add regression line
            from scipy.stats import linregress
            try:
                slope, intercept, r_value, p_value, std_err = linregress(actual_arr, predicted_arr)
                line_x = np.array([min_val, max_val])
                line_y = slope * line_x + intercept
                plt.plot(line_x, line_y, 'g-', lw=2, alpha=0.7, label=f'Regression (R={r_value:.3f})')
            except Exception:
                pass
            
            plt.xlabel(f'Actual {actual_field}')
            plt.ylabel(f'Predicted')
            r2_val = metrics.get("r2_score", float("nan"))
            corr_val = metrics.get("correlation", float("nan"))
            plt.title(f'Rate Prediction Quality\nR² = {r2_val:.4f}, Correlation = {corr_val:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(figure_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            artifacts["figure_path"] = str(figure_path)
        except Exception as e:
            # If plotting fails, continue without visualization
            print(f"Warning: Failed to generate rate prediction visualization: {e}")
    
    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }
    
    artifacts: Dict[str, Any] = {
        "actual_field": actual_field,
        "predicted_values": predicted_arr.tolist(),
        "actual_values": actual_arr.tolist(),
    }
    
    if predicted_from_step is not None:
        artifacts["predicted_from_step"] = predicted_from_step
    else:
        artifacts["predicted_field"] = predicted_field_raw
    
    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }


# -----------------------------
# Population models
# -----------------------------

def fit_population_model_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fit a simple population model to predict population responses from stimulus/task variables.
    
    This tool fits models that describe how a population of neurons responds to stimuli.
    Currently supports:
    - "linear": Linear population response model (weighted sum of individual neuron responses)
    - "population_vector": Population vector decoding (for direction/position coding)
    
    Args:
        neuron_response_fields: List of field names for individual neuron responses (firing rates)
        stimulus_field: Field name for stimulus/task variable (e.g., direction, position)
        model_type: Type of population model - "linear" (default) or "population_vector"
        regularization: L2 regularization strength for linear model (default: 1.0)
    
    Returns:
        Dict with:
            - metrics: Model fit metrics (R², etc.)
            - artifacts: Model parameters (weights, preferred directions/positions, etc.)
    """
    nd = state.extra.get("neural_dataset")
    if nd is None or not hasattr(nd, "raw"):
        raise RuntimeError(
            "neural_dataset not found in state.extra; run dataloader first."
        )
    
    neuron_response_fields_raw = params.get("neuron_response_fields")
    stimulus_field_raw = params.get("stimulus_field")
    
    if not neuron_response_fields_raw:
        raise RuntimeError("neuron_response_fields parameter is required and must be non-empty")
    if not stimulus_field_raw:
        raise RuntimeError("stimulus_field parameter is required")
    
    # Normalize neuron fields to list[str]
    if isinstance(neuron_response_fields_raw, str):
        neuron_fields_list: List[str] = [neuron_response_fields_raw]
    else:
        neuron_fields_list = [str(f).strip() for f in neuron_response_fields_raw]
    
    neuron_fields: List[str] = [
        _resolve_field_name_via_plan_alias(state, name) for name in neuron_fields_list
    ]
    stimulus_field: str = _resolve_field_name_via_plan_alias(state, str(stimulus_field_raw).strip())
    
    model_type = params.get("model_type", "linear")
    regularization = float(params.get("regularization", 1.0))
    
    # Extract trials
    field_catalog = state.extra.get("field_catalog")
    trials = _extract_trials_generic(nd.raw, field_catalog=field_catalog)
    trials = _maybe_apply_preprocess(trials, state)
    if len(trials) == 0:
        raise RuntimeError(
            "No trial dicts extracted. Check dataset structure or field names."
        )
    
    # Build design matrix: rows = trials, cols = neurons
    X_rows: List[List[float]] = []
    stimulus_values: List[float] = []
    
    # Track which fields are missing for better error messages
    missing_neuron_fields: Dict[str, int] = {}
    missing_stimulus_count = 0
    
    for rec in trials:
        stim_val = _safe_float(rec.get(stimulus_field, None))
        if stim_val is None:
            missing_stimulus_count += 1
            continue
        
        row: List[float] = []
        valid = True
        missing_field = None
        for neuron_field in neuron_fields:
            neuron_val = _safe_float(rec.get(neuron_field, None))
            if neuron_val is None:
                valid = False
                missing_field = neuron_field
                if neuron_field not in missing_neuron_fields:
                    missing_neuron_fields[neuron_field] = 0
                missing_neuron_fields[neuron_field] += 1
                break
            if neuron_val < 0:
                neuron_val = 0.0  # Clamp negative values
            row.append(float(neuron_val))
        
        if not valid:
            continue
        
        X_rows.append(row)
        stimulus_values.append(float(stim_val))
    
    if len(X_rows) < 10:
        # Provide detailed error message
        error_parts = [
            f"Too few usable samples for population model: n={len(X_rows)} (out of {len(trials)} trials)"
        ]
        
        if missing_stimulus_count > 0:
            error_parts.append(f"Missing stimulus_field '{stimulus_field}' in {missing_stimulus_count} trials")
        
        if missing_neuron_fields:
            error_parts.append(f"Missing neuron_response_fields:")
            for field, count in missing_neuron_fields.items():
                error_parts.append(f"  - '{field}': missing in {count} trials")
        
        # Show available fields from first trial (if any)
        if len(trials) > 0:
            first_trial = trials[0]
            available_fields = [k for k in first_trial.keys() if isinstance(first_trial.get(k), (int, float, list, np.ndarray))]
            if available_fields:
                error_parts.append(f"Available numeric/array fields in first trial: {available_fields[:10]}")
        
        raise RuntimeError("\n".join(error_parts))
    
    X = np.asarray(X_rows, dtype=np.float64)  # n_trials x n_neurons
    y = np.asarray(stimulus_values, dtype=np.float64)  # n_trials
    
    if X.ndim != 2 or y.ndim != 1:
        raise RuntimeError(
            f"Design matrix / stimulus have wrong shape: X={X.shape}, y={y.shape}"
        )
    
    if model_type == "linear":
        # Linear population response model: predict stimulus from weighted sum of neuron responses
        # y = X @ w + b
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=regularization, max_iter=1000)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        # Metrics
        try:
            r2 = float(r2_score(y, y_pred))
        except Exception:
            r2 = float("nan")
        
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        
        try:
            corr, _ = pearsonr(y, y_pred)
            correlation = float(corr)
        except Exception:
            correlation = float("nan")
        
        metrics: Dict[str, Any] = {
            "n_samples": int(X.shape[0]),
            "n_neurons": int(X.shape[1]),
            "r2_score": r2,
            "rmse": rmse,
            "correlation": correlation,
            "regularization_alpha": regularization,
        }
        
        artifacts: Dict[str, Any] = {
            "model_type": "linear",
            "neuron_response_fields": neuron_fields,
            "stimulus_field": stimulus_field,
            "weights": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "predicted_stimulus": y_pred.tolist(),
            "actual_stimulus": y.tolist(),
        }
    
    elif model_type == "population_vector":
        # Population vector: each neuron has a preferred direction/position
        # Population response = weighted sum of preferred directions
        # For circular variables (directions), use vector sum
        
        # Estimate preferred directions from tuning curves
        n_neurons = X.shape[1]
        preferred_directions: List[float] = []
        weights: List[float] = []
        
        for neuron_idx in range(n_neurons):
            neuron_responses = X[:, neuron_idx]
            
            # Find preferred direction as the stimulus value with maximum average response
            # For circular variables, we'll use a simple approach: find peak
            # More sophisticated: fit tuning curve and extract preferred direction
            
            # Simple method: weighted average of stimulus values
            # For circular variables, convert to vectors first
            # For now, assume linear variable
            if np.max(neuron_responses) > 0:
                # Weighted average
                weights_norm = neuron_responses / (np.sum(neuron_responses) + 1e-10)
                preferred = float(np.sum(weights_norm * y))
                preferred_directions.append(preferred)
                
                # Weight = peak response
                peak_response = float(np.max(neuron_responses))
                weights.append(peak_response)
            else:
                preferred_directions.append(0.0)
                weights.append(0.0)
        
        # Compute population vector prediction
        # For each trial, compute weighted vector sum
        y_pred = []
        for trial_idx in range(X.shape[0]):
            trial_responses = X[trial_idx, :]
            # Normalize responses
            total_response = np.sum(trial_responses)
            if total_response > 0:
                normalized_responses = trial_responses / total_response
                # Weighted sum of preferred directions
                pred = float(np.sum(normalized_responses * np.array(preferred_directions)))
            else:
                pred = float(np.mean(y))
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        
        # Metrics
        try:
            r2 = float(r2_score(y, y_pred))
        except Exception:
            r2 = float("nan")
        
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        
        try:
            corr, _ = pearsonr(y, y_pred)
            correlation = float(corr)
        except Exception:
            correlation = float("nan")
        
        metrics = {
            "n_samples": int(X.shape[0]),
            "n_neurons": int(n_neurons),
            "r2_score": r2,
            "rmse": rmse,
            "correlation": correlation,
        }
        
        artifacts = {
            "model_type": "population_vector",
            "neuron_response_fields": neuron_fields,
            "stimulus_field": stimulus_field,
            "preferred_directions": preferred_directions,
            "weights": weights,
            "predicted_stimulus": y_pred.tolist(),
            "actual_stimulus": y.tolist(),
        }
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return {
        "metrics": metrics,
        "artifacts": artifacts,
    }

