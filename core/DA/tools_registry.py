# neuro_copilot/tools_registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neuro_copilot.core.tools.decoding import (
    decode_logistic_regression_tool,
    # decode_odr_cue_location_tool,  # ODR-specific tool removed to avoid dataset-specific heuristics
    decode_multiclass_tool,
    decode_svm_tool,
    describe_trial_columns_tool,
    decode_with_holdout_tool,
)
from neuro_copilot.core.tools.encoding import (
    fit_glm_poisson_tool,
    fit_tuning_function_tool,
    compute_rate_prediction_metrics_tool,
    fit_population_model_tool,
)
from neuro_copilot.core.tools.plot import plot_confusion_matrix_tool, plot_tuning_curve_tool
from neuro_copilot.core.tools.preprocessing import (
    set_preprocess_rules_tool,
)
from neuro_copilot.core.tools.pca_tools import (
    pca_sklearn_full,
    pca_sklearn_randomized,
    pca_incremental,
    plot_explained_variance,
    plot_pc_trajectories_over_time,
    decode_f1_linear,
)

# Import generic tools (schema-aware, no hardcoding)
from neuro_copilot.core.tools.generic_tools import (
    decode_logistic_regression,
    decode_svm,
    decode_with_holdout,
    compute_firing_rates,
    bin_spike_times,
    extract_trial_features,
    pca_decomposition,
    plot_confusion_matrix,
    plot_raster,
)

# -----------------------------
# Tool spec / registry
# -----------------------------

@dataclass
class ToolSpec:
    """
    Dataset-agnostic tool specification.

    op_name: stable identifier used in plan JSON.
    description: short human-readable summary.
    input_spec: minimal contract (types, required keys, shapes) for validator + LLM prompting.
    output_spec: minimal contract for downstream consumers.
    callable: actual Python function to run (Execution Agent will call it). Can be None for stub tools.
    tags: e.g., ["decoding"], ["encoding"], ["viz"], ["utility"].
    """
    op_name: str
    description: str
    input_spec: Dict[str, Any] = field(default_factory=dict)
    output_spec: Dict[str, Any] = field(default_factory=dict)
    callable: Optional[Callable[..., Any]] = None
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    A strict whitelist of analysis operations.
    Reasoning Agent sees registry as a "menu".
    Execution Agent dispatches ONLY by looking up op_name in this registry.
    """
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.op_name or not isinstance(spec.op_name, str):
            raise ValueError("ToolSpec.op_name must be a non-empty string.")
        if spec.op_name in self._tools:
            raise ValueError(f"Duplicate tool op_name: {spec.op_name}")
        self._tools[spec.op_name] = spec

    def get(self, op_name: str) -> ToolSpec:
        if op_name not in self._tools:
            raise KeyError(f"op_name not in registry (not whitelisted): {op_name}")
        return self._tools[op_name]

    def has(self, op_name: str) -> bool:
        return op_name in self._tools

    def list(self, tags: Optional[List[str]] = None) -> List[ToolSpec]:
        tools = list(self._tools.values())
        if not tags:
            return sorted(tools, key=lambda x: x.op_name)
        tag_set = set(tags)
        return sorted(
            [t for t in tools if tag_set.intersection(set(t.tags))],
            key=lambda x: x.op_name
        )

    def to_whitelist_text(self, tags: Optional[List[str]] = None) -> str:
        """
        Render a compact LLM-facing whitelist menu.
        This is NOT code; it's "tool cards" with input contracts.
        """
        items = self.list(tags=tags)
        lines: List[str] = []
        for t in items:
            lines.append(f"- op_name: {t.op_name}")
            if t.description:
                lines.append(f"  desc: {t.description}")
            if t.input_spec:
                lines.append(f"  input_spec: {t.input_spec}")
            if t.output_spec:
                lines.append(f"  output_spec: {t.output_spec}")
        return "\n".join(lines)


# -----------------------------
# Default registry (v0)
# -----------------------------

def build_default_registry() -> ToolRegistry:
    """
    v0 registry:
    - Keep it small.
    - Only define contracts; callables can be filled later.
    - No dataset-specific assumptions (no hardcoded field names).
    """
    reg = ToolRegistry()

    # Utility / dataset inspection
    reg.register(ToolSpec(
        op_name="describe_dataset",
        description="Summarize dataset structure and available fields (uses loader-provided summaries).",
        input_spec={
            "required": [],
            "optional": ["top_k_fields"],
            "notes": "Should not read full raw data; prefer field_catalog_text/field_catalog."
        },
        output_spec={
            "type": "dict",
            "keys": ["summary_text", "high_level_stats"]
        },
        callable=None,
        tags=["utility"]
    ))

    reg.register(ToolSpec(
        op_name="describe_trial_columns",
        description="List trial/interval columns with unique values and counts. Identifies potential label columns (stimulus side, response side, trial type, outcome).",
        input_spec={
            "required": [],
            "optional": ["columns", "max_unique"],
            "types": {"columns": "list[string]", "max_unique": "int"},
            "notes": (
                "columns: Optional list of specific columns to describe (default: all trial-related). "
                "max_unique: Maximum unique values to show per column (default: 20). "
                "Returns value counts and identifies candidate label columns for left/right, correct/incorrect, etc."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["n_columns", "columns", "label_candidates", "summary_text"]
        },
        callable=describe_trial_columns_tool,
        tags=["utility", "exploration"]
    ))

    reg.register(ToolSpec(
        op_name="list_fields",
        description="List candidate fields/paths for a given keyword or type signature.",
        input_spec={
            "required": [],
            "optional": ["query", "top_k"],
            "notes": "Search over field_catalog (not raw)."
        },
        output_spec={"type": "dict", "keys": ["candidates"]},
        callable=None,
        tags=["utility"]
    ))

    reg.register(ToolSpec(
        op_name="set_preprocess_rules",
        description="Store dataset preprocessing rules (filters) for downstream tools.",
        input_spec={
            "required": ["rules"],
            "optional": ["mode"],
            "types": {"rules": "dict", "mode": "string"},
            "notes": (
                "rules schema: {filters: [{field, op, value/values}], logic: 'and'|'or', drop_na: bool}. "
                "Supported ops: ==, !=, <, <=, >, >=, in, not_in, between, contains, exists. "
                "mode: 'set' (default), 'append', or 'clear'."
            ),
        },
        output_spec={
            "type": "dict",
            "keys": ["status", "mode", "rules"]
        },
        callable=set_preprocess_rules_tool,
        tags=["utility", "preprocess"]
    ))

    # Decoding (neural -> behavior/stimulus)
    reg.register(ToolSpec(
        op_name="decode_logistic_regression",
        description="Decode a target variable from neural features using logistic regression with CV. Supports any feature and target field names.",
        input_spec={
            "required": ["feature", "target"],
            "optional": ["cv_folds", "seed"],
            "types": {"feature": "string", "target": "string", "cv_folds": "int", "seed": "int"},
            "notes": "Feature should be a time series or array field. Target will be binarized by median threshold for binary classification."
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
        },
        callable=decode_logistic_regression_tool,
        tags=["decoding"]
    ))
    
    reg.register(ToolSpec(
        op_name="decode_svm",
        description="Decode a target variable from neural features using SVM (Support Vector Machine) with CV. Supports any feature and target field names, similar to decode_multiclass but uses SVM classifier.",
        input_spec={
            "required": ["feature", "target"],
            "optional": ["cv_folds", "kernel", "C", "feature_type", "time_window_fields", "target_from_structure", "standardize", "seed"],
            "types": {
                "feature": "string|list",
                "target": "string",
                "feature_type": "string",
                "cv_folds": "int",
                "kernel": "string",
                "C": "float",
                "standardize": "bool",
                "seed": "int"
            },
            "notes": (
                "IMPORTANT: 'feature' is a FIELD NAME (e.g., 'TS', 'spike_times'), NOT a processing type. "
                "'feature_type' is a SEPARATE parameter that specifies HOW to process the feature field. "
                "Example: feature='TS', feature_type='firing_rates' means use field 'TS' and compute firing rates from it. "
                "feature: Field name(s) for neural data (string or list of strings). "
                "target: Field name for target variable (string). "
                "feature_type: Processing method - 'fixed_features' (default, for time series), 'firing_rates' (requires time_window_fields), "
                "'raw' (use field directly), or 'precomputed_rates' (requires feature to be a list). "
                "time_window_fields: Dict for firing_rates, e.g., {'fixation': ('start_field', 'end_field')}. "
                "kernel: 'linear', 'rbf' (default), 'poly', or 'sigmoid'. "
                "C: Regularization parameter (default 1.0). "
                "standardize: Whether to standardize features (default True)."
            )
        },
        output_spec={"type": "dict", "keys": ["metrics", "artifacts"]},
        callable=decode_svm_tool,
        tags=["decoding"]
    ))

    # ODR-specific tool commented out to avoid dataset-specific heuristics
    # reg.register(ToolSpec(
    #     op_name="decode_odr_cue_location",
    #     description="Decode cue location (1-8) from neural activity in ODR task.",
    #     ...
    #     callable=decode_odr_cue_location_tool,
    #     tags=["decoding"]
    # ))

    # Encoding / modeling (task variables -> firing rates)
    reg.register(ToolSpec(
        op_name="fit_glm_poisson",
        description="Fit a Poisson GLM to predict firing rates from task/stimulus variables.",
        input_spec={
            "required": ["predictors", "response"],
            "optional": ["regularization", "time_window", "seed", "response_from_spikes"],
            "types": {"predictors": "list[string]", "response": "string"},
            "notes": (
                "predictors: List of field names for predictors. "
                "Fields not in raw data but detected in mappings will be inferred from data structure. "
                "response: Field name for response variable (firing rate). "
                "response_from_spikes: Optional dict to compute response from spike times if response field is missing. "
                "Example: {'ts_field':'spikes','start_field':'SO1','end_field':None,"
                "'start_offset_ms':-500,'end_offset_ms':4500,'aggregate':'mean'}."
                "Use actual firing rate field names from field_catalog (e.g., precomputed rate fields if available)."
            )
        },
        output_spec={"type": "dict", "keys": ["metrics", "model_path_optional", "artifacts"]},
        callable=fit_glm_poisson_tool,
        tags=["encoding"]
    ))

    reg.register(ToolSpec(
        op_name="fit_tuning_function",
        description="Fit a tuning function (e.g., Gaussian) to neuronal responses across conditions (e.g., directions, positions).",
        input_spec={
            "required": ["condition_field", "response_field"],
            "optional": ["function_type", "initial_params"],
            "types": {
                "condition_field": "string",
                "response_field": "string",
                "function_type": "string",
                "initial_params": "dict"
            },
            "notes": (
                "condition_field: Field name for condition values (e.g., location, direction). "
                "Fields not in raw data but detected in mappings will be inferred from data structure. "
                "response_field: Field name for response variable (firing rate). "
                "Use actual firing rate field names from field_catalog. "
                "function_type: 'gaussian' (default) or 'von_mises' (for circular variables, not yet implemented). "
                "initial_params: Optional dict with initial guesses: {'baseline': float, 'peak': float, 'preferred': float, 'width': float}."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
            "notes": "Metrics include R², RMSE, MAE, correlation. Artifacts include fitted parameters (baseline, peak, preferred, width) and predicted responses."
        },
        callable=fit_tuning_function_tool,
        tags=["encoding"]
    ))

    reg.register(ToolSpec(
        op_name="compute_rate_prediction_metrics",
        description="Compute metrics for rate prediction quality (R², log-likelihood, correlation, MSE, etc.). ONLY use after ENCODING tools (fit_glm_poisson, fit_tuning_function, fit_population_model). DO NOT use after DECODING tools.",
        input_spec={
            "required": ["actual_field"],
            "optional": ["predicted_field", "predicted_from_step", "metric_type"],
            "types": {
                "predicted_field": "string",
                "predicted_from_step": "int",
                "actual_field": "string",
                "metric_type": "string"
            },
            "notes": (
                "⚠️ IMPORTANT: This tool is ONLY for ENCODING tasks (predicting firing rates). "
                "DO NOT use this after DECODING tools (decode_*, which already include their own metrics). "
                "actual_field: Field name for actual/observed firing rates (required). "
                "predicted_from_step: Step index (int) to read predicted values from previous step output. "
                "RECOMMENDED: Use this when evaluating predictions from fit_glm_poisson, fit_tuning_function, etc. "
                "Reads from step output JSON and extracts from artifacts.predicted_values or artifacts.predicted_responses. "
                "Example: After step 0 with fit_glm_poisson, use predicted_from_step=0. "
                "predicted_field: Field name for predicted firing rates from raw data (only use if predicted_from_step is not provided). "
                "metric_type: Which metrics to compute - 'all' (default), 'r2', 'log_likelihood', 'correlation', 'mse', 'mae', 'poisson_deviance'. "
                "IMPORTANT: You MUST provide either predicted_from_step OR predicted_field (not both, but at least one)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
            "notes": "Metrics include R², log-likelihood, correlation, MSE, RMSE, MAE, Poisson deviance, and summary statistics."
        },
        callable=compute_rate_prediction_metrics_tool,
        tags=["encoding"]
    ))

    reg.register(ToolSpec(
        op_name="fit_population_model",
        description="Fit a simple population model to predict population responses from stimulus/task variables. Supports linear models and population vector decoding.",
        input_spec={
            "required": ["neuron_response_fields", "stimulus_field"],
            "optional": ["model_type", "regularization"],
            "types": {
                "neuron_response_fields": "list[string]",
                "stimulus_field": "string",
                "model_type": "string",
                "regularization": "float"
            },
            "notes": (
                "neuron_response_fields: List of field names for individual neuron responses (firing rates). "
                "stimulus_field: Field name for stimulus/task variable (e.g., direction, position). "
                "model_type: 'linear' (default, Ridge regression) or 'population_vector' (vector sum decoding). "
                "regularization: L2 regularization strength for linear model (default: 1.0)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
            "notes": "Metrics include R², RMSE, correlation. Artifacts include model weights, preferred directions/positions (for population_vector), and predictions."
        },
        callable=fit_population_model_tool,
        tags=["encoding"]
    ))

    # Visualization (purely from execution outputs / arrays)
    reg.register(ToolSpec(
        op_name="plot_confusion_matrix",
        description="Render a confusion matrix figure from decoding results.",
        input_spec={
            "required": ["confusion_matrix"],
            "optional": ["title", "normalize", "save_path", "class_labels", "figsize", "cmap"],
            "types": {
                "confusion_matrix": "list[list[int|float]]|ndarray",
                "normalize": "bool",
                "title": "string",
                "save_path": "string",
                "class_labels": "list[string]",
                "figsize": "tuple[float, float]",
                "cmap": "string"
            },
            "notes": "confusion_matrix can be a 2D array or list of lists. If normalize=True, shows percentages instead of counts."
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path", "confusion_matrix_shape", "normalized", "class_labels", "metadata"]
        },
        callable=plot_confusion_matrix_tool,
        tags=["viz"]
    ))

    reg.register(ToolSpec(
        op_name="plot_tuning_curve",
        description="Plot a tuning curve showing neuronal responses across different conditions/stimuli.",
        input_spec={
            "required": ["responses", "conditions"],
            "optional": [
                "error_bars", "title", "save_path", "xlabel", "ylabel",
                "figsize", "style", "show_individual", "aggregate",
                "marker", "linestyle", "linewidth", "markersize",
                "capsize", "capthick", "alpha", "individual_alpha",
                "color", "individual_color", "individual_size",
                "xlabel_rotation", "show_grid", "grid_alpha", "dpi"
            ],
            "types": {
                "responses": "list[float]|list[list[float]]|ndarray",
                "conditions": "list[string|float]",
                "error_bars": "string|list[float]",
                "style": "string",
                "aggregate": "string",
                "show_individual": "bool",
                "marker": "string",
                "linestyle": "string",
                "linewidth": "float",
                "markersize": "float",
                "capsize": "float",
                "capthick": "float",
                "alpha": "float",
                "individual_alpha": "float",
                "color": "string",
                "individual_color": "string",
                "individual_size": "float",
                "xlabel_rotation": "float",
                "show_grid": "bool",
                "grid_alpha": "float",
                "dpi": "int"
            },
            "notes": (
                "responses: 1D array (one value per condition) or 2D array "
                "(multiple neurons/trials x conditions). "
                "conditions: List of condition labels (strings or numbers). "
                "error_bars: 'sem' (standard error), 'std' (standard deviation), or array of error values. "
                "style: 'line' (default) or 'bar'. "
                "aggregate: 'mean' (default) or 'median' for 2D responses."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path", "n_conditions", "n_neurons", "style", "error_bars", "conditions", "mean_responses", "metadata"]
        },
        callable=plot_tuning_curve_tool,
        tags=["viz"]
    ))

    # Generic multi-class decoding tool (works with any dataset)
    reg.register(ToolSpec(
        op_name="decode_multiclass",
        description="Generic multi-class decoding tool that works with any dataset structure. Supports various feature types and can infer targets from dataset structure.",
        input_spec={
            "required": ["feature"],
            "optional": ["target", "feature_type", "time_window_fields", "target_from_structure", "cv_folds", "seed"],
            "types": {
                "feature": "string|list[string]",
                "target": "string",
                "feature_type": "string",
                "cv_folds": "int",
                "seed": "int"
            },
            "notes": (
                "IMPORTANT: 'feature' is a FIELD NAME (e.g., 'TS', 'spike_times'), NOT a processing type. "
                "'feature_type' is a SEPARATE parameter that specifies HOW to process the feature field. "
                "Example: feature='TS', feature_type='firing_rates' means use field 'TS' and compute firing rates from it. "
                "feature: Field name(s) for neural features (string or list of strings). "
                "target: Field name for target variable (optional if target_from_structure=True). "
                "feature_type: Processing method - 'fixed_features' (default, for time series), 'firing_rates' (requires time_window_fields), "
                "'raw' (use field directly), or 'precomputed_rates' (requires feature to be a list). "
                "time_window_fields: Dict for firing_rates, e.g., {'fixation': ('start_field', 'end_field')}. "
                "target_from_structure: If True, infer target from dataset structure (e.g., column index)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
            "notes": "Metrics include per-class accuracy, confusion matrix, class labels"
        },
        callable=decode_multiclass_tool,
        tags=["decoding"]
    ))

    # Comprehensive decoding with train/val/test split
    reg.register(ToolSpec(
        op_name="decode_with_holdout",
        description="Decode with stratified train/val/test split (70/15/15) and comprehensive metrics. Reports accuracy, balanced accuracy, macro/micro F1, AUROC, confusion matrix, plus 5-fold CV with mean±std. Ideal for NWB datasets with trial_type labels.",
        input_spec={
            "required": [],
            "optional": [
                "feature", "target", "target_values", "filter_field", "filter_values",
                "time_window", "alignment_field", "bin_size_ms", "unit_filter", "trial_filter",
                "train_ratio", "val_ratio", "test_ratio",
                "cv_folds", "seed", "classifier", "C", "penalty"
            ],
            "types": {
                "feature": "string",
                "target": "string",
                "target_values": "dict",
                "filter_field": "string",
                "filter_values": "list[string]",
                "time_window": "tuple[float, float] | dict",
                "alignment_field": "string",
                "bin_size_ms": "int",
                "unit_filter": "dict",
                "trial_filter": "dict",
                "train_ratio": "float",
                "val_ratio": "float",
                "test_ratio": "float",
                "cv_folds": "int",
                "seed": "int",
                "classifier": "string",
                "C": "float",
                "penalty": "string"
            },
            "notes": (
                "feature: Spike times field (default: 'spike_times_by_unit'). "
                "target: Label field (default: 'trial_type'). "
                "target_values: Mapping from label strings to integers. If not provided, auto-inferred from data (e.g., for binary classification, the two unique values become 0 and 1). "
                "filter_field: Field to filter trials (default: None, no filtering). "
                "filter_values: Values to keep (default: None, no filtering). Example: filter_field='trial_response', filter_values=['correct', 'incorrect']. "
                "time_window: Two formats supported: "
                "  (1) Absolute: tuple (start, end) in seconds, e.g., (0.0, 1.0). "
                "  (2) Trial-aligned: dict {'start': '<event_column>', 'end': '<event_column> + offset'}, "
                "      e.g., {'start': '<event_col>', 'end': '<event_col> + 1'} for 1s after the event. "
                "alignment_field: Alternative way to specify alignment (use an event column from the dataset), use with time_window tuple for offset. "
                "bin_size_ms: If provided, bin spikes into this window size (e.g., 100 or 200ms), creating multiple features per unit. "
                "unit_filter: Filter units by a column, e.g., {'<quality_col>': ['value1', 'value2']}. "
                "trial_filter: Additional trial filter, e.g., {'<filter_col>': <filter_value>}. "
                "train_ratio/val_ratio/test_ratio: Split ratios (default: 0.70/0.15/0.15). "
                "cv_folds: Cross-validation folds (default: 5). "
                "seed: Random seed (default: 42). "
                "classifier: 'logistic_regression' (default) or 'svm'. "
                "C: Regularization (default: 1.0). penalty: 'l2' (default) or 'l1'. "
                "⚠️ IMPORTANT: For proper decoding, use trial-aligned time_window with a task-relevant field. "
                "Returns comprehensive metrics including balanced accuracy, F1, AUROC, CV scores."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["metrics", "artifacts"],
            "notes": (
                "Metrics: n_samples, n_features, n_classes, class_labels, class_counts_overall, "
                "split_sizes, split_class_counts, test_accuracy, test_balanced_accuracy, "
                "test_f1_macro, test_f1_micro, test_auroc, confusion_matrix, "
                "cv_accuracy_mean, cv_accuracy_std, cv_f1_macro_mean, cv_f1_macro_std. "
                "Artifacts: model info (type, C, penalty), feature/target fields, figure_path."
            )
        },
        callable=decode_with_holdout_tool,
        tags=["decoding"]
    ))

    # PCA analysis tools
    reg.register(ToolSpec(
        op_name="pca_sklearn_full",
        description="Standard sklearn PCA with full SVD decomposition for dimensionality reduction.",
        input_spec={
            "required": ["X2D"],
            "optional": ["n_components", "whiten", "random_state"],
            "types": {
                "X2D": "ndarray",
                "n_components": "int",
                "whiten": "bool",
                "random_state": "int"
            },
            "notes": (
                "X2D: Data matrix [(trials*time_bins) × neurons]. "
                "n_components: Number of components (None = min(n_samples, n_features)). "
                "whiten: Whether to whiten the components. "
                "Returns PCA model, explained variance, components, and transformed data."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["pca_model", "explained_variance", "explained_variance_ratio", "cumulative_variance", "components", "transformed"]
        },
        callable=pca_sklearn_full,
        tags=["pca", "dimensionality_reduction"]
    ))

    reg.register(ToolSpec(
        op_name="pca_sklearn_randomized",
        description="Randomized PCA (faster for large datasets) using randomized SVD.",
        input_spec={
            "required": ["X2D", "n_components"],
            "optional": ["random_state"],
            "types": {
                "X2D": "ndarray",
                "n_components": "int",
                "random_state": "int"
            },
            "notes": "Same as pca_sklearn_full but uses randomized SVD for speed on large datasets."
        },
        output_spec={
            "type": "dict",
            "keys": ["pca_model", "explained_variance", "explained_variance_ratio", "cumulative_variance", "components", "transformed"]
        },
        callable=pca_sklearn_randomized,
        tags=["pca", "dimensionality_reduction"]
    ))

    reg.register(ToolSpec(
        op_name="pca_incremental",
        description="Incremental PCA for memory-efficient processing of very large datasets.",
        input_spec={
            "required": ["X2D", "n_components"],
            "optional": ["batch_size", "random_state"],
            "types": {
                "X2D": "ndarray",
                "n_components": "int",
                "batch_size": "int",
                "random_state": "int"
            },
            "notes": "batch_size: Number of samples to process at once (default 1000)."
        },
        output_spec={
            "type": "dict",
            "keys": ["pca_model", "explained_variance", "explained_variance_ratio", "cumulative_variance", "components", "transformed"]
        },
        callable=pca_incremental,
        tags=["pca", "dimensionality_reduction"]
    ))

    reg.register(ToolSpec(
        op_name="plot_explained_variance",
        description="Plot PCA explained variance curves (individual and cumulative).",
        input_spec={
            "required": ["explained_variance_ratio", "cumulative_variance"],
            "optional": ["save_path", "title", "target_variance", "figsize"],
            "types": {
                "explained_variance_ratio": "list[float]",
                "cumulative_variance": "list[float]",
                "save_path": "string",
                "title": "string",
                "target_variance": "float",
                "figsize": "tuple[float, float]"
            },
            "notes": (
                "explained_variance_ratio: Variance explained by each PC. "
                "cumulative_variance: Cumulative variance explained. "
                "target_variance: Target cumulative variance (default 0.80, draws horizontal line). "
                "Returns k_for_target: Number of PCs needed to reach target variance."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path", "k_for_target", "variance_at_target", "top_10_variance"]
        },
        callable=plot_explained_variance,
        tags=["pca", "viz"]
    ))

    reg.register(ToolSpec(
        op_name="plot_pc_trajectories_over_time",
        description="Plot PC trajectories over time for different experimental conditions.",
        input_spec={
            "required": ["X_pca", "trial_labels", "time_bins"],
            "optional": ["save_path", "title", "pcs_to_plot", "unique_labels", "figsize"],
            "types": {
                "X_pca": "ndarray",
                "trial_labels": "ndarray",
                "time_bins": "int",
                "pcs_to_plot": "list[int]",
                "save_path": "string",
                "title": "string",
                "figsize": "tuple[float, float]"
            },
            "notes": (
                "X_pca: Transformed data [n_samples × n_components]. "
                "trial_labels: Trial condition labels [n_trials]. "
                "time_bins: Number of time bins per trial. "
                "pcs_to_plot: Which PCs to plot (0-indexed, default [0, 1, 2])."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path", "n_conditions", "n_trials", "time_bins", "pcs_plotted"]
        },
        callable=plot_pc_trajectories_over_time,
        tags=["pca", "viz"]
    ))

    reg.register(ToolSpec(
        op_name="decode_f1_linear",
        description="Decode stimulus labels from PCA features using linear classifier with cross-validation.",
        input_spec={
            "required": ["X_pca", "labels"],
            "optional": ["cv", "classifier", "random_state"],
            "types": {
                "X_pca": "ndarray",
                "labels": "ndarray",
                "cv": "int",
                "classifier": "string",
                "random_state": "int"
            },
            "notes": (
                "X_pca: PCA-transformed features [n_samples × n_components]. "
                "labels: Target labels [n_samples]. "
                "cv: Number of cross-validation folds (default 5). "
                "classifier: 'logreg' (LogisticRegression) or 'linear_svm' (LinearSVC)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["accuracy_mean", "accuracy_std", "accuracies_per_fold", "n_classes", "n_samples", "n_features"]
        },
        callable=decode_f1_linear,
        tags=["pca", "decoding"]
    ))

    # =============================================================================
    # Generic Tools (Schema-aware, no hardcoding)
    # These tools have generic X, y interfaces and work with any dataset
    # when combined with code_generator for data extraction
    # Note: Named with "generic_" prefix to avoid conflicts with existing tools
    # =============================================================================
    
    reg.register(ToolSpec(
        op_name="generic_decode_logistic",
        description="[GENERIC] Decode categorical labels from neural features using logistic regression with cross-validation. Use when parsed_schema is available.",
        input_spec={
            "required": [],  # X, y are provided by code generation, not as params
            "optional": ["cv_folds", "seed", "max_iter", "description"],
            "types": {
                "cv_folds": "int",
                "seed": "int",
                "max_iter": "int",
                "description": "string"
            },
            "notes": (
                "Generic tool that works with any dataset via code generation. "
                "Provide 'description' to specify what features to extract. "
                "Example: description='decode <label_column> from firing rates in <time_window> window'. "
                "cv_folds: Number of cross-validation folds (default 5). "
                "max_iter: Max iterations for solver (default 1000)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["accuracy", "accuracy_std", "chance_level", "confusion_matrix", "cv_scores"]
        },
        callable=None,  # Called via code generation
        tags=["decoding", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_decode_svm",
        description="[GENERIC] Decode using Support Vector Machine with RBF kernel and cross-validation. Use when parsed_schema is available.",
        input_spec={
            "required": [],
            "optional": ["cv_folds", "C", "gamma", "seed", "description"],
            "types": {
                "cv_folds": "int",
                "C": "float",
                "gamma": "string or float",
                "seed": "int",
                "description": "string"
            },
            "notes": (
                "Generic SVM decoder via code generation. "
                "Provide 'description' to specify feature extraction. "
                "C: Regularization parameter (default 1.0). "
                "gamma: Kernel coefficient ('scale', 'auto', or float)."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["accuracy", "accuracy_std", "confusion_matrix", "cv_scores"]
        },
        callable=None,
        tags=["decoding", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_decode_holdout",
        description="[GENERIC] Train/test split decoding with holdout validation. Use when parsed_schema is available.",
        input_spec={
            "required": [],
            "optional": ["test_size", "model", "seed", "description"],
            "types": {
                "test_size": "float (0-1)",
                "model": "string (logistic, svm)",
                "seed": "int",
                "description": "string"
            },
            "notes": (
                "Generic holdout decoder. "
                "test_size: Fraction of data for testing (default 0.2). "
                "model: Classifier type - 'logistic' or 'svm' (default 'logistic')."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["train_accuracy", "test_accuracy", "confusion_matrix"]
        },
        callable=None,
        tags=["decoding", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_compute_firing_rates",
        description="[GENERIC] Compute mean firing rates from spike times within time windows.",
        input_spec={
            "required": [],
            "optional": ["description"],
            "types": {
                "description": "string"
            },
            "notes": (
                "Compute firing rates via code generation. "
                "Specify in 'description' which spike times and time windows to use. "
                "Returns firing rate (spikes/second) per unit per window."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["firing_rates", "unit_ids", "window_labels"]
        },
        callable=None,
        tags=["preprocessing", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_bin_spike_times",
        description="[GENERIC] Bin spike times into counts per time bin.",
        input_spec={
            "required": ["bin_size"],
            "optional": ["t_start", "t_end", "description"],
            "types": {
                "bin_size": "float (seconds)",
                "t_start": "float",
                "t_end": "float",
                "description": "string"
            },
            "notes": (
                "bin_size: Width of each bin in seconds. "
                "Returns spike count histogram."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["counts", "bin_edges", "bin_centers"]
        },
        callable=None,
        tags=["preprocessing", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_extract_trial_features",
        description="[GENERIC] Extract features from each trial (e.g., firing rates in a window).",
        input_spec={
            "required": [],
            "optional": ["window", "feature_type", "description"],
            "types": {
                "window": "tuple (start, end) relative to trial onset",
                "feature_type": "string (rate, count, latency)",
                "description": "string"
            },
            "notes": (
                "Extracts per-trial features for decoding. "
                "window: Time window relative to stimulus onset. "
                "feature_type: 'rate' (spikes/s), 'count', or 'latency'."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["X", "trial_ids", "feature_names"]
        },
        callable=None,
        tags=["preprocessing", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_pca",
        description="[GENERIC] Apply PCA dimensionality reduction to feature matrix.",
        input_spec={
            "required": [],
            "optional": ["n_components", "whiten", "description"],
            "types": {
                "n_components": "int or float (0-1 for variance)",
                "whiten": "bool",
                "description": "string"
            },
            "notes": (
                "n_components: Number of components or variance ratio to retain. "
                "whiten: Whether to whiten the components."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["X_transformed", "explained_variance_ratio", "components", "n_components"]
        },
        callable=None,
        tags=["preprocessing", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_plot_confusion_matrix",
        description="[GENERIC] Plot confusion matrix from decoding results.",
        input_spec={
            "required": [],
            "optional": ["labels", "title", "normalize", "output_path", "description"],
            "types": {
                "labels": "list of strings",
                "title": "string",
                "normalize": "bool",
                "output_path": "string",
                "description": "string"
            },
            "notes": "Generates and optionally saves a confusion matrix visualization."
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path", "confusion_matrix"]
        },
        callable=None,
        tags=["visualization", "generic"]
    ))
    
    reg.register(ToolSpec(
        op_name="generic_plot_raster",
        description="[GENERIC] Create a spike raster plot.",
        input_spec={
            "required": [],
            "optional": ["time_range", "output_path", "title", "description"],
            "types": {
                "time_range": "tuple (start, end)",
                "output_path": "string",
                "title": "string",
                "description": "string"
            },
            "notes": "Creates a raster plot showing spike times across trials or units."
        },
        output_spec={
            "type": "dict",
            "keys": ["figure_path"]
        },
        callable=None,
        tags=["visualization", "generic"]
    ))

    # ---- Preprocessing ----
    from neuro_copilot.core.tools.generic_tools import normalize_array as _normalize_array
    reg.register(ToolSpec(
        op_name="normalize_array",
        description="Normalize a feature matrix (z-score, min-max, or robust).",
        input_spec={
            "required": ["X"],
            "optional": ["method", "axis"],
            "types": {
                "X": "2-D array (n_samples, n_features)",
                "method": "string",
                "axis": "int"
            },
            "notes": (
                "Normalize features before classification. "
                "method: 'zscore' (default), 'minmax', or 'robust'. "
                "axis: 0 = per-feature (column-wise, default), 1 = per-sample."
            )
        },
        output_spec={
            "type": "dict",
            "keys": ["X_normalized", "method", "shape"]
        },
        callable=_normalize_array,
        tags=["preprocessing", "generic"]
    ))

    return reg