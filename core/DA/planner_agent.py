# neuro_copilot/planner_agent.py
"""
Simplified Planner Agent for NS-Copilot.

Generates high-level analysis plans that the Code Generator translates into
executable Python code.  The planner outputs:
  - Which generic tools to call (by name)
  - A natural-language *description* of what each step should do
  - Tool-level configuration only (cv_folds, seed, n_components, …)

Field-level details (column names, time windows, trial filters) are resolved
later by the Code Generator using the ParsedSchema and raw data summary.
NO hardcoded dataset-specific field names.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neuro_copilot.core.state import NeuroGlobalState
from neuro_copilot.core.llm_client import generate_json, LLMResponse


class PlannerAgentError(Exception):
    pass


@dataclass
class PlannerAgentConfig:
    model: str = "auto"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 180.0
    temperature: float = float(os.getenv("PLANNER_TEMPERATURE", "0.2"))
    seed: Optional[int] = int(os.getenv("DA_SEED", "0"))
    retry_model: str = "max"
    use_adaptive_model: bool = True


# =========================================================================
# Plan schema
# =========================================================================

PLAN_SCHEMA_HINT = """
{
  "task_type": "decoding|encoding|pca|statistics|other",
  "steps": [
    {
      "op_name": "generic_tool_name",
      "description": "Natural-language description of what this step should do",
      "params": { "cv_folds": 5 }
    }
  ],
  "questions_for_user": [
    {
      "field": "string",
      "question": "string"
    }
  ]
}
""".strip()


# =========================================================================
# System prompt — concise, generic, no dataset-specific field names
# =========================================================================

SYSTEM_PROMPT = """
You are the Planner Agent for a data analysis system.

YOUR JOB:
Generate a JSON analysis plan that tells the Code Generator WHAT to do,
not HOW to do it at the field level.  The Code Generator has access to
the raw data, the dataset schema, and the full tool library — it will
write the data-extraction code itself.

PLAN FORMAT:
{
  "task_type": "decoding|encoding|pca|statistics|other",
  "required_metrics": ["<metric1>", "<metric2>", ...],
  "steps": [
    {
      "op_name": "<tool_name>",
      "description": "<natural-language description of the analysis step>",
      "params": { <tool configuration only, e.g. cv_folds, seed, n_components> }
    }
  ],
  "questions_for_user": []
}

"required_metrics": List of ALL metric names the user explicitly asked to report
  (e.g. ["accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "auroc"]).
  Extract these directly from the user's request. If the user did not specify
  any metrics, use an empty list [].

AVAILABLE TOOLS (use these as op_name):
  Decoding / Classification:
    decode_logistic_regression  — Logistic regression with cross-validation (supports groups=, pca_components=, C_values=, cv_repeats= for hyperparameter search; returns full metrics incl. y_true/y_pred)
    decode_svm                  — SVM classifier with cross-validation (supports groups=, pca_components=, C_values=, cv_repeats= for hyperparameter search; returns full metrics incl. y_true/y_pred)
    decode_random_forest        — Random Forest classifier with cross-validation (supports groups=, pca_components=, cv_repeats=; returns full metrics incl. y_true/y_pred; no scaling needed)
    decode_xgboost              — XGBoost gradient boosting classifier with cross-validation (supports groups=, pca_components=, cv_repeats=; handles high-dim features and class imbalance well; returns full metrics incl. y_true/y_pred)
    decode_with_holdout         — Train/val/test split with comprehensive metrics (recommended only for larger datasets where a held-out test set is statistically reliable)
    decode_with_ndt3            — Pretrained foundation model for spike decoding (recommended for primate/multi-species data with spike times + trial structure + event alignment; outperforms traditional methods with limited data)
    decode_with_mtm             — Pretrained foundation model for spike decoding (recommended for mouse Neuropixels/silicon-probe data; uses NDT1 encoder with multi-task masking, trained on IBL recordings)
    decode_with_poyo            — Pretrained foundation model for spike decoding (recommended for multi-species data, variable neuron counts, or macaque Utah array recordings; PerceiverIO architecture pretrained on 178 sessions from mouse and macaque data)
    decode_with_labram          — Pretrained EEG foundation model (~180M params, ~2,500h pretraining; outputs 200-dim features; requires standard 10-20 channels)
    decode_with_reve            — Pretrained EEG foundation model (~900M params, ~60,000h pretraining; outputs 1536-dim features; supports any electrode montage)
    decode_with_cbramod         — Pretrained EEG foundation model (~4M params, ~9,000h pretraining; outputs 600-dim features; channel-agnostic)
    decode_with_eegpt           — Pretrained EEG foundation model (~750M params, NeurIPS 2024; outputs 2048-dim features; requires standard 10-10/10-20 channels)
    decode_with_eegmamba        — Pretrained EEG foundation model (~3.3M params, ~16,700h pretraining; outputs 600-dim features; channel-agnostic, SSM architecture)
    decode_with_brainomni      — Pretrained EEG foundation model (~60M params, ~2,650h pretraining, NeurIPS 2025; outputs 4096-dim features; requires standard electrode positions)
  # Encoding (not yet implemented):
    # encode_poisson_glm        — Poisson GLM encoding model
    # fit_gaussian_tuning       — Fit Gaussian tuning curve
  Feature Extraction:
    compute_firing_rates        — Firing rates from spike times
    bin_spike_times             — Bin spikes into time bins
    extract_trial_features      — Per-trial feature matrix from spike times
    extract_spectral_features   — Frequency-domain power features from multi-channel time-series (any signal type: EEG, EMG, LFP, etc.). Returns (n_subjects, n_channels × n_bands) feature matrix. Recommended for time-series classification tasks.
  PCA:
    pca_decomposition           — Standard PCA
    # pca_randomized            — NOT YET IMPLEMENTED
    # pca_incremental           — NOT YET IMPLEMENTED
  # Statistics (not yet implemented):
    # statistical_test          — t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, Wilcoxon
    # compute_correlation       — Pearson or Spearman correlation
  Plotting:
    plot_confusion_matrix       — Confusion matrix heatmap
    plot_raster                 — Spike raster plot
    # plot_tuning_curve         — NOT YET IMPLEMENTED
    # plot_explained_variance   — NOT YET IMPLEMENTED
    # plot_pc_trajectories      — NOT YET IMPLEMENTED

RULES:
1. "description" is the most important field.  Write a clear, specific
   sentence about what the step should accomplish, referencing the user's
   intent and the data columns described in the dataset summary.
   Example: "Decode <label_column> from mean firing rates computed
   during the appropriate event-aligned time window (see dataset summary
   for available event columns and their ranges)."
2. "params" should contain ONLY tool-level configuration (cv_folds,
   seed, n_components, kernel, test, method, etc.).
   Do NOT put field names (column names) in params — put them in the
   description instead.
3. Keep plans focused but thorough.  A simple decoding task may need
   only 1 step, but you should add steps when good methodology demands
   it (e.g. comparing classifiers, using cross-validation for small
   datasets).  Multi-step plans are encouraged when they improve the
   reliability of the analysis.
4. If the user's request is ambiguous or a required column is unclear,
   add a question to "questions_for_user" instead of guessing.
5. Do NOT invent field names.  Refer to the DATASET SUMMARY provided.
6. Output ONLY a single JSON object — no markdown, no extra text.
7. WHEN TO USE FOUNDATION MODELS:

   a) decode_with_ndt3 (NDT3):
      Use when the dataset has:
      - Spike timing data (roles: SPIKE_TIME, UNIT_ID)
      - Trial-based structure (roles: TRIAL_ID, TRIAL_START or TRIAL_END)
      - Event alignment available (roles: STIMULUS_ONSET, RESPONSE_TIME, or EVENT_TIME)
      - Classification or regression labels (roles: TRIAL_TYPE, OUTCOME, or LABEL)
      - At least 10 neurons
      NDT3 is a 45M-parameter Transformer pretrained on 2000+ hours of multi-species
      spike data. Best for primate data, multi-species data, or when species is unknown.
      DO NOT use NDT3 for: EEG/LFP data, fMRI data, continuous recordings without
      trials, or datasets with < 10 neurons.

   b) decode_with_mtm (MtM):
      Use when the dataset has the same requirements as NDT3, PLUS:
      - Species is mouse
      - Recording type matches MtM's pretraining data (mouse extracellular electrophysiology)
      DO NOT use MtM for: non-spike data or datasets with < 5 neurons.

   c) decode_with_poyo (POYO-1):
      Use when the dataset has the same requirements as NDT3/MtM, PLUS:
      - Multi-species data
      - Variable neuron counts across sessions
      - At least 5 neurons
      POYO-1's PerceiverIO cross-attention architecture handles variable input
      sizes naturally and was pretrained on multi-species data (mouse and macaque).
      DO NOT use POYO for: EEG/LFP data, fMRI data, continuous recordings
      without trials, or datasets with < 5 neurons.

   d) decode_with_labram (LaBraM):
      - Input: EEG data, standard 10-20 channels, any sampling rate (resampled to 200Hz)
      - Output: 200-dim features per subject
      - ~180M params, pretrained on ~2,500h EEG
      - NOT for: spike data, fMRI, or non-EEG signals

   e) decode_with_reve (REVE):
      - Input: EEG data, any electrode montage, any sampling rate (resampled to 200Hz)
      - Output: 1536-dim features per subject (512 mean + 512 std + 512 max)
      - ~900M params, pretrained on ~60,000h EEG from 92 datasets
      - NOT for: spike data, fMRI, or non-EEG signals

   f) decode_with_cbramod (CBraMod):
      - Input: EEG data, any electrode layout (no position info needed), any sampling rate (resampled to 200Hz)
      - Output: 600-dim features per subject (200 mean + 200 std + 200 max)
      - ~4M params, pretrained on ~9,000h EEG
      - NOT for: spike data, fMRI, or non-EEG signals

   g) decode_with_eegpt (EEGPT):
      - Input: EEG data, standard 10-10/10-20 channels, any sampling rate (resampled to 256Hz)
      - Output: 2048-dim features per subject (4 summary tokens × 512-dim)
      - ~750M params, NeurIPS 2024
      - NOT for: non-standard channel names, spike data, fMRI, or non-EEG signals

   h) decode_with_eegmamba (EEGMamba):
      - Input: EEG data, any electrode layout (no position info needed), any sampling rate (resampled to 200Hz)
      - Output: 600-dim features per subject (200 mean + 200 std + 200 max)
      - ~3.3M params, SSM (Mamba2) architecture, pretrained on ~16,700h EEG
      - NOT for: spike data, fMRI, or non-EEG signals

   i) decode_with_brainomni (BrainOmni):
      - Input: EEG data, standard electrode names (positions auto-detected from MNE), any sampling rate (resampled to 256Hz)
      - Output: 4096-dim features per subject (16 latent sources × 256 dims)
      - ~60M params, pretrained on ~2,650h EEG/MEG, NeurIPS 2025
      - NOT for: spike data, fMRI, or non-standard electrode names

   HOW TO CHOOSE between foundation models:
   - EEG data (.set/.edf/.bdf) → all six EEG models are candidates; choose based on data compatibility (channel requirements, electrode positions) and consider trying multiple models when the user does not specify one
   - Spike data → choose among spike foundation models based on each model's pretrained data characteristics described in the tool specs above. Match the model whose training data best aligns with the dataset's species, recording equipment, and neuron counts.
   - If species/recording info is unavailable → try multiple spike models and compare
   - If the user explicitly requests a specific model → use that model
   - If the user explicitly forbids pretrained models → use only traditional feature extraction
   - If the user neither specifies nor forbids pretrained models → consider running multiple compatible foundation models alongside traditional feature extraction, comparing their results, and selecting the best-performing approach. Generate a single plan that loops through all compatible models.

8. EVALUATION STRATEGY — choose an appropriate evaluation method based on dataset
   characteristics (sample size, class balance, etc.). Consider the trade-offs
   between different cross-validation strategies and holdout splits. Be mindful
   of class imbalance when selecting classifiers and metrics.

9. DIMENSIONALITY REDUCTION — if dimensionality reduction is needed, ensure it is
   applied inside cross-validation folds to prevent data leakage, rather than on
   the full dataset before splitting.

10. CLASSIFIER SELECTION — choose classifiers appropriate for the data characteristics.
   If the user does not specify a particular classifier, consider comparing multiple
   approaches. If the user specifies a particular classifier, use only that one.

11. FOUNDATION MODEL EVALUATION — when a foundation model is used for feature
   extraction, the classifier input should be the model's output features only.
   Mixing in external metadata (e.g. clinical scores, demographics) confounds
   the evaluation of the model's learned representations.
""".strip()


# =========================================================================
# User prompt builder
# =========================================================================

def build_plan_prompt(
    field_catalog_text: str,
    user_idea: str,
    *,
    schema_description: str = "",
    validation_feedback: Optional[str] = None,
    previous_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the user-facing prompt for the Planner LLM call.

    Args:
        field_catalog_text: Textual summary of the loaded dataset fields.
        user_idea: The user's analysis request in natural language.
        schema_description: Optional ParsedSchema description (from schema_parser).
        validation_feedback: Error feedback when retrying a failed plan.
        previous_plan: The previous plan JSON (for retry context).
    """
    sections: List[str] = []

    # --- dataset summary ---
    sections.append("## Dataset Summary")
    sections.append(field_catalog_text)

    if schema_description:
        sections.append("")
        sections.append("## Parsed Schema")
        sections.append(schema_description)

    # --- retry context ---
    if validation_feedback:
        sections.append("")
        sections.append("## RETRY — Previous Plan Failed")
        if previous_plan:
            sections.append(
                f"Previous plan:\n```json\n{json.dumps(previous_plan, indent=2, ensure_ascii=False)}\n```"
            )
        sections.append(f"Errors:\n{validation_feedback}")
        sections.append(
            "Fix the errors above and generate a CORRECTED plan.  "
            "Pay special attention to missing parameters or incorrect tool names."
        )

    # --- user request ---
    sections.append("")
    sections.append("## User Request")
    sections.append(user_idea)

    # --- final reminder ---
    sections.append("")
    sections.append(
        "Return a single JSON plan object with \"task_type\", \"steps\" "
        "(non-empty list), and \"questions_for_user\".  "
        "Each step must have \"op_name\", \"description\", and \"params\"."
    )

    return "\n".join(sections)


# =========================================================================
# Model name resolution
# =========================================================================

def _resolve_model_name(model: str) -> str:
    """Resolve 'auto' / 'max' aliases to concrete model names."""
    from neuro_copilot.core.config import (
        LLM_BACKEND, PLANNER_MODEL,
        _get_default_model, _get_advanced_model,
    )

    model_lower = model.lower().strip()

    if model_lower == "auto":
        return PLANNER_MODEL
    elif model_lower == "max":
        return _get_advanced_model()
    return model


# =========================================================================
# Main entry point
# =========================================================================

def run_planner_agent(
    state: NeuroGlobalState,
    *,
    user_idea: str,
    tools_whitelist_text: Optional[str] = None,
    tools_tags: Optional[list[str]] = None,
    config: Optional[PlannerAgentConfig] = None,
    validation_feedback: Optional[str] = None,
    previous_plan: Optional[Dict[str, Any]] = None,
) -> NeuroGlobalState:
    """
    Run the Planner Agent and store the resulting plan in ``state``.

    The function signature is kept identical to the previous version so
    that callers (pipeline.py, main.py) do not need changes.

    Args:
        state: Global state containing dataset info.
        user_idea: User's analysis idea in natural language.
        tools_whitelist_text: Ignored (kept for backward compatibility).
        tools_tags: Ignored (kept for backward compatibility).
        config: Planner Agent configuration.
        validation_feedback: Optional error feedback for retry mode.
        previous_plan: Optional previous plan that failed (for retry).
    """
    if config is None:
        config = PlannerAgentConfig()

    # --- field catalog text (required) ---
    field_catalog_text = state.extra.get("field_catalog_text")
    if not field_catalog_text:
        raise PlannerAgentError(
            "field_catalog_text not found in state.extra. "
            "Run dataloader first so it can generate the dataset structure summary."
        )

    # Truncate if excessively long to stay within context limits
    MAX_CHARS = 20_000
    if len(field_catalog_text) > MAX_CHARS:
        truncated = field_catalog_text[:MAX_CHARS]
        last_nl = truncated.rfind("\n")
        if last_nl > MAX_CHARS * 0.8:
            truncated = truncated[:last_nl]
        field_catalog_text = (
            truncated
            + f"\n\n... [TRUNCATED: {len(truncated)} of "
            f"{len(state.extra.get('field_catalog_text', ''))} chars]\n"
        )

    # --- optional schema description ---
    schema_desc = ""
    parsed_schema = state.extra.get("parsed_schema")
    if parsed_schema is not None:
        # Reuse the Code Generator's schema builder if available
        try:
            from neuro_copilot.core.DA.code_generator import CodeGenerator
            schema_desc = CodeGenerator._build_schema_description(
                CodeGenerator, parsed_schema
            )
        except Exception:
            schema_desc = str(parsed_schema)

    # --- resolve LLM model ---
    actual_model = _resolve_model_name(config.model)

    # --- build prompt & call LLM ---
    prompt = build_plan_prompt(
        field_catalog_text,
        user_idea,
        schema_description=schema_desc,
        validation_feedback=validation_feedback,
        previous_plan=previous_plan,
    )

    resp: LLMResponse = generate_json(
        model=actual_model,
        base_url=config.base_url,
        timeout_s=config.timeout_s,
        temperature=config.temperature,
        seed=config.seed,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        schema_hint=PLAN_SCHEMA_HINT,
    )

    # --- store results in state ---
    state.extra["analysis_plan_raw"] = resp.raw_text
    state.extra["analysis_plan_json"] = resp.json_obj
    state.extra["analysis_plan_llm_meta"] = {
        "model": resp.model,
        "requested_model": config.model,
        "actual_model": actual_model,
        "prompt_tokens": resp.prompt_tokens,
        "eval_tokens": resp.eval_tokens,
        "total_duration_ms": resp.total_duration_ms,
        "is_retry": validation_feedback is not None,
    }

    # Audit trail
    state.extra["planner_prompt"] = prompt
    state.extra["planner_system_prompt"] = SYSTEM_PROMPT

    return state


# =========================================================================
# Backward compatibility aliases
# =========================================================================
ReasoningAgentError = PlannerAgentError
ReasoningAgentConfig = PlannerAgentConfig
run_reasoning_agent = run_planner_agent
