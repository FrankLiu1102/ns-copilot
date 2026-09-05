"""
Code Generator for NS-Copilot

This module uses GPT-4 to generate executable Python code for:
1. Data extraction from raw datasets based on schema
2. Tool invocation with extracted data

The generated code bridges the gap between:
- Raw data (loaded by dataloader)
- Generic tool interfaces (X, y parameters)

No hardcoded field names - GPT-4 generates dataset-specific code.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from neuro_copilot.core.llm_client import generate_text as _llm_generate_text, LLMResponse
from neuro_copilot.core.config import CODE_GENERATOR_MODEL
from neuro_copilot.core.DA.schema_parser import ParsedSchema, ColumnRole
from neuro_copilot.core.tools.generic_tools import get_all_tool_signatures


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str                    # The executable Python code
    explanation: str             # Human-readable explanation
    imports: List[str]           # Required imports
    tool_calls: List[str]        # Names of tools being called
    confidence: float            # LLM's confidence (0-1)
    raw_response: str            # Full LLM response for debugging
    prompt: str = ""             # The prompt sent to GPT-4
    system_prompt: str = ""      # The system prompt used


@dataclass
class AnalysisPlan:
    """Simplified analysis plan for code generation."""
    task: str                    # e.g., "decode trial_type from spike rates"
    tool_name: str               # e.g., "decode_logistic_regression"
    parameters: Dict[str, Any]   # Tool parameters (cv_folds, seed, etc.)
    

# =============================================================================
# Prompt Templates
# =============================================================================

_CODE_GENERATION_SYSTEM_PROMPT_TEMPLATE = """You are an expert Python programmer specializing in neural data analysis.
Your task is to generate executable Python code that:
1. Extracts data from raw datasets based on the provided schema
2. Transforms data into the format required by analysis tools
3. Calls the appropriate analysis tools

CRITICAL RULES:
- Generate ONLY valid, executable Python code
- Common libraries are pre-loaded (see below), but you MAY use `import` / `from X import Y` for additional submodules (e.g. `from sklearn.model_selection import RepeatedStratifiedKFold`)
- Use pandas/numpy for data manipulation
- Access raw data from the `raw_data` variable (dict of DataFrames)
- Store final results in the `result` variable
- Handle edge cases (missing data, type conversion)
- Add brief comments explaining each step
- REPRODUCIBILITY: The variable `RANDOM_SEED` is pre-loaded in the execution environment with the correct seed for this run. You MUST use `RANDOM_SEED` (NOT a hardcoded number like 42) for ALL random operations: train/test splits, cross-validation, model instantiation, etc. NEVER write `random_state=42` or `seed=42` — ALWAYS write `random_state=RANDOM_SEED` or `seed=RANDOM_SEED`. Example: `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)`, `LogisticRegression(random_state=RANDOM_SEED)`, `train_test_split(X, y, random_state=RANDOM_SEED)`

STRICT PLAN ADHERENCE:
- You MUST implement the analysis plan EXACTLY as specified in the task description
- DO NOT modify, substitute, or "improve" any parameters from the plan

**CRITICAL - TOOL SELECTION (MANDATORY)**:
⚠️ If the plan specifies "decode_with_ndt3", you MUST call tools.decode_with_ndt3()
⚠️ If the plan specifies "decode_with_mtm", you MUST call tools.decode_with_mtm()
⚠️ If the plan specifies "decode_with_poyo", you MUST call tools.decode_with_poyo()
⚠️ If the plan specifies "decode_with_labram", you MUST call tools.decode_with_labram()
⚠️ If the plan specifies "decode_with_brainomni", you MUST call tools.decode_with_brainomni()
⚠️ NDT3, MtM, POYO-1, LaBraM, and BrainOmni ARE FULLY AVAILABLE and FUNCTIONAL in the execution environment
⚠️ NEVER add comments like "NDT3/MtM/POYO/LaBraM unavailable" or "skip" - this is FALSE
⚠️ DO NOT substitute with decode_logistic_regression or any other tool
⚠️ DO NOT create fallback code - implement the plan EXACTLY as written
⚠️ Ignoring this instruction will cause execution failure

- If the plan specifies certain values, columns, or configurations, use them literally
- Your role is to TRANSLATE the plan into working code, NOT to reinterpret or override it
- If the plan appears suboptimal or incorrect, implement it anyway - the Controller will handle corrections
- If implementation is technically impossible, raise a clear error rather than silently changing the approach
- ⚠️ ERROR VISIBILITY: When wrapping tool calls in try/except, ALWAYS print the
  full error message: `except Exception as e: print(f"ERROR in {tool_name}: {e}"); import traceback; traceback.print_exc()`
  Never silently swallow exceptions — errors must be visible in stdout for debugging.
- When a foundation model is used for feature extraction, the classifier input
  should be the model's output features only (not augmented with external metadata).
- Be mindful of the feature-to-sample ratio and class imbalance when choosing
  classifiers and preprocessing steps.
- Ensure any preprocessing (scaling, dimensionality reduction) is fitted inside
  cross-validation folds to prevent data leakage.

- ⚠️ PERFORMANCE — AVOID NESTED PYTHON LOOPS:
  Do NOT use nested Python for-loops to call scipy/numpy functions one element at a time.
  Instead, use VECTORIZED computation: pass multi-dimensional arrays to scipy/numpy
  functions and use the `axis` parameter. Nested Python loops on large arrays are
  10-100x slower than equivalent vectorized operations.


PRE-LOADED LIBRARIES (available without import):
- `np` / `numpy`: NumPy for numerical operations
- `pd` / `pandas`: Pandas for DataFrames
- `scipy`: SciPy (use `scipy.signal.welch(...)` or `from scipy.signal import welch`)
- `plt` / `matplotlib`: Matplotlib for plotting (Agg backend, non-interactive)
- `sns`: Seaborn for visualization
- `StandardScaler`, `LabelEncoder`: sklearn preprocessing
- `Pipeline`: sklearn.pipeline.Pipeline (for wrapping scaler + classifier)
- `train_test_split`, `cross_val_score`, `StratifiedKFold`, `LeaveOneOut`: sklearn model selection
- `LogisticRegression`, `SVC`: sklearn classifiers
- `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`: sklearn metrics

For any class/function NOT listed above, use standard import statements:
  e.g. `from sklearn.model_selection import RepeatedStratifiedKFold`
  e.g. `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier`
Allowed packages: numpy, pandas, scipy, matplotlib, seaborn, sklearn, xgboost, lightgbm, mne, and Python stdlib.
- `tools`: Module with generic analysis functions (see TOOL CATEGORIES below)
- `os`, `glob`: File system operations (os.path.join, glob.glob, os.walk)
- `re`: Regular expressions (re.search, re.match, etc.)
- `mne`: MNE-Python for EEG data loading (mne.io.read_raw_eeglab, etc.)

AVAILABLE VARIABLES:
- `raw_data`: Dict mapping filename -> pandas DataFrame (or numpy arrays for .mat/.set files)
- `tools`: Module with analysis functions (use tools.function_name())
- `OUTPUT_DIR`: String path to the output directory for saving figures
- `DATASET_ROOT_DIR`: String path to the extracted dataset root directory.
  Use this to discover and load additional files (e.g., iterate over data files,
  read metadata files, etc.) via os.walk() or glob.glob().
- `save_figure(fig, name)`: Helper to save a matplotlib figure and close it
- `SHARED_CACHE`: Dict that persists across retry attempts. Use it to cache expensive
  operations (data loading, feature extraction) so they are not repeated on retries.
  IMPORTANT: Use descriptive cache keys that reflect the feature extraction method,
  so that different approaches use different cache entries and do not reuse stale data.
  Pattern:
    if 'bandpower_X' not in SHARED_CACHE:
        # ... expensive loading and feature extraction ...
        SHARED_CACHE['bandpower_X'] = X
        SHARED_CACHE['bandpower_y'] = y
    else:
        X = SHARED_CACHE['bandpower_X']
        y = SHARED_CACHE['bandpower_y']
  ALWAYS check SHARED_CACHE first before loading data from disk.

⚠️ PREFER tools.decode_*() OVER HAND-WRITTEN TRAINING LOOPS:
  tools.decode_logistic_regression() and tools.decode_svm() already handle StandardScaler,
  StratifiedKFold CV (or GroupKFold when groups are provided), class_weight='balanced',
  and FULL metric computation internally. They return a dict with:
    cv_accuracy_mean, cv_balanced_accuracy_mean, cv_f1_macro_mean, cv_auroc_macro_mean,
    per_class_recall, confusion_matrix, n_samples, n_features, n_classes, class_labels.

  ALWAYS prefer calling tools.decode_*() instead of writing your own sklearn training loop.
  Your job is to prepare the feature matrix X and label vector y, then call:
    result_lr = tools.decode_logistic_regression(X, y)
    result_svm = tools.decode_svm(X, y)
  USE THEIR RETURN VALUES DIRECTLY for reporting metrics. Do NOT re-train.

  When using window-level features (multiple rows per subject), pass the groups parameter
  with subject IDs so the tool uses GroupKFold (no subject leakage across folds):
    result_lr = tools.decode_logistic_regression(X_windows, y_windows, groups=subject_ids)
    result_svm = tools.decode_svm(X_windows, y_windows, groups=subject_ids)

  When using subject-level features (one row per subject), omit groups — the tool
  uses StratifiedKFold automatically. This is simpler and faster.

  When dimensionality reduction is needed, use the pca_components parameter instead of
  applying PCA externally. PCA is fitted inside each CV fold to prevent data leakage:
    result = tools.decode_logistic_regression(X, y, pca_components=20)      # fixed 20 components
    result = tools.decode_svm(X, y, pca_components=0.95)                    # retain 95% variance
  DO NOT apply PCA or other dimensionality reduction outside of tools.decode_*() and then
  pass the transformed features — this causes CV data leakage.

  DO NOT normalize or standardize features before passing them to tools.decode_*().
  The tools already apply StandardScaler internally. Pre-normalizing (e.g., z-scoring
  per channel, dividing by total power) destroys discriminative information and leads
  to near-chance classification.

  When comparing multiple hyperparameter values, use C_values instead of manual loops:
    result = tools.decode_logistic_regression(X, y, C_values=[...])
    result = tools.decode_svm(X, y, C_values=[...])
  Pass a list of C values to search over. The tool runs CV for each and returns the best result.

  For more stable metric estimates, use cv_repeats to run multiple repetitions of
  cross-validation with different random shuffles:
    result = tools.decode_logistic_regression(X, y, cv_repeats=3)  # 3x5-fold = 15 total folds
    result = tools.decode_xgboost(X, y, cv_repeats=3)
  This reduces evaluation noise without changing the model, especially useful when
  n_samples < 200.

  All tools.decode_*() return y_true and y_pred lists (out-of-fold predictions).
  Use these directly for plotting confusion matrices instead of reconstructing from counts.

AVAILABLE TOOLS (all called via tools.function_name()):
{TOOL_SIGNATURES}

DATA-DRIVEN FILTERING (MANDATORY):
Before applying any equality or membership filter on a column, ALWAYS check
the actual values in the data first. If the target value does not exist,
skip that filter condition and print a warning. Example pattern:

  if col in df.columns and target_val in df[col].values:
      df = df[df[col] == target_val]
  else:
      print(f"Warning: column '{col}' has values {df[col].unique().tolist()}, "
            f"skipping filter {col}=={target_val}")

This prevents zero-row results from impossible filter conditions (e.g.,
filtering stim_present==1 when the column is always 0). NEVER assume a
flag or category column contains a specific value without verifying it
in the actual data first.

MULTI-STEP ANALYSIS RULES:
When the task involves multiple analysis steps:
- Use intermediate Python variables to pass data between steps
  (e.g., step1_result = tools.pca_decomposition(X); X_pca = step1_result["transformed"])
- Collect all step results into the final `result` dict:
  result = {"step1": step1_result, "step2": step2_result, ...}
- If a step produces a figure, use save_figure() or pass save_path=OUTPUT_DIR + "/name.png"
- When comparing multiple models or pipelines, choose the selection criterion
  that best fits the data characteristics (e.g., consider class balance when
  deciding which metric to prioritize for model selection).

AUROC COMPUTATION (IMPORTANT — sklearn roc_auc_score usage):
When computing AUROC with roc_auc_score:
- Binary classification (2 classes): pass the POSITIVE CLASS probability only:
    roc_auc_score(y_true, y_proba[:, 1])
  Do NOT pass the full (n, 2) probability matrix — it will silently fail or error.
- Multiclass (3+ classes): pass the full probability matrix with multi_class='ovr':
    roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
Always wrap in try/except and fall back to np.nan on failure.

========================================================================
NDT3 FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ NDT3 (tools.decode_with_ndt3) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_ndt3", you MUST call tools.decode_with_ndt3()
⚠️ Never add comments saying "NDT3 unavailable" or substitute with other tools

NDT3 ONLY does feature extraction. YOU must handle classification and metrics.

tools.decode_with_ndt3(spike_data) returns:
  {'features': np.ndarray (n_trials, hidden_dim), 'n_trials': int, 'hidden_dim': int}

WORKFLOW:
1. Use schema-driven column mapping to prepare data:
   mapper = SchemaColumnMapper(parsed_schema)  # Both are pre-loaded, use directly
   spike_time_col = mapper.get_column_by_role('spike_time', file_name='...')
   unit_id_col = mapper.get_column_by_role('unit_id', file_name='...')
2. Bin spikes into (n_trials, n_time_bins, n_neurons) array
3. Call: ndt3_result = tools.decode_with_ndt3(spike_data)
4. Get features: features = ndt3_result['features']
5. Train YOUR OWN classifier on features (sklearn, etc.)
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

⚠️ Do NOT redefine parsed_schema or SchemaColumnMapper - they exist in the environment.
⚠️ NEVER hardcode field names - always use mapper.get_column_by_role().

========================================================================
MtM FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ MtM (tools.decode_with_mtm) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_mtm", you MUST call tools.decode_with_mtm()
⚠️ Never add comments saying "MtM unavailable" or substitute with other tools

MtM ONLY does feature extraction. YOU must handle classification and metrics.

tools.decode_with_mtm(spike_data) returns:
  {'features': np.ndarray (n_trials, hidden_dim), 'n_trials': int, 'hidden_dim': int}

IMPORTANT: MtM expects 20ms bins and up to 100 time bins (2s window).
Use bin_size=0.02 when binning spikes for MtM.

WORKFLOW (identical to NDT3):
1. Use schema-driven column mapping to prepare data:
   mapper = SchemaColumnMapper(parsed_schema)
   spike_time_col = mapper.get_column_by_role('spike_time', file_name='...')
   unit_id_col = mapper.get_column_by_role('unit_id', file_name='...')
2. Bin spikes into (n_trials, n_time_bins, n_neurons) array (bin_size=0.02)
3. Call: mtm_result = tools.decode_with_mtm(spike_data)
4. Get features: features = mtm_result['features']
5. Train YOUR OWN classifier on features (sklearn, etc.)
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

⚠️ Do NOT redefine parsed_schema or SchemaColumnMapper - they exist in the environment.
⚠️ NEVER hardcode field names - always use mapper.get_column_by_role().

========================================================================
POYO-1 FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ POYO-1 (tools.decode_with_poyo) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_poyo", you MUST call tools.decode_with_poyo()
⚠️ Never add comments saying "POYO unavailable" or substitute with other tools

POYO-1 ONLY does feature extraction. YOU must handle classification and metrics.

tools.decode_with_poyo(spike_data) returns:
  {'features': np.ndarray (n_trials, hidden_dim), 'n_trials': int, 'hidden_dim': int}

IMPORTANT: POYO-1 accepts any bin size. Default recommendation is 20ms bins.
POYO-1 handles variable neuron counts (up to 1000 neurons) natively.

WORKFLOW (identical to NDT3/MtM):
1. Use schema-driven column mapping to prepare data:
   mapper = SchemaColumnMapper(parsed_schema)
   spike_time_col = mapper.get_column_by_role('spike_time', file_name='...')
   unit_id_col = mapper.get_column_by_role('unit_id', file_name='...')
2. Bin spikes into (n_trials, n_time_bins, n_neurons) array
3. Call: poyo_result = tools.decode_with_poyo(spike_data)
4. Get features: features = poyo_result['features']
5. Train YOUR OWN classifier on features (sklearn, etc.)
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

⚠️ Do NOT redefine parsed_schema or SchemaColumnMapper - they exist in the environment.
⚠️ NEVER hardcode field names - always use mapper.get_column_by_role().

========================================================================
LaBraM FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ LaBraM (tools.decode_with_labram) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_labram", you MUST call tools.decode_with_labram()
⚠️ Never substitute with manual bandpower features when LaBraM is specified

LaBraM ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_labram(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 200), 'n_subjects': int, 'embed_dim': int}

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 200Hz)
- ch_names: list of channel names in standard 10-20 notation (e.g., ['FP1', 'FP2', 'F3', ...])
- Channel names must be UPPERCASE (e.g., 'FP1' not 'Fp1')
- Pass the FULL recording per subject — the tool internally splits it into segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 200-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 200) — each row is one subject's aggregated embedding.
- Classification should be done at the SUBJECT level, not at the segment level.
- Do NOT create segment-level samples or expand the dataset — use the subject-level features directly.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = [ch.upper() for ch in common_channels]  # LaBraM needs uppercase
3. Call: labram_result = tools.decode_with_labram(eeg_list, sfreq, ch_names)
4. Get features: X = labram_result['features']  # (n_subjects, 200) — one row per subject
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

========================================================================
REVE FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ REVE (tools.decode_with_reve) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_reve", you MUST call tools.decode_with_reve()
⚠️ Never substitute with manual bandpower features when REVE is specified

REVE ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_reve(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 1536), 'n_subjects': int, 'embed_dim': int}
  Features are 1536-dim: 512 mean + 512 std + 512 max pooled across channels and time patches.

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 200Hz)
- ch_names: list of channel names in standard notation (e.g., ['Fp1', 'Fp2', 'F3', ...])
- Channel names can be any case — REVE handles them via its position bank
- Pass the FULL recording per subject — the tool internally splits it into segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 1536-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 1536) — each row is one subject's aggregated embedding (mean+std+max).
- Classification should be done at the SUBJECT level, not at the segment level.
- Do NOT create segment-level samples or expand the dataset — use the subject-level features directly.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = common_channels  # REVE accepts any case
3. Call: reve_result = tools.decode_with_reve(eeg_list, sfreq, ch_names)
4. Get features: X = reve_result['features']  # (n_subjects, 1536) — one row per subject
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

========================================================================
CBraMod FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ CBraMod (tools.decode_with_cbramod) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_cbramod", you MUST call tools.decode_with_cbramod()
⚠️ Never substitute with manual bandpower features when CBraMod is specified

CBraMod ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_cbramod(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 600), 'n_subjects': int, 'embed_dim': int}
  Features are 600-dim: 200 mean + 200 std + 200 max pooled across channels and time patches.

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 200Hz)
- ch_names: list of channel names (any naming convention — CBraMod is channel-agnostic)
- Pass the FULL recording per subject — the tool internally splits it into segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 600-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 600) — each row is one subject's aggregated embedding (mean+std+max).
- Classification should be done at the SUBJECT level, not at the segment level.
- Do NOT create segment-level samples or expand the dataset — use the subject-level features directly.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = common_channels  # CBraMod accepts any naming convention
3. Call: cbramod_result = tools.decode_with_cbramod(eeg_list, sfreq, ch_names)
4. Get features: X = cbramod_result['features']  # (n_subjects, 600) — one row per subject
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

========================================================================
EEGPT FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ EEGPT (tools.decode_with_eegpt) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_eegpt", you MUST call tools.decode_with_eegpt()
⚠️ Never substitute with manual bandpower features when EEGPT is specified

EEGPT ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_eegpt(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 2048), 'n_subjects': int, 'embed_dim': int}
  Features are 2048-dim: 4 summary tokens x 512-dim embedding, mean-pooled across time patches.

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 256Hz)
- ch_names: list of channel names in standard 10-10 or 10-20 naming (e.g., Fp1, F3, C3, P3, O1)
  Old channel names (T3, T4, T5, T6) are auto-mapped to modern equivalents (T7, T8, P7, P8).
  Only channels matching EEGPT's 62-channel 10-10 dictionary will be used.
- Pass the FULL recording per subject — the tool internally splits it into 4s segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 2048-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 2048) — each row is one subject's aggregated embedding.
- Classification should be done at the SUBJECT level, not at the segment level.
- Do NOT create segment-level samples or expand the dataset — use the subject-level features directly.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = common_channels  # EEGPT auto-maps standard 10-10/10-20 channel names
3. Call: eegpt_result = tools.decode_with_eegpt(eeg_list, sfreq, ch_names)
4. Get features: X = eegpt_result['features']  # (n_subjects, 2048) — one row per subject
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

========================================================================
EEGMamba FOUNDATION MODEL USAGE (MANDATORY WHEN SPECIFIED IN PLAN)
========================================================================

⚠️ EEGMamba (tools.decode_with_eegmamba) IS FULLY AVAILABLE - DO NOT SKIP IT
⚠️ If the plan says "decode_with_eegmamba", you MUST call tools.decode_with_eegmamba()
⚠️ Never substitute with manual bandpower features when EEGMamba is specified

EEGMamba ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_eegmamba(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 600), 'n_subjects': int, 'embed_dim': int}
  Features are 600-dim: 200 mean + 200 std + 200 max pooled across channels and time patches.

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 200Hz)
- ch_names: list of channel names (any naming convention — EEGMamba is channel-agnostic)
- Pass the FULL recording per subject — the tool internally splits it into segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 600-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 600) — each row is one subject's aggregated embedding (mean+std+max).
- Classification should be done at the SUBJECT level, not at the segment level.
- Do NOT create segment-level samples or expand the dataset — use the subject-level features directly.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = common_channels  # EEGMamba accepts any naming convention
3. Call: eegmamba_result = tools.decode_with_eegmamba(eeg_list, sfreq, ch_names)
4. Get features: X = eegmamba_result['features']  # (n_subjects, 600) — one row per subject
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested from y_true, y_pred, y_proba

========================================================================
BRAINOMNI FOUNDATION MODEL USAGE (decode_with_brainomni)
========================================================================
BrainOmni is a Criss-Cross Transformer with Sensor Encoder pretrained on ~2,650
hours of EEG/MEG data. It produces 4096-dim features via frozen feature extraction.

BrainOmni ONLY does feature extraction from EEG. YOU must handle classification and metrics.

tools.decode_with_brainomni(eeg_data, sfreq, ch_names) returns:
  {'features': np.ndarray (n_subjects, 4096), 'n_subjects': int, 'embed_dim': int}
  Features are 4096-dim: 16 latent sources × 256 dims, mean-pooled over time.

IMPORTANT:
- eeg_data must be a LIST of (n_channels, n_samples) arrays, one per subject
- sfreq: sampling frequency in Hz (any value, auto-resampled to 256Hz)
- ch_names: list of channel names (STANDARD names like 'Fp1', 'Cz' — positions
  are obtained from MNE standard montages; non-standard names will use fallback positions)
- Pass the FULL recording per subject — the tool internally splits it into segments,
  extracts features per segment, and averages them into ONE feature vector per subject.
  Do NOT manually segment the EEG before calling this tool.

OUTPUT:
- Returns ONE 4096-dim feature vector PER SUBJECT (not per segment).
- features shape: (n_subjects, 4096) — each row is one subject's aggregated embedding.
- Classification should be done at the SUBJECT level, not at the segment level.

WORKFLOW:
1. Discover all .set files: set_files = glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
2. TWO-PASS loading (handles heterogeneous channels across subjects):
   # PASS 1: find common EEG channels (preload=False saves memory)
   common_channels = None
   sfreq = None
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
       eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
       sfreq = raw.info['sfreq']
       if common_channels is None:
           common_channels = eeg_chs
       else:
           common_channels = [ch for ch in common_channels if ch in eeg_chs]
   # PASS 2: load data with common channels only
   eeg_list = []
   for path in sorted(set_files):
       raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
       raw.pick(common_channels)
       eeg_list.append(raw.get_data())
   ch_names = common_channels
3. Call: brainomni_result = tools.decode_with_brainomni(eeg_list, sfreq, ch_names)
4. Get features: X = brainomni_result['features']  # (n_subjects, 4096)
5. Train your own classifier on X with subject-level labels
6. Compute whatever metrics the user requested

========================================================================
MULTI-FILE EEG DATASET HANDLING (e.g., EEGLAB .set files)
========================================================================

When the dataset contains multiple EEG files (e.g., one .set file per subject):
- Use `DATASET_ROOT_DIR` to discover all files: glob.glob(os.path.join(DATASET_ROOT_DIR, '**/*.set'), recursive=True)
- Load each .set file with mne: raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
- Get EEG data: data = raw.get_data()  # shape (n_channels, n_samples)
- Get metadata: sfreq = raw.info['sfreq'], ch_names = raw.ch_names
- Look for metadata files (.tsv, .csv) in the dataset directory to find subject labels. Check the data summary below for available metadata files and their column names.
- Extract subject ID from file path using regex patterns visible in the filenames (check data summary for examples).
- IMPORTANT: For large datasets, extract features per-subject individually rather than concatenating all raw data into memory. Do NOT vstack/concatenate all raw time-series across subjects before feature extraction — this causes memory overflow. Process each subject's data independently, extract a fixed-size feature vector, then stack the feature vectors.
- IMPORTANT: All code runs in a SINGLE execution block. Define ALL helper functions in the same block where you use them. Do NOT assume functions from previous attempts exist.

⚠️ EEG CHANNEL FILTERING — USE MNE CHANNEL TYPES (NOT HARDCODED NAMES):
  Datasets may contain non-EEG channels (Resp, ECG, EMG, accelerometer X/Y/Z, etc.)
  mixed with EEG channels. Different subjects may have different channel counts.
  ALWAYS filter to EEG-only channels using MNE's built-in channel type detection:
    eeg_ch_names = [name for name, ch_type in zip(raw.ch_names, raw.get_channel_types())
                    if ch_type == 'eeg']
  Do NOT rely on hardcoded non-EEG channel name lists — MNE already knows the types.

⚠️ HETEROGENEOUS CHANNEL HANDLING — TWO-PASS APPROACH:
  When subjects may have different channel configurations, use a two-pass approach:
  PASS 1 (channel discovery, preload=False to save memory):
    common_channels = None
    for path in sorted(set_files):
        raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
        eeg_chs = [name for name, ct in zip(raw.ch_names, raw.get_channel_types()) if ct == 'eeg']
        if common_channels is None:
            common_channels = eeg_chs
        else:
            common_channels = [ch for ch in common_channels if ch in eeg_chs]
  PASS 2 (load data with only common channels):
    for path in sorted(set_files):
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        raw.pick(common_channels)
        data = raw.get_data()  # now all subjects have identical channel count and order
  This ensures consistent feature dimensions across all subjects, even when some subjects
  have extra non-EEG channels (e.g., Resp, ECG) or different EEG montages.
  NOTE: For foundation model tools (decode_with_brainomni, decode_with_eegpt, etc.),
  pass the common_channels as ch_names so the model receives consistent channel info.

CRITICAL RULES FOR SANDBOX:
- All libraries are pre-loaded. NEVER use `import` or `from X import Y` — these statements are stripped and will cause NameError.
- Use `scipy.signal.welch(...)` directly, NOT `from scipy.signal import welch`.
- Use `re.search(...)` directly, NOT `import re`.
- Use `mne.io.read_raw_eeglab(...)` directly, NOT `import mne`.

OUTPUT FORMAT:
```python
# Your code here (NO imports!)
result = ...
```

Do not include markdown formatting or import statements."""


def _get_code_generation_system_prompt() -> str:
    """Build the system prompt with dynamically generated tool signatures."""
    try:
        tool_sigs = get_all_tool_signatures()
    except Exception:
        tool_sigs = "  (Tool signatures unavailable — refer to tool documentation)"
    return _CODE_GENERATION_SYSTEM_PROMPT_TEMPLATE.replace("{TOOL_SIGNATURES}", tool_sigs)


# Lazy-evaluated property for backward compatibility
CODE_GENERATION_SYSTEM_PROMPT = _get_code_generation_system_prompt()


CODE_GENERATION_USER_PROMPT = """Generate Python code for this analysis task.

## Task
{task_description}

## Dataset Schema
{schema_description}

## Data Summary
{data_summary}

## Available Data Keys
The `raw_data` dict contains these keys (use EXACTLY one of these):
{available_data_keys}
{target_tool_section}
## Available Tools
{available_tools}

Generate the complete Python code:"""


EXTRACTION_ONLY_PROMPT = """Generate Python code to extract and prepare data.

## Dataset Schema
{schema_description}

## Data Summary  
{data_summary}

## Required Output
- X: Feature matrix [n_samples, n_features] - {feature_description}
- y: Labels array [n_samples] - {label_description}

## Extraction Requirements
{extraction_requirements}

Generate code that creates X and y variables:"""


# =============================================================================
# Code Generator Class
# =============================================================================

class CodeGenerator:
    """
    LLM-based code generator for data extraction and tool invocation.
    
    Workflow:
    1. Receive task + schema + raw data summary
    2. Generate data extraction code
    3. Generate tool invocation code
    4. Return complete executable code
    """
    
    def __init__(self, model: Optional[str] = None, llm_client: Optional[object] = None):
        """
        Initialize code generator.
        
        Args:
            model: LLM model name. Defaults to CODE_GENERATOR_MODEL from config.
            llm_client: Deprecated – ignored. Kept for backward compatibility.
        """
        self.model = model or CODE_GENERATOR_MODEL

    def _call_llm(self, *, system_prompt: str, user_prompt: str,
                  temperature: float = None, max_tokens: int = 8000) -> str:
        """Call the unified LLM client (text mode, no JSON) and return raw text."""
        if temperature is None:
            temperature = float(os.getenv("CODE_GENERATOR_TEMPERATURE", "0.2"))
        resp: LLMResponse = _llm_generate_text(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=120.0,
            retries=2,
        )
        return resp.raw_text
    
    def generate_analysis_code(
        self,
        task_description: str,
        parsed_schema: Optional[ParsedSchema],
        data_summary: str,
        tool_name: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        available_data_keys: Optional[List[str]] = None,
        previous_code: Optional[str] = None,
        user_instructions: Optional[str] = None,
    ) -> GeneratedCode:
        """
        Generate complete analysis code.

        Args:
            task_description: What analysis to perform
            parsed_schema: Parsed schema from schema_parser (can be None)
            data_summary: Summary of raw data (column stats, samples)
            tool_name: Name of target tool, or None for multi-tool / free-form tasks
            tool_parameters: Parameters to pass to tool (cv_folds, seed, etc.)
            available_data_keys: Actual keys in the raw_data dict passed to the
                executor.  If provided these are injected into the prompt so the
                LLM uses the correct keys.
            user_instructions: Original user prompt, so the Code Generator can
                see metric/output requirements that Planner may have omitted.

        Returns:
            GeneratedCode with executable Python code
        """
        if tool_parameters is None:
            tool_parameters = {}

        # Build schema description (handle None case)
        if parsed_schema is not None:
            schema_desc = self._build_schema_description(parsed_schema)
        else:
            schema_desc = "No formal schema provided. Infer data structure from data_summary below."

        # Build the "Target Tool" section — omitted when tool_name is None
        if tool_name:
            from neuro_copilot.core.tools.generic_tools import get_tool_signature
            try:
                tool_sig = get_tool_signature(tool_name)
            except KeyError:
                tool_sig = f"{tool_name}(X, y, ...)"
            target_tool_section = (
                f"## Target Tool\n"
                f"{tool_name}({tool_sig})\n\n"
                f"## Tool Parameters\n"
                f"{tool_parameters}\n"
            )
        else:
            target_tool_section = (
                "## Target Tool\n"
                "Multiple tools may be needed. Choose from the Available Tools "
                "list below based on the task description.\n"
            )

        # Build prompt
        if available_data_keys:
            keys_text = "\n".join(f"  - '{k}'" for k in available_data_keys)
        else:
            keys_text = "(not specified — check data_summary for file names)"

        prompt = CODE_GENERATION_USER_PROMPT.format(
            task_description=task_description,
            schema_description=schema_desc,
            data_summary=data_summary,
            target_tool_section=target_tool_section,
            available_tools=get_all_tool_signatures(),
            available_data_keys=keys_text,
        )

        # Pass through the user's original instructions so metric/output
        # requirements are not lost during Planner summarization.
        if user_instructions:
            prompt += (
                "\n\n## Original User Instructions\n"
                "The following is the user's original request. Pay special "
                "attention to any metric output requirements listed here.\n\n"
                + user_instructions
            )

        # Memory module disabled per advisor decision
        # try:
        #     from neuro_copilot.core.memory.retriever import get_memory_section
        #     ...
        #     memory_section = get_memory_section(modality=modality, model=tool_name)
        #     if memory_section:
        #         prompt += "\n\n" + memory_section
        # except Exception:
        #     pass

        # If previous code is provided (improvement retry), append it so the
        # LLM can refine rather than rewrite from scratch.
        if previous_code:
            prompt += (
                "\n\n## Previous Working Code\n"
                "The following code ran successfully in the previous attempt. "
                "Make MINIMAL changes to improve it based on the task description above.\n"
                "- NEVER rewrite from scratch.\n"
                "- NEVER change data loading, file discovery, label mapping, or subject alignment code — copy these sections VERBATIM.\n"
                "- ONLY change classifier/evaluation parameters.\n"
                "- Do NOT add validation code, sanity checks, or defensive assertions.\n"
                f"```python\n{previous_code}\n```"
            )

        # Call LLM
        raw_text = self._call_llm(
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=8000,
        )

        # Parse response and attach prompt info
        result = self._parse_code_response_from_text(raw_text, tool_name)
        result.prompt = prompt
        result.system_prompt = CODE_GENERATION_SYSTEM_PROMPT
        return result
    
    def generate_extraction_code(
        self,
        parsed_schema: ParsedSchema,
        data_summary: str,
        feature_description: str,
        label_description: str,
        extraction_requirements: str = "",
    ) -> GeneratedCode:
        """
        Generate only data extraction code (without tool call).
        
        Useful when you want to extract data once, then call multiple tools.
        
        Args:
            parsed_schema: Parsed schema
            data_summary: Data summary
            feature_description: What features to extract (e.g., "firing rates per neuron per trial")
            label_description: What labels to extract (e.g., "trial type")
            extraction_requirements: Additional requirements
            
        Returns:
            GeneratedCode with extraction code only
        """
        schema_desc = self._build_schema_description(parsed_schema)
        
        prompt = EXTRACTION_ONLY_PROMPT.format(
            schema_description=schema_desc,
            data_summary=data_summary,
            feature_description=feature_description,
            label_description=label_description,
            extraction_requirements=extraction_requirements,
        )
        
        raw_text = self._call_llm(
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=2000,
        )
        
        return self._parse_code_response_from_text(raw_text, None)
    
    def generate_from_plan(
        self,
        plan: Dict[str, Any],
        parsed_schema: Optional[ParsedSchema],
        data_summary: str,
        available_data_keys: Optional[List[str]] = None,
        previous_code: Optional[str] = None,
        user_instructions: Optional[str] = None,
    ) -> GeneratedCode:
        """
        Generate code from a structured plan (from Planner agent).

        Supports both single-step and multi-step plans.  For multi-step
        plans the steps are merged into one comprehensive task description
        and GPT-4 generates a single block of code that chains all steps
        via intermediate Python variables.

        Args:
            plan: Plan dict with 'steps', each step has 'tool'/'op_name'
                  and 'params'/'parameters'
            parsed_schema: Parsed schema (can be None)
            data_summary: Data summary
            available_data_keys: Actual keys in raw_data (passed to prompt)
            user_instructions: Original user prompt (for metric requirements)

        Returns:
            GeneratedCode for entire plan
        """
        steps = plan.get("steps", [])
        if not steps:
            raise ValueError("Plan has no steps")

        # Build primary_metric instruction from the Planner's runtime decision
        _pm = plan.get("primary_metric")
        _pm_instruction = ""
        if _pm:
            _pm_instruction = (
                f"\n\nIMPORTANT: The primary evaluation metric is '{_pm}'. "
                f"You MUST store this metric's value under the EXACT key '{_pm}' "
                f"in the result dict (e.g., result['{_pm}'] = <value>)."
            )

        # Build required_metrics instruction from the Planner's extraction
        _req_metrics = plan.get("required_metrics") or []
        _req_instruction = ""
        if _req_metrics:
            _metric_list = ", ".join(_req_metrics)
            _req_instruction = (
                f"\n\nREQUIRED OUTPUT METRICS: You MUST compute and store ALL of "
                f"the following metrics in the result dict: {_metric_list}. "
                f"Do not omit any of these metrics from the output."
            )

        _extra_instructions = _pm_instruction + _req_instruction

        if len(steps) == 1:
            # ---- single-step (existing fast path) ----
            step = steps[0]
            tool_name = step.get("tool", step.get("op_name", ""))
            params = step.get("params", step.get("parameters", {}))
            task_desc = step.get("description", f"Run {tool_name}") + _extra_instructions

            return self.generate_analysis_code(
                task_description=task_desc,
                parsed_schema=parsed_schema,
                data_summary=data_summary,
                tool_name=tool_name,
                tool_parameters=params,
                available_data_keys=available_data_keys,
                previous_code=previous_code,
                user_instructions=user_instructions,
            )

        # ---- multi-step plan ----
        task_desc = self._build_multi_step_task_description(steps, plan) + _extra_instructions

        return self.generate_analysis_code(
            task_description=task_desc,
            parsed_schema=parsed_schema,
            data_summary=data_summary,
            tool_name=None,  # multiple tools — no single target
            tool_parameters={},
            available_data_keys=available_data_keys,
            previous_code=previous_code,
            user_instructions=user_instructions,
        )

    # ------------------------------------------------------------------
    # Helpers for multi-step plan generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_multi_step_task_description(
        steps: List[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> str:
        """
        Convert a list of plan steps into a combined natural-language task
        description that GPT-4 can translate into a single code block.
        """
        task_type = plan.get("task_type", "analysis")
        lines = [
            f"Perform a multi-step {task_type} analysis with {len(steps)} steps.",
            "Chain the steps so that each step can use results from previous steps.",
            "Collect ALL step results into a single `result` dict at the end.",
            "",
        ]

        for idx, step in enumerate(steps, 1):
            tool = step.get("tool", step.get("op_name", "unknown"))
            desc = step.get("description", f"Run {tool}")
            params = step.get("params", step.get("parameters", {}))

            lines.append(f"### Step {idx}: {desc}")
            lines.append(f"- Tool: tools.{tool}()")
            if params:
                # Only include non-field-name params that are tool config
                param_strs = [f"{k}={v!r}" for k, v in params.items()]
                lines.append(f"- Parameters: {', '.join(param_strs)}")
            lines.append("")

        lines.append(
            "Store the combined results as: "
            "result = {'step1': <step1_result>, 'step2': <step2_result>, ...}"
        )
        return "\n".join(lines)
    
    def regenerate_with_error(
        self,
        original_code: str,
        error_message: str,
        error_traceback: str,
        parsed_schema: Optional[ParsedSchema],
        data_summary: str,
        tool_name: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> GeneratedCode:
        """
        Regenerate code after an execution error (self-correction).

        Args:
            original_code: The code that failed
            error_message: Error message from execution
            error_traceback: Full traceback
            parsed_schema: Parsed schema (can be None)
            data_summary: Data summary
            tool_name: Target tool name
            task_description: Original task description from the plan

        Returns:
            GeneratedCode with corrected code
        """
        # Build schema description (handle None case)
        if parsed_schema is not None:
            schema_desc = self._build_schema_description(parsed_schema)
        else:
            schema_desc = "No formal schema provided. Infer data structure from data_summary."

        # Include task description if available so the LLM understands
        # the original goal, not just the failing code.
        task_section = ""
        if task_description:
            task_section = f"\n## Original Task\n{task_description}\n"

        # Include tool signatures so the LLM knows each tool's expected
        # input format (e.g. decode_with_ndt3 expects per-trial binned data).
        available_tools = get_all_tool_signatures()

        error_correction_prompt = f"""The previous code failed with an error. Fix the code.
{task_section}
## Previous Code
```python
{original_code}
```

## Error
{error_message}

## Traceback
{error_traceback}

## Dataset Schema
{schema_desc}

## Data Summary
{data_summary}

## Target Tool
{tool_name or "Choose the appropriate tool(s) from the available generic_tools"}

## Available Tools
{available_tools}

CRITICAL: DO NOT use import statements! All libraries (np, pd, scipy, sklearn classes) are pre-loaded.

Generate ONLY the corrected Python code. Common fixes:
- REMOVE all import statements (libraries are pre-loaded)
- Check column names match the schema exactly
- Ensure proper data types (numeric vs string)
- Handle missing values with dropna() or fillna()
- Verify array shapes match expected dimensions
- Use proper pandas indexing (loc/iloc)

Corrected code (NO imports):"""

        raw_text = self._call_llm(
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            user_prompt=error_correction_prompt,
            temperature=0.3,
            max_tokens=8000,
        )
        
        # Parse response and attach prompt info
        result = self._parse_code_response_from_text(raw_text, tool_name)
        result.prompt = error_correction_prompt
        result.system_prompt = CODE_GENERATION_SYSTEM_PROMPT
        return result

    def modify_existing_code(
        self,
        original_code: str,
        recommendations: list,
        metrics_summary: str,
        parsed_schema: Optional[ParsedSchema],
        data_summary: str,
        tool_name: Optional[str] = None,
    ) -> GeneratedCode:
        """
        Modify existing working code based on controller recommendations.

        Unlike generate_from_plan() which can rewrite from scratch, this method
        instructs the LLM to make targeted modifications to preserve working logic
        (data loading, feature extraction) and only change what the recommendations
        suggest (classifier params, PCA, class weighting, etc.).

        Args:
            original_code: The working code from the previous attempt
            recommendations: List of specific improvement recommendations from controller
            metrics_summary: Summary of current metrics for context
            parsed_schema: Parsed schema (can be None)
            data_summary: Data summary
            tool_name: Target tool name

        Returns:
            GeneratedCode with modified code
        """
        if parsed_schema is not None:
            schema_desc = self._build_schema_description(parsed_schema)
        else:
            schema_desc = "No formal schema provided. Infer data structure from data_summary."

        recommendations_text = "\n".join(f"- {r}" for r in recommendations)
        available_tools = get_all_tool_signatures()

        modify_prompt = f"""You are given working Python code and a list of specific improvements to apply.

## Current Working Code
```python
{original_code}
```

## Current Metrics
{metrics_summary}

## Recommended Improvements
{recommendations_text}

## Dataset Schema
{schema_desc}

## Data Summary
{data_summary}

## Available Tools
{available_tools}

## Instructions — MINIMAL CHANGES ONLY
Apply ONLY the recommended improvements. Make the SMALLEST possible changes.
- NEVER touch data loading, file discovery, label mapping, subject alignment, or feature extraction code. These sections are CORRECT — copy them verbatim.
- ONLY change classifier/evaluation parameters (e.g., arguments passed to tools.decode_* calls).
- Do NOT add validation code, sanity checks, logging, or defensive assertions.
- Do NOT restructure, reorder, or refactor any part of the code.
- Do NOT add imports — all libraries are pre-loaded.
- If a recommendation requires changing data loading or alignment, SKIP it.
- Output the complete code with your minimal changes applied.

Modified code (NO imports):"""

        raw_text = self._call_llm(
            system_prompt=CODE_GENERATION_SYSTEM_PROMPT,
            user_prompt=modify_prompt,
            temperature=0.2,
            max_tokens=8000,
        )

        result = self._parse_code_response_from_text(raw_text, tool_name)
        result.prompt = modify_prompt
        result.system_prompt = CODE_GENERATION_SYSTEM_PROMPT
        return result

    def _build_schema_description(self, parsed_schema: ParsedSchema) -> str:
        """Build human-readable schema description for prompt."""
        lines = []
        
        lines.append(f"Data type: {parsed_schema.data_type}")
        lines.append(f"Time unit: {parsed_schema.time_unit}")
        lines.append("")
        
        for fname, fschema in parsed_schema.files.items():
            lines.append(f"### {fname}")
            if fschema.row_semantics:
                lines.append(f"Row semantics: {fschema.row_semantics}")
            lines.append("Columns:")
            for col in fschema.columns:
                relative_str = " (relative time)" if col.is_relative_time else ""
                lines.append(f"  - {col.column_name}: {col.role.value} - {col.description}{relative_str}")
            lines.append("")
        
        # Add semantic shortcuts
        if parsed_schema.spike_time_column:
            lines.append(f"Spike time column: {parsed_schema.spike_time_column.column_name} (in {parsed_schema.spike_time_column.file_name})")
        if parsed_schema.unit_id_column:
            lines.append(f"Unit ID column: {parsed_schema.unit_id_column.column_name} (in {parsed_schema.unit_id_column.file_name})")
        if parsed_schema.trial_type_column:
            lines.append(f"Trial type column: {parsed_schema.trial_type_column.column_name} (in {parsed_schema.trial_type_column.file_name})")
        if parsed_schema.stimulus_onset_column:
            lines.append(f"Stimulus onset column: {parsed_schema.stimulus_onset_column.column_name} (in {parsed_schema.stimulus_onset_column.file_name})")
        
        return "\n".join(lines)
    
    def _parse_code_response_from_text(
        self,
        raw_content: str,
        tool_name: Optional[str],
    ) -> GeneratedCode:
        """Parse raw LLM text response into GeneratedCode."""
        
        # Extract code block
        code = self._extract_code_block(raw_content)
        
        # Extract imports from code
        imports = self._extract_imports(code)
        
        # Identify tool calls
        tool_calls = []
        if tool_name:
            tool_calls.append(tool_name)
        # Also scan for other tool calls
        from neuro_copilot.core.tools.generic_tools import list_tools
        for t in list_tools():
            if f"tools.{t}" in code or f"{t}(" in code:
                if t not in tool_calls:
                    tool_calls.append(t)
        
        # Extract explanation (text before code block)
        explanation = self._extract_explanation(raw_content)
        
        return GeneratedCode(
            code=code,
            explanation=explanation,
            imports=imports,
            tool_calls=tool_calls,
            confidence=0.8,  # Could be improved with LLM self-evaluation
            raw_response=raw_content,
        )
    
    def _extract_code_block(self, text: str) -> str:
        """Extract Python code block from LLM response."""
        # Try to find ```python ... ``` block
        pattern = r'```python\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try to find ``` ... ``` block
        pattern = r'```\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # No code block found, assume entire response is code
        # Remove any markdown-like lines
        lines = []
        for line in text.split('\n'):
            if line.strip().startswith('#') or not line.strip().startswith(('##', '**', '-')):
                lines.append(line)
        return '\n'.join(lines).strip()
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation text before code block."""
        # Find where code block starts
        idx = text.find('```')
        if idx > 0:
            return text[:idx].strip()
        return ""


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_analysis_code(
    task_description: str,
    parsed_schema: Optional[ParsedSchema],
    data_summary: str,
    tool_name: Optional[str] = None,
    tool_parameters: Optional[Dict[str, Any]] = None,
    llm_client: Optional[object] = None,
    model: Optional[str] = None,
) -> GeneratedCode:
    """
    Convenience function to generate analysis code.

    Args:
        task_description: What analysis to perform
        parsed_schema: Parsed schema from schema_parser (can be None)
        data_summary: Summary of raw data
        tool_name: Name of tool to call, or None for multi-tool tasks
        tool_parameters: Tool parameters
        llm_client: Deprecated – ignored.
        model: Optional model override. Defaults to CODE_GENERATOR_MODEL.

    Returns:
        GeneratedCode
    """
    generator = CodeGenerator(model=model)
    return generator.generate_analysis_code(
        task_description=task_description,
        parsed_schema=parsed_schema,
        data_summary=data_summary,
        tool_name=tool_name,
        tool_parameters=tool_parameters,
    )


def generate_code_from_plan(
    plan: Dict[str, Any],
    parsed_schema: ParsedSchema,
    data_summary: str,
    llm_client: Optional[object] = None,
    model: Optional[str] = None,
) -> GeneratedCode:
    """
    Generate code from analysis plan.
    
    Args:
        plan: Plan dict from Planner agent
        parsed_schema: Parsed schema
        data_summary: Data summary
        llm_client: Deprecated – ignored.
        model: Optional model override. Defaults to CODE_GENERATOR_MODEL.
        
    Returns:
        GeneratedCode
    """
    generator = CodeGenerator(model=model)
    return generator.generate_from_plan(
        plan=plan,
        parsed_schema=parsed_schema,
        data_summary=data_summary,
    )


# =============================================================================
# Code Templates for Common Patterns
# =============================================================================
#
# These templates serve as reference examples for the LLM and as documentation.
# All use generic placeholders ({field_name}) — NO dataset-specific field names.

SPIKE_RATE_EXTRACTION_TEMPLATE = '''
# Load data
spike_df = raw_data['{spike_file}']
trial_df = raw_data['{trial_file}']

# Extract spike times grouped by unit
spike_times_by_unit = spike_df.groupby('{unit_id_col}')['{spike_time_col}'].apply(list).tolist()

# Extract trial information
trial_starts = trial_df['{trial_start_col}'].values
trial_ends = trial_df['{trial_end_col}'].values
y = trial_df['{label_col}'].values

# Compute firing rates for each trial
n_trials = len(trial_starts)
n_units = len(spike_times_by_unit)
X = np.zeros((n_trials, n_units))

for t_idx in range(n_trials):
    t_start, t_end = trial_starts[t_idx], trial_ends[t_idx]
    duration = t_end - t_start
    for u_idx, unit_spikes in enumerate(spike_times_by_unit):
        spikes = np.array(unit_spikes)
        n_spikes = np.sum((spikes >= t_start) & (spikes < t_end))
        X[t_idx, u_idx] = n_spikes / duration if duration > 0 else 0

# Call decoding tool
result = tools.decode_logistic_regression(X, y, cv_folds={cv_folds}, seed={seed})
'''

SIMPLE_DECODE_TEMPLATE = '''
# Load data
df = raw_data['{data_file}']

# Extract features and labels
X = df[{feature_cols}].values
y = df['{label_col}'].values

# Call decoding tool
result = tools.{tool_name}(X, y, **{params})
'''

ENCODING_GLM_TEMPLATE = '''
# Load data
df = raw_data['{data_file}']

# Extract predictors and response
predictor_cols = {predictor_cols}  # list of column names
X = df[predictor_cols].values.astype(float)
y = df['{response_col}'].values.astype(float)

# Ensure non-negative response for Poisson model
y = np.clip(y, 0, None)

# Fit Poisson GLM encoding model
result = tools.encode_poisson_glm(X, y, alpha={alpha}, test_ratio={test_ratio})
'''

GAUSSIAN_TUNING_TEMPLATE = '''
# Load data
df = raw_data['{data_file}']

# Compute mean response per condition
grouped = df.groupby('{condition_col}')['{response_col}'].mean()
conditions = grouped.index.values.astype(float)
responses = grouped.values.astype(float)

# Fit Gaussian tuning curve
fit_result = tools.fit_gaussian_tuning(conditions, responses)

# Plot tuning curve
plot_result = tools.plot_tuning_curve(
    conditions, responses,
    save_path=OUTPUT_DIR + '/tuning_curve.png',
    title='Tuning Curve',
    xlabel='{condition_col}',
    ylabel='Mean {response_col}',
)

result = {{"fit": fit_result, "plot": plot_result}}
'''

PCA_ANALYSIS_TEMPLATE = '''
# Load data
df = raw_data['{data_file}']

# Extract feature matrix
feature_cols = {feature_cols}
X = df[feature_cols].values.astype(float)

# Handle missing values
X = np.nan_to_num(X, nan=0.0)

# Run PCA
pca_result = tools.pca_decomposition(X, n_components={n_components})

# Plot explained variance
plot_result = tools.plot_explained_variance(
    pca_result['explained_variance_ratio'],
    save_path=OUTPUT_DIR + '/pca_variance.png',
)

result = {{"pca": pca_result, "variance_plot": plot_result}}
'''

STATISTICAL_TEST_TEMPLATE = '''
# Load data
df = raw_data['{data_file}']

# Split into groups by condition
groups = []
for condition_value in df['{condition_col}'].unique():
    group_data = df[df['{condition_col}'] == condition_value]['{measure_col}'].dropna().values
    groups.append(group_data)

# Run statistical test
result = tools.statistical_test(groups, test='{test_type}')
'''

MULTI_STEP_DECODE_TEMPLATE = '''
# Step 1: Load and prepare data
spike_df = raw_data['{spike_file}']
trial_df = raw_data['{trial_file}']

# Extract spike times grouped by unit
spike_times_by_unit = spike_df.groupby('{unit_id_col}')['{spike_time_col}'].apply(list).tolist()

# Build trial windows and labels
trial_starts = trial_df['{trial_start_col}'].values
trial_ends = trial_df['{trial_end_col}'].values
y = trial_df['{label_col}'].values

# Compute firing rate features [n_trials, n_units]
trial_windows = list(zip(trial_starts, trial_ends))
X = tools.extract_trial_features(spike_times_by_unit, trial_windows, feature_type='rate')

# Step 2: PCA dimensionality reduction
pca_result = tools.pca_decomposition(X, n_components={n_components})
X_pca = pca_result['transformed']

# Step 3: Decode from PCA features
decode_result = tools.decode_logistic_regression(X_pca, y, cv_folds={cv_folds})

# Collect all results
result = {{
    'step1_features': {{'shape': X.shape, 'n_trials': len(y), 'n_units': X.shape[1]}},
    'step2_pca': {{
        'n_components': pca_result['n_components'],
        'cumulative_variance': pca_result['cumulative_variance'],
    }},
    'step3_decode': decode_result,
}}
'''

NDT3_DECODE_TEMPLATE = '''
# SCHEMA-DRIVEN NDT3 FEATURE EXTRACTION + DOWNSTREAM ANALYSIS
# NDT3 only extracts features. Classification/metrics are done here.

mapper = SchemaColumnMapper(parsed_schema)

# Get column names by semantic role
spike_cols = {{
    'spike_time': mapper.get_column_by_role('spike_time', file_name='{spike_file}'),
    'unit_id': mapper.get_column_by_role('unit_id', file_name='{spike_file}'),
}}
trial_cols = {{
    'trial_id': mapper.get_column_by_role('trial_id', file_name='{trial_file}'),
    'trial_start': mapper.get_column_by_role('trial_start', file_name='{trial_file}'),
    'trial_type': mapper.get_column_by_role('trial_type', file_name='{trial_file}'),
}}

alignment = mapper.get_alignment_strategy()
stimulus_col = alignment['event_col']
is_relative = alignment['is_relative']

spike_df = raw_data['{spike_file}']
trial_df = raw_data['{trial_file}']

# Bin spikes: (n_trials, n_time_bins, n_neurons)
bin_size_s = {bin_size_ms} / 1000.0
window_start, window_end = {window_start}, {window_end}
n_bins = int((window_end - window_start) / bin_size_s)
unit_ids = np.sort(spike_df[spike_cols['unit_id']].unique())
n_neurons = len(unit_ids)
n_trials = len(trial_df)

spike_array = np.zeros((n_trials, n_bins, n_neurons))
labels = []

for trial_idx, (_, trial) in enumerate(trial_df.iterrows()):
    if is_relative:
        align_time = trial[trial_cols['trial_start']] + trial[stimulus_col]
    else:
        align_time = trial[stimulus_col]
    t_start = align_time + window_start
    t_end = align_time + window_end
    time_edges = np.linspace(t_start, t_end, n_bins + 1)
    trial_spikes = spike_df[
        (spike_df[spike_cols['spike_time']] >= t_start) &
        (spike_df[spike_cols['spike_time']] < t_end)
    ]
    for neuron_idx, unit_id in enumerate(unit_ids):
        unit_spikes = trial_spikes[
            trial_spikes[spike_cols['unit_id']] == unit_id
        ][spike_cols['spike_time']].values
        if len(unit_spikes) > 0:
            counts, _ = np.histogram(unit_spikes, bins=time_edges)
            spike_array[trial_idx, :, neuron_idx] = counts
    labels.append(trial[trial_cols['trial_type']])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Step 1: NDT3 feature extraction
ndt3_result = tools.decode_with_ndt3(spike_data=spike_array)
features = ndt3_result['features']

# Step 2: Downstream classification (generated code handles this)
# ... classifier training, metrics computation, etc.
'''
