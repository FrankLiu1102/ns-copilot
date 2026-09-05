# neuro_copilot/dataloader.py

import os
import tempfile
import tarfile
import zipfile
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Iterable

import numpy as np
import pandas as pd

from scipy.io import loadmat
import h5py

from neuro_copilot.core.state import NeuroGlobalState, UserDataState, DatasetInfo
from neuro_copilot.core.tools.preprocessing import normalize_preprocess_rules

import random
from collections import defaultdict

class DatasetLoadError(Exception):
    pass


@dataclass
class NeuralDataset:
    """
    Unified internal dataset object for neuron science data.

    raw: stores all variables loaded from the .mat file (no discarding).
    meta: optional extracted metadata (can be extended later).
    var_stats: lightweight per-variable structural stats (shape/dtype/min/max/mean/std when applicable).
    """
    path: str
    raw: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    var_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# -----------------------------
# MAT loading helpers
# -----------------------------

def _mat_struct_to_dict(ms, depth: int, max_depth: int) -> Dict[str, Any]:
    out = {}
    fieldnames = getattr(ms, "_fieldnames", []) or []
    for fn in fieldnames:
        try:
            out[fn] = _mat_to_py(getattr(ms, fn), depth=depth + 1, max_depth=max_depth)
        except Exception:
            out[fn] = None
    return out


def _mat_to_py(x: Any, depth: int = 0, max_depth: int = 20) -> Any:
    if depth > max_depth:
        return "<MAX_DEPTH_REACHED>"

    # robust mat_struct detection
    if type(x).__name__ == "mat_struct" and hasattr(x, "_fieldnames"):
        return _mat_struct_to_dict(x, depth=depth, max_depth=max_depth)

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            if x.ndim == 0:
                return _mat_to_py(x.item(), depth=depth + 1, max_depth=max_depth)
            if x.ndim == 1:
                return [_mat_to_py(x[i], depth=depth + 1, max_depth=max_depth) for i in range(x.shape[0])]
            if x.ndim == 2:
                return [[_mat_to_py(x[i, j], depth=depth + 1, max_depth=max_depth)
                         for j in range(x.shape[1])]
                        for i in range(x.shape[0])]
            # fallback for higher dims
            return [_mat_to_py(e, depth=depth + 1, max_depth=max_depth) for e in x.flat]
        return x

    if isinstance(x, (str, int, float, bytes, np.number)):
        return x

    return x

def _load_mat_v72(path: str) -> Dict[str, Any]:
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        for k in ["__header__", "__version__", "__globals__"]:
            if k in data:
                del data[k]

        # NEW: recursively decode matlab structs/cells
        data = {k: _mat_to_py(v) for k, v in data.items()}

        return data
    except Exception as e:
        raise DatasetLoadError(f"Failed to load v7.2 MAT file: {e}")


def _h5_to_py(obj: Any) -> Any:
    """
    Recursively convert h5py objects (Group/Dataset) into Python objects.

    Note:
    - This is a minimal, robust conversion that preserves hierarchy.
    - MATLAB v7.3 structs/cells may require additional decoding later.
    """
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        # Convert scalar bytes to str when possible
        if isinstance(arr, (bytes, np.bytes_)):
            try:
                return arr.decode("utf-8", errors="ignore")
            except Exception:
                return arr
        return np.array(arr)

    if isinstance(obj, h5py.Group):
        out = {}
        for k in obj.keys():
            out[k] = _h5_to_py(obj[k])
        return out

    return obj


def _load_mat_v73(path: str) -> Dict[str, Any]:
    """
    Load MATLAB v7.3 .mat files (HDF5-based).
    """
    try:
        with h5py.File(path, "r") as f:
            out = {}
            for key in f.keys():
                out[key] = _h5_to_py(f[key])
            return out
    except Exception as e:
        raise DatasetLoadError(f"Failed to load v7.3 MAT file (HDF5): {e}")


def _load_mat_file(path: str) -> Dict[str, Any]:
    """
    Automatically detect whether .mat is v7.2 or v7.3 and load accordingly.
    """
    try:
        return _load_mat_v72(path)
    except Exception:
        return _load_mat_v73(path)


def _generate_data_summary(out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a factual summary of the loaded data.
    
    This provides raw statistics only - no interpretations or recommendations.
    The LLM must independently analyze this data to identify any issues.
    """
    summary = {
        "spike_times_info": {},
        "time_fields": {},
    }
    
    # 1. Analyze spike_times
    if "spike_times_data" in out:
        spike_data = out["spike_times_data"]
        if len(spike_data) > 0:
            summary["spike_times_info"] = {
                "total_spikes": len(spike_data),
                "range": [float(np.min(spike_data)), float(np.max(spike_data))],
                "sample": [float(x) for x in spike_data[:5]]
            }
    elif "spike_times_by_unit" in out:
        all_spikes = []
        for unit_spikes in out["spike_times_by_unit"]:
            if len(unit_spikes) > 0:
                all_spikes.extend(unit_spikes)
        if all_spikes:
            all_spikes_arr = np.array(all_spikes)
            summary["spike_times_info"] = {
                "total_spikes": len(all_spikes_arr),
                "range": [float(np.min(all_spikes_arr)), float(np.max(all_spikes_arr))],
                "sample": [float(x) for x in all_spikes_arr[:5]]
            }
    
    # 2. Analyze time fields - get trial_start_time for reference
    trial_start_times = out.get("trial_start_time")
    if trial_start_times is None:
        trial_start_times = out.get("start_time")
    
    trial_start_arr = None
    if trial_start_times is not None and hasattr(trial_start_times, '__len__') and len(trial_start_times) > 0:
        trial_start_arr = np.array([float(t) if t is not None else 0.0 for t in trial_start_times])
        start_time_range = [float(np.min(trial_start_arr)), float(np.max(trial_start_arr))]
        summary["time_fields"]["trial_start_time"] = {
            "range": start_time_range,
            "is_relative": False,  # start_time is always absolute
            "n_values": len(trial_start_arr)
        }
    
    # 3. Scan all trial_*_time fields
    for key in out.keys():
        if key.startswith("trial_") and "_time" in key and key != "trial_start_time":
            values = out[key]
            if values is not None and hasattr(values, '__len__') and len(values) > 0:
                try:
                    # Convert to float array, handling None values
                    float_values = []
                    for v in values:
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            try:
                                float_values.append(float(v))
                            except:
                                pass
                    
                    if float_values:
                        arr = np.array(float_values)
                        min_val = float(np.min(arr))
                        max_val = float(np.max(arr))
                        
                        # Heuristic: if max < 100, likely relative to trial start
                        # If max > 100, likely absolute time
                        is_relative = max_val < 100
                        
                        field_info = {
                            "range": [min_val, max_val],
                            "is_relative": is_relative,
                            "n_values": len(float_values)
                        }
                        
                        # If relative and we have start_times, compute absolute range
                        if is_relative and trial_start_arr is not None and len(trial_start_arr) >= len(float_values):
                            abs_values = trial_start_arr[:len(float_values)] + arr
                            field_info["absolute_range"] = [float(np.min(abs_values)), float(np.max(abs_values))]
                        
                        summary["time_fields"][key] = field_info
                except Exception:
                    # Skip fields that can't be analyzed
                    pass
    
    # No pre-computed diagnostics - let LLM analyze the raw data
    return summary


def _load_nwb_file(path: str) -> Dict[str, Any]:
    """
    Load NWB (Neurodata Without Borders) file.
    NWB uses HDF5 format with standardized structure.
    
    Common NWB structure:
    - /intervals/trials: trial metadata (type, response, timestamps, etc.)
    - /units: neural unit data (spike_times, cell_type, quality, etc.)
    - /acquisition: behavioral data (lick_times, etc.)
    - /general: metadata about the experiment
    
    Returns a flattened dict structure compatible with existing field catalog system.
    """
    try:
        with h5py.File(path, "r") as f:
            out = {}
            
            # 1. Extract trial data from /intervals/trials
            if "intervals" in f and "trials" in f["intervals"]:
                trials_group = f["intervals"]["trials"]
                trial_data = {}
                
                # Get number of trials from any field
                n_trials = None
                for key in trials_group.keys():
                    if isinstance(trials_group[key], h5py.Dataset):
                        n_trials = trials_group[key].shape[0]
                        break
                
                if n_trials is not None:
                    # Extract all trial fields
                    for key in trials_group.keys():
                        try:
                            if isinstance(trials_group[key], h5py.Dataset):
                                data = trials_group[key][()]
                                # Decode bytes to strings
                                if isinstance(data, np.ndarray) and data.dtype.kind in ('S', 'O'):
                                    try:
                                        data = np.array([d.decode('utf-8') if isinstance(d, bytes) else d for d in data])
                                    except:
                                        pass
                                trial_data[key] = data
                        except Exception as e:
                            # Skip fields that fail to load
                            pass
                    
                    # Store as structured array or dict of arrays
                    if trial_data:
                        # Create a list of dicts (one per trial) for compatibility
                        trials_list = []
                        for i in range(n_trials):
                            trial_dict = {}
                            for key, values in trial_data.items():
                                try:
                                    trial_dict[key] = values[i] if hasattr(values, '__getitem__') else values
                                except:
                                    pass
                            trials_list.append(trial_dict)
                        out["trials"] = trials_list
                        
                        # Also store as individual arrays for easier access
                        for key, values in trial_data.items():
                            out[f"trial_{key}"] = values
            
            # 2. Extract neural unit data from /units
            if "units" in f:
                units_group = f["units"]
                unit_data = {}
                
                for key in units_group.keys():
                    try:
                        if isinstance(units_group[key], h5py.Dataset):
                            data = units_group[key][()]
                            
                            # Handle spike_times which uses ragged arrays with index
                            if key == "spike_times":
                                # NWB stores spike_times as flat array with index
                                out["spike_times_data"] = np.array(data)
                                
                            elif key == "spike_times_index":
                                # Index to split spike_times by unit
                                out["spike_times_index"] = np.array(data)
                                
                                # If we have both data and index, split into per-unit arrays
                                if "spike_times_data" in out:
                                    spike_data = out["spike_times_data"]
                                    indices = np.array(data)
                                    
                                    # Split spike times by unit
                                    spike_times_by_unit = []
                                    start_idx = 0
                                    for end_idx in indices:
                                        spike_times_by_unit.append(spike_data[start_idx:end_idx])
                                        start_idx = end_idx
                                    out["spike_times_by_unit"] = spike_times_by_unit
                                    
                            else:
                                # Regular fields (cell_type, quality, etc.)
                                # Decode bytes to strings if needed
                                if data.dtype.kind in ('S', 'O'):
                                    try:
                                        data = np.array([d.decode('utf-8') if isinstance(d, bytes) else d for d in data])
                                    except:
                                        pass
                                unit_data[key] = data
                                out[f"unit_{key}"] = data
                    except Exception as e:
                        # Skip fields that fail to load
                        pass
            
            # 3. Extract behavioral data from /acquisition
            if "acquisition" in f:
                acq_group = f["acquisition"]
                
                # Look for behavioral events (lick_times, etc.)
                for key in acq_group.keys():
                    try:
                        obj = acq_group[key]
                        
                        # Navigate to data field if this is a group
                        if isinstance(obj, h5py.Group):
                            # For nested groups (like lick_times with lick_left_times/lick_right_times)
                            for subkey in obj.keys():
                                subobj = obj[subkey]
                                if isinstance(subobj, h5py.Group):
                                    # Extract timestamps from nested behavioral data
                                    if "timestamps" in subobj:
                                        data = subobj["timestamps"][()]
                                        out[f"behavior_{subkey}"] = np.array(data)
                                    elif "data" in subobj:
                                        data = subobj["data"][()]
                                        out[f"behavior_{subkey}_data"] = np.array(data)
                                elif isinstance(subobj, h5py.Dataset):
                                    data = subobj[()]
                                    out[f"behavior_{subkey}"] = np.array(data)
                            
                            # Also check top-level timestamps/data in this group
                            if "timestamps" in obj:
                                data = obj["timestamps"][()]
                                out[f"behavior_{key}"] = np.array(data)
                            elif "data" in obj:
                                data = obj["data"][()]
                                out[f"behavior_{key}"] = np.array(data)
                        elif isinstance(obj, h5py.Dataset):
                            data = obj[()]
                            out[f"behavior_{key}"] = np.array(data)
                    except Exception as e:
                        # Skip fields that fail to load
                        pass
            
            # 4. Extract metadata from /general (optional)
            if "general" in f:
                general_group = f["general"]
                metadata = {}
                
                # Extract simple metadata fields
                for key in general_group.keys():
                    try:
                        obj = general_group[key]
                        if isinstance(obj, h5py.Dataset):
                            data = obj[()]
                            if isinstance(data, bytes):
                                data = data.decode('utf-8', errors='ignore')
                            metadata[key] = data
                    except:
                        pass
                
                if metadata:
                    out["metadata"] = metadata
            
            # 5. Add NWB format indicator
            out["_nwb_format"] = True
            out["_nwb_version"] = f.attrs.get("nwb_version", "unknown").decode() if isinstance(f.attrs.get("nwb_version"), bytes) else str(f.attrs.get("nwb_version", "unknown"))
            
            # 6. Generate data summary for Controller diagnosis
            out["_data_summary"] = _generate_data_summary(out)
            
            return out
            
    except Exception as e:
        raise DatasetLoadError(f"Failed to load NWB file: {e}")


# -----------------------------
# Dataset summarization helpers
# -----------------------------

def _is_scalar_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_))

def _is_record_dict(d: Dict[str, Any]) -> bool:
    # A lightweight heuristic: string keys, and values are mostly scalars/ndarrays/lists/dicts
    if not isinstance(d, dict) or len(d) == 0:
        return False
    if not all(isinstance(k, str) for k in d.keys()):
        return False
    return True

def _infer_list_matrix_shape(x: Any) -> Optional[Tuple[int, int]]:
    # If x is list of lists with consistent second dimension, treat as matrix-like
    if not isinstance(x, list) or len(x) == 0:
        return None
    if not all(isinstance(r, list) for r in x):
        return None
    n_rows = len(x)
    n_cols = len(x[0]) if n_rows > 0 else 0
    if n_cols == 0:
        return None
    # allow a small number of irregular rows without failing hard
    ok = sum(1 for r in x if isinstance(r, list) and len(r) == n_cols)
    if ok / max(n_rows, 1) >= 0.95:
        return (n_rows, n_cols)
    return None

def _sample_numeric_minmax(arr: np.ndarray, max_elems: int = 200) -> Dict[str, Any]:
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return {}
    if not np.issubdtype(arr.dtype, np.number):
        return {}
    flat = arr.ravel()
    if flat.size > max_elems:
        _rng = np.random.RandomState(42)
        idx = _rng.choice(flat.size, size=max_elems, replace=False)
        flat = flat[idx]
    try:
        return {"min": float(np.nanmin(flat)), "max": float(np.nanmax(flat))}
    except Exception:
        return {}

def _value_signature(v: Any) -> Dict[str, Any]:
    """
    Describe value type without semantics.
    Returns a dict with coarse type info.
    """
    sig: Dict[str, Any] = {}
    if _is_scalar_number(v):
        sig["kind"] = "scalar_number"
        sig["dtype"] = type(v).__name__
        return sig
    if isinstance(v, (bool, np.bool_)):
        sig["kind"] = "scalar_bool"
        return sig
    if isinstance(v, str):
        sig["kind"] = "string"
        return sig
    if isinstance(v, np.ndarray):
        sig["kind"] = "ndarray"
        sig["ndim"] = int(v.ndim)
        sig["dtype"] = str(v.dtype)
        sig["shape"] = tuple(v.shape)
        # 1D length summary is handled elsewhere
        return sig
    if isinstance(v, list):
        sig["kind"] = "list"
        sig["len"] = len(v)
        # only record element type loosely
        if len(v) > 0:
            sig["elem_type"] = type(v[0]).__name__
        return sig
    if isinstance(v, dict):
        sig["kind"] = "dict"
        sig["n_keys"] = len(v)
        return sig
    sig["kind"] = type(v).__name__
    return sig

def _traverse_for_record_dicts(
    obj: Any,
    path: str,
    out_records: List[Tuple[str, Dict[str, Any]]],
    max_depth: int,
    depth: int,
    rng: random.Random,
    sample_budget: int,
) -> None:
    """
    Collect record-like dict samples with paths (without dataset-specific rules).
    Uses sampling budget to avoid explosion.
    """
    if len(out_records) >= sample_budget:
        return
    if depth > max_depth:
        return

    # record dict
    if isinstance(obj, dict) and _is_record_dict(obj):
        out_records.append((path, obj))
        # still traverse a bit deeper, but keep it cheap
        if len(out_records) >= sample_budget:
            return

    if isinstance(obj, dict):
        # sample some keys to traverse
        keys = list(obj.keys())
        rng.shuffle(keys)
        for k in keys[: min(len(keys), 10)]:  # cap branching
            _traverse_for_record_dicts(
                obj[k], f"{path}.{k}" if path else str(k),
                out_records, max_depth, depth + 1, rng, sample_budget
            )
            if len(out_records) >= sample_budget:
                return

    elif isinstance(obj, list):
        # If matrix-like list, sample a few cells
        shape = _infer_list_matrix_shape(obj)
        if shape is not None:
            n_rows, n_cols = shape
            # sample up to 12 cells
            for _ in range(min(12, n_rows * n_cols)):
                i = rng.randrange(n_rows)
                j = rng.randrange(n_cols)
                _traverse_for_record_dicts(
                    obj[i][j], f"{path}[{i}][{j}]",
                    out_records, max_depth, depth + 1, rng, sample_budget
                )
                if len(out_records) >= sample_budget:
                    return
        else:
            # 1D list: sample a few elements
            if len(obj) == 0:
                return
            for _ in range(min(12, len(obj))):
                i = rng.randrange(len(obj))
                _traverse_for_record_dicts(
                    obj[i], f"{path}[{i}]",
                    out_records, max_depth, depth + 1, rng, sample_budget
                )
                if len(out_records) >= sample_budget:
                    return

    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # sample a few entries
            if obj.size == 0:
                return
            # choose indices uniformly in flattened space
            for _ in range(min(12, obj.size)):
                idx = rng.randrange(obj.size)
                v = obj.flat[idx]
                _traverse_for_record_dicts(
                    v, f"{path}.flat[{idx}]",
                    out_records, max_depth, depth + 1, rng, sample_budget
                )
                if len(out_records) >= sample_budget:
                    return

def _extract_matrix_structure_metadata(
    raw_data: Dict[str, Any],
    rng: random.Random,
    max_samples_per_col: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extract metadata about matrix-like structures where column indices 
    might represent implicit categorical variables (e.g., cue_location).
    
    Returns:
        List of matrix structure info dicts
    """
    matrix_infos = []
    
    for var_name, var_value in raw_data.items():
        matrix_info = None
        
        # Check for list-of-lists matrix structure
        if isinstance(var_value, list) and len(var_value) > 0:
            shape = _infer_list_matrix_shape(var_value)
            if shape is not None:
                n_rows, n_cols = shape
                matrix_info = {
                    "variable_name": var_name,
                    "structure_type": "list_matrix",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "column_indices": list(range(n_cols)),
                    "column_index_range": f"0-{n_cols-1}" if n_cols > 0 else "empty",
                    "note": f"Column index may represent categorical variable (0-{n_cols-1}, or 1-{n_cols} if 1-indexed)",
                }
                
                # Sample a few cells from different columns to check content type
                col_samples = {}
                for col_idx in range(min(n_cols, 8)):  # Sample up to 8 columns
                    samples = []
                    for _ in range(min(max_samples_per_col, n_rows)):
                        row_idx = rng.randrange(n_rows)
                        if row_idx < len(var_value) and col_idx < len(var_value[row_idx]):
                            cell = var_value[row_idx][col_idx]
                            if isinstance(cell, dict) and _looks_like_trial(cell):
                                # Extract some key field names to show structure
                                sample_keys = list(cell.keys())[:10]
                                samples.append(f"dict with keys: {sample_keys}")
                                break  # One sample per column is enough
                    if samples:
                        col_samples[f"col_{col_idx}"] = samples[0]
                
                if col_samples:
                    matrix_info["column_content_samples"] = col_samples
        
        # Check for ndarray with object dtype (another common matrix structure)
        elif isinstance(var_value, np.ndarray) and var_value.dtype == object:
            if var_value.ndim == 2:
                n_rows, n_cols = var_value.shape
                matrix_info = {
                    "variable_name": var_name,
                    "structure_type": "ndarray_object_2d",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "column_indices": list(range(n_cols)),
                    "column_index_range": f"0-{n_cols-1}" if n_cols > 0 else "empty",
                    "note": f"Column index may represent categorical variable (0-{n_cols-1}, or 1-{n_cols} if 1-indexed)",
                }
                
                # Sample cells
                col_samples = {}
                for col_idx in range(min(n_cols, 8)):
                    samples = []
                    for _ in range(min(max_samples_per_col, n_rows)):
                        row_idx = rng.randrange(n_rows)
                        if row_idx < n_rows and col_idx < n_cols:
                            cell = var_value[row_idx, col_idx]
                            if isinstance(cell, dict) and _looks_like_trial(cell):
                                sample_keys = list(cell.keys())[:10]
                                samples.append(f"dict with keys: {sample_keys}")
                                break
                    if samples:
                        col_samples[f"col_{col_idx}"] = samples[0]
                
                if col_samples:
                    matrix_info["column_content_samples"] = col_samples
        
        if matrix_info:
            matrix_infos.append(matrix_info)
    
    return matrix_infos


def _looks_like_trial(d: Dict[str, Any]) -> bool:
    """
    Heuristic: check if a dict looks like a trial record.
    (Moved to top level to be reused by matrix structure extraction)
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


def build_field_catalog(
    nd: "NeuralDataset",
    sample_budget: int = 60,
    max_depth: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Build a dataset-agnostic structural summary for LLM reasoning:
    - Top-level variables + coarse type/shape
    - Sampled record-dicts: key frequency, value type signatures
    - Minimal numeric summaries: scalar min/max, 1D length distribution, array min/max (sampled)
    """
    rng = random.Random(seed)

    catalog: Dict[str, Any] = {}
    top_level: List[Dict[str, Any]] = []

    # 1) Top-level variable overview
    for k, v in nd.raw.items():
        item: Dict[str, Any] = {"name": k, "type": type(v).__name__}
        if isinstance(v, np.ndarray):
            item["kind"] = "ndarray"
            item["shape"] = tuple(v.shape)
            item["dtype"] = str(v.dtype)
        elif isinstance(v, list):
            item["kind"] = "list"
            shape = _infer_list_matrix_shape(v)
            if shape is not None:
                item["shape_like"] = {"type": "list_matrix", "rows": shape[0], "cols": shape[1]}
            else:
                item["len"] = len(v)
        elif isinstance(v, dict):
            item["kind"] = "dict"
            item["n_keys"] = len(v)
            item["keys_head"] = list(v.keys())[:30]
        else:
            item["kind"] = "scalar_or_other"
        top_level.append(item)

    catalog["top_level"] = top_level

    # 2) Collect record-like dict samples with paths
    record_samples: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in nd.raw.items():
        _traverse_for_record_dicts(
            v, k, record_samples, max_depth=max_depth, depth=0,
            rng=rng, sample_budget=sample_budget
        )
        if len(record_samples) >= sample_budget:
            break

    catalog["record_dict_samples"] = {
        "sample_budget": sample_budget,
        "n_collected": len(record_samples),
        "example_paths": [p for p, _ in record_samples[:10]],
    }

    if len(record_samples) == 0:
        # nothing more to do
        return catalog

    # 3) Key frequency + per-field value summaries (dataset-agnostic)
    key_count = defaultdict(int)
    field_value_sigs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # For numeric summaries
    scalar_minmax: Dict[str, Dict[str, float]] = {}
    array_len_list: Dict[str, List[int]] = defaultdict(list)
    array_minmax: Dict[str, Dict[str, float]] = {}

    # helper accumulators
    scalar_vals: Dict[str, List[float]] = defaultdict(list)
    array_min_candidates: Dict[str, List[float]] = defaultdict(list)
    array_max_candidates: Dict[str, List[float]] = defaultdict(list)

    array_samples_count: Dict[str, int] = defaultdict(int)

    for _, d in record_samples:
        for key, val in d.items():
            key_count[key] += 1
            sig = _value_signature(val)
            field_value_sigs[key].append(sig)

            # numeric scalar min/max
            if _is_scalar_number(val):
                scalar_vals[key].append(float(val))
                continue

            # numeric arrays: length distribution + sampled min/max
            if isinstance(val, np.ndarray):
                array_samples_count[key] += 1
                if val.ndim == 1:
                    array_len_list[key].append(int(val.shape[0]))
                mm = _sample_numeric_minmax(val, max_elems=200)
                if "min" in mm:
                    array_min_candidates[key].append(mm["min"])
                if "max" in mm:
                    array_max_candidates[key].append(mm["max"])

    # compute scalar min/max
    for key, vals in scalar_vals.items():
        if len(vals) == 0:
            continue
        scalar_minmax[key] = {"min": float(min(vals)), "max": float(max(vals))}

    # compute array min/max (over sampled mins/maxs)
    for key, mins in array_min_candidates.items():
        if len(mins) > 0:
            array_minmax.setdefault(key, {})["min"] = float(min(mins))
    for key, maxs in array_max_candidates.items():
        if len(maxs) > 0:
            array_minmax.setdefault(key, {})["max"] = float(max(maxs))

    # length distribution summary (for 1D arrays/lists)
    length_summary: Dict[str, Dict[str, Any]] = {}
    for key, lens in array_len_list.items():
        if len(lens) == 0:
            continue
        lens_sorted = sorted(lens)
        mid = lens_sorted[len(lens_sorted) // 2]
        length_summary[key] = {
            "n": len(lens),
            "min": int(lens_sorted[0]),
            "median": int(mid),
            "max": int(lens_sorted[-1]),
        }

    n_records = len(record_samples)
    common_keys_90 = [k for k, c in key_count.items() if c / n_records >= 0.90]
    common_keys_50 = [k for k, c in key_count.items() if c / n_records >= 0.50]

    # 4) Build per-field report (without semantics)
    fields: Dict[str, Any] = {}
    for key in sorted(key_count.keys(), key=lambda x: key_count[x], reverse=True):
        sigs = field_value_sigs[key]
        # count kinds
        kind_count = defaultdict(int)
        for s in sigs:
            kind_count[s.get("kind", "unknown")] += 1

        kinds_dict = dict(kind_count)
        fields[key] = {
            "presence": {"count": int(key_count[key]), "ratio": float(key_count[key] / n_records)},
            "kinds": kinds_dict,
            "mixed_kinds": (len(kinds_dict) > 1),
        }

        if key in scalar_minmax:
            fields[key]["scalar_minmax"] = scalar_minmax[key]
        if key in length_summary:
            fields[key]["array_1d_length"] = length_summary[key]
        if key in array_minmax:
            fields[key]["array_numeric_minmax_sampled"] = {
                **array_minmax[key],
                "sample_elems_per_array_max": 200,
                "n_arrays_sampled": int(array_samples_count.get(key, 0)),
            }

    catalog["record_field_stats"] = {
        "n_records_sampled": n_records,
        "common_keys_ratio_ge_0.90": common_keys_90[:200],
        "common_keys_ratio_ge_0.50": common_keys_50[:200],
        "fields": fields,
    }
    
    # 5) NEW: Extract matrix structure metadata (for implicit field inference)
    matrix_structures = _extract_matrix_structure_metadata(nd.raw, rng=rng)
    if matrix_structures:
        catalog["matrix_structures"] = matrix_structures

    return catalog

def render_field_catalog_text(
    field_catalog: Dict[str, Any],
    max_fields: int = 30,
    max_lines: int = 1000,
) -> str:
    """
    Render field_catalog into a compact, LLM-friendly text with a strict size budget.
    Dataset-agnostic. No semantics, only structure/stats.
    """
    lines: List[str] = []
    push = lines.append

    push("DATASET STRUCTURE SUMMARY")
    push("")

    # Top-level
    push("Top-level variables:")
    for item in field_catalog.get("top_level", [])[:20]:
        name = item.get("name")
        kind = item.get("kind", item.get("type"))
        if kind == "ndarray":
            push(f"- {name}: ndarray shape={item.get('shape')} dtype={item.get('dtype')}")
        elif kind == "list":
            shape_like = item.get("shape_like")
            if shape_like and shape_like.get("type") == "list_matrix":
                push(f"- {name}: list_matrix rows={shape_like.get('rows')} cols={shape_like.get('cols')}")
            else:
                push(f"- {name}: list len={item.get('len')}")
        elif kind == "dict":
            push(f"- {name}: dict n_keys={item.get('n_keys')} keys_head={item.get('keys_head')}")
        else:
            push(f"- {name}: {kind}")

    push("")
    
    # NEW: Matrix structure metadata (for implicit field inference)
    matrix_structures = field_catalog.get("matrix_structures", [])
    if matrix_structures:
        push("⚠️ IMPORTANT: Matrix structures with implicit column indices:")
        push("(Column indices in these matrices may represent categorical variables)")
        for ms in matrix_structures[:5]:  # Show up to 5 matrix structures
            var_name = ms.get("variable_name", "unknown")
            n_rows = ms.get("n_rows", 0)
            n_cols = ms.get("n_cols", 0)
            col_range = ms.get("column_index_range", "")
            note = ms.get("note", "")
            push(f"- Variable '{var_name}': {n_rows}×{n_cols} matrix")
            push(f"  Column indices: {col_range}")
            push(f"  Note: {note}")
            # Show a sample of column content if available
            col_samples = ms.get("column_content_samples", {})
            if col_samples:
                sample_keys = list(col_samples.keys())[:3]
                push(f"  Sample columns: {', '.join(sample_keys)}")
        push("")
        push("⚠️ If user mentions a field that's not explicitly in the data,")
        push("   it might be represented by COLUMN INDEX in one of these matrices.")
        push("")

    # Record dict sampling info
    rds = field_catalog.get("record_dict_samples", {})
    push("Record-like dict sampling:")
    push(f"- sampled_records={rds.get('n_collected')} (budget={rds.get('sample_budget')})")
    ex_paths = rds.get("example_paths", [])
    if ex_paths:
        push(f"- example_paths={ex_paths[:6]}")
    push("")

    # Field stats
    rfs = field_catalog.get("record_field_stats", {})
    push("Record field statistics:")
    push(f"- n_records_sampled={rfs.get('n_records_sampled')}")
    common90 = rfs.get("common_keys_ratio_ge_0.90", [])[:max_fields]
    push(f"- common_keys(>=90% presence) top{len(common90)}={common90}")
    push("")

    fields = rfs.get("fields", {})
    if not fields:
        push("(No record fields detected.)")
        return "\n".join(lines[:max_lines])

    # Print per-field details (for common keys first, then stop at max_fields)
    def emit_field(key: str, info: Dict[str, Any]):
        pres = info.get("presence", {})
        ratio = pres.get("ratio", None)
        kinds = info.get("kinds", {})
        mixed = info.get("mixed_kinds", False)
        push(f"* {key}: presence={ratio:.3f} kinds={kinds} mixed_kinds={mixed}")

        if "scalar_minmax" in info:
            mm = info["scalar_minmax"]
            push(f"  - scalar_minmax: min={mm.get('min')} max={mm.get('max')}")
        if "array_1d_length" in info:
            ls = info["array_1d_length"]
            push(f"  - array_1d_length: n={ls.get('n')} min={ls.get('min')} median={ls.get('median')} max={ls.get('max')}")
        if "array_numeric_minmax_sampled" in info:
            amm = info["array_numeric_minmax_sampled"]
            push(
                f"  - array_numeric_minmax_sampled: "
                f"min={amm.get('min')} max={amm.get('max')} "
                f"(n_arrays_sampled={amm.get('n_arrays_sampled')}, "
                f"sample_elems_per_array_max={amm.get('sample_elems_per_array_max')})"
            )

    push("Per-field details (common keys first):")
    emitted = 0
    for key in common90:
        if key in fields:
            emit_field(key, fields[key])
            emitted += 1
            if emitted >= max_fields:
                break

    # Enforce line budget (line-safe)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("... (truncated by max_lines budget)")
    return "\n".join(lines)

def _safe_numeric_stats(arr: np.ndarray) -> Dict[str, Any]:
    """
    Compute lightweight numeric stats for numeric arrays.
    Avoid heavy computation on huge arrays by sampling if needed (future extension).
    """
    stats: Dict[str, Any] = {}
    if not isinstance(arr, np.ndarray):
        return stats
    if arr.size == 0:
        return stats
    if not np.issubdtype(arr.dtype, np.number):
        return stats

    # Use nan-safe ops
    try:
        stats["min"] = float(np.nanmin(arr))
        stats["max"] = float(np.nanmax(arr))
        stats["mean"] = float(np.nanmean(arr))
        stats["std"] = float(np.nanstd(arr))
    except Exception:
        # If any numeric issues arise, skip
        pass
    return stats


def _collect_var_stats(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    var_stats: Dict[str, Dict[str, Any]] = {}

    def visit(prefix: str, x: Any):
        name = prefix
        info: Dict[str, Any] = {}

        if isinstance(x, np.ndarray):
            info["type"] = "ndarray"
            info["shape"] = tuple(x.shape)
            info["dtype"] = str(x.dtype)
            info.update(_safe_numeric_stats(x))
        elif isinstance(x, (int, float, str, bytes, np.number)):
            info["type"] = type(x).__name__
        elif isinstance(x, dict):
            info["type"] = "dict"
            info["keys"] = list(x.keys())[:50]
        else:
            info["type"] = type(x).__name__

        var_stats[name] = info

        # Recurse a bit for nested dicts (v7.3 groups)
        if isinstance(x, dict):
            for k, v in x.items():
                visit(f"{name}.{k}" if name else str(k), v)

    for k, v in raw.items():
        visit(k, v)

    return var_stats


def _build_dataset_info_for_mat(path: str, nd: NeuralDataset) -> DatasetInfo:
    """
    Keep DatasetInfo structure unchanged (state.py), but adapt fields for .mat:
    - n_rows/n_cols are not meaningful for general .mat -> set to -1.
    - dtypes stores variable-level structural descriptions.
    """
    dtypes: Dict[str, str] = {}
    for var_name, info in nd.var_stats.items():
        dtype_str = info.get("dtype")
        shape = info.get("shape")
        typ = info.get("type")
        parts = [typ]
        if shape is not None:
            parts.append(f"shape={shape}")
        if dtype_str is not None:
            parts.append(f"dtype={dtype_str}")
        dtypes[var_name] = ", ".join(parts)

    return DatasetInfo(
        path=os.path.abspath(path),
        n_rows=-1,
        n_cols=-1,
        dtypes=dtypes,
    )


def _extract_tabular_metadata_if_any(raw: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Optional: if we can confidently identify a tabular trial metadata variable,
    convert it to a DataFrame. Otherwise return None.

    Minimal heuristic (safe):
    - Find a 2D numeric ndarray with reasonable shape (rows >= 5, cols <= 200)
    - and variable name suggests trial info.
    """
    candidates = []
    for k, v in raw.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim != 2:
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        rows, cols = v.shape
        if rows >= 5 and cols <= 200:
            lname = k.lower()
            score = 0
            if "trial" in lname or "meta" in lname or "info" in lname or "label" in lname:
                score += 2
            candidates.append((score, k, v))

    if not candidates:
        return None

    # take the highest score, then larger rows
    candidates.sort(key=lambda x: (x[0], x[2].shape[0]), reverse=True)
    _, key, arr = candidates[0]
    cols = arr.shape[1]
    col_names = [f"{key}_col_{i}" for i in range(cols)]
    return pd.DataFrame(arr, columns=col_names)


# -----------------------------
# Archive extraction helpers
# -----------------------------

def _extract_archive(archive_path: str, extract_dir: str) -> str:
    """
    Extract compressed archive (tar.gz, tgz, zip) to directory.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        
    Returns:
        Path to extraction directory
        
    Raises:
        DatasetLoadError: If extraction fails
    """
    try:
        if archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(path=extract_dir)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            raise DatasetLoadError(f"Unsupported archive format: {archive_path}")
        
        return extract_dir
    except Exception as e:
        raise DatasetLoadError(f"Failed to extract archive {archive_path}: {e}")


def _find_dataset_files(directory: str, extensions: Tuple[str, ...] = ('.nwb', '.mat')) -> List[str]:
    """
    Recursively find dataset files in directory.
    
    Args:
        directory: Directory to search
        extensions: File extensions to look for
        
    Returns:
        List of absolute paths to dataset files
    """
    dataset_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                dataset_files.append(os.path.join(root, file))
    return sorted(dataset_files)


# -----------------------------
# Public API
# -----------------------------


def _load_csv_file(path: str) -> Dict[str, Any]:
    """
    Load CSV file into a dict structure compatible with NeuralDataset.
    
    This function performs RAW LOADING ONLY - no preprocessing, no assumptions
    about data type (spike times, trials, etc.). All semantic interpretation
    is left to downstream LLM-driven tools.
    
    The CSV is read into a pandas DataFrame, then stored as:
    - 'dataframe': The raw pandas DataFrame
    - 'columns': List of column names
    - 'column_info': Dict with column metadata (dtype, sample values, stats)
    - 'rows': List of dicts (one per row) for small files
    
    Args:
        path: Path to CSV file
        
    Returns:
        Dict with raw data and metadata for LLM to interpret
    """
    try:
        # Load full CSV
        df = pd.read_csv(path)
        filename = os.path.basename(path)
        
        print(f"  Loading CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)", flush=True)
        
        out = {}
        
        # Store raw DataFrame for downstream processing
        out["_dataframe"] = df
        out["_columns"] = list(df.columns)
        out["_n_rows"] = len(df)
        out["_source_file"] = filename
        out["_csv_format"] = True
        
        # Generate column metadata for LLM interpretation
        column_info = {}
        for col in df.columns:
            values = df[col]
            info = {
                "dtype": str(values.dtype),
                "n_unique": int(values.nunique()),
                "n_null": int(values.isnull().sum()),
            }
            
            # Add sample values (first 5 non-null)
            non_null = values.dropna()
            if len(non_null) > 0:
                sample_vals = non_null.head(5).tolist()
                info["sample_values"] = sample_vals
                
                # Add stats for numeric columns
                if np.issubdtype(values.dtype, np.number):
                    info["min"] = float(values.min()) if not pd.isna(values.min()) else None
                    info["max"] = float(values.max()) if not pd.isna(values.max()) else None
                    info["mean"] = float(values.mean()) if not pd.isna(values.mean()) else None
                
                # Add unique values for categorical columns (if few enough)
                if info["n_unique"] <= 20:
                    info["unique_values"] = non_null.unique().tolist()
            
            column_info[col] = info
        
        out["_column_info"] = column_info
        
        # Store each column as array (with original column name, no prefix)
        for col in df.columns:
            values = df[col].values
            if np.issubdtype(values.dtype, np.number):
                out[col] = values
            elif values.dtype == object:
                out[col] = np.array([str(v) if pd.notna(v) else '' for v in values])
            else:
                out[col] = values
        
        # Create rows list (one dict per row) - only for small files
        if len(df) <= 50000:
            out["_rows"] = df.to_dict('records')
        else:
            print(f"  Skipping rows list for large file ({len(df)} rows)", flush=True)
            out["_rows"] = None
        
        # Generate summary text for LLM
        summary_lines = [f"CSV File: {filename}", f"Shape: {len(df)} rows × {len(df.columns)} columns", "Columns:"]
        for col, info in column_info.items():
            dtype_str = info["dtype"]
            sample = info.get("sample_values", [])[:3]
            summary_lines.append(f"  - {col} ({dtype_str}): {sample}")
        out["_summary_text"] = "\n".join(summary_lines)
        
        return out
        
    except Exception as e:
        raise DatasetLoadError(f"Failed to load CSV file: {e}")


def _merge_csv_datasets(paths: List[str]) -> Dict[str, Any]:
    """
    Merge multiple CSV files into a single dataset.
    
    This function performs RAW MERGING ONLY - no preprocessing assumptions.
    Each CSV is stored with its filename as a namespace prefix.
    
    Args:
        paths: List of CSV file paths to merge
    
    Returns a merged dict suitable for NeuralDataset:
        - _files: Dict mapping filename -> {dataframe, columns, column_info, ...}
        - _source_files: List of source filenames
        - _combined_summary: Overall data description
    """
    merged = {
        "_files": {},
        "_csv_format": True,
        "_multi_file": True,
        "_source_files": [],
    }
    
    all_summaries = []
    
    for path in paths:
        csv_data = _load_csv_file(path)
        source_file = csv_data.get("_source_file", os.path.basename(path))
        
        merged["_source_files"].append(source_file)
        merged["_files"][source_file] = csv_data
        
        # Collect summary for combined view
        if "_summary_text" in csv_data:
            all_summaries.append(csv_data["_summary_text"])
    
    merged["_combined_summary"] = "\n\n".join(all_summaries)
    
    return merged


def _load_set_file(path: str) -> Dict[str, Any]:
    """
    Load EEGLAB .set file into a dict structure compatible with NeuralDataset.

    EEGLAB .set files are MATLAB files containing an EEG struct with fields like
    data, srate, chanlocs, nbchan, pnts, etc.  This loader extracts the EEG
    time-series matrix and metadata so that downstream LLM-generated code can
    work with the data (e.g. band-power extraction, classification).

    Strategy:
    1. Try MNE (``mne.io.read_raw_eeglab``) — richest metadata.
    2. Fall back to scipy ``loadmat`` — works for MATLAB ≤ v7.2 .set files.
    3. Fall back to h5py — works for MATLAB v7.3 (HDF5-based) .set files.
    """
    filename = os.path.basename(path)

    # ------- attempt 1: MNE -------
    try:
        import mne
        raw_eeg = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
        data = raw_eeg.get_data()  # (n_channels, n_samples)
        ch_names = raw_eeg.ch_names
        srate = raw_eeg.info["sfreq"]

        out: Dict[str, Any] = {
            "_eeg_format": True,
            "_source_file": filename,
            "_loader": "mne",
            "data": data,
            "srate": float(srate),
            "nbchan": int(data.shape[0]),
            "pnts": int(data.shape[1]),
            "ch_names": list(ch_names),
            "duration_sec": float(data.shape[1] / srate),
        }
        print(f"  Loaded .set (MNE): {filename}  channels={out['nbchan']}  "
              f"samples={out['pnts']}  srate={out['srate']}Hz  "
              f"duration={out['duration_sec']:.1f}s", flush=True)
        return out
    except ImportError:
        pass
    except Exception as mne_err:
        print(f"  MNE failed for {filename}: {mne_err}. Trying scipy...", flush=True)

    # ------- attempt 2: scipy.io.loadmat -------
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        eeg = mat.get("EEG", None)
        if eeg is None:
            # Some .set files store top-level fields directly
            return mat

        data = np.array(eeg.data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        srate = float(eeg.srate)

        ch_names = []
        if hasattr(eeg, "chanlocs"):
            chanlocs = eeg.chanlocs
            if hasattr(chanlocs, "__len__"):
                ch_names = [str(getattr(c, "labels", f"Ch{i}")) for i, c in enumerate(chanlocs)]
            elif hasattr(chanlocs, "labels"):
                ch_names = [str(chanlocs.labels)]
        if not ch_names:
            ch_names = [f"Ch{i}" for i in range(data.shape[0])]

        out = {
            "_eeg_format": True,
            "_source_file": filename,
            "_loader": "scipy",
            "data": data,
            "srate": srate,
            "nbchan": int(data.shape[0]),
            "pnts": int(data.shape[1]),
            "ch_names": ch_names,
            "duration_sec": float(data.shape[1] / srate),
        }
        print(f"  Loaded .set (scipy): {filename}  channels={out['nbchan']}  "
              f"samples={out['pnts']}  srate={out['srate']}Hz  "
              f"duration={out['duration_sec']:.1f}s", flush=True)
        return out
    except NotImplementedError:
        pass
    except Exception as scipy_err:
        print(f"  scipy failed for {filename}: {scipy_err}. Trying h5py...", flush=True)

    # ------- attempt 3: h5py (MATLAB v7.3 / HDF5) -------
    try:
        with h5py.File(path, "r") as f:
            if "EEG" not in f:
                return _h5_to_py(f)
            eeg_grp = f["EEG"]
            data = np.array(eeg_grp["data"], dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            srate = float(np.array(eeg_grp["srate"]).flat[0])

            ch_names = [f"Ch{i}" for i in range(data.shape[0])]
            if "chanlocs" in eeg_grp and "labels" in eeg_grp["chanlocs"]:
                try:
                    labels_ref = eeg_grp["chanlocs"]["labels"]
                    ch_names = [
                        "".join(chr(c) for c in f[ref][:].flat)
                        for ref in labels_ref[:, 0]
                    ]
                except Exception:
                    pass

            out = {
                "_eeg_format": True,
                "_source_file": filename,
                "_loader": "h5py",
                "data": data,
                "srate": srate,
                "nbchan": int(data.shape[0]),
                "pnts": int(data.shape[1]),
                "ch_names": ch_names,
                "duration_sec": float(data.shape[1] / srate),
            }
            print(f"  Loaded .set (h5py): {filename}  channels={out['nbchan']}  "
                  f"samples={out['pnts']}  srate={out['srate']}Hz  "
                  f"duration={out['duration_sec']:.1f}s", flush=True)
            return out
    except Exception as h5_err:
        raise DatasetLoadError(
            f"Failed to load EEGLAB .set file {filename} "
            f"(tried MNE, scipy, h5py): {h5_err}"
        )


def _load_dataset(path: str) -> NeuralDataset:
    """
    Load neural dataset from file.

    This function performs RAW LOADING ONLY - no preprocessing assumptions.

    Supports:
    - .mat files (MATLAB v7.2 and v7.3)
    - .nwb files (Neurodata Without Borders HDF5 format)
    - .csv files (tabular data)
    - .set files (EEGLAB format)

    Args:
        path: Path to dataset file
    """
    if not os.path.exists(path):
        raise DatasetLoadError(f"Dataset file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".mat":
        raw = _load_mat_file(path)
    elif ext == ".nwb":
        raw = _load_nwb_file(path)
    elif ext == ".csv":
        raw = _load_csv_file(path)
    elif ext == ".set":
        raw = _load_set_file(path)
    else:
        raise NotImplementedError(f"Unsupported file format: {ext}. Supported: .mat, .nwb, .csv, .set")

    nd = NeuralDataset(
        path=os.path.abspath(path),
        raw=raw,
        meta={},
        var_stats=_collect_var_stats(raw),
    )
    return nd


def load_dataset_into_state(
    state: NeuroGlobalState,
    path: str,
    preprocess_rules: Optional[Dict[str, Any]] = None,
    additional_paths: Optional[List[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> NeuroGlobalState:
    """
    Load dataset into state, with support for compressed archives and multiple files.
    
    This function performs RAW LOADING ONLY - no preprocessing assumptions.
    The schema (if provided) is stored in state.meta for downstream LLM interpretation.
    
    Supports:
    - Direct files: .mat, .nwb, .csv, .set
    - Archives: .tar.gz, .tgz, .zip (will extract and find first dataset file)
    - Multiple CSV files: When additional_paths is provided, merges all CSV files
    
    Args:
        state: Current global state
        path: Path to primary dataset file or archive
        preprocess_rules: Optional preprocessing rules (legacy, not used in raw loading)
        additional_paths: Optional list of additional file paths (for multi-CSV support)
        schema: Optional schema description parsed from prompt. Stored in state for 
                downstream LLM tools to interpret column semantics.
        
    Returns:
        Updated state with loaded dataset
    """
    temp_dir = None
    actual_dataset_path = path
    
    try:
        # Check if we have multiple CSV files to merge
        all_paths = [path]
        if additional_paths:
            all_paths.extend(additional_paths)
        
        # Check if all files are CSVs for merging
        all_csv = all(p.lower().endswith('.csv') for p in all_paths)
        
        if len(all_paths) > 1 and all_csv:
            # Multiple CSV files - merge them
            print(f"Merging {len(all_paths)} CSV files...")
            for i, p in enumerate(all_paths, 1):
                print(f"  {i}. {os.path.basename(p)}")
            
            raw = _merge_csv_datasets(all_paths)
            nd = NeuralDataset(
                path=os.path.abspath(path),  # Use first path as primary
                raw=raw,
                meta={"multi_file": True, "source_files": [os.path.basename(p) for p in all_paths]},
                var_stats=_collect_var_stats(raw),
            )
        else:
            # Single file or archive - use original logic
            # Check if this is an archive file
            if path.lower().endswith(('.tar.gz', '.tgz', '.tar', '.zip')):
                # Create temporary directory for extraction
                temp_dir = tempfile.mkdtemp(prefix='neuro_copilot_extract_')
                print(f"Extracting archive {path} to {temp_dir}...")
                
                # Extract archive
                _extract_archive(path, temp_dir)
                
                # Find dataset files in extracted content
                dataset_files = _find_dataset_files(temp_dir, extensions=('.nwb', '.mat', '.csv', '.set'))
                
                if not dataset_files:
                    raise DatasetLoadError(
                        f"No dataset files (.nwb, .mat, .csv, .set) found in archive {path}"
                    )
                
                # Use the first dataset file found
                actual_dataset_path = dataset_files[0]
                print(f"Found {len(dataset_files)} dataset file(s), using: {os.path.basename(actual_dataset_path)}")
                
                if len(dataset_files) > 1:
                    print(f"Note: Archive contains {len(dataset_files)} dataset files. Using first one:")
                    for i, df in enumerate(dataset_files[:5], 1):
                        print(f"  {i}. {os.path.relpath(df, temp_dir)}")
                    if len(dataset_files) > 5:
                        print(f"  ... and {len(dataset_files) - 5} more")
            
            # Load the dataset (either original path or extracted file)
            nd = _load_dataset(actual_dataset_path)
        
        # Store schema in state for downstream tools to use
        if schema:
            state.extra["dataset_schema"] = schema

        if preprocess_rules is not None:
            try:
                normalized_rules = normalize_preprocess_rules(preprocess_rules)
                state.extra["preprocess_rules"] = normalized_rules
                nd.meta["preprocess_rules"] = normalized_rules
            except Exception as e:
                state.extra["preprocess_rules"] = preprocess_rules
                nd.meta["preprocess_rules_error"] = str(e)

        # Optional: extract tabular metadata if exists
        df_meta = _extract_tabular_metadata_if_any(nd.raw)

        info = _build_dataset_info_for_mat(actual_dataset_path, nd)

        # Cache the full neural dataset for execution/debugging
        state.extra["neural_dataset"] = nd

        # Build dataset-agnostic structural summary for LLM
        try:
            nd.meta["field_catalog"] = build_field_catalog(nd, sample_budget=60, max_depth=4, seed=0)
            state.extra["field_catalog"] = nd.meta["field_catalog"]  # optional

            nd.meta["field_catalog_text"] = render_field_catalog_text(nd.meta["field_catalog"])
            state.extra["field_catalog_text"] = nd.meta["field_catalog_text"]
            
            # Auto-detect dataset type and field mappings
            from .dataset_detector import auto_detect_dataset_fields
            try:
                field_mapping = auto_detect_dataset_fields(nd.meta["field_catalog"])
                state.extra["field_mapping"] = field_mapping
                nd.meta["field_mapping"] = field_mapping
            except Exception as e:
                # Auto-detection failed, but don't fail the whole load
                nd.meta["field_mapping_error"] = str(e)

        except Exception as e:
            nd.meta["field_catalog_error"] = str(e)

        state.user_data = UserDataState(
            raw_data=df_meta,          # may be None
            dataset_info=info,
        )
        
        # Store original path and temp dir info for cleanup
        if temp_dir:
            state.extra["_temp_extract_dir"] = temp_dir
            state.extra["_archive_path"] = path
            state.extra["_extracted_file"] = actual_dataset_path

        return state
        
    except Exception as e:
        # Clean up temp directory on error
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        raise