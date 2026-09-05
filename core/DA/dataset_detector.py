# neuro_copilot/dataset_detector.py

"""
Automatic dataset type detection and field mapping.
This module detects dataset types and field patterns from field_catalog,
eliminating the need for hardcoded field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import re


@dataclass
class FieldMapping:
    """Detected field mappings for a dataset."""
    dataset_type: str  # e.g., "odr", "generic", "memory_task"
    
    # Neural activity fields
    neural_activity_fields: List[str]  # e.g., ["TS", "spike_times"]
    primary_neural_activity: str  # Primary field to use, e.g., "TS"
    
    # Target/label fields (for decoding)
    target_fields: List[str]  # e.g., ["cue_location", "stimulus_location"]
    primary_target: Optional[str]  # Primary target field, e.g., "cue_location"
    
    # Firing rate fields (if precomputed)
    firing_rate_fields: Dict[str, List[str]]  # e.g., {"fixation": ["fixrate"], "cue": ["cuerate"]}
    
    # Time event fields (for computing firing rates)
    time_event_fields: Dict[str, List[str]]  # e.g., {"cue_on": ["Cue_onT"], "fix_off": ["Fix_offT"]}
    
    # Response time fields
    response_time_fields: List[str]  # e.g., ["RT", "reaction_time"]
    
    # Confidence score
    confidence: float  # 0.0 to 1.0, how confident we are in this detection


def _normalize_field_name(name: str) -> str:
    """Normalize field name for comparison."""
    return name.lower().strip().replace("_", "").replace("-", "")


def _matches_pattern(field_name: str, patterns: List[str]) -> bool:
    """Check if field name matches any pattern."""
    normalized = _normalize_field_name(field_name)
    for pattern in patterns:
        pattern_norm = _normalize_field_name(pattern)
        # Exact match
        if normalized == pattern_norm:
            return True
        # Wildcard match (e.g., "*rate" matches "fixrate", "cuerate")
        if "*" in pattern_norm:
            regex = pattern_norm.replace("*", ".*")
            if re.match(regex, normalized):
                return True
    return False


def _detect_neural_activity_fields(fields: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Detect neural activity fields (spike times, neural signals).
    
    Returns:
        (list of field names, primary field name)
    """
    # Common patterns for neural activity
    patterns = [
        "spike_times", "spiketimes", "spikes",
        "neural_activity", "neuralactivity", "neural",
        "spike_times_array", "spike_times_list"
    ]

    detected = []
    for field_name in fields.keys():
        if _matches_pattern(field_name, patterns):
            detected.append(field_name)

    # Use first match (no field-name preference)
    if detected:
        primary = detected[0]
    else:
        # Fallback: look for any array-like field that might be neural data
        for field_name, field_info in fields.items():
            if isinstance(field_info, dict):
                dtype = field_info.get("dtype", "")
                shape = field_info.get("shape", [])
                # Look for array fields that might be neural data
                if "array" in dtype.lower() or (isinstance(shape, list) and len(shape) > 0):
                    detected.append(field_name)
                    if not primary:
                        primary = field_name
    
    return detected if detected else [], primary if detected else None


def _detect_target_fields(fields: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
    """
    Detect target/label fields for decoding tasks.
    
    Returns:
        (list of field names, primary field name)
    """
    # Common patterns for target fields
    patterns = [
        "target", "label", "class", "category",
        "condition", "choice", "direction", "location"
    ]
    
    detected = []
    for field_name in fields.keys():
        normalized = _normalize_field_name(field_name)
        # Check exact matches
        if _matches_pattern(field_name, patterns):
            detected.append(field_name)
        # Check if field name contains location-related keywords
        elif any(keyword in normalized for keyword in ["location", "position", "direction", "angle"]):
            detected.append(field_name)
    
    # Use first detected target field (no field-name preference)
    primary = detected[0] if detected else None
    
    return detected, primary


def _detect_firing_rate_fields(fields: Dict[str, Any]) -> Dict[str, List[str]]:
    """Detect precomputed firing rate fields. Currently disabled."""
    return {}


def _detect_time_event_fields(fields: Dict[str, Any]) -> Dict[str, List[str]]:
    """Detect time event fields. Currently disabled."""
    return {}


def _detect_response_time_fields(fields: Dict[str, Any]) -> List[str]:
    """Detect response time fields."""
    patterns = ["rt", "reaction_time", "responsetime", "response_time", "reactiontime"]
    
    detected = []
    for field_name in fields.keys():
        if _matches_pattern(field_name, patterns):
            detected.append(field_name)
    
    return detected if detected else []


def _detect_dataset_type(fields: Dict[str, Any], field_mapping: FieldMapping) -> str:
    """
    Detect dataset type based on field patterns.
    
    Returns:
        Dataset type string (e.g., "odr", "generic")
    """
    # Dataset type detection — generic only (no task-specific heuristics)
    return "generic"


def auto_detect_dataset_fields(field_catalog: Dict[str, Any]) -> FieldMapping:
    """
    Automatically detect dataset type and field mappings from field_catalog.
    
    Args:
        field_catalog: Field catalog dictionary from dataloader
        
    Returns:
        FieldMapping object with detected fields
    """
    # Extract fields from catalog
    record_stats = field_catalog.get("record_field_stats", {})
    fields = record_stats.get("fields", {})
    
    if not fields:
        # Fallback: try to get fields from top level
        fields = {k: v for k, v in field_catalog.items() if isinstance(v, dict)}
    
    # Detect different field types
    neural_activity_fields, primary_neural = _detect_neural_activity_fields(fields)
    target_fields, primary_target = _detect_target_fields(fields)
    firing_rate_fields = _detect_firing_rate_fields(fields)
    time_event_fields = _detect_time_event_fields(fields)
    response_time_fields = _detect_response_time_fields(fields)
    
    # Create initial mapping
    field_mapping = FieldMapping(
        dataset_type="generic",  # Will be updated
        neural_activity_fields=neural_activity_fields,
        primary_neural_activity=primary_neural,
        target_fields=target_fields,
        primary_target=primary_target,
        firing_rate_fields=firing_rate_fields,
        time_event_fields=time_event_fields,
        response_time_fields=response_time_fields,
        confidence=0.5,  # Default confidence
    )
    
    # Detect dataset type
    dataset_type = _detect_dataset_type(fields, field_mapping)
    field_mapping.dataset_type = dataset_type
    
    # Calculate confidence based on how many fields we detected
    total_detected = (
        len(neural_activity_fields) +
        len(target_fields) +
        len(firing_rate_fields) +
        len(time_event_fields) +
        len(response_time_fields)
    )
    
    if total_detected >= 5:
        field_mapping.confidence = 0.9
    elif total_detected >= 3:
        field_mapping.confidence = 0.7
    elif total_detected >= 1:
        field_mapping.confidence = 0.5
    else:
        field_mapping.confidence = 0.3
    
    return field_mapping


def get_field_mapping(state: "NeuroGlobalState") -> Optional[FieldMapping]:
    """
    Get field mapping from state, or auto-detect if not present.
    
    Args:
        state: Global state
        
    Returns:
        FieldMapping object or None if detection fails
    """
    # Check if already detected and cached
    if "field_mapping" in state.extra:
        return state.extra["field_mapping"]
    
    # Get field catalog
    field_catalog = state.extra.get("field_catalog")
    if not field_catalog:
        return None
    
    # Auto-detect
    field_mapping = auto_detect_dataset_fields(field_catalog)
    
    # Cache in state
    state.extra["field_mapping"] = field_mapping
    
    return field_mapping









