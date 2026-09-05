"""
REVE Tool Specification for NS-Copilot Tool Registry

This provides a GENERIC interface description for REVE (Representation from
EEG with Vision-inspired Encoding). The tool ONLY does feature extraction.
All downstream logic (classification, metrics, visualization) is generated
by the LLM.
"""

from typing import Dict, Any, List
import numpy as np


# Tool Registry Entry
REVE_TOOL_SPEC = {
    "name": "decode_with_reve",
    "description": """
Extract features from EEG data using the pretrained REVE foundation model.

REVE is a Transformer-based foundation model pretrained on 60,000+ hours of
EEG data from 92 datasets spanning 25,000 subjects — the largest EEG pretraining
effort to date. It outputs 512-dimensional latent representations that capture
universal EEG patterns across tasks, subjects, and electrode configurations.

REVE uses 4D positional encoding (3D electrode coordinates + temporal position),
enabling it to handle ANY electrode montage without retraining.

Features are extracted using multi-statistic pooling (mean + std + max) across
channels and time patches, producing 1536-dim vectors (512 * 3) that capture
central tendency, variability, and peak activations.

**When to use:**
- Data modality: Raw EEG signals (.set, .edf, .bdf)
- Channel system: Any standard system (10-20, 10-10, 10-05, BioSemi, EGI)
- Task type: Any EEG classification, regression, or feature extraction
- Especially strong with frozen features (no fine-tuning needed)
- Preferred over LaBraM for linear probing tasks

**Input:**
- eeg_data: List of (n_channels, n_samples) arrays, one per subject
  Data should be in microvolts. Any sampling rate supported (auto-resampled to 200Hz).
- sfreq: Sampling frequency in Hz
- ch_names: List of channel names in standard notation

**Output:**
- features: (n_subjects, 1536) latent representations (512-dim mean + 512-dim std + 512-dim max)
  The Code Generator must handle all downstream analysis.
""",
    "input_spec": {
        "eeg_data": (
            "list of numpy arrays, each (n_channels, n_samples), raw EEG per subject. "
            "Any sampling rate supported (auto-resampled to 200Hz internally)."
        ),
        "sfreq": "float, sampling frequency in Hz",
        "ch_names": (
            "list of str, channel names in standard notation "
            "(e.g., ['Fp1', 'Fp2', 'F3', 'F4', ...]). Supports 10-20, 10-10, 10-05."
        ),
        "max_seconds": (
            "float or None, max duration per subject in seconds (default: use all)"
        ),
    },
    "output_spec": {
        "features": "numpy array (n_subjects, 1536), latent features (mean+std+max pooling)",
    },
    "constraints": {
        "data_modality": ["eeg", "raw_eeg", "eeglab_set", "edf", "bdf"],
        "min_channels": 1,
        "max_channels": 256,
        "recommended_channels": 19,
        "channel_system": ["10-20", "10-10", "10-05", "biosemi", "egi"],
        "target_sfreq": 200,
        "requires_gpu": True,
    },
    "tags": [
        "feature_extraction", "foundation_model", "eeg",
        "pretrained", "transformer", "10-20", "10-10",
        "frozen_probing", "universal",
    ],
}


def reve_extract_features(
    eeg_data: list,
    sfreq: float,
    ch_names: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract REVE features from EEG data.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays per subject
        sfreq: Sampling frequency
        ch_names: Channel names in standard notation

    Returns dict with 'features' key: (n_subjects, 1536) numpy array.
    """
    from neuro_copilot.core.models.reve_wrapper import REVEDecoder

    model_name = kwargs.get('model_name', 'brain-bzh/reve-base')
    max_seconds = kwargs.get('max_seconds', None)

    decoder = REVEDecoder(model_name)
    features = decoder.extract_features_batch(
        eeg_data, sfreq, ch_names, max_seconds
    )

    return {
        'features': features,
        'n_subjects': features.shape[0],
        'embed_dim': features.shape[1],
        'checkpoint_path': decoder.model_name,
    }
