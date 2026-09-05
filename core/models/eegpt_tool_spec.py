"""
EEGPT Tool Specification for NS-Copilot Tool Registry

This provides a GENERIC interface description for EEGPT (NeurIPS 2024).
The tool ONLY does feature extraction.
All downstream logic (classification, metrics, visualization) is generated
by the LLM.
"""

from typing import Dict, Any, List
import numpy as np


# Tool Registry Entry
EEGPT_TOOL_SPEC = {
    "name": "decode_with_eegpt",
    "description": """
Extract features from EEG data using the pretrained EEGPT foundation model.

EEGPT is a ViT-based foundation model pretrained on 58-channel EEG data using
mask-based dual self-supervised learning (NeurIPS 2024). It uses channel-specific
embeddings from a dictionary of 62 standard 10-10 channels and processes
4-second segments at 256Hz.

Features are extracted from summary tokens (4 tokens x 512-dim = 2048-dim),
mean-pooled across time patches and segments.

**When to use:**
- Data modality: Raw EEG signals (.set, .edf, .bdf)
- Channel system: Standard 10-10 or 10-20 system (auto-mapped to known channels)
- Task type: Any EEG classification, regression, or feature extraction
- Higher-dimensional features (2048-dim) than LaBraM/CBraMod/REVE
- Particularly suitable for multi-channel resting-state EEG

**Input:**
- eeg_data: List of (n_channels, n_samples) arrays, one per subject
  Data should be in microvolts. Any sampling rate supported (auto-resampled to 256Hz).
- sfreq: Sampling frequency in Hz
- ch_names: List of channel names (standard 10-10/10-20 naming required)

**Output:**
- features: (n_subjects, 2048) latent representations (4 summary tokens x 512-dim)
  The Code Generator must handle all downstream analysis.
""",
    "input_spec": {
        "eeg_data": (
            "list of numpy arrays, each (n_channels, n_samples), raw EEG per subject. "
            "Any sampling rate supported (auto-resampled to 256Hz internally)."
        ),
        "sfreq": "float, sampling frequency in Hz",
        "ch_names": (
            "list of str, channel names in standard 10-10 or 10-20 naming. "
            "Old names (T3, T4, T5, T6) are auto-mapped to modern equivalents (T7, T8, P7, P8)."
        ),
        "max_seconds": (
            "float or None, max duration per subject in seconds (default: use all)"
        ),
    },
    "output_spec": {
        "features": "numpy array (n_subjects, 2048), latent features from summary tokens",
    },
    "constraints": {
        "data_modality": ["eeg", "raw_eeg", "eeglab_set", "edf", "bdf"],
        "min_channels": 5,
        "max_channels": 64,
        "channel_system": ["10-20", "10-10"],
        "target_sfreq": 256,
        "requires_gpu": True,
    },
    "tags": [
        "feature_extraction", "foundation_model", "eeg",
        "pretrained", "transformer", "self_supervised",
        "NeurIPS_2024", "frozen_probing",
    ],
}


def eegpt_extract_features(
    eeg_data: list,
    sfreq: float,
    ch_names: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract EEGPT features from EEG data.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays per subject
        sfreq: Sampling frequency
        ch_names: Channel names (standard 10-10 or 10-20)

    Returns dict with 'features' key: (n_subjects, 2048) numpy array.
    """
    from neuro_copilot.core.models.eegpt_wrapper import EEGPTDecoder

    checkpoint_path = kwargs.get('checkpoint_path', None)
    max_seconds = kwargs.get('max_seconds', None)

    decoder = EEGPTDecoder(checkpoint_path)
    features = decoder.extract_features_batch(
        eeg_data, sfreq, ch_names, max_seconds
    )

    return {
        'features': features,
        'n_subjects': features.shape[0],
        'embed_dim': features.shape[1],
        'checkpoint_path': decoder.checkpoint_path,
    }
