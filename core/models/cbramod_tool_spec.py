"""
CBraMod Tool Specification for NS-Copilot Tool Registry

This provides a GENERIC interface description for CBraMod (Criss-Cross Brain
Foundation Model). The tool ONLY does feature extraction.
All downstream logic (classification, metrics, visualization) is generated
by the LLM.
"""

from typing import Dict, Any, List
import numpy as np


# Tool Registry Entry
CBRAMOD_TOOL_SPEC = {
    "name": "decode_with_cbramod",
    "description": """
Extract features from EEG data using the pretrained CBraMod foundation model.

CBraMod is a Criss-Cross Transformer pretrained on ~9,000 hours of EEG data
from the Temple University Hospital EEG Corpus (TUEG). It uses separate spatial
and temporal attention streams to capture both inter-channel and temporal patterns.

CBraMod uses ACPE (Asymmetric Conditional Positional Encoding) via depthwise
convolution, enabling it to handle ANY number of channels without requiring
explicit electrode positions or channel name mappings.

Features are extracted using multi-statistic pooling (mean + std + max) across
channels and time patches, producing 600-dim vectors (200 * 3).

**When to use:**
- Data modality: Raw EEG signals (.set, .edf, .bdf)
- Channel system: Any system, any number of channels (no position info needed)
- Task type: Any EEG classification, regression, or feature extraction
- Lightweight model (~4M parameters, fast inference)
- Good alternative when REVE is unavailable or LaBraM channels don't match

**Input:**
- eeg_data: List of (n_channels, n_samples) arrays, one per subject
  Data should be in microvolts. Any sampling rate supported (auto-resampled to 200Hz).
- sfreq: Sampling frequency in Hz
- ch_names: List of channel names (any naming convention)

**Output:**
- features: (n_subjects, 600) latent representations (200-dim mean + 200-dim std + 200-dim max)
  The Code Generator must handle all downstream analysis.
""",
    "input_spec": {
        "eeg_data": (
            "list of numpy arrays, each (n_channels, n_samples), raw EEG per subject. "
            "Any sampling rate supported (auto-resampled to 200Hz internally)."
        ),
        "sfreq": "float, sampling frequency in Hz",
        "ch_names": (
            "list of str, channel names (any naming convention — CBraMod is channel-agnostic)."
        ),
        "max_seconds": (
            "float or None, max duration per subject in seconds (default: use all)"
        ),
    },
    "output_spec": {
        "features": "numpy array (n_subjects, 600), latent features (mean+std+max pooling)",
    },
    "constraints": {
        "data_modality": ["eeg", "raw_eeg", "eeglab_set", "edf", "bdf"],
        "min_channels": 1,
        "max_channels": 256,
        "channel_system": ["any"],
        "target_sfreq": 200,
        "requires_gpu": True,
    },
    "tags": [
        "feature_extraction", "foundation_model", "eeg",
        "pretrained", "criss_cross_transformer", "channel_agnostic",
        "frozen_probing", "lightweight",
    ],
}


def cbramod_extract_features(
    eeg_data: list,
    sfreq: float,
    ch_names: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract CBraMod features from EEG data.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays per subject
        sfreq: Sampling frequency
        ch_names: Channel names (any convention)

    Returns dict with 'features' key: (n_subjects, 600) numpy array.
    """
    from neuro_copilot.core.models.cbramod_wrapper import CBraModDecoder

    checkpoint_path = kwargs.get('checkpoint_path', None)
    max_seconds = kwargs.get('max_seconds', None)

    decoder = CBraModDecoder(checkpoint_path)
    features = decoder.extract_features_batch(
        eeg_data, sfreq, ch_names, max_seconds
    )

    return {
        'features': features,
        'n_subjects': features.shape[0],
        'embed_dim': features.shape[1],
        'checkpoint_path': decoder.checkpoint_path,
    }
