"""
BrainOmni Tool Specification for NS-Copilot Tool Registry

This provides a GENERIC interface description for BrainOmni (Brain Foundation
Model for Unified EEG and MEG Signals). The tool ONLY does feature extraction.
All downstream logic (classification, metrics, visualization) is generated
by the LLM.
"""

from typing import Dict, Any, List
import numpy as np


# Tool Registry Entry
BRAINOMNI_TOOL_SPEC = {
    "name": "decode_with_brainomni",
    "description": """
Extract features from EEG data using the pretrained BrainOmni foundation model.

BrainOmni is a Criss-Cross Transformer with Sensor Encoder pretrained on
~2,650 hours of EEG and MEG data from multiple public datasets. It uses
BrainTokenizer (RVQ) + masked prediction for self-supervised pretraining.

BrainOmni uses 3D electrode positions obtained automatically from MNE standard
montages, supporting any standard EEG electrode system (10-20, 10-10, etc.).

Features are extracted by encoding EEG into 16 latent source variables with
256-dim representations, then mean-pooling over time to produce 4096-dim vectors.

**When to use:**
- Data modality: Raw EEG signals (.set, .edf, .bdf)
- Channel system: Any standard EEG system (10-20, 10-10, etc.)
- Task type: Any EEG classification, regression, or feature extraction
- Best frozen-feature performance among available EEG foundation models

**Input:**
- eeg_data: List of (n_channels, n_samples) arrays, one per subject
  Data should be in microvolts. Any sampling rate supported (auto-resampled to 256Hz).
- sfreq: Sampling frequency in Hz
- ch_names: List of channel names (standard names like 'Fp1', 'Cz', etc.)

**Output:**
- features: (n_subjects, 4096) latent representations (16 sources × 256 dims)
  The Code Generator must handle all downstream analysis.
""",
    "input_spec": {
        "eeg_data": (
            "list of numpy arrays, each (n_channels, n_samples), raw EEG per subject. "
            "Any sampling rate supported (auto-resampled to 256Hz internally)."
        ),
        "sfreq": "float, sampling frequency in Hz",
        "ch_names": (
            "list of str, channel names (standard names — positions obtained from MNE montages)."
        ),
        "max_seconds": (
            "float or None, max duration per subject in seconds (default: use all)"
        ),
    },
    "output_spec": {
        "features": "numpy array (n_subjects, 4096), latent features (16 sources × 256 dims)",
    },
    "constraints": {
        "data_modality": ["eeg", "raw_eeg", "eeglab_set", "edf", "bdf"],
        "min_channels": 1,
        "max_channels": 256,
        "channel_system": ["standard_1020", "standard_1005", "standard_10-20"],
        "target_sfreq": 256,
        "requires_gpu": True,
    },
    "tags": [
        "feature_extraction", "foundation_model", "eeg",
        "pretrained", "criss_cross_transformer", "sensor_encoder",
        "frozen_probing", "clinical", "neurips_2025",
    ],
}


def brainomni_extract_features(
    eeg_data: list,
    sfreq: float,
    ch_names: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract BrainOmni features from EEG data.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays per subject
        sfreq: Sampling frequency
        ch_names: Channel names (standard names)

    Returns dict with 'features' key: (n_subjects, 4096) numpy array.
    """
    from neuro_copilot.core.models.brainomni_wrapper import BrainOmniDecoder

    checkpoint_dir = kwargs.get('checkpoint_dir', None)
    max_seconds = kwargs.get('max_seconds', None)

    decoder = BrainOmniDecoder(checkpoint_dir)
    features = decoder.extract_features_batch(
        eeg_data, sfreq, ch_names, max_seconds
    )

    return {
        'features': features,
        'n_subjects': features.shape[0],
        'embed_dim': features.shape[1],
        'checkpoint_path': str(decoder.checkpoint_path),
    }
