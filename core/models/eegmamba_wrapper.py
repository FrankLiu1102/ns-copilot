"""
EEGMamba Wrapper for NS-Copilot

Provides a generic interface to extract features from the pretrained EEGMamba
foundation model for EEG data classification and analysis.

EEGMamba uses a bidirectional Mamba (SSM) architecture with the same input/output
format as CBraMod (same author). It accepts any number of EEG channels without
requiring channel position information.

Reference: Wang et al. (2025) "EEGMamba: An EEG Foundation Model with Mamba"
           (Neural Networks 2025)
GitHub: https://github.com/wjq-learning/EEGMamba
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List

from core.models.base_eeg_wrapper import BaseEEGDecoder, find_repo, find_checkpoint


# Locate EEGMamba repo
EEGMAMBA_PATH = find_repo(
    'EEGMAMBA_PATH',
    ['/app/eegmamba', './models/eegmamba', '../models/eegmamba',
     os.path.expanduser('~/data/EEGMamba')],
    os.path.join('models', 'eegmamba.py'),
)

EEGMAMBA_AVAILABLE = False
_EEGMamba_cls = None
if EEGMAMBA_PATH is not None:
    try:
        from neuro_copilot.core.models.base_eeg_wrapper import import_from_repo
        _eegmamba_mod = import_from_repo(EEGMAMBA_PATH, 'models/eegmamba.py')
        _EEGMamba_cls = _eegmamba_mod.EEGMamba
        EEGMAMBA_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import EEGMamba: {e}")
else:
    print("Warning: EEGMamba repo not found. Set EEGMAMBA_PATH environment variable.")


def _find_checkpoint() -> Optional[str]:
    """Find EEGMamba checkpoint in standard locations."""
    candidates = []
    if EEGMAMBA_PATH is not None:
        candidates.append(os.path.join(EEGMAMBA_PATH, 'pretrained_weights', 'pretrained_EEGMamba.pth'))
    candidates += [
        '/app/eegmamba/pretrained_weights/pretrained_EEGMamba.pth',
        './models/eegmamba/pretrained_weights/pretrained_EEGMamba.pth',
        '../models/eegmamba/pretrained_weights/pretrained_EEGMamba.pth',
    ]
    return find_checkpoint('EEGMAMBA_CHECKPOINT_PATH', candidates)


class EEGMambaDecoder(BaseEEGDecoder):
    """
    Generic wrapper for EEGMamba feature extraction from EEG data.

    EEGMamba uses a bidirectional Mamba (SSM) encoder with the same
    input/output format as CBraMod: (batch, channels, patches, 200).
    It is channel-agnostic via depthwise Conv2d positional encoding.

    Usage:
        decoder = EEGMambaDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', ...])
        # features: (600,) numpy array (200 mean + 200 std + 200 max)
    """

    TARGET_SFREQ = 200
    PATCH_SIZE = 200
    MAX_SEGMENT_SECONDS = 16.0
    SCALE_FACTOR = 100.0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device)

        if not EEGMAMBA_AVAILABLE:
            raise ImportError(
                "EEGMamba is not available. Please either:\n"
                "  1. Clone https://github.com/wjq-learning/EEGMamba\n"
                "  2. Set EEGMAMBA_PATH environment variable"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "EEGMamba checkpoint not found. Please either:\n"
                "  1. Download from https://huggingface.co/weighting666/EEGMamba\n"
                "  2. Set EEGMAMBA_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self._embed_dim = 200

        print(f"Loading EEGMamba from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained EEGMamba model from checkpoint."""
        EEGMamba = _EEGMamba_cls

        self.model = EEGMamba(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
        )
        self.model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location='cpu', weights_only=True)
        )
        self.model.proj_out = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        print(f"EEGMamba model loaded: embed_dim={self._embed_dim}, device={self.device}")

    @property
    def feature_dim(self) -> int:
        return self._embed_dim * 3  # 200 * 3 = 600

    def _preprocess_segment(self, segment, sfreq, **ctx):
        n_channels, n_samples = segment.shape
        segment = self._resample(segment, sfreq, self.TARGET_SFREQ)
        n_samples_200 = segment.shape[1]

        n_patches = n_samples_200 // self.PATCH_SIZE
        if n_patches < 1:
            raise ValueError(
                f"EEG segment too short for EEGMamba. "
                f"Need at least {self.PATCH_SIZE} samples at {self.TARGET_SFREQ}Hz, "
                f"got {n_samples_200}."
            )
        n_patches = min(n_patches, int(self.MAX_SEGMENT_SECONDS))
        segment = segment[:, :n_patches * self.PATCH_SIZE]
        segment = segment.reshape(n_channels, n_patches, self.PATCH_SIZE)
        return torch.FloatTensor(segment).unsqueeze(0) / self.SCALE_FACTOR

    def _encode_segment(self, x, **ctx):
        feats = self.model(x)
        return self._multi_stat_pool(feats)
