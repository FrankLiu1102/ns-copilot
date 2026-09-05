"""
CBraMod (Criss-Cross Brain Modulation) Wrapper for NS-Copilot

Provides a generic interface to extract features from the pretrained CBraMod
foundation model for EEG data classification and analysis.

This wrapper ONLY handles CBraMod-specific logic:
  - Model loading (from local checkpoint)
  - EEG preprocessing (resample to 200Hz, normalize, segment into 1s patches)
  - Feature extraction (raw EEG -> 200-dim latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: Wang et al. (2025) "CBraMod: A Criss-Cross Brain Foundation Model
           for EEG Decoding" (ICLR 2025)
GitHub: https://github.com/wjq-learning/CBraMod
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List

from core.models.base_eeg_wrapper import BaseEEGDecoder, find_repo, find_checkpoint


# Locate CBraMod repo
CBRAMOD_PATH = find_repo(
    'CBRAMOD_PATH',
    ['/app/cbramod', './models/cbramod', '../models/cbramod',
     os.path.expanduser('~/data/cbramod')],
    os.path.join('models', 'cbramod.py'),
)

CBRAMOD_AVAILABLE = False
_CBraMod_cls = None
if CBRAMOD_PATH is not None:
    try:
        from neuro_copilot.core.models.base_eeg_wrapper import import_from_repo
        _cbramod_mod = import_from_repo(CBRAMOD_PATH, 'models/cbramod.py')
        _CBraMod_cls = _cbramod_mod.CBraMod
        CBRAMOD_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import CBraMod: {e}")
else:
    print("Warning: CBraMod repo not found. Set CBRAMOD_PATH environment variable.")


def _find_checkpoint() -> Optional[str]:
    """Find CBraMod checkpoint in standard locations."""
    candidates = []
    if CBRAMOD_PATH is not None:
        candidates.append(os.path.join(CBRAMOD_PATH, 'pretrained_weights', 'pretrained_weights.pth'))
    candidates += [
        '/app/cbramod/pretrained_weights/pretrained_weights.pth',
        './models/cbramod/pretrained_weights/pretrained_weights.pth',
        '../models/cbramod/pretrained_weights/pretrained_weights.pth',
    ]
    return find_checkpoint('CBRAMOD_CHECKPOINT_PATH', candidates)


class CBraModDecoder(BaseEEGDecoder):
    """
    Generic wrapper for CBraMod feature extraction from EEG data.

    CBraMod uses a Criss-Cross Transformer with separate spatial and temporal
    attention streams. It accepts any number of EEG channels without requiring
    channel position information (ACPE learns positions via depthwise convolution).

    Usage:
        decoder = CBraModDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', 'Fp2', ...])
        # features: (600,) numpy array (200 mean + 200 std + 200 max)
    """

    TARGET_SFREQ = 200
    PATCH_SIZE = 200  # 1 second at 200Hz
    MAX_SEGMENT_SECONDS = 16.0
    SCALE_FACTOR = 100.0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device)

        if not CBRAMOD_AVAILABLE:
            raise ImportError(
                "CBraMod is not available. Please either:\n"
                "  1. Clone https://github.com/wjq-learning/CBraMod\n"
                "  2. Set CBRAMOD_PATH environment variable"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "CBraMod checkpoint not found. Please either:\n"
                "  1. Download from https://huggingface.co/weighting666/CBraMod\n"
                "  2. Set CBRAMOD_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)

        print(f"Loading CBraMod from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained CBraMod model from checkpoint."""
        CBraMod = _CBraMod_cls

        # Load checkpoint to infer embed_dim from state_dict
        state_dict = torch.load(str(self.checkpoint_path), map_location='cpu', weights_only=True)
        # Infer d_model from a known layer's weight shape
        for key in state_dict:
            if 'encoder_layer' in key and key.endswith('.weight') and state_dict[key].ndim == 2:
                self._embed_dim = state_dict[key].shape[-1]
                break
        else:
            self._embed_dim = 200  # fallback if inference fails

        self.model = CBraMod(
            in_dim=self._embed_dim, out_dim=self._embed_dim, d_model=self._embed_dim,
            dim_feedforward=self._embed_dim * 4, seq_len=30, n_layer=12, nhead=8,
        )
        self.model.load_state_dict(state_dict)
        self.model.proj_out = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        print(f"CBraMod model loaded: embed_dim={self._embed_dim}, device={self.device}")

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
                f"EEG segment too short for CBraMod. "
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
