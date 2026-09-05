"""
REVE (Representation for EEG with Versatile Embeddings) Wrapper for NS-Copilot

Provides a generic interface to extract features from the pretrained REVE foundation model
for EEG data classification and analysis.

This wrapper ONLY handles REVE-specific logic:
  - Model loading (from HuggingFace via transformers AutoModel)
  - EEG preprocessing (resample to 200Hz, z-score normalize, clip at 15 std)
  - Electrode position lookup (3D coordinates from HuggingFace position bank)
  - Feature extraction (raw EEG -> 512-dim latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: "REVE: A Foundation Model for EEG -- Adapting to Any Setup with
           Large-Scale Pretraining on 25,000 Subjects" (2025)
HuggingFace: https://huggingface.co/brain-bzh/reve-base
"""

import os
import torch
import numpy as np
from typing import Optional, List

from core.models.base_eeg_wrapper import BaseEEGDecoder


# Clean up broken deepspeed module that may be injected by other models
# (e.g., BrainOmni's mamba-ssm dependency). A phantom deepspeed with
# __spec__=None causes transformers to crash on import.
import sys as _sys
if "deepspeed" in _sys.modules and getattr(_sys.modules["deepspeed"], "__spec__", None) is None:
    del _sys.modules["deepspeed"]

REVE_AVAILABLE = False
try:
    from transformers import AutoModel  # noqa: F401
    REVE_AVAILABLE = True
except ImportError:
    pass


class REVEDecoder(BaseEEGDecoder):
    """
    Generic wrapper for REVE feature extraction from EEG data.

    Uses transformers AutoModel with trust_remote_code=True to load
    the REVE model from HuggingFace. Works with Python 3.10+.

    Usage:
        decoder = REVEDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', 'Fp2', ...])
        # features: (1536,) numpy array (512 mean + 512 std + 512 max)
    """

    TARGET_SFREQ = 200
    PATCH_SIZE = 200
    CLIP_STD = 15.0
    MAX_SEGMENT_SECONDS = 16.0

    def __init__(
        self,
        model_name: str = "brain-bzh/reve-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device)

        if not REVE_AVAILABLE:
            raise ImportError(
                "REVE requires the transformers library.\n"
                "  pip install transformers\n"
                "You also need HuggingFace authentication:\n"
                "  huggingface-cli login"
            )

        self.model_name = model_name
        self.pos_bank = None
        self._embed_dim = None
        self._feature_dim = None
        self._positions_cache = {}

        print(f"Loading REVE model from {self.model_name}")
        self._load_model()

    def _load_model(self):
        """Load pretrained REVE model and position bank from HuggingFace."""
        from transformers import AutoModel

        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(self.device)

        embed_dim = 512
        if hasattr(self.model, 'config'):
            cfg = self.model.config
            for attr in ('embed_dim', 'hidden_size', 'd_model'):
                if hasattr(cfg, attr):
                    embed_dim = getattr(cfg, attr)
                    break
        self._embed_dim = embed_dim
        self._feature_dim = embed_dim * 3
        print(f"REVE model loaded: embed_dim={self._embed_dim}, feature_dim={self._feature_dim}, device={self.device}")

    @property
    def feature_dim(self) -> int:
        return self._feature_dim  # 1536

    def _get_positions(self, ch_names: List[str]) -> torch.Tensor:
        """Look up 3D electrode positions for given channel names."""
        cache_key = tuple(ch_names)
        if cache_key in self._positions_cache:
            return self._positions_cache[cache_key]
        positions = self.pos_bank(ch_names)
        self._positions_cache[cache_key] = positions
        return positions

    def _prepare(self, eeg_data, sfreq, ch_names):
        positions = self._get_positions(ch_names)
        return eeg_data, sfreq, {'positions': positions}

    def _preprocess_segment(self, segment, sfreq, **ctx):
        segment = self._resample(segment, sfreq, self.TARGET_SFREQ)
        n_samples_200 = segment.shape[1]

        if n_samples_200 < self.PATCH_SIZE:
            raise ValueError(
                f"EEG segment too short for REVE. "
                f"Need at least {self.PATCH_SIZE} samples at {self.TARGET_SFREQ}Hz, "
                f"got {n_samples_200}."
            )

        # Per-channel z-score + clip
        segment = segment.astype(np.float32)
        mean = segment.mean(axis=1, keepdims=True)
        std = segment.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        segment = (segment - mean) / std
        segment = np.clip(segment, -self.CLIP_STD, self.CLIP_STD)

        return torch.FloatTensor(segment).unsqueeze(0)

    def _encode_segment(self, x, **ctx):
        positions = ctx['positions']
        pos = positions.unsqueeze(0).to(self.device)

        if self.device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(x, pos)
        else:
            outputs = self.model(x, pos)

        return self._multi_stat_pool(outputs)
