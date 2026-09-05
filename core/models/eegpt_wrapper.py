"""
EEGPT Wrapper for NS-Copilot

Provides a generic interface to extract features from the pretrained EEGPT
foundation model for EEG data classification and analysis.

This wrapper ONLY handles EEGPT-specific logic:
  - Model loading (from local checkpoint)
  - Channel name mapping (standard 10-10 system)
  - EEG preprocessing (resample to 256Hz, segment into 4s windows)
  - Feature extraction (raw EEG -> 2048-dim latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: Zhang et al. (2024) "EEGPT: Unleashing the Potential of EEG
           Generalist Foundation Model by Autoregressive Pre-training"
           (NeurIPS 2024)
GitHub: https://github.com/jackbauer0/EEGPT
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple

from core.models.base_eeg_wrapper import BaseEEGDecoder, find_repo, find_checkpoint


# Old 10-20 channel names -> modern 10-10 equivalents
_OLD_CHANNEL_MAP = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    'A1': 'TP9', 'A2': 'TP10',
}


# Locate EEGPT repo
EEGPT_PATH = find_repo(
    'EEGPT_PATH',
    ['/app/eegpt', './models/eegpt', '../models/eegpt',
     os.path.expanduser('~/data/eegpt')],
    os.path.join('downstream', 'Modules', 'models', 'EEGPT_mcae.py'),
)

EEGPT_AVAILABLE = False
EEGPT_MODULE_PATH = None
if EEGPT_PATH is not None:
    _downstream_path = os.path.join(EEGPT_PATH, 'downstream')
    if os.path.isdir(_downstream_path):
        EEGPT_MODULE_PATH = _downstream_path
        if _downstream_path not in sys.path:
            sys.path.insert(0, _downstream_path)
        try:
            from Modules.models.EEGPT_mcae import EEGTransformer, CHANNEL_DICT  # noqa: F401
            EEGPT_AVAILABLE = True
        except ImportError as e:
            print(f"Warning: Could not import EEGPT: {e}")
    else:
        print(f"Warning: EEGPT downstream directory not found at {_downstream_path}")
else:
    print("Warning: EEGPT repo not found. Set EEGPT_PATH environment variable.")


def _find_checkpoint() -> Optional[str]:
    """Find EEGPT checkpoint in standard locations."""
    checkpoint_name = 'eegpt_mcae_58chs_4s_large4E.ckpt'
    candidates = []
    if EEGPT_PATH is not None:
        candidates.append(os.path.join(EEGPT_PATH, 'checkpoint', checkpoint_name))
    candidates += [
        f'/app/eegpt/checkpoint/{checkpoint_name}',
        f'./models/eegpt/checkpoint/{checkpoint_name}',
        f'../models/eegpt/checkpoint/{checkpoint_name}',
    ]
    return find_checkpoint('EEGPT_CHECKPOINT_PATH', candidates)


def _normalize_channel_name(ch: str) -> str:
    """Normalize a channel name to uppercase, strip dots, map old names."""
    ch = ch.upper().strip().strip('.')
    return _OLD_CHANNEL_MAP.get(ch, ch)


class EEGPTDecoder(BaseEEGDecoder):
    """
    Generic wrapper for EEGPT feature extraction from EEG data.

    EEGPT is a ViT-based foundation model pretrained on 58-channel EEG data
    using mask-based dual self-supervised learning. It uses channel-specific
    embeddings from a 62-entry dictionary (standard 10-10 system) and processes
    4-second segments at 256Hz.

    Features are 2048-dimensional (4 summary tokens x 512 embedding dim).

    Usage:
        decoder = EEGPTDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', 'Fp2', ...])
        # features: (2048,) numpy array
    """

    TARGET_SFREQ = 256
    SEGMENT_SECONDS = 4
    SEGMENT_SAMPLES = 1024  # 4 * 256
    PATCH_SIZE = 64
    EMBED_DIM = 512
    EMBED_NUM = 4
    FEATURE_DIM = 2048  # EMBED_NUM * EMBED_DIM

    MAX_SEGMENT_SECONDS = 4.0
    MIN_SEGMENT_SECONDS = 2.0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device)

        if not EEGPT_AVAILABLE:
            raise ImportError(
                "EEGPT is not available. Please either:\n"
                "  1. Clone https://github.com/jackbauer0/EEGPT\n"
                "  2. Set EEGPT_PATH environment variable"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "EEGPT checkpoint not found. Please either:\n"
                "  1. Download eegpt_mcae_58chs_4s_large4E.ckpt from Figshare\n"
                "  2. Set EEGPT_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self._channel_dict = None

        print(f"Loading EEGPT from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained EEGPT model from checkpoint."""
        from Modules.models.EEGPT_mcae import EEGTransformer, CHANNEL_DICT

        self._channel_dict = CHANNEL_DICT

        self.model = EEGTransformer(
            img_size=[len(CHANNEL_DICT), self.SEGMENT_SAMPLES],
            patch_size=self.PATCH_SIZE,
            embed_num=self.EMBED_NUM,
            embed_dim=self.EMBED_DIM,
            depth=8, num_heads=8, mlp_ratio=4.0,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
            init_std=0.02, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        ckpt = torch.load(
            str(self.checkpoint_path), map_location='cpu', weights_only=False,
        )

        state_dict = {}
        raw_state = ckpt.get('state_dict', ckpt)
        for k, v in raw_state.items():
            if k.startswith('target_encoder.'):
                state_dict[k[len('target_encoder.'):]] = v

        if not state_dict:
            state_dict = raw_state

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        print(
            f"EEGPT model loaded: embed_dim={self.EMBED_DIM}, "
            f"embed_num={self.EMBED_NUM}, feature_dim={self.FEATURE_DIM}, "
            f"device={self.device}"
        )

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM  # 2048

    def _map_channels(
        self, ch_names: List[str],
    ) -> Tuple[List[int], torch.Tensor]:
        """Map input channel names to EEGPT's channel dictionary."""
        matched_indices = []
        chan_ids = []
        for i, ch in enumerate(ch_names):
            normalized = _normalize_channel_name(ch)
            if normalized in self._channel_dict:
                matched_indices.append(i)
                chan_ids.append(self._channel_dict[normalized])
        if not matched_indices:
            raise ValueError(
                f"No channels match EEGPT's 10-10 dictionary. "
                f"Input channels: {ch_names[:10]}... "
                f"Expected: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, etc."
            )
        return matched_indices, torch.tensor(chan_ids).unsqueeze(0).long()

    def _prepare(self, eeg_data, sfreq, ch_names):
        matched_indices, chan_ids = self._map_channels(ch_names)
        chan_ids = chan_ids.to(self.device)
        eeg_data = eeg_data[matched_indices, :]

        print(
            f"  EEGPT: {len(matched_indices)}/{len(ch_names)} channels matched, "
            f"{eeg_data.shape[1] / sfreq:.1f}s recording"
        )

        return eeg_data, sfreq, {'chan_ids': chan_ids}

    def _preprocess_segment(self, segment, sfreq, **ctx):
        segment = self._resample(segment, sfreq, self.TARGET_SFREQ)
        n_samples_256 = segment.shape[1]

        if n_samples_256 >= self.SEGMENT_SAMPLES:
            segment = segment[:, :self.SEGMENT_SAMPLES]
        else:
            pad_width = self.SEGMENT_SAMPLES - n_samples_256
            segment = np.pad(segment, ((0, 0), (0, pad_width)), mode='constant')

        # Per-channel z-score normalization
        means = segment.mean(axis=1, keepdims=True)
        stds = segment.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1.0
        segment = (segment - means) / stds

        return torch.FloatTensor(segment).unsqueeze(0)

    def _encode_segment(self, x, **ctx):
        chan_ids = ctx['chan_ids']
        n_channels = x.shape[1]

        # Dynamically adjust num_patches to match input channel count
        original_num_patches = self.model.num_patches
        self.model.num_patches = (n_channels, self.SEGMENT_SAMPLES // self.PATCH_SIZE)
        self.model.patch_embed.num_patches = self.model.num_patches

        try:
            out = self.model(x, chan_ids=chan_ids)
            out = out.flatten(2)  # (1, 16, 4, 512) -> (1, 16, 2048)
            return out.mean(dim=1)  # (1, 2048)
        finally:
            self.model.num_patches = original_num_patches
            self.model.patch_embed.num_patches = original_num_patches
