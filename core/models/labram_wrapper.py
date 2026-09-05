"""
LaBraM (Large Brain Model) Wrapper for NS-Copilot

Provides a generic interface to extract features from pretrained LaBraM models
for EEG data classification and analysis.

This wrapper ONLY handles LaBraM-specific logic:
  - Model loading (checkpoint -> NeuralTransformer)
  - EEG preprocessing (resample to 200Hz, patch, normalize)
  - Feature extraction (raw EEG -> latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: Jiang et al. (2024) "Large Brain Model for Learning Generic
           Representations with Tremendous EEG Data in BCI" (ICLR 2024 Spotlight)
GitHub: https://github.com/935963004/LaBraM
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List

from core.models.base_eeg_wrapper import BaseEEGDecoder, find_repo, find_checkpoint


# Standard 10-20 channel order used by LaBraM for positional embeddings.
LABRAM_STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
]

# Locate LaBraM repo
LABRAM_PATH = find_repo(
    'LABRAM_PATH',
    ['/app/labram', './models/labram', '../models/labram',
     os.path.expanduser('~/data/labram')],
    'modeling_finetune.py',
)

LABRAM_AVAILABLE = False
if LABRAM_PATH is not None:
    if LABRAM_PATH not in sys.path:
        sys.path.insert(0, LABRAM_PATH)
    try:
        from modeling_finetune import labram_base_patch200_200  # noqa: F401
        LABRAM_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import LaBraM: {e}")
else:
    print("Warning: LaBraM repo not found. Set LABRAM_PATH environment variable.")


def _find_checkpoint() -> Optional[str]:
    """Find LaBraM checkpoint in standard locations."""
    candidates = []
    if LABRAM_PATH is not None:
        candidates.append(os.path.join(LABRAM_PATH, 'checkpoints', 'labram-base.pth'))
    candidates += [
        '/app/labram/checkpoints/labram-base.pth',
        './models/labram/checkpoints/labram-base.pth',
        '../models/labram/checkpoints/labram-base.pth',
    ]
    return find_checkpoint('LABRAM_CHECKPOINT_PATH', candidates)


def get_input_chans(ch_names: List[str]) -> List[int]:
    """
    Map channel names to LaBraM positional embedding indices.
    Returns list of indices: [0 (CLS token), ch1_idx+1, ch2_idx+1, ...]
    """
    input_chans = [0]  # CLS token
    upper_names = [name.upper() for name in LABRAM_STANDARD_1020]
    for ch_name in ch_names:
        ch_upper = ch_name.upper()
        if ch_upper in upper_names:
            input_chans.append(upper_names.index(ch_upper) + 1)
        else:
            raise ValueError(
                f"Channel '{ch_name}' not found in LaBraM's standard 10-20 system. "
                f"Supported channels: {LABRAM_STANDARD_1020[:20]}... (128 total)"
            )
    return input_chans


class LaBraMDecoder(BaseEEGDecoder):
    """
    Generic wrapper for LaBraM feature extraction from EEG data.

    Usage:
        decoder = LaBraMDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', 'Fp2', ...])
        # features: (200,) numpy array
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

        if not LABRAM_AVAILABLE:
            raise ImportError(
                "LaBraM is not available. Please either:\n"
                "  1. Clone https://github.com/935963004/LaBraM\n"
                "  2. Set LABRAM_PATH environment variable"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "LaBraM checkpoint not found. Please either:\n"
                "  1. Run setup.sh to download models\n"
                "  2. Set LABRAM_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self._embed_dim = None
        self._pool_method = "mean"

        print(f"Loading LaBraM from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained LaBraM-Base model from checkpoint."""
        from modeling_finetune import labram_base_patch200_200

        self.model = labram_base_patch200_200(
            num_classes=0, init_values=0.1, drop_path_rate=0.0,
        )

        checkpoint = torch.load(
            str(self.checkpoint_path), map_location='cpu', weights_only=False,
        )

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Strip 'student.' prefix and remap 'norm' -> 'fc_norm'
        model_state = self.model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            clean_k = k.replace('student.', '', 1) if k.startswith('student.') else k
            if clean_k.startswith('norm.') and clean_k not in model_state:
                remapped = clean_k.replace('norm.', 'fc_norm.', 1)
                if remapped in model_state and v.shape == model_state[remapped].shape:
                    filtered[remapped] = v
                    continue
            if clean_k in model_state and v.shape == model_state[clean_k].shape:
                filtered[clean_k] = v

        missing, _ = self.model.load_state_dict(filtered, strict=False)
        if missing:
            non_head_missing = [k for k in missing if not k.startswith('head.')]
            if non_head_missing:
                print(f"Warning: Missing keys in LaBraM: {non_head_missing[:5]}...")

        self.model.eval()
        self.model.to(self.device)
        self._embed_dim = self.model.embed_dim
        print(f"LaBraM model loaded: embed_dim={self._embed_dim}, device={self.device}")

    @property
    def feature_dim(self) -> int:
        return self._embed_dim  # 200

    def _prepare(self, eeg_data, sfreq, ch_names):
        input_chans = get_input_chans(ch_names)
        return eeg_data, sfreq, {'input_chans': input_chans}

    def _preprocess_segment(self, segment, sfreq, **ctx):
        n_channels, n_samples = segment.shape
        segment = self._resample(segment, sfreq, self.TARGET_SFREQ)
        n_samples_200 = segment.shape[1]

        n_patches = n_samples_200 // self.PATCH_SIZE
        if n_patches < 1:
            raise ValueError(
                f"EEG segment too short for LaBraM. "
                f"Need at least {self.PATCH_SIZE} samples at {self.TARGET_SFREQ}Hz, "
                f"got {n_samples_200}."
            )
        n_patches = min(n_patches, int(self.MAX_SEGMENT_SECONDS))
        segment = segment[:, :n_patches * self.PATCH_SIZE]
        segment = segment.reshape(n_channels, n_patches, self.PATCH_SIZE)
        return torch.FloatTensor(segment).unsqueeze(0) / self.SCALE_FACTOR

    def _encode_segment(self, x, **ctx):
        input_chans = ctx['input_chans']
        if self._pool_method == "cls":
            features = self.model.forward_features(
                x, input_chans=input_chans, return_all_tokens=True
            )
            return features[:, 0, :]
        else:
            return self.model.forward_features(x, input_chans=input_chans)

    def extract_features(
        self,
        eeg_data: np.ndarray,
        sfreq: float,
        ch_names: List[str],
        max_seconds: Optional[float] = None,
        pool_method: str = "mean",
    ) -> np.ndarray:
        self._pool_method = pool_method
        return super().extract_features(eeg_data, sfreq, ch_names, max_seconds)

    def extract_features_batch(
        self,
        eeg_list: list,
        sfreq: float,
        ch_names: List[str],
        max_seconds: Optional[float] = None,
        pool_method: str = "mean",
        **kwargs,
    ) -> np.ndarray:
        self._pool_method = pool_method
        return super().extract_features_batch(
            eeg_list, sfreq, ch_names, max_seconds, pool_method=pool_method, **kwargs
        )
