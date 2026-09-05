"""
BrainOmni Wrapper for NS-Copilot

Provides a generic interface to extract features from the pretrained BrainOmni
foundation model for EEG data classification and analysis.

This wrapper handles:
  - Model loading (BrainOmni tiny + frozen BrainTokenizer)
  - Electrode position generation from MNE standard montages
  - EEG preprocessing (resample to 256Hz, normalize, bandpass)
  - Feature extraction (raw EEG -> 4096-dim latent features)

BrainOmni uses a Criss-Cross Transformer with Sensor Encoder that accepts
3D electrode positions, making it channel-agnostic across any EEG montage.

Reference: "BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals"
           (NeurIPS 2025)
GitHub: https://github.com/OpenTSLab/BrainOmni
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from core.models.base_eeg_wrapper import BaseEEGDecoder, find_repo, find_checkpoint


# Locate BrainOmni repo
BRAINOMNI_PATH = find_repo(
    'BRAINOMNI_PATH',
    ['/app/brainomni', './models/brainomni', '../models/brainomni',
     os.path.expanduser('~/data/brainomni')],
    os.path.join('brainomni', 'model.py'),
)

BRAINOMNI_AVAILABLE = False
if BRAINOMNI_PATH is not None:
    if BRAINOMNI_PATH not in sys.path:
        sys.path.insert(0, BRAINOMNI_PATH)

    # Provide a lightweight deepspeed stub for single-GPU inference
    if 'deepspeed' not in sys.modules:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            import types
            _ds = types.ModuleType('deepspeed')
            _ds_comm = types.ModuleType('deepspeed.comm')
            _ds_comm.is_initialized = lambda: False
            _ds_comm.get_world_size = lambda: 1
            _ds_comm.all_reduce = lambda *a, **k: None
            _ds_comm.broadcast = lambda *a, **k: None
            _ds_comm.ReduceOp = type('ReduceOp', (), {'SUM': 0})()
            _ds.comm = _ds_comm
            sys.modules['deepspeed'] = _ds
            sys.modules['deepspeed.comm'] = _ds_comm

    try:
        from brainomni.model import BrainOmni  # noqa: F401
        BRAINOMNI_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import BrainOmni: {e}")
else:
    print("Warning: BrainOmni repo not found. Set BRAINOMNI_PATH environment variable.")


def _find_checkpoint_dir() -> Optional[str]:
    """Find BrainOmni checkpoint directory (should contain tiny/ and braintokenizer/)."""
    candidates = []
    if BRAINOMNI_PATH is not None:
        candidates.append(os.path.join(BRAINOMNI_PATH, 'pretrained_weights'))
    candidates += [
        '/app/brainomni/pretrained_weights',
        './models/brainomni/pretrained_weights',
        '../models/brainomni/pretrained_weights',
    ]
    return find_checkpoint('BRAINOMNI_CHECKPOINT_PATH',
                           [os.path.join(c, 'tiny', 'BrainOmni.pt') for c in candidates])


def _get_electrode_positions(
    ch_names: List[str],
    montage_name: str = 'standard_1020',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get 3D electrode positions and sensor types from MNE standard montages.

    Positions are normalized following BrainOmni's convention:
    center-mean, then scale by sqrt(3 * mean(||pos||^2)).

    Returns:
        pos: (n_channels, 6) float32 array
        sensor_type: (n_channels,) int32 array (all zeros for EEG)
    """
    import mne

    montage_candidates = [montage_name, 'standard_1005', 'standard_1020']
    seen = set()
    montage_candidates = [m for m in montage_candidates if m not in seen and not seen.add(m)]

    best_positions = {}
    for mname in montage_candidates:
        try:
            montage = mne.channels.make_standard_montage(mname)
            ch_pos = montage.get_positions()['ch_pos']
            for name in ch_names:
                if name not in best_positions:
                    if name in ch_pos:
                        best_positions[name] = ch_pos[name]
                    else:
                        for mch, mpos in ch_pos.items():
                            if mch.lower() == name.lower():
                                best_positions[name] = mpos
                                break
        except Exception as e:
            print(f"Warning: Failed to load montage '{mname}': {e}")
            continue

    pos_list = []
    for name in ch_names:
        if name in best_positions:
            xyz = best_positions[name].astype(np.float32)
            pos_list.append(np.concatenate([xyz, np.zeros(3, dtype=np.float32)]))
        else:
            print(f"Warning: Channel '{name}' not found in standard montages, using origin position")
            pos_list.append(np.zeros(6, dtype=np.float32))

    pos = np.stack(pos_list, axis=0)

    # Normalize positions
    xyz = pos[:, :3]
    mean_pos = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - mean_pos
    scale = np.sqrt(3 * np.mean(np.sum(xyz ** 2, axis=1)))
    if scale > 1e-10:
        xyz = xyz / scale
    pos[:, :3] = xyz

    sensor_type = np.zeros(len(ch_names), dtype=np.int32)
    return pos, sensor_type


def _normalize_eeg_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize EEG data following BrainOmni's sensortype_wise_normalize:
    global average reference + global z-score.
    """
    data = data.copy().astype(np.float64)
    mean_ref = np.mean(data, axis=0, keepdims=True)
    data = data - mean_ref
    std = np.std(data) + 1e-5
    data = data / std
    return data.astype(np.float32)


# Non-EEG channel names to filter out
_NON_EEG = {'Resp', 'ECG', 'EKG', 'EMG', 'EOG', 'HEOG', 'VEOG',
             'HEO', 'VEO', 'X', 'Y', 'Z', 'Status', 'STI 014',
             'GSR', 'Trigger', 'Event'}


class BrainOmniDecoder(BaseEEGDecoder):
    """
    Generic wrapper for BrainOmni feature extraction from EEG data.

    BrainOmni uses a Criss-Cross Transformer with Sensor Encoder that accepts
    3D electrode positions + sensor type.

    Output: 4096-dim features (16 latent sources x 256 dims, mean-pooled over time).

    Usage:
        decoder = BrainOmniDecoder()
        features = decoder.extract_features(eeg_data, sfreq=500, ch_names=['Fp1', 'Fp2', ...])
        # features: (4096,) numpy array
    """

    TARGET_SFREQ = 256
    MAX_SEGMENT_SECONDS = 30.0
    MIN_SEGMENT_SECONDS = 2.0
    N_NEURO = 16
    LM_DIM = 256

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(device)

        if not BRAINOMNI_AVAILABLE:
            raise ImportError(
                "BrainOmni is not available. Please either:\n"
                "  1. Clone https://github.com/OpenTSLab/BrainOmni\n"
                "  2. Set BRAINOMNI_PATH environment variable"
            )

        if checkpoint_dir is None:
            ckpt_path = _find_checkpoint_dir()
            if ckpt_path:
                # find_checkpoint returns the full .pt path; we need the parent dir
                checkpoint_dir = str(Path(ckpt_path).parent.parent)
        if checkpoint_dir is None:
            raise FileNotFoundError(
                "BrainOmni checkpoints not found. Please either:\n"
                "  1. Download from https://huggingface.co/OpenTSLab/BrainOmni\n"
                "  2. Set BRAINOMNI_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        print(f"Loading BrainOmni from {self.checkpoint_dir}")
        self._load_model()

    def _load_model(self):
        """Load pretrained BrainOmni (tiny) model with frozen tokenizer."""
        from brainomni.model import BrainOmni

        cfg_path = self.checkpoint_dir / 'tiny' / 'model_cfg.json'
        with open(cfg_path) as f:
            model_config = json.load(f)

        self.model = BrainOmni(**model_config)

        self.checkpoint_path = self.checkpoint_dir / 'tiny' / 'BrainOmni.pt'
        checkpoint = torch.load(str(self.checkpoint_path), map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint, strict=False)

        for p in self.model.tokenizer.parameters():
            p.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        self.LM_DIM = model_config.get('lm_dim', 256)
        self.N_NEURO = model_config.get('n_neuro', 16)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"BrainOmni loaded: {n_params:.1f}M params, "
              f"lm_dim={self.LM_DIM}, n_neuro={self.N_NEURO}, device={self.device}")

    @property
    def feature_dim(self) -> int:
        return self.N_NEURO * self.LM_DIM  # 16 * 256 = 4096

    @staticmethod
    def _align_channels(
        eeg_data: np.ndarray,
        ch_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Align EEG data and channel names, filtering out non-EEG channels
        and handling mismatches between data rows and channel name count.
        """
        n_data = eeg_data.shape[0]
        n_names = len(ch_names)

        if n_data != n_names:
            if n_data < n_names:
                ch_names = ch_names[:n_data]
            else:
                eeg_data = eeg_data[:n_names, :]

        keep_idx = [i for i, name in enumerate(ch_names) if name not in _NON_EEG]
        if len(keep_idx) < len(ch_names):
            eeg_data = eeg_data[keep_idx, :]
            ch_names = [ch_names[i] for i in keep_idx]

        return eeg_data, ch_names

    def _prepare(self, eeg_data, sfreq, ch_names):
        # Align channels
        eeg_data, ch_names = self._align_channels(eeg_data, ch_names)

        # Resample + normalize the full recording before segmenting
        eeg_data = self._resample(eeg_data, sfreq, self.TARGET_SFREQ)
        eeg_data = _normalize_eeg_data(eeg_data)

        # Get electrode positions
        pos, sensor_type_arr = _get_electrode_positions(ch_names)
        pos_tensor = torch.FloatTensor(pos).unsqueeze(0).to(self.device)
        st_tensor = torch.IntTensor(sensor_type_arr).unsqueeze(0).to(self.device)

        return eeg_data, self.TARGET_SFREQ, {'pos': pos_tensor, 'sensor_type': st_tensor}

    def _preprocess_segment(self, segment, sfreq, **ctx):
        # Data is already resampled and normalized in _prepare
        return torch.FloatTensor(segment).unsqueeze(0)

    def _encode_segment(self, x, **ctx):
        # encode returns (B, n_neuro, seq_len, lm_dim)
        features = self.model.encode(x, ctx['pos'], ctx['sensor_type'])
        # Mean pool over temporal dimension, then flatten
        return features.mean(dim=2).flatten(1)
