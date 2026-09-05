"""
MtM (Multi-task Masking) Model Wrapper for NS-Copilot

Provides a generic interface to extract features from pretrained MtM models
(NDT1 architecture with multi-task masking training strategy).

This wrapper ONLY handles MtM-specific logic:
  - Model loading (checkpoint → NDT1)
  - Feature extraction (spike tensor → latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: Zhang et al. (2024) "Multi-task Masking for Neural Foundation Models"
GitHub: https://github.com/colehurwitz/IBL_MtM_model
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

import os

# Locate MtM repo for model code and data files
MTM_PATH = os.environ.get('MTM_PATH', None)
if MTM_PATH is None:
    _candidates = [
        '/app/mtm',                    # Docker mount
        './models/mtm',                # Relative to project
        '../models/mtm',               # Relative to project (from core/)
        os.path.expanduser('~/data/mtm_model'),  # User data directory
    ]
    for _c in _candidates:
        if os.path.isdir(_c) and os.path.isdir(os.path.join(_c, 'src')):
            MTM_PATH = os.path.abspath(_c)
            break

MTM_AVAILABLE = False
_NDT1_cls = None
if MTM_PATH is not None:
    try:
        from neuro_copilot.core.models.base_eeg_wrapper import import_from_repo
        _src = os.path.join(MTM_PATH, 'src')
        _prev_cwd = os.getcwd()
        os.chdir(MTM_PATH)  # ndt1.py uses relative path: open('data/target_eids.txt')
        try:
            _ndt1_mod = import_from_repo(_src, 'models/ndt1.py')
            _NDT1_cls = _ndt1_mod.NDT1
            MTM_AVAILABLE = True
        finally:
            os.chdir(_prev_cwd)
    except (ImportError, AttributeError, FileNotFoundError) as e:
        print(f"Warning: Could not import MtM (NDT1): {e}")
else:
    print("Warning: MtM repo not found. Set MTM_PATH environment variable.")


def _find_checkpoint() -> Optional[str]:
    """Find MtM checkpoint in standard locations."""
    env_path = os.environ.get('MTM_CHECKPOINT_PATH', None)
    if env_path and os.path.exists(env_path):
        return env_path

    checkpoint_name = 'multi-NDT1-MtM-10-sessions/model_best.pt'
    if MTM_PATH is not None:
        candidates = [
            os.path.join(MTM_PATH, 'checkpoints', checkpoint_name),
        ]
    else:
        candidates = []

    candidates += [
        f'/app/mtm/checkpoints/{checkpoint_name}',
        f'./models/mtm/checkpoints/{checkpoint_name}',
        f'../models/mtm/checkpoints/{checkpoint_name}',
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


class MTMDecoder:
    """
    Generic wrapper for MtM (NDT1) feature extraction.

    Usage:
        decoder = MTMDecoder(checkpoint_path)
        features = decoder.extract_features(spike_tensor)
        # features: (n_trials, hidden_dim) numpy array
        # Then use features with any downstream model in generated code
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if not MTM_AVAILABLE:
            raise ImportError(
                "MtM is not available. Please either:\n"
                "  1. Clone https://github.com/colehurwitz/IBL_MtM_model\n"
                "  2. Set MTM_PATH environment variable"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "MtM checkpoint not found. Please either:\n"
                "  1. Download from HuggingFace: ibl-foundation-model/multi-NDT1-MtM-10-sessions\n"
                "  2. Set MTM_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None

        # Model config values (populated after loading)
        self._n_channels = None
        self._max_time_bins = None
        self._hidden_size = None
        self._use_prompt = None
        self._use_session = None
        self._stitcher_supported = None
        self._default_eid = None

        print(f"Loading MtM from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained MtM model from checkpoint."""
        # MtM code requires cwd to be repo root for data/target_eids.txt
        orig_cwd = os.getcwd()
        os.chdir(MTM_PATH)
        try:
            checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        finally:
            os.chdir(orig_cwd)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model = checkpoint['model']
        else:
            raise ValueError(
                f"Unexpected checkpoint format: {type(checkpoint)}. "
                "Expected dict with 'model' key."
            )

        self.model.eval()
        self.model.to(self.device)

        # Read config values from the loaded model (no hardcoding)
        self._n_channels = self.model.n_channels
        self._max_time_bins = self.model.encoder.max_F
        self._hidden_size = self.model.encoder.hidden_size
        self._use_prompt = self.model.encoder.embedder.use_prompt
        self._use_session = self.model.encoder.embedder.use_session

        if hasattr(self.model, 'stitching') and self.model.stitching:
            supported = self.model.encoder.stitcher.stitcher_dict.keys()
            self._stitcher_supported = set(int(k) for k in supported)
        else:
            self._stitcher_supported = None

        if self._use_session:
            self._default_eid = self.model.encoder.embedder.eid_lookup[0]

        # Count extra tokens prepended by embedder
        self._n_extra_tokens = int(self._use_prompt) + int(self._use_session)

        print(
            f"MtM model loaded: hidden_size={self._hidden_size}, "
            f"n_channels={self._n_channels}, max_time_bins={self._max_time_bins}, "
            f"extra_tokens={self._n_extra_tokens}"
        )

    def extract_features(
        self,
        spikes: torch.Tensor,
        pool_method: str = "mean",
    ) -> np.ndarray:
        """
        Extract latent features from spike data using MtM's encoder.

        Args:
            spikes: (n_trials, n_time_bins, n_neurons) tensor of binned spike counts.
                    Values should be non-negative (spike counts per bin).
                    Bin size should match MtM training: 20ms.
                    n_time_bins should not exceed model's max_F (typically 100).
            pool_method: 'mean', 'max', or 'last' for temporal pooling

        Returns:
            features: (n_trials, hidden_dim) numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        B, T, N = spikes.shape

        # --- Input validation ---
        if T > self._max_time_bins:
            raise ValueError(
                f"Input n_time_bins={T} exceeds the model's positional "
                f"encoding limit (max_F={self._max_time_bins}). MtM expects "
                f"per-trial binned spike counts at 20ms bins within a 2s window "
                f"(yielding ≤{self._max_time_bins} bins). Your code may have "
                f"concatenated all spikes into a single long sequence instead "
                f"of binning within each trial's time window."
            )

        if N > self._n_channels:
            raise ValueError(
                f"Input n_neurons={N} exceeds the model's maximum "
                f"({self._n_channels}). MtM cannot process more neurons "
                f"than its embedding dimension."
            )

        # --- Dynamic GPU memory check ---
        if self.device != "cpu" and torch.cuda.is_available():
            bytes_per_element = 4  # float32
            # Input + embedded + transformer activations (conservative 4x)
            estimated_bytes = (
                B * T * self._n_channels * bytes_per_element * 4
            )
            dev_idx = torch.device(self.device).index or 0
            free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
            if estimated_bytes > free_mem * 0.9:
                raise RuntimeError(
                    f"Insufficient GPU memory for MtM inference. "
                    f"Input shape ({B}, {T}, {N}). Estimated need: "
                    f"{estimated_bytes / 1e9:.2f} GB, available: "
                    f"{free_mem / 1e9:.2f} GB. Try reducing the number "
                    f"of trials, units, or time bins."
                )

        self.model.eval()

        # --- Prepare input ---
        # Pad neuron dimension to n_channels if needed
        if N < self._n_channels:
            padded = torch.zeros(B, T, self._n_channels, device=self.device)
            padded[:, :, :N] = spikes.to(self.device)
            spikes_input = padded
        else:
            spikes_input = spikes.to(self.device)

        # Attention masks: 1 for real data, 0 for padding
        time_mask = torch.ones(B, T, dtype=torch.long, device=self.device)
        timestamps = (
            torch.arange(T, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
            .long()
        )

        # --- Handle stitcher ---
        # If num_neurons is supported by stitcher, use it normally.
        # Otherwise, bypass stitcher (data is already padded to n_channels).
        stitcher_removed = False
        use_stitcher = (
            self._stitcher_supported is not None
            and N in self._stitcher_supported
        )
        if not use_stitcher and hasattr(self.model.encoder, 'stitcher'):
            self._orig_stitcher = self.model.encoder.stitcher
            delattr(self.model.encoder, 'stitcher')
            stitcher_removed = True
            num_neuron = self._n_channels
        else:
            num_neuron = N

        # Disable masking for inference
        orig_mask = self.model.encoder.mask
        self.model.encoder.mask = False

        # Masking mode (needed for prompt token selection)
        masking_mode = "neuron" if self._use_prompt else None
        eid = self._default_eid if self._use_session else None

        try:
            with torch.no_grad():
                features, _ = self.model.encoder(
                    spikes=spikes_input,
                    spikes_mask=time_mask,
                    spikes_timestamp=timestamps,
                    neuron_regions=None,
                    masking_mode=masking_mode,
                    num_neuron=num_neuron,
                    eid=eid,
                )
        finally:
            # Restore model state
            self.model.encoder.mask = orig_mask
            if stitcher_removed:
                self.model.encoder.stitcher = self._orig_stitcher
                del self._orig_stitcher

        # --- Strip extra tokens (prompt + session) ---
        if self._n_extra_tokens > 0:
            features = features[:, self._n_extra_tokens:, :]

        # --- Temporal pooling ---
        features = features.float()
        if pool_method == "mean":
            pooled = features.mean(dim=1)
        elif pool_method == "max":
            pooled = features.max(dim=1)[0]
        elif pool_method == "last":
            pooled = features[:, -1, :]
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        pooled_cpu = pooled.cpu()
        try:
            return pooled_cpu.numpy()
        except RuntimeError:
            return np.array(pooled_cpu.tolist())
