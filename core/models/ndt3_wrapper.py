"""
NDT3 Model Wrapper for NS-Copilot
Provides a generic interface to extract features from pretrained NDT3 models.

This wrapper ONLY handles NDT3-specific logic:
  - Model loading (checkpoint → BrainBertInterface)
  - Feature extraction (spike tensor → latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Add NDT3 to path (support both local and Docker environments)
import os

# Try to locate NDT3 in the following order:
# 1. Environment variable NDT3_PATH
# 2. Docker mount: /app/ndt3
# 3. Relative to project: ./models/ndt3
NDT3_PATH = os.environ.get('NDT3_PATH', None)
if NDT3_PATH is None:
    if os.path.exists('/app/ndt3'):
        NDT3_PATH = '/app/ndt3'
    elif os.path.exists('./models/ndt3'):
        NDT3_PATH = os.path.abspath('./models/ndt3')
    elif os.path.exists('../models/ndt3'):
        NDT3_PATH = os.path.abspath('../models/ndt3')
    else:
        raise RuntimeError(
            "NDT3 not found. Please either:\n"
            "  1. Run ./setup.sh to download NDT3\n"
            "  2. Set NDT3_PATH environment variable\n"
            "  3. Ensure models/ndt3 directory exists"
        )

if NDT3_PATH not in sys.path:
    sys.path.insert(0, NDT3_PATH)

try:
    # flash_attn is pinned to 2.7.4 in Dockerfile for NDT3 compatibility.
    # Fallback patch for environments where flash_attn >= 2.8 is installed:
    # flash_attn 2.8+ removed `pos_idx_in_fp32` from RotaryEmbedding.__init__,
    # but NDT3's TemporalRotaryEmbedding still passes it to super().__init__.
    import inspect
    from flash_attn.layers.rotary import RotaryEmbedding as _OrigRotaryEmbedding
    _rotary_params = inspect.signature(_OrigRotaryEmbedding.__init__).parameters
    if 'pos_idx_in_fp32' not in _rotary_params:
        _orig_init = _OrigRotaryEmbedding.__init__
        def _patched_init(self, *args, **kwargs):
            kwargs.pop('pos_idx_in_fp32', None)
            if len(args) > 4:
                args = args[:4] + args[5:]
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, 'pos_idx_in_fp32'):
                self.pos_idx_in_fp32 = True
        _OrigRotaryEmbedding.__init__ = _patched_init

    from context_general_bci.model import load_from_checkpoint, BrainBertInterface
    NDT3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NDT3: {e}")
    NDT3_AVAILABLE = False


def _find_checkpoint() -> Optional[str]:
    """Find NDT3 checkpoint in standard locations."""
    env_path = os.environ.get('NDT3_CHECKPOINT_PATH', None)
    if env_path and os.path.exists(env_path):
        return env_path

    checkpoint_name = '753jmg4u/checkpoints/val-epoch=397-val_loss=0.4987.ckpt'
    candidates = [
        f'/app/ndt3/data/pretrained/{checkpoint_name}',
        f'./models/ndt3/data/pretrained/{checkpoint_name}',
        f'../models/ndt3/data/pretrained/{checkpoint_name}',
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


class NDT3Decoder:
    """
    Generic wrapper for NDT3 feature extraction.

    Usage:
        decoder = NDT3Decoder(checkpoint_path)
        features = decoder.extract_features(spike_tensor)
        # features: (n_trials, hidden_dim) numpy array
        # Then use features with any downstream model in generated code
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if not NDT3_AVAILABLE:
            raise ImportError("NDT3 is not available. Please install context_general_bci.")

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "NDT3 checkpoint not found. Please either:\n"
                "  1. Run ./setup.sh to download the checkpoint\n"
                "  2. Set NDT3_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None

        print(f"Loading NDT3 from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained NDT3 model from .ckpt checkpoint."""
        import functools
        original_load = torch.load

        @functools.wraps(original_load)
        def load_with_weights_only_false(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = load_with_weights_only_false
        try:
            self.model = load_from_checkpoint(str(self.checkpoint_path))
        finally:
            torch.load = original_load

        self.model.eval()
        self.model.to(self.device)
        print("NDT3 model loaded successfully")

    def extract_features(
        self,
        spikes: torch.Tensor,
        pool_method: str = "mean",
    ) -> np.ndarray:
        """
        Extract latent features from spike data using NDT3's backbone.

        Args:
            spikes: (n_trials, n_time_bins, n_neurons) tensor of binned spike counts
                    Values should be non-negative integers (spike counts per bin).
            pool_method: 'mean', 'max', or 'last' for temporal pooling

        Returns:
            features: (n_trials, hidden_dim) numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        from einops import rearrange

        self.model.eval()

        B, T, N = spikes.shape

        # --- Input shape sanity check ---
        # NDT3's positional encoding supports up to max_trial_length
        # time bins (cfg.transformer.max_trial_length). If T exceeds
        # this, the caller likely concatenated an entire session
        # instead of binning per trial.
        max_T = self.model.cfg.transformer.max_trial_length
        if T > max_T:
            raise ValueError(
                f"Input n_time_bins={T} exceeds the model's positional "
                f"encoding limit (max_trial_length={max_T}). Input shape "
                f"({B}, {T}, {N}). NDT3 expects per-trial binned spike "
                f"counts (typically 10-200 time bins per trial). Your code "
                f"may have concatenated all spikes into a single long "
                f"sequence instead of binning within each trial's time "
                f"window. Please fix the data preparation logic."
            )

        # --- Dynamic GPU memory check ---
        if self.device != "cpu" and torch.cuda.is_available():
            npt = self.model.cfg.neurons_per_token
            n_groups = (N + npt - 1) // npt
            seq_len = T * n_groups
            # Estimate memory: input tensor + embedded + backbone output
            # Use a conservative 3x multiplier for intermediate activations
            bytes_per_element = 2  # bfloat16
            estimated_bytes = B * seq_len * 1024 * bytes_per_element * 3
            dev_idx = torch.device(self.device).index or 0
            free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
            if estimated_bytes > free_mem * 0.9:
                raise RuntimeError(
                    f"Insufficient GPU memory for NDT3 inference. "
                    f"Input shape ({B}, {T}, {N}) → sequence length "
                    f"{seq_len} per trial. Estimated need: "
                    f"{estimated_bytes / 1e9:.2f} GB, available: "
                    f"{free_mem / 1e9:.2f} GB. Try reducing the number "
                    f"of trials, units, or time bins."
                )

        spikes = spikes.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # --- Step 1: Tokenize spikes into NDT3 flat format ---
            npt = self.model.cfg.neurons_per_token

            # Pad neuron dimension to be divisible by neurons_per_token
            n_groups = (N + npt - 1) // npt
            pad_n = n_groups * npt - N
            if pad_n > 0:
                spikes = torch.nn.functional.pad(spikes, (0, pad_n))

            # Reshape: (B, T, N_padded) -> (B, T, n_groups, npt)
            spikes_4d = spikes.view(B, T, n_groups, npt)

            # Flat serving: (B, T, n_groups, npt) -> (B, T*n_groups, 1, npt)
            spikes_flat = rearrange(spikes_4d, 'b t c h -> b (t c) 1 h')

            # --- Step 2: Encode spikes via SpikeContext pipeline ---
            spike_pipeline = None
            for k, v in self.model.task_pipelines.items():
                if hasattr(v, 'encode_direct'):
                    spike_pipeline = v
                    break
            if spike_pipeline is None:
                raise RuntimeError("Could not find SpikeContext pipeline in model")

            embedded = spike_pipeline.encode_direct(spikes_flat)

            # --- Step 3: Time and position indices ---
            time_idx = torch.arange(T, device=self.device).unsqueeze(1).expand(T, n_groups).flatten()
            time_idx = time_idx.unsqueeze(0).expand(B, -1)

            space_idx = torch.arange(n_groups, device=self.device).unsqueeze(0).expand(T, n_groups).flatten()
            space_idx = space_idx.unsqueeze(0).expand(B, -1)

            # --- Step 4: Backbone ---
            features = self.model.backbone(
                embedded,
                times=time_idx,
                positions=space_idx,
            )

            # --- Step 5: Temporal pooling (float32 for precision) ---
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
