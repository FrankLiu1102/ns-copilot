"""
POYO-1 Model Wrapper for NS-Copilot

Provides a generic interface to extract features from pretrained POYO-1 models
(PerceiverIO architecture with cross-attention on spike tokens).

This wrapper ONLY handles POYO-specific logic:
  - Model loading (checkpoint → POYO)
  - Feature extraction (spike tensor → latent features)

Everything downstream (classification, metrics, visualization) is left to
the LLM-generated code, which can adapt to any user request.

Reference: Azabou et al. (2023) "A Unified, Scalable Framework for Neural
Population Decoding" (NeurIPS 2024)
GitHub: https://github.com/neuro-galaxy/torch_brain
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import os


# Locate POYO checkpoint directory
POYO_PATH = os.environ.get('POYO_PATH', None)
if POYO_PATH is None:
    _candidates = [
        '/app/poyo',                            # Docker mount
        './models/poyo',                        # Relative to project
        '../models/poyo',                       # Relative (from core/)
        os.path.expanduser('~/data/poyo'),      # User data directory
    ]
    for _c in _candidates:
        if os.path.isdir(_c):
            POYO_PATH = os.path.abspath(_c)
            break

# torch_brain is installed via pip (no sys.path hacking needed)
POYO_AVAILABLE = False
try:
    from torch_brain.models import POYO  # noqa: F401
    POYO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import POYO (torch_brain): {e}")


def _find_checkpoint(variant: str = "poyo_1") -> Optional[str]:
    """Find POYO checkpoint in standard locations."""
    env_path = os.environ.get('POYO_CHECKPOINT_PATH', None)
    if env_path and os.path.exists(env_path):
        return env_path

    checkpoint_name = f'{variant}.ckpt'
    candidates = []
    if POYO_PATH is not None:
        candidates.append(
            os.path.join(POYO_PATH, 'checkpoints', checkpoint_name)
        )

    candidates += [
        f'/app/poyo/checkpoints/{checkpoint_name}',
        f'./models/poyo/checkpoints/{checkpoint_name}',
        f'../models/poyo/checkpoints/{checkpoint_name}',
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


class POYODecoder:
    """
    Generic wrapper for POYO-1 feature extraction.

    POYO-1 uses a PerceiverIO architecture that tokenizes individual spikes
    and compresses them into a fixed set of latent representations via
    cross-attention.

    Usage:
        decoder = POYODecoder(checkpoint_path)
        features = decoder.extract_features(spike_tensor)
        # features: (n_trials, latent_dim) numpy array
        # Then use features with any downstream model in generated code
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        variant: str = "poyo_1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if not POYO_AVAILABLE:
            raise ImportError(
                "POYO-1 is not available. Please either:\n"
                "  1. Install torch_brain: pip install torch_brain\n"
                "  2. See https://github.com/neuro-galaxy/torch_brain"
            )

        if checkpoint_path is None:
            checkpoint_path = _find_checkpoint(variant)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "POYO-1 checkpoint not found. Please either:\n"
                "  1. Download from: https://nyu1.osn.mghpcc.org/"
                "brainsets-public/model-zoo/poyo_1.ckpt\n"
                "  2. Set POYO_CHECKPOINT_PATH environment variable"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None

        # Model config (populated after loading)
        self._dim = None
        self._num_latents_per_step = None
        self._sequence_length = None
        self._latent_step = None

        print(f"Loading POYO-1 from {self.checkpoint_path}")
        self._load_model()

    def _load_model(self):
        """Load pretrained POYO-1 model from checkpoint."""
        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        # PyTorch Lightning checkpoint format
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError(
                f"Unexpected checkpoint format: {type(checkpoint)}. "
                "Expected a PyTorch Lightning checkpoint dict."
            )

        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint['state_dict']

        # Model config is nested under hparams['model'] in the checkpoint
        model_cfg = hparams.get('model', hparams)

        # Read config from checkpoint (no hardcoding)
        self._dim = model_cfg.get('dim', 128)
        self._num_latents_per_step = model_cfg.get('num_latents_per_step', 32)
        self._sequence_length = model_cfg.get('sequence_length', 1.0)
        self._latent_step = model_cfg.get('latent_step', 0.125)

        # Build model from checkpoint hyperparameters.
        # POYO requires readout_spec for construction. We create a dummy
        # matching the checkpoint's readout shape (2-dim continuous).
        from torch_brain.models import POYO
        from torch_brain.registry import ModalitySpec, DataType

        # Infer readout dim from checkpoint weights
        readout_dim = 2
        readout_key = 'model.readout.weight'
        if readout_key in state_dict:
            readout_dim = state_dict[readout_key].shape[0]

        dummy_readout = ModalitySpec(
            id=0,
            dim=readout_dim,
            type=DataType.CONTINUOUS,
            timestamp_key='dummy',
            value_key='dummy',
            loss_fn=torch.nn.MSELoss(),
        )

        # Filter to only POYO.__init__ accepted params
        build_params = {'readout_spec': dummy_readout}
        for key in (
            'sequence_length', 'latent_step', 'num_latents_per_step',
            'dim', 'depth', 'dim_head', 'cross_heads', 'self_heads',
            'ffn_dropout', 'lin_dropout', 'atn_dropout',
            'emb_init_scale', 't_min', 't_max',
        ):
            if key in model_cfg:
                build_params[key] = model_cfg[key]

        self.model = POYO(**build_params)

        # Load state dict — strip 'model.' prefix (Lightning wrapper)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix('model.')] = v
        load_info = self.model.load_state_dict(cleaned, strict=False)

        # --- Diagnostic: report state_dict mismatches ---
        if load_info.missing_keys:
            print(
                f"[POYO diag] state_dict missing_keys "
                f"({len(load_info.missing_keys)}): "
                f"{load_info.missing_keys[:10]}"
                + (" ..." if len(load_info.missing_keys) > 10 else "")
            )
        if load_info.unexpected_keys:
            print(
                f"[POYO diag] state_dict unexpected_keys "
                f"({len(load_info.unexpected_keys)}): "
                f"{load_info.unexpected_keys[:10]}"
                + (" ..." if len(load_info.unexpected_keys) > 10 else "")
            )
        if not load_info.missing_keys and not load_info.unexpected_keys:
            print("[POYO diag] state_dict loaded perfectly (no mismatches)")

        # --- Diagnostic: check for NaN/Inf in loaded parameters ---
        nan_params = []
        for pname, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(f"{pname}(NaN)")
            elif torch.isinf(param).any():
                nan_params.append(f"{pname}(Inf)")
        if nan_params:
            print(f"[POYO diag] WARNING: params with NaN/Inf: {nan_params}")
        else:
            print(f"[POYO diag] All {sum(1 for _ in self.model.parameters())} params are finite")

        self.model.eval()
        self.model.to(self.device)

        # --- Diagnostic: print model submodule names ---
        submodule_names = [n for n, _ in self.model.named_children()]
        print(f"[POYO diag] Model submodules: {submodule_names}")

        print(
            f"POYO-1 model loaded: dim={self._dim}, "
            f"num_latents_per_step={self._num_latents_per_step}, "
            f"sequence_length={self._sequence_length}s, "
            f"depth={model_cfg.get('depth', '?')}"
        )

    def _bins_to_spike_tokens(
        self,
        spikes: torch.Tensor,
        bin_size_s: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert binned spike tensor to POYO's spike-token format.

        Args:
            spikes: (B, T, N) binned spike counts
            bin_size_s: bin width in seconds

        Returns:
            unit_indices: (B, max_spikes) int tensor of unit IDs
            timestamps:   (B, max_spikes) float tensor of spike times
            token_type:   (B, max_spikes) int tensor (all zeros for spikes)
            mask:         (B, max_spikes) bool tensor (True = real spike)
        """
        B, T, N = spikes.shape
        spikes_np = spikes.cpu().numpy().astype(int)

        all_unit_indices = []
        all_timestamps = []
        max_spikes = 0

        for b in range(B):
            trial_units = []
            trial_times = []
            for t in range(T):
                bin_start = t * bin_size_s
                for n in range(N):
                    count = spikes_np[b, t, n]
                    if count > 0:
                        # Distribute spikes uniformly within the bin
                        for k in range(count):
                            spike_time = (
                                bin_start
                                + (k + 0.5) * bin_size_s / max(count, 1)
                            )
                            trial_units.append(n)
                            trial_times.append(spike_time)
            all_unit_indices.append(trial_units)
            all_timestamps.append(trial_times)
            max_spikes = max(max_spikes, len(trial_units))

        # Ensure at least 1 token (avoids empty tensors)
        max_spikes = max(max_spikes, 1)

        unit_idx = torch.zeros(B, max_spikes, dtype=torch.long)
        times = torch.zeros(B, max_spikes, dtype=torch.float)
        mask = torch.zeros(B, max_spikes, dtype=torch.bool)

        for b in range(B):
            n_spk = len(all_unit_indices[b])
            if n_spk > 0:
                unit_idx[b, :n_spk] = torch.tensor(all_unit_indices[b])
                times[b, :n_spk] = torch.tensor(all_timestamps[b])
                mask[b, :n_spk] = True

        token_type = torch.zeros_like(unit_idx)
        return unit_idx, times, token_type, mask

    def extract_features(
        self,
        spikes: torch.Tensor,
        pool_method: str = "mean",
        bin_size_s: float = 0.02,
    ) -> np.ndarray:
        """
        Extract latent features from binned spike data using POYO's encoder.

        Args:
            spikes: (n_trials, n_time_bins, n_neurons) tensor of binned spike
                    counts. Values should be non-negative integers.
            pool_method: 'mean', 'max', or 'last' for latent token pooling
            bin_size_s: bin width in seconds (for timestamp reconstruction)

        Returns:
            features: (n_trials, latent_dim) numpy array (dim from checkpoint)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        B, T, N = spikes.shape
        session_length = T * bin_size_s

        # --- GPU memory check ---
        if self.device != "cpu" and torch.cuda.is_available():
            bytes_per_element = 4  # float32
            # Conservative estimate: spikes + tokens + latents + activations
            estimated_spikes = int(spikes.sum().item())
            estimated_bytes = max(
                estimated_spikes * self._dim * bytes_per_element * 4,
                B * self._num_latents_per_step * self._dim
                * bytes_per_element * 4,
            )
            dev_idx = torch.device(self.device).index or 0
            free_mem, total_mem = torch.cuda.mem_get_info(dev_idx)
            if estimated_bytes > free_mem * 0.9:
                raise RuntimeError(
                    f"Insufficient GPU memory for POYO-1 inference. "
                    f"Input shape ({B}, {T}, {N}). Estimated need: "
                    f"{estimated_bytes / 1e9:.2f} GB, available: "
                    f"{free_mem / 1e9:.2f} GB. Try reducing the number "
                    f"of trials, units, or time bins."
                )

        self.model.eval()

        # --- Convert binned spikes to spike tokens ---
        unit_indices, timestamps, token_type, input_mask = (
            self._bins_to_spike_tokens(spikes, bin_size_s)
        )

        # --- Diagnostic: token conversion stats ---
        spikes_per_trial = input_mask.sum(dim=1)  # (B,)
        print(
            f"[POYO diag] Tokenization: shape=({B}, {timestamps.shape[1]}), "
            f"spikes/trial min={spikes_per_trial.min().item()}, "
            f"max={spikes_per_trial.max().item()}, "
            f"mean={spikes_per_trial.float().mean().item():.1f}, "
            f"zero_spike_trials={int((spikes_per_trial == 0).sum().item())}"
        )
        print(
            f"[POYO diag] Timestamps range: "
            f"[{timestamps[input_mask].min().item():.4f}, "
            f"{timestamps[input_mask].max().item():.4f}]"
            if input_mask.any()
            else "[POYO diag] WARNING: ALL tokens are masked (no real spikes)"
        )
        print(
            f"[POYO diag] Unit indices range: "
            f"[{unit_indices[input_mask].min().item()}, "
            f"{unit_indices[input_mask].max().item()}]"
            if input_mask.any()
            else "[POYO diag] WARNING: No real spikes to report unit range"
        )

        unit_indices = unit_indices.to(self.device)
        timestamps = timestamps.to(self.device)
        token_type = token_type.to(self.device)
        input_mask = input_mask.to(self.device)

        # --- Build latent query grid ---
        n_latent = self._num_latents_per_step
        latent_index = (
            torch.arange(n_latent, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        latent_timestamps = (
            torch.linspace(0, session_length, n_latent, device=self.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        print(
            f"[POYO diag] Latent grid: n_latent={n_latent}, "
            f"session_length={session_length:.3f}s, "
            f"latent_ts range=[{latent_timestamps[0,0].item():.4f}, "
            f"{latent_timestamps[0,-1].item():.4f}]"
        )

        # --- Extract latent features via forward pre-hook on dec_atn ---
        # POYO's forward() pipeline: encoder → processor → decoder → readout.
        # The decoder cross-attention (dec_atn) receives the post-processor
        # latents as its 2nd argument: dec_atn(output_queries, latents, ...).
        # We capture latents from the pre-hook args before decoding.
        latent_features = {}
        hook_call_count = [0]

        def _capture_latents(module, args):
            hook_call_count[0] += 1
            latent_features['value'] = args[1]
            latent_features['n_args'] = len(args)
            # Capture all arg shapes for diagnostics
            latent_features['arg_shapes'] = [
                tuple(a.shape) if hasattr(a, 'shape') else type(a).__name__
                for a in args
            ]

        # --- Diagnostic: verify dec_atn exists ---
        if not hasattr(self.model, 'dec_atn'):
            available = [n for n, _ in self.model.named_children()]
            raise RuntimeError(
                f"POYO model has no 'dec_atn' attribute. "
                f"Available submodules: {available}. "
                f"torch_brain API may have changed."
            )

        hook = self.model.dec_atn.register_forward_pre_hook(_capture_latents)

        # Dummy output queries (decoder will run but we ignore its output)
        output_timestamps = torch.zeros(
            B, 1, device=self.device, dtype=torch.float
        )
        output_session_index = torch.zeros(
            B, 1, device=self.device, dtype=torch.long
        )

        # --- Diagnostic: also capture encoder output ---
        encoder_output = {}
        if hasattr(self.model, 'enc_atn'):
            def _capture_encoder(module, args, output):
                encoder_output['value'] = output
            enc_hook = self.model.enc_atn.register_forward_hook(
                _capture_encoder
            )
        else:
            enc_hook = None
            print(
                "[POYO diag] No 'enc_atn' submodule found, "
                "skipping encoder output capture"
            )

        try:
            with torch.no_grad():
                model_output = self.model(
                    input_unit_index=unit_indices,
                    input_timestamps=timestamps,
                    input_token_type=token_type,
                    input_mask=input_mask,
                    latent_index=latent_index,
                    latent_timestamps=latent_timestamps,
                    output_session_index=output_session_index,
                    output_timestamps=output_timestamps,
                )
        finally:
            hook.remove()
            if enc_hook is not None:
                enc_hook.remove()

        # --- Diagnostic: forward pass results ---
        print(f"[POYO diag] dec_atn hook called {hook_call_count[0]} time(s)")
        if model_output is not None:
            if hasattr(model_output, 'shape'):
                mo_nan = torch.isnan(model_output).sum().item()
                print(
                    f"[POYO diag] model_output shape={tuple(model_output.shape)}, "
                    f"NaN count={mo_nan}/{model_output.numel()}"
                )
            else:
                print(f"[POYO diag] model_output type={type(model_output).__name__}")

        if 'value' in encoder_output:
            enc_val = encoder_output['value']
            if hasattr(enc_val, 'shape'):
                enc_nan = torch.isnan(enc_val).sum().item()
                print(
                    f"[POYO diag] enc_atn output: shape={tuple(enc_val.shape)}, "
                    f"NaN={enc_nan}/{enc_val.numel()}, "
                    f"range=[{enc_val[~torch.isnan(enc_val)].min().item():.4f}, "
                    f"{enc_val[~torch.isnan(enc_val)].max().item():.4f}]"
                    if enc_nan < enc_val.numel()
                    else f"[POYO diag] enc_atn output: ALL NaN ({enc_val.numel()} elements)"
                )

        if 'value' not in latent_features:
            raise RuntimeError(
                "Failed to capture latent features from POYO processor. "
                "The model's internal structure may have changed. "
                "Please check torch_brain version compatibility."
            )

        # --- Diagnostic: latent features from hook ---
        print(
            f"[POYO diag] dec_atn pre-hook: n_args={latent_features['n_args']}, "
            f"arg_shapes={latent_features['arg_shapes']}"
        )

        # latent_features['value'] shape: (B, n_latent, dim)
        features = latent_features['value'].float()
        feat_nan = torch.isnan(features).sum().item()
        feat_inf = torch.isinf(features).sum().item()
        nan_trials = torch.isnan(features).any(dim=-1).any(dim=-1).sum().item()
        print(
            f"[POYO diag] Captured latent features: shape={tuple(features.shape)}, "
            f"NaN={feat_nan}/{features.numel()}, "
            f"Inf={feat_inf}/{features.numel()}, "
            f"NaN trials={nan_trials}/{B}"
        )
        if feat_nan < features.numel():
            finite_vals = features[~torch.isnan(features) & ~torch.isinf(features)]
            if finite_vals.numel() > 0:
                print(
                    f"[POYO diag] Finite feature stats: "
                    f"min={finite_vals.min().item():.4f}, "
                    f"max={finite_vals.max().item():.4f}, "
                    f"mean={finite_vals.mean().item():.4f}, "
                    f"std={finite_vals.std().item():.4f}"
                )
        else:
            print("[POYO diag] ALL features are NaN — model output is completely invalid")

        # --- Pool across latent tokens ---
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
