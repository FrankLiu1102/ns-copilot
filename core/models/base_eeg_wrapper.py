"""
Base class for EEG foundation model wrappers in NS-Copilot.

Provides:
  - Repo and checkpoint discovery utilities
  - Segment-based feature extraction with mean pooling
  - Batch feature extraction with error handling

Subclasses implement model-specific logic: loading, preprocessing, encoding.
"""

import os
import sys
import importlib.util
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any


def import_from_repo(repo_path: str, module_file: str, module_name: str = None):
    """Import a module from a specific repo path, isolating sys.path changes.

    Temporarily adds repo_path to sys.path during import so that the module's
    internal relative imports (e.g. ``from models.xxx import ...``) resolve
    correctly, then removes it afterwards to prevent cross-repo collisions.

    Args:
        repo_path: Absolute path to the repo root (e.g. '/app/eegmamba')
        module_file: Relative path to the .py file within the repo
                     (e.g. 'models/eegmamba.py')
        module_name: Optional module name for sys.modules cache.
                     Defaults to a unique name based on repo + file path.

    Returns:
        The imported module object.
    """
    full_path = os.path.join(repo_path, module_file)
    if not os.path.isfile(full_path):
        raise ImportError(f"Cannot find {full_path}")

    if module_name is None:
        # Use a unique name to avoid collisions between repos
        repo_basename = os.path.basename(repo_path)
        module_name = f"_repo_{repo_basename}_{module_file.replace('/', '.').replace('.py', '')}"

    # Return cached module if already imported
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Temporarily add repo_path to sys.path for internal relative imports
    added = False
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        added = True

    try:
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Remove repo_path so it doesn't pollute later imports
        if added and repo_path in sys.path:
            sys.path.remove(repo_path)


def find_repo(env_var: str, candidates: List[str], check_file: str) -> Optional[str]:
    """Find a model repository by checking env var, then candidate paths."""
    path = os.environ.get(env_var)
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, check_file)):
            return os.path.abspath(c)
    return None


def find_checkpoint(env_var: str, candidates: List[str]) -> Optional[str]:
    """Find a model checkpoint by checking env var, then candidate paths."""
    path = os.environ.get(env_var)
    if path and os.path.exists(path):
        return path
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


class BaseEEGDecoder:
    """
    Abstract base class for EEG foundation model feature extraction.

    Subclasses must implement:
      - feature_dim (property): output feature dimension
      - _load_model(): load the pretrained model
      - _preprocess_segment(segment, sfreq, **ctx): raw segment -> model-ready tensor
      - _encode_segment(x, **ctx): model tensor -> (1, feature_dim) feature tensor

    Optionally override:
      - _prepare(eeg_data, sfreq, ch_names): pre-segmentation hook for channel
        mapping, position lookup, or full-recording preprocessing. Returns
        (eeg_data, sfreq, ctx_dict).
    """

    MAX_SEGMENT_SECONDS: float = 16.0
    MIN_SEGMENT_SECONDS: float = 1.0

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError

    def _prepare(
        self,
        eeg_data: np.ndarray,
        sfreq: float,
        ch_names: List[str],
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Pre-segmentation hook. Override for channel mapping, position lookup,
        or full-recording preprocessing (e.g. resample before segmenting).

        Returns:
            (eeg_data, sfreq, ctx) — ctx is passed as **kwargs to
            _preprocess_segment and _encode_segment.
        """
        return eeg_data, sfreq, {}

    def _preprocess_segment(
        self, segment: np.ndarray, sfreq: float, **ctx
    ) -> torch.Tensor:
        """Convert a raw EEG segment to model input tensor."""
        raise NotImplementedError

    def _encode_segment(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        """Run model on preprocessed input. Returns (1, feature_dim) tensor."""
        raise NotImplementedError

    def extract_features(
        self,
        eeg_data: np.ndarray,
        sfreq: float,
        ch_names: List[str],
        max_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract features by splitting into segments, encoding each, and averaging.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # Truncate before any processing
        if max_seconds is not None:
            max_samples = int(max_seconds * sfreq)
            eeg_data = eeg_data[:, :max_samples]

        # Model-specific preparation
        eeg_data, sfreq, ctx = self._prepare(eeg_data, sfreq, ch_names)
        n_samples = eeg_data.shape[1]

        # Split into non-overlapping segments
        segment_samples = int(self.MAX_SEGMENT_SECONDS * sfreq)
        min_samples = int(self.MIN_SEGMENT_SECONDS * sfreq)

        segments = []
        for start in range(0, n_samples, segment_samples):
            seg = eeg_data[:, start:start + segment_samples]
            if seg.shape[1] >= min_samples:
                segments.append(seg)

        if not segments:
            raise ValueError(
                f"EEG recording too short. Need at least {self.MIN_SEGMENT_SECONDS}s, "
                f"got {n_samples / sfreq:.1f}s."
            )

        # Extract features per segment and average
        self.model.eval()
        seg_features = []
        with torch.no_grad():
            for seg in segments:
                x = self._preprocess_segment(seg, sfreq, **ctx)
                if isinstance(x, torch.Tensor):
                    x = x.to(self.device)
                feat = self._encode_segment(x, **ctx)
                seg_features.append(feat)

        stacked = torch.cat(seg_features, dim=0)
        pooled = stacked.mean(dim=0)
        return self._to_numpy(pooled)

    def extract_features_batch(
        self,
        eeg_list: list,
        sfreq: float,
        ch_names: List[str],
        max_seconds: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Extract features for multiple EEG samples with error handling."""
        all_features = []
        for i, eeg in enumerate(eeg_list):
            try:
                feat = self.extract_features(
                    eeg, sfreq, ch_names, max_seconds, **kwargs
                )
                all_features.append(feat)
            except Exception as e:
                print(f"Warning: Failed to extract features for sample {i}: {e}")
                all_features.append(np.zeros(self.feature_dim, dtype=np.float32))
        return np.vstack(all_features)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to numpy, handling potential issues."""
        tensor = tensor.float().cpu()
        try:
            return tensor.numpy()
        except RuntimeError:
            return np.array(tensor.tolist())

    @staticmethod
    def _resample(data: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
        """Resample data along axis=1 if needed."""
        if abs(sfreq - target_sfreq) <= 1.0:
            return data
        if target_sfreq < sfreq:
            nyquist_orig = sfreq / 2.0
            nyquist_new = target_sfreq / 2.0
            print(f"ℹ️ Resampling EEG from {sfreq}Hz to {target_sfreq}Hz. "
                  f"Nyquist frequency changes from {nyquist_orig:.0f}Hz to {nyquist_new:.0f}Hz "
                  f"(frequencies above {nyquist_new:.0f}Hz will be lost).",
                  flush=True)
        from scipy.signal import resample
        n_target = int(data.shape[1] * target_sfreq / sfreq)
        return resample(data, n_target, axis=1)

    @staticmethod
    def _multi_stat_pool(feats: torch.Tensor) -> torch.Tensor:
        """
        Multi-statistic pooling: mean + std + max across spatial/temporal dims.

        Args:
            feats: (batch, ..., embed_dim) — will be flattened to (batch, N, embed_dim)

        Returns:
            (batch, embed_dim * 3) tensor
        """
        if feats.dim() > 3:
            flat = feats.flatten(1, -2)  # (batch, N, embed_dim)
        elif feats.dim() == 3:
            flat = feats
        else:
            # (batch, embed_dim) — no dims to pool, repeat
            return torch.cat([feats, torch.zeros_like(feats), feats], dim=-1)

        feat_mean = flat.mean(dim=1)
        feat_std = flat.std(dim=1)
        feat_max = flat.max(dim=1).values
        return torch.cat([feat_mean, feat_std, feat_max], dim=-1)
