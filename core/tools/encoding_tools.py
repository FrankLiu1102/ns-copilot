"""
[LEGACY] Encoding Tools for PFC-4 Dataset

NOTE: This is a legacy/experimental module hardcoded for the PFC-4 MATLAB dataset.
It is NOT used by the main pipeline or any benchmark experiments in the paper.
Not intended for reuse with other datasets.

Neural Encoding Task:
    Input: (F1 frequency, time within delay)
    Output: Neural population activity (PCA latent representation)

This module provides tools for:
- Loading data subsets
- Extracting delay period activity
- Preparing encoding datasets
- Training encoding models
- Evaluating encoding performance
- Shuffle controls
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings


@dataclass
class EncodingDataset:
    """Container for encoding dataset"""
    X_train: np.ndarray  # (n_train, 2) - [F1, time]
    X_test: np.ndarray   # (n_test, 2)
    y_train: np.ndarray  # (n_train, n_pcs) - PC scores
    y_test: np.ndarray   # (n_test, n_pcs)
    time_bins_train: np.ndarray  # (n_train,) - time bin indices
    time_bins_test: np.ndarray   # (n_test,)
    f1_train: np.ndarray  # (n_train,) - F1 values
    f1_test: np.ndarray   # (n_test,)
    pca_model: PCA
    explained_variance_ratio: np.ndarray
    n_pcs: int
    session_ids: List[str]
    n_sessions: int


@dataclass
class EncodingResults:
    """Container for encoding evaluation results"""
    r2_per_pc: np.ndarray
    r2_overall: float
    predictions: np.ndarray
    targets: np.ndarray
    test_time_bins: np.ndarray
    test_f1: np.ndarray
    model_name: str


def extract_delay_period_data(
    sessions: Dict[str, Any],
    bin_size_ms: int = 125,
    delay_start_ms: int = 0,
    delay_end_ms: int = 3000,
    sqrt_transform: bool = True,
    z_score: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract delay period neural activity from multiple sessions
    
    PFC-4 data format:
        - 'result': (n_trials, n_columns) cell array
        - First row is headers: ['class', 'trial', 'hit', 'f1', 'f2', 'spikes', 'PD', 'KD', ...]
        - Column 3 (f1): F1 frequency
        - Column 5 (spikes): List of spike time arrays (one per neuron)
        - Column 7 (KD): Delay period start time (ms)
        - Column 8 (SO1): Delay period end time (ms)
    
    Args:
        sessions: Dictionary of session data
        bin_size_ms: Bin size in milliseconds
        delay_start_ms: Delay period start (ms after KD onset)
        delay_end_ms: Delay period end (ms after KD onset)
        sqrt_transform: Apply square root transform to stabilize variance
        z_score: Apply z-scoring to firing rates
        
    Returns:
        Tuple of (neural_data, f1_labels, time_bins, session_ids)
        - neural_data: (n_samples, n_bins, n_neurons)
        - f1_labels: (n_samples,) - F1 frequency for each trial
        - time_bins: (n_bins,) - time bin centers in ms
        - session_ids: List of session identifiers for each trial
    """
    all_trials = []
    all_f1 = []
    all_session_ids = []
    
    n_bins = (delay_end_ms - delay_start_ms) // bin_size_ms
    time_bins = np.arange(delay_start_ms, delay_end_ms, bin_size_ms) + bin_size_ms / 2
    
    for session_name, mat_data in sessions.items():
        try:
            # Extract result array
            if 'result' not in mat_data:
                warnings.warn(f"Skipping {session_name}: no 'result' field")
                continue
            
            result = mat_data['result']
            
            # First row is headers, skip it
            if result.shape[0] < 2:
                warnings.warn(f"Skipping {session_name}: insufficient data")
                continue
            
            # Extract data rows (skip header row 0)
            n_trials_session = result.shape[0] - 1
            
            for trial_idx in range(1, result.shape[0]):  # Start from 1 to skip header
                try:
                    # Extract F1 frequency (column 3)
                    f1_val = result[trial_idx, 3]
                    if hasattr(f1_val, 'shape') and f1_val.size > 0:
                        f1_val = float(f1_val.flatten()[0])
                    else:
                        continue
                    
                    # Extract spike times (column 5)
                    spikes_cell = result[trial_idx, 5]
                    if not hasattr(spikes_cell, 'shape') or spikes_cell.size == 0:
                        continue
                    
                    # spikes_cell is (1, n_neurons) array of arrays
                    spike_trains = spikes_cell.flatten()
                    n_neurons = len(spike_trains)
                    
                    if n_neurons == 0:
                        continue
                    
                    # Extract delay period timing
                    # KD (column 7): delay period start
                    kd_val = result[trial_idx, 7]
                    if hasattr(kd_val, 'shape') and kd_val.size > 0:
                        kd_time = float(kd_val.flatten()[0])
                    else:
                        continue
                    
                    # Define delay period window
                    delay_start_abs = kd_time + delay_start_ms
                    delay_end_abs = kd_time + delay_end_ms
                    
                    # Bin spike times into firing rates
                    firing_rates = np.zeros((n_bins, n_neurons))
                    
                    for neuron_idx, spike_times_arr in enumerate(spike_trains):
                        # Extract spike times for this neuron
                        if hasattr(spike_times_arr, 'flatten'):
                            spike_times = spike_times_arr.flatten()
                        else:
                            spike_times = np.array([])
                        
                        # Filter spikes within delay period
                        delay_spikes = spike_times[
                            (spike_times >= delay_start_abs) & 
                            (spike_times < delay_end_abs)
                        ]
                        
                        # Bin spikes
                        for bin_idx in range(n_bins):
                            bin_start = delay_start_abs + bin_idx * bin_size_ms
                            bin_end = bin_start + bin_size_ms
                            
                            spike_count = np.sum(
                                (delay_spikes >= bin_start) & 
                                (delay_spikes < bin_end)
                            )
                            
                            # Convert to firing rate (Hz)
                            firing_rates[bin_idx, neuron_idx] = spike_count / (bin_size_ms / 1000.0)
                    
                    # Apply square root transform if requested
                    if sqrt_transform:
                        firing_rates = np.sqrt(firing_rates)
                    
                    all_trials.append(firing_rates)
                    all_f1.append(f1_val)
                    all_session_ids.append(session_name)
                
                except Exception as e:
                    warnings.warn(f"Error processing trial {trial_idx} in {session_name}: {e}")
                    continue
        
        except Exception as e:
            warnings.warn(f"Error processing {session_name}: {e}")
            continue
    
    if not all_trials:
        raise ValueError("No valid trials extracted from any session")
    
    # Convert to arrays
    neural_data = np.array(all_trials)  # (n_trials, n_bins, n_neurons)
    f1_labels = np.array(all_f1)
    
    print(f"Extracted {len(all_trials)} trials from {len(set(all_session_ids))} sessions")
    
    # Z-score across neurons (if requested)
    if z_score:
        # Flatten to (n_trials * n_bins, n_neurons)
        n_trials, n_bins_actual, n_neurons = neural_data.shape
        neural_flat = neural_data.reshape(-1, n_neurons)
        
        mean = neural_flat.mean(axis=0)
        std = neural_flat.std(axis=0) + 1e-8
        neural_flat = (neural_flat - mean) / std
        
        neural_data = neural_flat.reshape(n_trials, n_bins_actual, n_neurons)
    
    return neural_data, f1_labels, time_bins[:neural_data.shape[1]], all_session_ids


def prepare_encoding_dataset(
    neural_data: np.ndarray,
    f1_labels: np.ndarray,
    time_bins: np.ndarray,
    session_ids: List[str],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42
) -> EncodingDataset:
    """
    Prepare dataset for encoding task: (F1, time) -> PC scores
    
    Args:
        neural_data: (n_trials, n_bins, n_neurons)
        f1_labels: (n_trials,)
        time_bins: (n_bins,)
        session_ids: List of session IDs per trial
        n_components: Number of PCs (if None, determined by variance_threshold)
        variance_threshold: Cumulative variance to retain
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        EncodingDataset object
    """
    n_trials, n_bins, n_neurons = neural_data.shape
    
    # Reshape to (n_trials * n_bins, n_neurons) for PCA
    X_flat = neural_data.reshape(-1, n_neurons)
    
    # Fit PCA on full dataset (train+test)
    # Note: In stricter protocols, PCA should be fit only on training data
    # But for encoding validation, this is acceptable
    pca = PCA(n_components=n_components or min(n_neurons, X_flat.shape[0]))
    pca.fit(X_flat)
    
    # Determine number of components
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    if n_components is None:
        n_pcs = np.searchsorted(cumsum_var, variance_threshold) + 1
        n_pcs = max(n_pcs, 2)  # At least 2 PCs
    else:
        n_pcs = n_components
    
    # Transform to PC space
    pc_scores = pca.transform(X_flat)[:, :n_pcs]  # (n_trials * n_bins, n_pcs)
    
    # Prepare input features: (F1, normalized_time)
    # Repeat F1 for each time bin within a trial
    f1_repeated = np.repeat(f1_labels, n_bins)  # (n_trials * n_bins,)
    
    # Normalized time within delay period (0 to 1)
    time_normalized = (time_bins - time_bins.min()) / (time_bins.max() - time_bins.min() + 1e-8)
    time_repeated = np.tile(time_normalized, n_trials)  # (n_trials * n_bins,)
    
    # Create input matrix
    X = np.column_stack([f1_repeated, time_repeated])  # (n_trials * n_bins, 2)
    y = pc_scores  # (n_trials * n_bins, n_pcs)
    
    # Store time bin indices for later analysis
    time_bin_indices = np.tile(np.arange(n_bins), n_trials)
    
    # Train/test split
    X_train, X_test, y_train, y_test, time_train, time_test, f1_train, f1_test = train_test_split(
        X, y, time_bin_indices, f1_repeated,
        test_size=test_size,
        random_state=random_state
    )
    
    return EncodingDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_bins_train=time_train,
        time_bins_test=time_test,
        f1_train=f1_train,
        f1_test=f1_test,
        pca_model=pca,
        explained_variance_ratio=pca.explained_variance_ratio_[:n_pcs],
        n_pcs=n_pcs,
        session_ids=list(set(session_ids)),
        n_sessions=len(set(session_ids))
    )


def train_encoding_model(
    dataset: EncodingDataset,
    model_type: str = "ridge",
    alpha: float = 1.0
) -> Any:
    """
    Train encoding model: (F1, time) -> PC scores
    
    Args:
        dataset: EncodingDataset object
        model_type: "ridge", "linear", or "mlp"
        alpha: Regularization strength for Ridge
        
    Returns:
        Trained model
    """
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(dataset.X_train, dataset.y_train)
    
    return model


def evaluate_encoding(
    model: Any,
    dataset: EncodingDataset,
    model_name: str = "Ridge"
) -> EncodingResults:
    """
    Evaluate encoding model performance
    
    Args:
        model: Trained encoding model
        dataset: EncodingDataset object
        model_name: Name for logging
        
    Returns:
        EncodingResults object
    """
    # Make predictions on test set
    y_pred = model.predict(dataset.X_test)
    
    # Compute R² per PC
    r2_per_pc = np.array([
        r2_score(dataset.y_test[:, i], y_pred[:, i])
        for i in range(dataset.n_pcs)
    ])
    
    # Overall R² (multioutput='uniform_average')
    r2_overall = r2_score(dataset.y_test, y_pred, multioutput='uniform_average')
    
    return EncodingResults(
        r2_per_pc=r2_per_pc,
        r2_overall=r2_overall,
        predictions=y_pred,
        targets=dataset.y_test,
        test_time_bins=dataset.time_bins_test,
        test_f1=dataset.f1_test,
        model_name=model_name
    )


def shuffle_control_encoding(
    neural_data: np.ndarray,
    f1_labels: np.ndarray,
    time_bins: np.ndarray,
    session_ids: List[str],
    n_components: int,
    model_type: str = "ridge",
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> EncodingResults:
    """
    Shuffle control: randomize F1 labels and re-run encoding
    
    This tests whether model performance depends on true F1 values
    or is driven by trivial temporal correlations.
    
    Args:
        Same as prepare_encoding_dataset + train_encoding_model
        
    Returns:
        EncodingResults for shuffled control
    """
    # Shuffle F1 labels
    rng = np.random.RandomState(random_state + 999)
    f1_shuffled = rng.permutation(f1_labels)
    
    # Prepare dataset with shuffled F1
    dataset_shuffled = prepare_encoding_dataset(
        neural_data=neural_data,
        f1_labels=f1_shuffled,
        time_bins=time_bins,
        session_ids=session_ids,
        n_components=n_components,
        test_size=test_size,
        random_state=random_state
    )
    
    # Train model
    model_shuffled = train_encoding_model(
        dataset=dataset_shuffled,
        model_type=model_type,
        alpha=alpha
    )
    
    # Evaluate
    results_shuffled = evaluate_encoding(
        model=model_shuffled,
        dataset=dataset_shuffled,
        model_name=f"{model_type}_shuffled"
    )
    
    return results_shuffled


def plot_encoding_performance(
    results_true: EncodingResults,
    results_shuffled: EncodingResults,
    save_path: Optional[str] = None
) -> None:
    """
    Plot encoding performance: R² per PC, true vs shuffled
    
    Args:
        results_true: Results from true F1 labels
        results_shuffled: Results from shuffled control
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Panel 1: R² per PC
    ax = axes[0]
    x = np.arange(len(results_true.r2_per_pc))
    width = 0.35
    
    ax.bar(x - width/2, results_true.r2_per_pc, width, label='True F1', alpha=0.8)
    ax.bar(x + width/2, results_shuffled.r2_per_pc, width, label='Shuffled F1', alpha=0.8)
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('R²')
    ax.set_title('Encoding Performance per PC')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i+1}' for i in x])
    ax.legend()
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 2: Overall R² comparison
    ax = axes[1]
    overall_r2 = [results_true.r2_overall, results_shuffled.r2_overall]
    colors = ['#2ecc71', '#e74c3c']
    
    ax.bar(['True F1', 'Shuffled F1'], overall_r2, color=colors, alpha=0.8)
    ax.set_ylabel('Overall R²')
    ax.set_title('Overall Encoding Performance')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved encoding performance plot to {save_path}")
    else:
        plt.show()


def plot_predicted_trajectories(
    results: EncodingResults,
    dataset: EncodingDataset,
    n_example_trials: int = 3,
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs true PC trajectories over time
    
    Args:
        results: EncodingResults object
        dataset: EncodingDataset object
        n_example_trials: Number of example trials to show
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Reconstruct trial structure from test set
    # Group by unique (F1, time_bin) pairs
    unique_f1 = np.unique(results.test_f1)
    
    # Select a few F1 values to plot
    selected_f1 = np.linspace(unique_f1.min(), unique_f1.max(), n_example_trials)
    
    n_pcs = min(3, dataset.n_pcs)  # Plot up to 3 PCs
    
    fig, axes = plt.subplots(n_pcs, 1, figsize=(10, 3 * n_pcs))
    if n_pcs == 1:
        axes = [axes]
    
    for pc_idx in range(n_pcs):
        ax = axes[pc_idx]
        
        for f1_val in selected_f1:
            # Find test samples with this F1 value
            mask = np.abs(results.test_f1 - f1_val) < 0.01 * (unique_f1.max() - unique_f1.min())
            
            if mask.sum() == 0:
                continue
            
            # Sort by time bin
            time_indices = results.test_time_bins[mask]
            sort_idx = np.argsort(time_indices)
            
            true_vals = results.targets[mask][sort_idx, pc_idx]
            pred_vals = results.predictions[mask][sort_idx, pc_idx]
            time_vals = time_indices[sort_idx]
            
            # Plot
            ax.plot(time_vals, true_vals, 'o-', alpha=0.6, label=f'F1={f1_val:.0f} (true)')
            ax.plot(time_vals, pred_vals, 's--', alpha=0.6, label=f'F1={f1_val:.0f} (pred)')
        
        ax.set_xlabel('Time Bin Index')
        ax.set_ylabel(f'PC{pc_idx + 1} Score')
        ax.set_title(f'PC{pc_idx + 1} Trajectories (R²={results.r2_per_pc[pc_idx]:.3f})')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()
