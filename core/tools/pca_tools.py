"""
PCA analysis tools for neural population activity

Implements:
- Multiple PCA variants (sklearn full, randomized, incremental)
- Explained variance analysis
- PC trajectory visualization
- Decoding from PCA features
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json


def pca_sklearn_full(
    X2D: np.ndarray,
    n_components: Optional[int] = None,
    whiten: bool = False,
    random_state: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Standard sklearn PCA with full SVD.
    
    Args:
        X2D: Data matrix [(trials*time_bins) × neurons]
        n_components: Number of components (None = min(n_samples, n_features))
        whiten: Whether to whiten the components
        random_state: Random seed
    
    Returns:
        dict with:
            - pca_model: Fitted PCA object
            - explained_variance: Variance explained by each PC
            - explained_variance_ratio: Fraction of variance explained
            - cumulative_variance: Cumulative variance explained
            - components: PC loadings [n_components × n_features]
            - transformed: Transformed data [n_samples × n_components]
    """
    if X2D.ndim != 2:
        raise ValueError(f"X2D must be 2D, got shape {X2D.shape}")
    
    # Fit PCA
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    X_pca = pca.fit_transform(X2D)
    
    # Compute cumulative variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    return {
        "pca_model": pca,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumsum_var.tolist(),
        "components": pca.components_,  # Keep as array
        "transformed": X_pca,
        "n_components": pca.n_components_,
        "n_features": X2D.shape[1],
        "n_samples": X2D.shape[0],
    }


def pca_sklearn_randomized(
    X2D: np.ndarray,
    n_components: int,
    random_state: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Randomized PCA (faster for large datasets).
    
    Args:
        X2D: Data matrix [(trials*time_bins) × neurons]
        n_components: Number of components (required)
        random_state: Random seed
    
    Returns:
        Same format as pca_sklearn_full
    """
    if X2D.ndim != 2:
        raise ValueError(f"X2D must be 2D, got shape {X2D.shape}")
    
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)
    X_pca = pca.fit_transform(X2D)
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    return {
        "pca_model": pca,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumsum_var.tolist(),
        "components": pca.components_,
        "transformed": X_pca,
        "n_components": pca.n_components_,
        "n_features": X2D.shape[1],
        "n_samples": X2D.shape[0],
    }


def pca_incremental(
    X2D: np.ndarray,
    n_components: int,
    batch_size: int = 1000,
    random_state: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Incremental PCA (memory-efficient for very large datasets).
    
    Args:
        X2D: Data matrix [(trials*time_bins) × neurons]
        n_components: Number of components
        batch_size: Batch size for incremental fitting
        random_state: Random seed
    
    Returns:
        Same format as pca_sklearn_full
    """
    if X2D.ndim != 2:
        raise ValueError(f"X2D must be 2D, got shape {X2D.shape}")
    
    np.random.seed(random_state)
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_pca = ipca.fit_transform(X2D)
    
    cumsum_var = np.cumsum(ipca.explained_variance_ratio_)
    
    return {
        "pca_model": ipca,
        "explained_variance": ipca.explained_variance_.tolist(),
        "explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumsum_var.tolist(),
        "components": ipca.components_,
        "transformed": X_pca,
        "n_components": ipca.n_components_,
        "n_features": X2D.shape[1],
        "n_samples": X2D.shape[0],
    }


def plot_explained_variance(
    explained_variance_ratio: List[float],
    cumulative_variance: List[float],
    save_path: Optional[str] = None,
    title: str = "PCA Explained Variance",
    target_variance: float = 0.80,
    figsize: Tuple[float, float] = (12, 5),
    **kwargs
) -> Dict[str, Any]:
    """
    Plot explained variance curves.
    
    Args:
        explained_variance_ratio: Variance explained by each PC
        cumulative_variance: Cumulative variance explained
        save_path: Path to save figure
        title: Plot title
        target_variance: Target cumulative variance (draws horizontal line)
        figsize: Figure size
    
    Returns:
        dict with:
            - figure_path: Path to saved figure
            - k_for_target: Number of PCs to reach target variance
            - variance_at_target: Actual variance at k_for_target
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_components = len(explained_variance_ratio)
    pcs = np.arange(1, n_components + 1)
    
    # Individual variance
    ax1.bar(pcs[:min(20, n_components)], explained_variance_ratio[:20], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual PC Variance (Top 20)')
    ax1.grid(alpha=0.3)
    
    # Cumulative variance
    ax2.plot(pcs, cumulative_variance, marker='o', markersize=3, color='darkorange', linewidth=2)
    ax2.axhline(y=target_variance, color='red', linestyle='--', label=f'{target_variance*100:.0f}% target')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, n_components + 1)
    ax2.set_ylim(0, 1.05)
    
    # Find k for target variance
    k_for_target = np.argmax(np.array(cumulative_variance) >= target_variance) + 1
    if k_for_target > 0:
        ax2.axvline(x=k_for_target, color='green', linestyle=':', alpha=0.5, label=f'k={k_for_target}')
        ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    variance_at_target = cumulative_variance[k_for_target - 1] if k_for_target > 0 else 0.0
    
    return {
        "figure_path": save_path,
        "k_for_target": int(k_for_target),
        "variance_at_target": float(variance_at_target),
        "top_10_variance": explained_variance_ratio[:10],
    }


def plot_pc_trajectories_over_time(
    X_pca: np.ndarray,
    trial_labels: np.ndarray,
    time_bins: int,
    save_path: Optional[str] = None,
    title: str = "PC Trajectories Over Time",
    pcs_to_plot: List[int] = [0, 1, 2],
    unique_labels: Optional[List[Any]] = None,
    figsize: Tuple[float, float] = (15, 5),
    **kwargs
) -> Dict[str, Any]:
    """
    Plot PC trajectories over time for different conditions.
    
    Args:
        X_pca: Transformed data [n_samples × n_components]
        trial_labels: Trial labels [n_trials]
        time_bins: Number of time bins per trial
        save_path: Path to save figure
        title: Plot title
        pcs_to_plot: Which PCs to plot (0-indexed)
        unique_labels: Unique condition labels (auto-detected if None)
        figsize: Figure size
    
    Returns:
        dict with figure_path and metadata
    """
    n_samples = X_pca.shape[0]
    n_trials = len(trial_labels)
    
    # Reshape to [n_trials, time_bins, n_components]
    expected_samples = n_trials * time_bins
    if n_samples != expected_samples:
        raise ValueError(f"Shape mismatch: X_pca has {n_samples} samples but expected {expected_samples} "
                        f"({n_trials} trials × {time_bins} time bins)")
    
    X_reshaped = X_pca.reshape(n_trials, time_bins, -1)
    
    if unique_labels is None:
        unique_labels = np.unique(trial_labels)
    
    n_pcs = len(pcs_to_plot)
    fig, axes = plt.subplots(1, n_pcs, figsize=figsize)
    if n_pcs == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for ax, pc_idx in zip(axes, pcs_to_plot):
        for label_idx, label in enumerate(unique_labels):
            # Get trials for this condition
            trial_mask = trial_labels == label
            pc_data = X_reshaped[trial_mask, :, pc_idx]  # [n_trials_cond, time_bins]
            
            # Plot mean trajectory
            mean_traj = pc_data.mean(axis=0)
            sem_traj = pc_data.std(axis=0) / np.sqrt(pc_data.shape[0])
            
            time_axis = np.arange(time_bins)
            ax.plot(time_axis, mean_traj, color=colors[label_idx], 
                   label=f'{label}', linewidth=2, alpha=0.8)
            ax.fill_between(time_axis, mean_traj - sem_traj, mean_traj + sem_traj,
                           color=colors[label_idx], alpha=0.2)
        
        ax.set_xlabel('Time Bin')
        ax.set_ylabel(f'PC{pc_idx + 1}')
        ax.set_title(f'PC{pc_idx + 1} Over Time')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        "figure_path": save_path,
        "n_conditions": len(unique_labels),
        "n_trials": n_trials,
        "time_bins": time_bins,
        "pcs_plotted": [pc + 1 for pc in pcs_to_plot],
    }


def decode_f1_linear(
    X_pca: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
    classifier: str = 'logreg',
    random_state: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Decode stimulus labels from PCA features using linear classifier.
    
    Args:
        X_pca: PCA-transformed features [n_samples × n_components]
        labels: Target labels [n_samples]
        cv: Number of cross-validation folds
        classifier: 'logreg' (LogisticRegression) or 'linear_svm' (LinearSVC)
        random_state: Random seed
    
    Returns:
        dict with:
            - accuracy_mean: Mean CV accuracy
            - accuracy_std: Std of CV accuracy
            - accuracies_per_fold: List of accuracies per fold
            - n_classes: Number of classes
            - n_samples: Number of samples
            - n_features: Number of PCA features
    """
    # Select classifier
    if classifier == 'logreg':
        clf = LogisticRegression(max_iter=1000, random_state=random_state, solver='lbfgs')
    elif classifier == 'linear_svm':
        clf = LinearSVC(max_iter=2000, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X_pca, labels, cv=cv_splitter, scoring='accuracy')
    
    return {
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "accuracies_per_fold": scores.tolist(),
        "n_classes": len(np.unique(labels)),
        "n_samples": X_pca.shape[0],
        "n_features": X_pca.shape[1],
        "classifier": classifier,
        "cv_folds": cv,
    }


