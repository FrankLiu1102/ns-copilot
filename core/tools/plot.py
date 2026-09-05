# core/tools/plot.py

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    # Create dummy classes for type hints
    class plt:
        @staticmethod
        def figure(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def close(): pass
        @staticmethod
        def title(*args, **kwargs): pass
        @staticmethod
        def ylabel(*args, **kwargs): pass
        @staticmethod
        def xlabel(*args, **kwargs): pass
        @staticmethod
        def tight_layout(): pass
        @staticmethod
        def plot(*args, **kwargs): pass
        @staticmethod
        def errorbar(*args, **kwargs): pass
        @staticmethod
        def legend(*args, **kwargs): pass
        @staticmethod
        def grid(*args, **kwargs): pass
    class sns:
        @staticmethod
        def heatmap(*args, **kwargs): pass

from ..state import NeuroGlobalState

if TYPE_CHECKING:
    # only for type hints; avoids circular import at runtime
    from ..execution_agent import ExecutionContext


def plot_confusion_matrix_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Render a confusion matrix figure from decoding results.
    
    Args:
        confusion_matrix: Confusion matrix data (2D array or list of lists)
        title: Optional title for the plot
        normalize: If True, normalize the matrix (show percentages)
        save_path: Optional path to save the figure (if not provided, saves to ctx.outputs_root)
        class_labels: Optional list of class labels (if not provided, uses indices)
        figsize: Optional figure size tuple (default: (8, 6))
        cmap: Optional colormap (default: 'Blues')
    
    Returns:
        Dict with figure_path and metadata
    """
    if not HAS_PLOTTING:
        raise RuntimeError(
            "matplotlib and seaborn are required for plotting. "
            "Please install them: pip install matplotlib seaborn"
        )
    
    confusion_matrix = params.get("confusion_matrix")
    if confusion_matrix is None:
        raise ValueError("confusion_matrix parameter is required")
    
    # Convert to numpy array if needed
    if isinstance(confusion_matrix, list):
        cm = np.array(confusion_matrix)
    elif isinstance(confusion_matrix, np.ndarray):
        cm = confusion_matrix.copy()
    else:
        raise ValueError(f"confusion_matrix must be a 2D array or list of lists, got {type(confusion_matrix)}")
    
    if cm.ndim != 2:
        raise ValueError(f"confusion_matrix must be 2D, got shape {cm.shape}")
    
    # Get optional parameters
    title = params.get("title", "Confusion Matrix")
    normalize = params.get("normalize", False)
    class_labels = params.get("class_labels", None)
    figsize = params.get("figsize", (8, 6))
    cmap = params.get("cmap", "Blues")
    dpi = params.get("dpi", 150)
    
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        cm_to_plot = cm_normalized
        fmt = '.2f'
        label = 'Normalized'
    else:
        cm_to_plot = cm
        fmt = 'd'
        label = 'Count'
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Use seaborn for better-looking heatmap
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': label},
        xticklabels=class_labels if class_labels else range(cm.shape[1]),
        yticklabels=class_labels if class_labels else range(cm.shape[0]),
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Determine save path
    if "save_path" in params and params["save_path"]:
        save_path = Path(params["save_path"])
    else:
        # Save to outputs directory with a default name
        save_path = ctx.outputs_root / "confusion_matrix.png"
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {
        "figure_path": str(save_path),
        "confusion_matrix_shape": cm.shape,
        "normalized": normalize,
        "class_labels": class_labels if class_labels else list(range(cm.shape[0])),
        "metadata": {
            "total_samples": int(cm.sum()),
            "n_classes": int(cm.shape[0]),
        }
    }


def plot_tuning_curve_tool(
    *,
    state: NeuroGlobalState,
    ctx: "ExecutionContext",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plot a tuning curve showing neuronal responses across different conditions/stimuli.
    
    Args:
        responses: Response data (firing rates, spike counts, etc.)
                  Can be:
                  - 1D array/list: single neuron, one value per condition
                  - 2D array/list: multiple neurons (rows) x conditions (cols), or
                                  single neuron with multiple trials per condition
        conditions: Condition/stimulus labels (e.g., directions, locations, contrasts)
                   Can be:
                   - List of labels (strings or numbers)
                   - List of numbers (will be used as x-axis positions)
        error_bars: Optional error bar data (SEM, SD, or confidence intervals)
                   Can be:
                   - "sem": Compute standard error of mean from responses
                   - "std": Compute standard deviation from responses
                   - Array/list: Precomputed error values
        title: Optional title for the plot
        save_path: Optional path to save the figure
        xlabel: Optional x-axis label (default: "Condition")
        ylabel: Optional y-axis label (default: "Response")
        figsize: Optional figure size tuple (default: (8, 6))
        style: Plot style - "line" (default) or "bar"
        show_individual: If True and responses is 2D, show individual data points
        aggregate: If responses is 2D, how to aggregate: "mean" (default) or "median"
        marker: Marker style for line plots (default: 'o')
        linestyle: Line style for line plots (default: '-')
        linewidth: Line width for line plots (default: 2)
        markersize: Marker size for line plots (default: 8)
        capsize: Error bar cap size (default: 5)
        capthick: Error bar cap thickness (default: 2)
        alpha: Transparency for bar plots and individual points (default: 0.7 for bars, 0.3 for points)
        color: Color for main plot (default: None, uses matplotlib default)
        individual_color: Color for individual data points (default: 'gray')
        individual_size: Size of individual data points (default: 20)
        xlabel_rotation: Rotation angle for x-axis labels if conditions are strings (default: 45)
        show_grid: Whether to show grid (default: True)
        grid_alpha: Grid transparency (default: 0.3)
        dpi: Figure resolution (default: 150)
    
    Returns:
        Dict with figure_path and metadata
    """
    if not HAS_PLOTTING:
        raise RuntimeError(
            "matplotlib and seaborn are required for plotting. "
            "Please install them: pip install matplotlib seaborn"
        )
    
    responses = params.get("responses")
    conditions = params.get("conditions")
    
    if responses is None:
        raise ValueError("responses parameter is required")
    if conditions is None:
        raise ValueError("conditions parameter is required")
    
    # Convert to numpy arrays
    if isinstance(responses, list):
        responses_arr = np.array(responses)
    elif isinstance(responses, np.ndarray):
        responses_arr = responses.copy()
    else:
        raise ValueError(f"responses must be an array or list, got {type(responses)}")
    
    if isinstance(conditions, list):
        conditions_arr = np.array(conditions)
    else:
        conditions_arr = conditions
    
    # Handle 1D vs 2D responses
    if responses_arr.ndim == 1:
        # Single neuron, one value per condition
        mean_responses = responses_arr
        n_neurons = 1
        n_conditions = len(responses_arr)
    elif responses_arr.ndim == 2:
        # Multiple neurons or multiple trials per condition
        aggregate = params.get("aggregate", "mean")
        if aggregate == "mean":
            mean_responses = np.mean(responses_arr, axis=0)
        elif aggregate == "median":
            mean_responses = np.median(responses_arr, axis=0)
        else:
            raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate}")
        
        n_neurons = responses_arr.shape[0]
        n_conditions = responses_arr.shape[1]
    else:
        raise ValueError(f"responses must be 1D or 2D, got shape {responses_arr.shape}")
    
    # Validate dimensions
    if len(mean_responses) != len(conditions_arr):
        raise ValueError(
            f"Number of conditions ({len(conditions_arr)}) must match "
            f"number of response values ({len(mean_responses)})"
        )
    
    # Get optional parameters
    title = params.get("title", "Tuning Curve")
    xlabel = params.get("xlabel", "Condition")
    ylabel = params.get("ylabel", "Response")
    figsize = params.get("figsize", (8, 6))
    style = params.get("style", "line")
    show_individual = params.get("show_individual", False)
    error_bars = params.get("error_bars", None)
    
    # Plotting style parameters (all configurable, no hardcoding)
    marker = params.get("marker", "o")
    linestyle = params.get("linestyle", "-")
    linewidth = params.get("linewidth", 2)
    markersize = params.get("markersize", 8)
    capsize = params.get("capsize", 5)
    capthick = params.get("capthick", 2)
    alpha_bar = params.get("alpha", 0.7)  # For bar plots
    alpha_individual = params.get("alpha", 0.3)  # For individual points (can be overridden)
    if "individual_alpha" in params:
        alpha_individual = params.get("individual_alpha")
    color = params.get("color", None)  # None means use matplotlib default
    individual_color = params.get("individual_color", "gray")
    individual_size = params.get("individual_size", 20)
    xlabel_rotation = params.get("xlabel_rotation", 45)
    show_grid = params.get("show_grid", True)
    grid_alpha = params.get("grid_alpha", 0.3)
    dpi = params.get("dpi", 150)
    
    # Compute error bars if requested
    error_values = None
    if error_bars:
        if error_bars == "sem":
            if responses_arr.ndim == 2:
                error_values = np.std(responses_arr, axis=0) / np.sqrt(responses_arr.shape[0])
            else:
                error_values = None  # Can't compute SEM from single values
        elif error_bars == "std":
            if responses_arr.ndim == 2:
                error_values = np.std(responses_arr, axis=0)
            else:
                error_values = None
        elif isinstance(error_bars, (list, np.ndarray)):
            error_arr = np.array(error_bars)
            if error_arr.shape == mean_responses.shape:
                error_values = error_arr
            else:
                raise ValueError(
                    f"error_bars shape {error_arr.shape} must match responses shape {mean_responses.shape}"
                )
        else:
            raise ValueError(
                f"error_bars must be 'sem', 'std', or an array, got {type(error_bars)}"
            )
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Convert conditions to numeric if needed (for proper x-axis positioning)
    if conditions_arr.dtype.kind in 'US':  # String type
        # Use indices as x positions
        x_positions = np.arange(len(conditions_arr))
        x_labels = conditions_arr
    else:
        # Use conditions directly as x positions
        x_positions = conditions_arr
        x_labels = conditions_arr
    
    # Plot (all parameters are configurable, no hardcoding)
    plot_kwargs = {}
    if color is not None:
        plot_kwargs["color"] = color
    
    if style == "line":
        if error_values is not None:
            errorbar_kwargs = {
                "x": x_positions,
                "y": mean_responses,
                "yerr": error_values,
                "marker": marker,
                "linestyle": linestyle,
                "capsize": capsize,
                "capthick": capthick,
                "label": "Mean ± Error",
            }
            if color is not None:
                errorbar_kwargs["color"] = color
            plt.errorbar(**errorbar_kwargs)
        else:
            plot_kwargs.update({
                "marker": marker,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "markersize": markersize,
                "label": "Response",
            })
            plt.plot(x_positions, mean_responses, **plot_kwargs)
    elif style == "bar":
        bar_kwargs = {
            "x": x_positions,
            "height": mean_responses,
            "alpha": alpha_bar,
            "label": "Mean ± Error" if error_values is not None else "Response",
        }
        if error_values is not None:
            bar_kwargs["yerr"] = error_values
            bar_kwargs["capsize"] = capsize
        if color is not None:
            bar_kwargs["color"] = color
        plt.bar(**bar_kwargs)
    else:
        raise ValueError(f"style must be 'line' or 'bar', got {style}")
    
    # Show individual data points if requested and available
    if show_individual and responses_arr.ndim == 2:
        for i in range(responses_arr.shape[0]):
            plt.scatter(
                x_positions,
                responses_arr[i, :],
                alpha=alpha_individual,
                s=individual_size,
                color=individual_color
            )
    
    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Set x-axis labels if conditions are strings
    if conditions_arr.dtype.kind in 'US':
        plt.xticks(x_positions, x_labels, rotation=xlabel_rotation, ha='right')
    
    # Add grid and legend
    if show_grid:
        plt.grid(True, alpha=grid_alpha)
    if error_values is not None or show_individual:
        plt.legend()
    
    plt.tight_layout()
    
    # Determine save path
    if "save_path" in params and params["save_path"]:
        save_path = Path(params["save_path"])
    else:
        save_path = ctx.outputs_root / "tuning_curve.png"
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {
        "figure_path": str(save_path),
        "n_conditions": int(n_conditions),
        "n_neurons": int(n_neurons) if responses_arr.ndim == 2 else 1,
        "style": style,
        "error_bars": error_bars if error_bars else None,
        "conditions": conditions_arr.tolist() if hasattr(conditions_arr, 'tolist') else list(conditions_arr),
        "mean_responses": mean_responses.tolist(),
        "metadata": {
            "response_range": [float(np.min(mean_responses)), float(np.max(mean_responses))],
            "response_mean": float(np.mean(mean_responses)),
        }
    }

