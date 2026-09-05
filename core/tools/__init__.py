"""
NS-Copilot Analysis Tools

This package contains analysis tools with two interface styles:

1. Legacy tools (decoding.py, etc.):
   - Accept state, ctx, params as arguments
   - Field names are specified in params
   - Used by existing execution_agent

2. Generic tools (generic_tools.py):
   - Accept numpy arrays directly (X, y, etc.)
   - No hardcoded field names
   - Designed for LLM-generated code execution
"""

from .generic_tools import (
    # Decoding
    decode_logistic_regression,
    decode_svm,
    decode_random_forest,
    decode_xgboost,
    decode_with_holdout,
    
    # Feature extraction
    compute_firing_rates,
    bin_spike_times,
    extract_trial_features,
    extract_spectral_features,
    
    # PCA
    pca_decomposition,
    
    # Preprocessing
    normalize_array,

    # Plotting
    plot_confusion_matrix,
    plot_raster,
    
    # Registry
    GENERIC_TOOLS,
    get_tool,
    list_tools,
    get_tool_signature,
    get_all_tool_signatures,
)

__all__ = [
    # Decoding
    "decode_logistic_regression",
    "decode_svm",
    "decode_random_forest",
    "decode_xgboost",
    "decode_with_holdout",
    
    # Feature extraction
    "compute_firing_rates",
    "bin_spike_times",
    "extract_trial_features",
    "extract_spectral_features",
    
    # PCA
    "pca_decomposition",
    
    # Preprocessing
    "normalize_array",

    # Plotting
    "plot_confusion_matrix",
    "plot_raster",
    
    # Registry
    "GENERIC_TOOLS",
    "get_tool",
    "list_tools",
    "get_tool_signature",
    "get_all_tool_signatures",
]
