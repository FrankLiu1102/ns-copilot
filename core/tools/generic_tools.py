"""
Generic Interface Tools for NS-Copilot

This module provides analysis tools with GENERIC INTERFACES that accept
numpy arrays directly, without any dataset-specific field names.

These tools are designed to be called by LLM-generated code:
1. LLM extracts data from raw dataset based on schema
2. LLM calls these tools with extracted arrays (X, y, etc.)
3. Tools return results

NO HARDCODED FIELD NAMES - all inputs are generic numpy arrays.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import time
import hashlib


# Module-level seed, injected by CodeExecutor before each run.
# Tool functions always use _RANDOM_SEED to ensure per-run reproducibility.
# The seed parameter in tool signatures is accepted for API compatibility
# but the framework-injected seed always takes precedence.
_RANDOM_SEED: int = 42

def _get_seed(seed: Optional[int] = None) -> int:
    """Return the framework-injected per-run seed.
    The seed argument is ignored — seed management is centralized
    at the pipeline level, not overridable by individual tool calls."""
    return _RANDOM_SEED


def _checkpoint_metadata(path: Optional[str]) -> Dict[str, Any]:
    """Compute file hash and size for a checkpoint path (reproducibility tracking)."""
    if not path or not Path(path).exists() or Path(path).is_dir():
        return {"checkpoint_path": str(path) if path else None, "checkpoint_hash": None}
    p = Path(path)
    size = p.stat().st_size
    # Use first+last 4MB for fast hashing of large files
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
        if size > 8 * 1024 * 1024:
            f.seek(-4 * 1024 * 1024, 2)
            h.update(f.read())
    return {
        "checkpoint_path": str(path),
        "checkpoint_hash": h.hexdigest()[:16],
        "checkpoint_size_bytes": size,
    }

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GroupKFold,
    train_test_split,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    confusion_matrix, 
    classification_report,
    f1_score,
    roc_auc_score,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None


# =============================================================================
# DECODING TOOLS (Generic Interface)
# =============================================================================


def _run_cv_with_full_metrics(
    model, X, y_encoded, le, cv, groups, seed, pca_components=None, cv_repeats=1,
    sample_weights=None,
) -> Dict[str, Any]:
    """Shared CV runner that returns full metrics for decode_* tools.

    If *pca_components* is not None, PCA is fitted on each training fold and
    applied to the test fold **before** the classifier, avoiding data leakage.

    If *cv_repeats* > 1 and the CV strategy is StratifiedKFold, uses
    RepeatedStratifiedKFold to run multiple repetitions with different shuffles,
    producing more stable metric estimates.
    """
    # Validate: need at least 2 classes for classification
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError(
            f"Classification requires at least 2 classes, but found {n_classes}: {le.classes_.tolist()}. "
            f"Check that the label column contains multiple distinct values."
        )

    # Upgrade to RepeatedStratifiedKFold if repeats > 1
    # Note: RepeatedStratifiedKFold does not support groups, so skip upgrade when groups are provided
    if cv_repeats > 1 and isinstance(cv, StratifiedKFold) and groups is None:
        cv = RepeatedStratifiedKFold(
            n_splits=cv.n_splits, n_repeats=cv_repeats, random_state=seed
        )
    from sklearn.decomposition import PCA

    n_classes = len(le.classes_)
    # Collect per-fold predictions
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_accuracies = []
    fold_bal_accs = []
    fold_f1s = []
    fold_aurocs = []

    splits = list(cv.split(X, y_encoded, groups))
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Apply PCA inside the fold if requested
        if pca_components is not None:
            n_comp = pca_components
            # Handle 0 < pca_components < 1 as variance ratio
            if isinstance(n_comp, float) and 0 < n_comp < 1:
                pca = PCA(n_components=n_comp, random_state=seed)
            else:
                n_comp = min(int(n_comp), X_train.shape[0], X_train.shape[1])
                pca = PCA(n_components=n_comp, random_state=seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        import copy
        fold_model = copy.deepcopy(model)
        if sample_weights is not None:
            fold_model.fit(X_train, y_train, sample_weight=sample_weights[train_idx])
        else:
            fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        # Probabilities for AUROC
        y_proba = None
        _proba_method = None
        try:
            y_proba = fold_model.predict_proba(X_test)
            _proba_method = "predict_proba"
        except Exception:
            try:
                dec = fold_model.decision_function(X_test)
                if dec.ndim == 1:
                    p = 1.0 / (1.0 + np.exp(-dec))
                    y_proba = np.vstack([1 - p, p]).T
                else:
                    exp_d = np.exp(dec - dec.max(axis=1, keepdims=True))
                    y_proba = exp_d / exp_d.sum(axis=1, keepdims=True)
                _proba_method = "decision_function"
            except Exception:
                print(f"⚠️ Warning: Could not extract probabilities for AUROC "
                      f"(model type: {type(fold_model).__name__}). AUROC will be NaN.",
                      flush=True)

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        if y_proba is not None:
            all_y_proba.append(y_proba)

        fold_accuracies.append(float(accuracy_score(y_test, y_pred)))
        fold_bal_accs.append(float(balanced_accuracy_score(y_test, y_pred)))
        fold_f1s.append(float(f1_score(y_test, y_pred, average="macro", zero_division=0)))

        if y_proba is not None:
            try:
                if n_classes == 2:
                    fold_aurocs.append(float(roc_auc_score(y_test, y_proba[:, 1])))
                else:
                    fold_aurocs.append(float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")))
            except Exception:
                fold_aurocs.append(float("nan"))
        else:
            fold_aurocs.append(float("nan"))

    # Aggregate across folds
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    # Per-class recall
    per_class_recall = {}
    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
    for i, cls_name in enumerate(le.classes_):
        denom = cm[i].sum()
        per_class_recall[str(cls_name)] = float(cm[i, i] / denom) if denom > 0 else 0.0

    # Overall AUROC from concatenated probabilities (use available folds even if some failed)
    auroc_overall = float("nan")
    if all_y_proba:
        try:
            y_proba_all = np.vstack(all_y_proba)
            # Match y_true to only the folds that have y_proba
            y_true_with_proba = np.concatenate(
                [all_y_true[i] for i in range(len(all_y_true)) if i < len(all_y_proba)]
            ) if len(all_y_proba) < len(splits) else y_true_all
            if n_classes == 2:
                auroc_overall = float(roc_auc_score(y_true_with_proba, y_proba_all[:, 1]))
            else:
                auroc_overall = float(roc_auc_score(y_true_with_proba, y_proba_all, multi_class="ovr", average="macro"))
            if len(all_y_proba) < len(splits):
                print(f"⚠️ AUROC computed from {len(all_y_proba)}/{len(splits)} folds "
                      f"(some folds lacked probability estimates)", flush=True)
        except Exception:
            pass

    return {
        "cv_accuracy_mean": float(np.mean(fold_accuracies)),
        "cv_accuracy_std": float(np.std(fold_accuracies)),
        "cv_balanced_accuracy_mean": float(np.mean(fold_bal_accs)),
        "cv_balanced_accuracy_std": float(np.std(fold_bal_accs)),
        "cv_f1_macro_mean": float(np.mean(fold_f1s)),
        "cv_f1_macro_std": float(np.std(fold_f1s)),
        "cv_auroc_macro_mean": float(np.nanmean(fold_aurocs)),
        "cv_auroc_macro_std": float(np.nanstd(fold_aurocs)),
        "cv_scores": fold_accuracies,
        "cv_fold_balanced_accuracies": fold_bal_accs,
        "cv_fold_f1_macros": fold_f1s,
        "cv_fold_aurocs": fold_aurocs,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true_all.tolist(),
        "y_pred": y_pred_all.tolist(),
        "y_proba": np.vstack(all_y_proba).tolist() if all_y_proba else None,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "class_labels": le.classes_.tolist(),
    }


def compare_models_significance(
    fold_scores_a: list,
    fold_scores_b: list,
    metric_name: str = "balanced_accuracy",
) -> Dict[str, Any]:
    """Paired t-test comparing two models' fold-level scores.

    Use cv_fold_balanced_accuracies (or other fold-level lists) from two
    decode_* results to test whether the difference is statistically significant.
    """
    from scipy import stats

    a = np.array(fold_scores_a, dtype=float)
    b = np.array(fold_scores_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    diff_mean = float(np.mean(a) - np.mean(b))
    t_stat, p_value = stats.ttest_rel(a, b)

    return {
        "metric": metric_name,
        "model_a_mean": float(np.mean(a)),
        "model_b_mean": float(np.mean(b)),
        "mean_difference": diff_mean,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_folds": n,
    }


def decode_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: Optional[int] = None,
    C: float = 1.0,
    C_values: Optional[List[float]] = None,
    penalty: str = "l2",
    max_iter: int = 1000,
    standardize: bool = True,
    pca_components: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """
    Binary/multiclass classification using logistic regression with full metrics.

    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target labels [n_samples] - integers or will be label-encoded
        groups: Optional group labels [n_samples] for GroupKFold CV.
                When provided, samples with the same group value are never
                split across train/test (e.g., subject IDs for window-level features).
        cv_folds: Number of cross-validation folds
        cv_repeats: Number of CV repetitions (default 1). When > 1, uses
                    RepeatedStratifiedKFold for more stable metric estimates.
        seed: Random seed for reproducibility
        C: Regularization strength (used when C_values is None)
        C_values: Optional list of C values to search over. Runs CV for each,
                  returns the best result by balanced accuracy.
        penalty: 'l1', 'l2', or 'elasticnet'
        max_iter: Maximum iterations for solver
        standardize: Whether to standardize features (recommended)
        pca_components: Optional PCA dimensionality reduction applied inside each
                        CV fold (no data leakage). int = fixed number of components,
                        float in (0,1) = variance ratio to retain.

    Returns:
        Dict with CV metrics: accuracy, balanced_accuracy, f1_macro,
        auroc_macro, per-class recall, confusion_matrix, y_true, y_pred, and metadata.
    """
    seed = _get_seed(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

    # Label encode if needed
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    # Choose CV strategy
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError(f"groups and y must have same length: {len(groups)} vs {len(y)}")
        n_groups = len(np.unique(groups))
        actual_folds = min(cv_folds, n_groups)
        cv = GroupKFold(n_splits=actual_folds)
    else:
        groups = None
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    solver = "lbfgs" if penalty == "l2" else "saga"

    def _build_model(c_val):
        if standardize:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=c_val, penalty=penalty, solver=solver,
                    max_iter=max_iter, random_state=seed, class_weight="balanced",
                )),
            ])
        return LogisticRegression(
            C=c_val, penalty=penalty, solver=solver,
            max_iter=max_iter, random_state=seed, class_weight="balanced",
        )

    # Grid search over C values if provided
    if C_values and len(C_values) > 1:
        best_result = None
        best_ba = -1.0
        best_c = C_values[0]
        for c_val in C_values:
            model = _build_model(c_val)
            res = _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats)
            ba = res.get("cv_balanced_accuracy_mean", -1.0)
            if ba > best_ba:
                best_ba = ba
                best_result = res
                best_c = c_val
        best_result["best_C"] = float(best_c)
        best_result["C_values_searched"] = C_values
        return best_result

    model = _build_model(C)
    return _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats)


def decode_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: Optional[int] = None,
    C: float = 1.0,
    C_values: Optional[List[float]] = None,
    kernel: str = "rbf",
    gamma: str = "scale",
    standardize: bool = True,
    pca_components: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """
    Classification using Support Vector Machine with full metrics.

    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target labels [n_samples]
        groups: Optional group labels [n_samples] for GroupKFold CV.
                When provided, samples with the same group value are never
                split across train/test (e.g., subject IDs for window-level features).
        cv_folds: Number of cross-validation folds
        cv_repeats: Number of CV repetitions (default 1). When > 1, uses
                    RepeatedStratifiedKFold for more stable metric estimates.
        seed: Random seed
        C: Regularization parameter (used when C_values is None)
        C_values: Optional list of C values to search over. Runs CV for each,
                  returns the best result by balanced accuracy.
        kernel: 'linear', 'rbf', 'poly', 'sigmoid'
        gamma: Kernel coefficient ('scale', 'auto', or float)
        standardize: Whether to standardize features
        pca_components: Optional PCA dimensionality reduction applied inside each
                        CV fold (no data leakage). int = fixed number of components,
                        float in (0,1) = variance ratio to retain.

    Returns:
        Dict with CV metrics: accuracy, balanced_accuracy, f1_macro,
        auroc_macro, per-class recall, confusion_matrix, y_true, y_pred, and metadata.
    """
    seed = _get_seed(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        actual_folds = min(cv_folds, n_groups)
        cv = GroupKFold(n_splits=actual_folds)
    else:
        groups = None
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def _build_model(c_val):
        if standardize:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=c_val, kernel=kernel, gamma=gamma,
                            random_state=seed, class_weight="balanced",
                            probability=True)),
            ])
        return SVC(C=c_val, kernel=kernel, gamma=gamma,
                   random_state=seed, class_weight="balanced",
                   probability=True)

    # Grid search over C values if provided
    if C_values and len(C_values) > 1:
        best_result = None
        best_ba = -1.0
        best_c = C_values[0]
        for c_val in C_values:
            model = _build_model(c_val)
            res = _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats)
            ba = res.get("cv_balanced_accuracy_mean", -1.0)
            if ba > best_ba:
                best_ba = ba
                best_result = res
                best_c = c_val
        best_result["best_C"] = float(best_c)
        best_result["C_values_searched"] = C_values
        best_result["kernel"] = kernel
        return best_result

    model = _build_model(C)
    result = _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats)
    result["kernel"] = kernel
    return result


def decode_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: Optional[int] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    pca_components: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """
    Classification using Random Forest with full metrics.

    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target labels [n_samples]
        groups: Optional group labels [n_samples] for GroupKFold CV.
        cv_folds: Number of cross-validation folds
        cv_repeats: Number of CV repetitions (default 1). When > 1, uses
                    RepeatedStratifiedKFold for more stable metric estimates.
        seed: Random seed
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_leaf: Minimum samples per leaf
        pca_components: Optional PCA inside each CV fold. int = fixed components,
                        float in (0,1) = variance ratio to retain.

    Returns:
        Dict with CV metrics: accuracy, balanced_accuracy, f1_macro,
        auroc_macro, per-class recall, confusion_matrix, y_true, y_pred, and metadata.
    """
    seed = _get_seed(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        actual_folds = min(cv_folds, n_groups)
        cv = GroupKFold(n_splits=actual_folds)
    else:
        groups = None
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    result = _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats)
    result["n_estimators"] = n_estimators
    result["max_depth"] = max_depth
    return result


def decode_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    seed: Optional[int] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    pca_components: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """
    Classification using XGBoost (gradient boosting) with full metrics.

    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target labels [n_samples]
        groups: Optional group labels [n_samples] for GroupKFold CV.
        cv_folds: Number of cross-validation folds
        cv_repeats: Number of CV repetitions (default 1). When > 1, uses
                    RepeatedStratifiedKFold for more stable metric estimates.
        seed: Random seed
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage (eta)
        pca_components: Optional PCA inside each CV fold. int = fixed components,
                        float in (0,1) = variance ratio to retain.

    Returns:
        Dict with CV metrics: accuracy, balanced_accuracy, f1_macro,
        auroc_macro, per-class recall, confusion_matrix, y_true, y_pred, and metadata.
    """
    seed = _get_seed(seed)
    from xgboost import XGBClassifier

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        actual_folds = min(cv_folds, n_groups)
        cv = GroupKFold(n_splits=actual_folds)
    else:
        groups = None
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Compute sample weights for class balancing
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_encoded)
    # XGBClassifier uses scale_pos_weight for binary; for general case we use sample_weight in fit
    # But since _run_cv_with_full_metrics calls model.fit(X_train, y_train), we wrap with a Pipeline-like approach

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )

    # For class balancing with XGBoost, set scale_pos_weight for binary
    if n_classes == 2:
        n_pos = np.sum(y_encoded == 1)
        n_neg = np.sum(y_encoded == 0)
        if n_pos > 0:
            model.set_params(scale_pos_weight=float(n_neg) / float(n_pos))

    result = _run_cv_with_full_metrics(model, X, y_encoded, le, cv, groups, seed, pca_components, cv_repeats,
                                       sample_weights=sample_weights)
    result["n_estimators"] = n_estimators
    result["max_depth"] = max_depth
    result["learning_rate"] = learning_rate
    return result


def decode_with_holdout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    cv_folds: int = 5,
    seed: Optional[int] = None,
    classifier: str = "logistic_regression",
    C: float = 1.0,
    penalty: str = "l2",
    standardize: bool = True,
    class_weight: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Comprehensive decoding with train/val/test split and cross-validation.
    
    Args:
        X: Feature matrix [n_samples, n_features]
        y: Target labels [n_samples]
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        cv_folds: Number of CV folds on training set
        seed: Random seed
        classifier: 'logistic_regression' or 'svm'
        C: Regularization strength
        penalty: 'l1' or 'l2'
        standardize: Whether to standardize features
        class_weight: Optional class weighting strategy ('balanced' or None)
        save_path: Optional path to save confusion matrix plot
        **kwargs: Additional keyword arguments (silently ignored for forward-compatibility)
    
    Returns:
        Dict with comprehensive metrics:
            - test_accuracy, test_balanced_accuracy
            - test_f1_macro, test_f1_micro
            - cv_accuracy_mean, cv_accuracy_std
            - confusion_matrix
            - class_labels, n_samples, n_features
    """
    seed = _get_seed(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    # Label encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # Sanitize ratios
    if test_ratio is None:
        test_ratio = 0.15
    if test_ratio <= 0:
        test_ratio = 0.15

    # Split BEFORE standardizing to prevent data leakage
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_encoded,
        test_size=test_ratio,
        random_state=seed,
        stratify=y_encoded
    )

    # Split: train vs val (skip if val_ratio is None or 0)
    if val_ratio and val_ratio > 0:
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=seed,
            stratify=y_trainval
        )
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = None, None

    # Standardize: fit on training data only
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    # Create classifier
    _cw = class_weight if class_weight in ("balanced",) else None
    if classifier == "svm":
        model = SVC(C=C, kernel="rbf", random_state=seed, probability=True,
                     class_weight=_cw)
    else:
        solver = "lbfgs" if penalty == "l2" else "saga"
        model = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=1000, random_state=seed,
            class_weight=_cw,
        )
    
    # Cross-validation on training set
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro")
    
    # Train on full training set and evaluate
    model.fit(X_train, y_train)
    
    # Validation metrics (skip if no validation set)
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
    else:
        val_accuracy = None
    
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_micro = f1_score(y_test, y_test_pred, average="micro")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # AUROC (if binary or has predict_proba)
    test_auroc = None
    if hasattr(model, "predict_proba"):
        try:
            y_test_proba = model.predict_proba(X_test)
            if n_classes == 2:
                test_auroc = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                test_auroc = roc_auc_score(
                    y_test, y_test_proba, multi_class="ovr", average="macro"
                )
        except:
            pass
    
    # Plot confusion matrix
    figure_path = None
    if save_path and HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Test Accuracy: {test_accuracy:.3f})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        figure_path = save_path
    
    # Per-class recall using original label names
    per_class_recall = {}
    for idx, label in enumerate(le.classes_):
        class_mask = (y_test == idx)
        if class_mask.sum() > 0:
            per_class_recall[str(label)] = float((y_test_pred[class_mask] == idx).sum() / class_mask.sum())

    result = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "class_labels": le.classes_.tolist(),
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val) if X_val is not None else 0,
            "test": len(X_test),
        },
        "val_accuracy": float(val_accuracy) if val_accuracy is not None else None,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_balanced_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_micro": float(test_f1_micro),
        "test_auroc": float(test_auroc) if test_auroc else None,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "cv_f1_macro_mean": float(np.mean(cv_f1_scores)),
        "cv_f1_macro_std": float(np.std(cv_f1_scores)),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "figure_path": figure_path,
    }
    # Also put per-class recall as top-level keys for easy access
    result.update(per_class_recall)
    return result


def decode_with_ndt3(
    spike_data: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from spike data using pretrained NDT3 foundation model.

    NDT3 is a 45M-parameter Transformer pretrained on 2000 hours of spike data.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        spike_data: Binned spike counts [n_trials, n_time_bins, n_neurons]
        **kwargs: checkpoint_path (str), pool_method ('mean'/'max'/'last')

    Returns:
        Dict with:
            - features: NDT3 latent features [n_trials, hidden_dim]
            - n_trials: Number of trials
            - hidden_dim: Feature dimension
    """
    from neuro_copilot.core.models.ndt3_tool_spec import ndt3_extract_features

    if not isinstance(spike_data, np.ndarray):
        spike_data = np.asarray(spike_data)

    if spike_data.ndim != 3:
        raise ValueError(
            f"spike_data must be 3D [n_trials, n_time_bins, n_neurons], "
            f"got shape {spike_data.shape}"
        )

    try:
        result = ndt3_extract_features(spike_data, **kwargs)
        # Track checkpoint for reproducibility
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "NDT3"
        return result
    except Exception as e:
        raise RuntimeError(
            f"NDT3 decoding failed: {str(e)}\n"
            f"Data shape: {spike_data.shape}\n"
            f"Make sure NDT3 checkpoint is downloaded and accessible."
        ) from e


def decode_with_mtm(
    spike_data: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from spike data using pretrained MtM foundation model.

    MtM (Multi-task Masking) uses an NDT1 encoder pretrained on IBL Neuropixels
    mouse recordings with 8 diverse masking strategies.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        spike_data: Binned spike counts [n_trials, n_time_bins, n_neurons]
                    Bin size: 20ms, max time bins: 100 (2s window).
        **kwargs: checkpoint_path (str), pool_method ('mean'/'max'/'last')

    Returns:
        Dict with:
            - features: MtM latent features [n_trials, hidden_dim]
            - n_trials: Number of trials
            - hidden_dim: Feature dimension
    """
    from neuro_copilot.core.models.mtm_tool_spec import mtm_extract_features

    if not isinstance(spike_data, np.ndarray):
        spike_data = np.asarray(spike_data)

    if spike_data.ndim != 3:
        raise ValueError(
            f"spike_data must be 3D [n_trials, n_time_bins, n_neurons], "
            f"got shape {spike_data.shape}"
        )

    try:
        result = mtm_extract_features(spike_data, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "MtM"
        return result
    except Exception as e:
        raise RuntimeError(
            f"MtM decoding failed: {str(e)}\n"
            f"Data shape: {spike_data.shape}\n"
            f"Make sure MtM checkpoint is downloaded and accessible."
        ) from e


def decode_with_poyo(
    spike_data: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from spike data using pretrained POYO-1 foundation model.

    POYO-1 is a PerceiverIO-based model pretrained on 178 sessions from mouse
    and macaque recordings. It uses cross-attention on individual spike tokens.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        spike_data: Binned spike counts [n_trials, n_time_bins, n_neurons]
        **kwargs: checkpoint_path (str), pool_method ('mean'/'max'/'last')

    Returns:
        Dict with:
            - features: POYO-1 latent features [n_trials, hidden_dim]
            - n_trials: Number of trials
            - hidden_dim: Feature dimension (determined by checkpoint)
    """
    from neuro_copilot.core.models.poyo_tool_spec import poyo_extract_features

    if not isinstance(spike_data, np.ndarray):
        spike_data = np.asarray(spike_data)

    if spike_data.ndim != 3:
        raise ValueError(
            f"spike_data must be 3D [n_trials, n_time_bins, n_neurons], "
            f"got shape {spike_data.shape}"
        )

    try:
        result = poyo_extract_features(spike_data, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "POYO-1"
        return result
    except Exception as e:
        raise RuntimeError(
            f"POYO-1 decoding failed: {str(e)}\n"
            f"Data shape: {spike_data.shape}\n"
            f"Make sure POYO checkpoint is downloaded and accessible."
        ) from e


def decode_with_labram(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained LaBraM foundation model.

    LaBraM (Large Brain Model) is a Neural Transformer pretrained on ~2,500 hours
    of EEG data. It outputs 200-dimensional latent representations.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names in standard 10-20 notation.
        **kwargs: checkpoint_path (str), max_seconds (float),
                  pool_method ('mean'/'cls')

    Returns:
        Dict with:
            - features: LaBraM latent features [n_subjects, 200]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (200)
    """
    from neuro_copilot.core.models.labram_tool_spec import labram_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = labram_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "LaBraM"
        return result
    except Exception as e:
        raise RuntimeError(
            f"LaBraM decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure LaBraM checkpoint is downloaded and accessible."
        ) from e


def decode_with_reve(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained REVE foundation model.

    REVE is a Transformer pretrained on 60,000+ hours of EEG data from 92 datasets
    spanning 25,000 subjects. It outputs 512-dimensional latent representations.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names in standard notation (10-20, 10-10, etc.).
        **kwargs: model_name (str), max_seconds (float)

    Returns:
        Dict with:
            - features: REVE latent features [n_subjects, 512]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (512)
    """
    from neuro_copilot.core.models.reve_tool_spec import reve_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = reve_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "REVE"
        return result
    except Exception as e:
        raise RuntimeError(
            f"REVE decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure braindecode is installed: pip install braindecode[hug]"
        ) from e


def decode_with_cbramod(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained CBraMod foundation model.

    CBraMod is a Criss-Cross Transformer pretrained on ~9,000 hours of EEG data.
    It outputs 200-dimensional latent representations per channel-patch, pooled
    into 600-dim features (mean+std+max). Channel-agnostic — works with any
    electrode configuration without position info.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names (any naming convention).
        **kwargs: checkpoint_path (str), max_seconds (float)

    Returns:
        Dict with:
            - features: CBraMod latent features [n_subjects, 600]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (600)
    """
    from neuro_copilot.core.models.cbramod_tool_spec import cbramod_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = cbramod_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "CBraMod"
        return result
    except Exception as e:
        raise RuntimeError(
            f"CBraMod decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure CBraMod repo is cloned and checkpoint is downloaded."
        ) from e


def decode_with_eegpt(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained EEGPT foundation model.

    EEGPT is a ViT-based foundation model pretrained on 58-channel EEG data
    using mask-based dual self-supervised learning (NeurIPS 2024). It uses
    channel-specific embeddings for standard 10-10 channels and processes
    4-second segments at 256Hz. Outputs 2048-dim features (4 summary tokens
    x 512-dim).
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names in standard 10-10 or 10-20 notation.
        **kwargs: checkpoint_path (str), max_seconds (float)

    Returns:
        Dict with:
            - features: EEGPT latent features [n_subjects, 2048]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (2048)
    """
    from neuro_copilot.core.models.eegpt_tool_spec import eegpt_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = eegpt_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "EEGPT"
        return result
    except Exception as e:
        raise RuntimeError(
            f"EEGPT decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure EEGPT repo is cloned and checkpoint is downloaded."
        ) from e


def decode_with_eegmamba(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained EEGMamba foundation model.

    EEGMamba is a bidirectional Mamba (SSM) model pretrained on ~16,700 hours of
    EEG data. It outputs 200-dimensional latent representations per channel-patch,
    pooled into 600-dim features (mean+std+max). Channel-agnostic — works with any
    electrode configuration without position info.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names (any naming convention).
        **kwargs: checkpoint_path (str), max_seconds (float)

    Returns:
        Dict with:
            - features: EEGMamba latent features [n_subjects, 600]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (600)
    """
    from neuro_copilot.core.models.eegmamba_tool_spec import eegmamba_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = eegmamba_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "EEGMamba"
        return result
    except Exception as e:
        raise RuntimeError(
            f"EEGMamba decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure EEGMamba repo is cloned and checkpoint is downloaded."
        ) from e


def decode_with_brainomni(
    eeg_data: list,
    sfreq: float,
    ch_names: list,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract features from EEG data using pretrained BrainOmni foundation model.

    BrainOmni is a Criss-Cross Transformer with Sensor Encoder pretrained on
    ~2,650 hours of EEG/MEG data. It produces 4096-dim feature vectors (16 latent
    sources × 256 dims). Uses 3D electrode positions from MNE standard montages.
    This function ONLY does feature extraction. All downstream analysis
    (classification, metrics, etc.) should be done in generated code.

    Args:
        eeg_data: List of (n_channels, n_samples) arrays, one per subject.
                  Data should be in microvolts. Any sampling rate supported.
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names (standard names like 'Fp1', 'Cz', etc.).
        **kwargs: checkpoint_dir (str), max_seconds (float)

    Returns:
        Dict with:
            - features: BrainOmni latent features [n_subjects, 4096]
            - n_subjects: Number of subjects
            - embed_dim: Feature dimension (4096)
    """
    from neuro_copilot.core.models.brainomni_tool_spec import brainomni_extract_features

    if not isinstance(eeg_data, list):
        raise ValueError(
            f"eeg_data must be a list of (n_channels, n_samples) arrays, "
            f"got {type(eeg_data)}"
        )

    if len(eeg_data) == 0:
        raise ValueError("eeg_data list is empty")

    if not isinstance(ch_names, list) or len(ch_names) == 0:
        raise ValueError("ch_names must be a non-empty list of channel names")

    try:
        result = brainomni_extract_features(eeg_data, sfreq, ch_names, **kwargs)
        ckpt = result.pop("checkpoint_path", None) or kwargs.get("checkpoint_path")
        result["model_metadata"] = _checkpoint_metadata(ckpt)
        result["model_metadata"]["model_name"] = "BrainOmni"
        return result
    except Exception as e:
        raise RuntimeError(
            f"BrainOmni decoding failed: {str(e)}\n"
            f"n_subjects: {len(eeg_data)}, sfreq: {sfreq}, "
            f"n_channels: {len(ch_names)}\n"
            f"Make sure BrainOmni repo is cloned and checkpoints are downloaded."
        ) from e


# =============================================================================
# FEATURE EXTRACTION HELPERS
# =============================================================================

def compute_firing_rates(
    spike_times: List[np.ndarray],
    time_window: Tuple[float, float],
) -> np.ndarray:
    """
    Compute firing rates from spike times for each unit.
    
    Args:
        spike_times: List of spike time arrays, one per unit
        time_window: (start, end) time window in seconds
    
    Returns:
        Firing rates array [n_units] in spikes/second
    """
    start, end = time_window
    duration = end - start
    
    if duration <= 0:
        raise ValueError(f"Invalid time window: ({start}, {end})")
    
    rates = []
    for unit_spikes in spike_times:
        unit_spikes = np.asarray(unit_spikes)
        n_spikes = np.sum((unit_spikes >= start) & (unit_spikes < end))
        rate = n_spikes / duration
        rates.append(rate)
    
    return np.array(rates)


def bin_spike_times(
    spike_times: List[np.ndarray],
    time_window: Tuple[float, float],
    bin_size: float,
) -> np.ndarray:
    """
    Bin spike times into histogram features.
    
    Args:
        spike_times: List of spike time arrays, one per unit
        time_window: (start, end) time window in seconds
        bin_size: Bin size in seconds
    
    Returns:
        Binned counts array [n_units, n_bins]
    """
    start, end = time_window
    n_bins = int(np.ceil((end - start) / bin_size))
    bin_edges = np.linspace(start, end, n_bins + 1)
    
    binned = []
    for unit_spikes in spike_times:
        unit_spikes = np.asarray(unit_spikes)
        counts, _ = np.histogram(unit_spikes, bins=bin_edges)
        binned.append(counts)
    
    return np.array(binned)


def extract_trial_features(
    spike_times_by_unit: List[np.ndarray],
    trial_windows: List[Tuple[float, float]],
    feature_type: str = "rate",
    bin_size: float = 0.1,
) -> np.ndarray:
    """
    Extract features for each trial from spike times.
    
    Args:
        spike_times_by_unit: List of spike time arrays, one per unit [n_units]
        trial_windows: List of (start, end) time windows for each trial [n_trials]
        feature_type: 'rate' (single rate per unit) or 'binned' (multiple bins per unit)
        bin_size: Bin size in seconds (for 'binned' type)
    
    Returns:
        Feature matrix [n_trials, n_features]
    """
    n_units = len(spike_times_by_unit)
    n_trials = len(trial_windows)
    
    if feature_type == "rate":
        # One feature per unit (firing rate)
        X = np.zeros((n_trials, n_units))
        
        for t_idx, (t_start, t_end) in enumerate(trial_windows):
            duration = t_end - t_start
            for u_idx, unit_spikes in enumerate(spike_times_by_unit):
                unit_spikes = np.asarray(unit_spikes)
                n_spikes = np.sum((unit_spikes >= t_start) & (unit_spikes < t_end))
                X[t_idx, u_idx] = n_spikes / duration if duration > 0 else 0
        
        return X
    
    elif feature_type == "binned":
        # Multiple bins per unit
        sample_window = trial_windows[0]
        n_bins = int(np.ceil((sample_window[1] - sample_window[0]) / bin_size))
        X = np.zeros((n_trials, n_units * n_bins))
        
        for t_idx, (t_start, t_end) in enumerate(trial_windows):
            bin_edges = np.linspace(t_start, t_end, n_bins + 1)
            for u_idx, unit_spikes in enumerate(spike_times_by_unit):
                unit_spikes = np.asarray(unit_spikes)
                counts, _ = np.histogram(unit_spikes, bins=bin_edges)
                X[t_idx, u_idx * n_bins:(u_idx + 1) * n_bins] = counts
        
        return X
    
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


# =============================================================================
# SPECTRAL FEATURE EXTRACTION
# =============================================================================

def extract_spectral_features(
    X: np.ndarray,
    sfreq: float,
    *,
    band_edges: Optional[List[float]] = None,
    segment_seconds: float = 2.0,
    segment_overlap_seconds: float = 0.0,
    max_seconds: Optional[float] = None,
    pooling: str = "mean",
) -> Dict[str, Any]:
    """
    Extract frequency-domain power features from multi-channel time-series data.

    Splits each subject's recording into fixed-length segments, computes the
    power spectral density (PSD) via Welch's method, integrates power within
    frequency bands, and aggregates across segments.

    This is a GENERAL-PURPOSE spectral feature extractor — it works on any
    multi-channel time-series signal (EEG, EMG, LFP, accelerometer, etc.).

    Args:
        X: list of arrays or 3D array.
            - If list: each element is one subject's data, shape (n_channels, n_samples).
              Subjects may have different n_samples.
            - If 3D array: shape (n_subjects, n_channels, n_samples).
        sfreq: Sampling frequency in Hz.
        band_edges: Sorted list of frequency boundaries defining bands.
            E.g. [1, 4, 8, 13, 30] defines 4 bands: 1-4, 4-8, 8-13, 13-30 Hz.
            Default: [1, 4, 8, 13, 30].
        segment_seconds: Length of each segment in seconds for Welch PSD.
        segment_overlap_seconds: Overlap between segments in seconds.
        max_seconds: If set, only use the first max_seconds of each recording.
        pooling: How to aggregate across segments — "mean", "median", or "concat"
            (concat appends mean + std + max per band).

    Returns:
        Dict with:
            - features: np.ndarray (n_subjects, n_features).
              n_features = n_channels * n_bands (for mean/median pooling)
              or n_channels * n_bands * 3 (for concat pooling).
            - n_subjects: int
            - n_channels: int
            - n_bands: int
            - band_ranges: list of [low, high] for each band
            - feature_dim: int
    """
    import scipy.signal

    # Default frequency bands
    if band_edges is None:
        band_edges = [1.0, 4.0, 8.0, 13.0, 30.0]
    band_edges = [float(b) for b in band_edges]
    if len(band_edges) < 2:
        raise ValueError("band_edges must have at least 2 values to define 1 band")

    n_bands = len(band_edges) - 1
    band_ranges = [[band_edges[i], band_edges[i + 1]] for i in range(n_bands)]

    # Normalize input to list of 2D arrays
    if isinstance(X, np.ndarray) and X.ndim == 3:
        subjects_data = [X[i] for i in range(X.shape[0])]
    elif isinstance(X, list):
        subjects_data = X
    else:
        raise ValueError("X must be a list of 2D arrays or a 3D array")

    seg_samples = int(segment_seconds * sfreq)
    overlap_samples = int(segment_overlap_seconds * sfreq)
    step_samples = seg_samples - overlap_samples
    if step_samples <= 0:
        raise ValueError("segment_seconds must be greater than segment_overlap_seconds")

    # Welch nperseg: use segment length or 4*sfreq, whichever is smaller
    nperseg = min(seg_samples, int(4 * sfreq))
    if nperseg < 4:
        nperseg = seg_samples

    all_features = []

    for subj_data in subjects_data:
        subj_data = np.asarray(subj_data, dtype=float)
        if subj_data.ndim != 2:
            raise ValueError(f"Each subject's data must be 2D (n_channels, n_samples), got shape {subj_data.shape}")

        n_ch, n_samp = subj_data.shape

        # Optionally truncate
        if max_seconds is not None:
            max_samp = int(max_seconds * sfreq)
            subj_data = subj_data[:, :min(n_samp, max_samp)]
            n_samp = subj_data.shape[1]

        if n_samp < seg_samples:
            # If recording is shorter than one segment, use entire recording
            segments = [subj_data]
        else:
            # Extract segments
            starts = np.arange(0, n_samp - seg_samples + 1, step_samples, dtype=int)
            if len(starts) == 0:
                segments = [subj_data]
            else:
                segments = [subj_data[:, s:s + seg_samples] for s in starts]

        # Compute bandpower for each segment: (n_segments, n_ch, n_bands)
        seg_bp = np.zeros((len(segments), n_ch, n_bands), dtype=float)

        for s_idx, seg in enumerate(segments):
            # Welch PSD: vectorized across channels
            freqs, psd = scipy.signal.welch(seg, fs=sfreq, nperseg=nperseg, axis=-1)
            # psd shape: (n_ch, n_freqs)

            for b_idx, (b_lo, b_hi) in enumerate(band_ranges):
                freq_mask = (freqs >= b_lo) & (freqs <= b_hi)
                if not np.any(freq_mask):
                    continue
                # Integrate PSD over band for all channels at once
                seg_bp[s_idx, :, b_idx] = np.trapz(psd[:, freq_mask], freqs[freq_mask], axis=-1)

        # Aggregate across segments
        if pooling == "mean":
            subj_bp = seg_bp.mean(axis=0)  # (n_ch, n_bands)
            subj_feat = subj_bp.reshape(-1)
        elif pooling == "median":
            subj_bp = np.median(seg_bp, axis=0)
            subj_feat = subj_bp.reshape(-1)
        elif pooling == "concat":
            bp_mean = seg_bp.mean(axis=0).reshape(-1)
            bp_std = seg_bp.std(axis=0).reshape(-1)
            bp_max = seg_bp.max(axis=0).reshape(-1)
            subj_feat = np.concatenate([bp_mean, bp_std, bp_max])
        else:
            raise ValueError(f"Unknown pooling: {pooling}. Use 'mean', 'median', or 'concat'.")

        all_features.append(subj_feat)

    features = np.vstack(all_features)

    return {
        "features": features,
        "n_subjects": features.shape[0],
        "n_channels": n_ch,
        "n_bands": n_bands,
        "band_ranges": band_ranges,
        "feature_dim": features.shape[1],
        "pooling": pooling,
        "sfreq": sfreq,
    }


# =============================================================================
# PCA TOOLS (Already generic, just re-export with cleaner interface)
# =============================================================================

def pca_decomposition(
    X: np.ndarray,
    n_components: Optional[int] = None,
    whiten: bool = False,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    PCA dimensionality reduction.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of components (None = min(n_samples, n_features))
        whiten: Whether to whiten the components
        random_state: Random seed
    
    Returns:
        Dict with:
            - transformed: Transformed data [n_samples, n_components]
            - components: PC loadings [n_components, n_features]
            - explained_variance_ratio: Fraction of variance per component
            - cumulative_variance: Cumulative variance explained
    """
    from sklearn.decomposition import PCA
    
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    X_transformed = pca.fit_transform(X)
    
    return {
        "transformed": X_transformed,
        "components": pca.components_,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components": pca.n_components_,
    }


# =============================================================================
# PLOTTING TOOLS
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> Dict[str, Any]:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Optional list of class label names
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Dict with confusion matrix and accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    figure_path = None
    if save_path and HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if class_labels is None:
            class_labels = [str(i) for i in range(cm.shape[0])]
        
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\nAccuracy: {accuracy:.3f}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        figure_path = save_path
    
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy),
        "figure_path": figure_path,
    }


def plot_raster(
    spike_times: List[np.ndarray],
    time_window: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    title: str = "Spike Raster",
    unit_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Plot spike raster.
    
    Args:
        spike_times: List of spike time arrays, one per unit
        time_window: Optional (start, end) to limit view
        save_path: Path to save figure
        title: Plot title
        unit_labels: Optional labels for each unit
    
    Returns:
        Dict with figure path and summary
    """
    figure_path = None
    
    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, unit_spikes in enumerate(spike_times):
            unit_spikes = np.asarray(unit_spikes)
            
            if time_window:
                mask = (unit_spikes >= time_window[0]) & (unit_spikes <= time_window[1])
                unit_spikes = unit_spikes[mask]
            
            ax.vlines(unit_spikes, i + 0.5, i + 1.5, linewidth=0.5)
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unit")
        ax.set_title(title)
        
        if time_window:
            ax.set_xlim(time_window)
        
        ax.set_ylim(0.5, len(spike_times) + 0.5)
        
        if unit_labels:
            ax.set_yticks(range(1, len(spike_times) + 1))
            ax.set_yticklabels(unit_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            figure_path = save_path
        
        plt.close()
    
    return {
        "n_units": len(spike_times),
        "figure_path": figure_path,
    }


# =============================================================================
# Normalization
# =============================================================================


def normalize_array(
    X: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> dict:
    """Normalize a feature matrix.

    Args:
        X: 2-D array (n_samples, n_features)
        method: 'zscore' (default), 'minmax', or 'robust'
        axis: 0 = per-feature (column-wise), 1 = per-sample (row-wise)

    Returns:
        dict with 'X_normalized' (np.ndarray) and 'method'.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if method == "zscore":
        mu = np.nanmean(X, axis=axis, keepdims=True)
        sd = np.nanstd(X, axis=axis, keepdims=True)
        sd[sd == 0] = 1.0
        X_out = (X - mu) / sd
    elif method == "minmax":
        xmin = np.nanmin(X, axis=axis, keepdims=True)
        xmax = np.nanmax(X, axis=axis, keepdims=True)
        rng = xmax - xmin
        rng[rng == 0] = 1.0
        X_out = (X - xmin) / rng
    elif method == "robust":
        med = np.nanmedian(X, axis=axis, keepdims=True)
        q75 = np.nanpercentile(X, 75, axis=axis, keepdims=True)
        q25 = np.nanpercentile(X, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        X_out = (X - med) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'zscore', 'minmax', or 'robust'.")

    return {
        "X_normalized": X_out,
        "method": method,
        "shape": list(X_out.shape),
    }


# =============================================================================
# TOOL REGISTRY FOR GENERIC TOOLS
# =============================================================================

GENERIC_TOOLS = {
    # Decoding
    "decode_logistic_regression": decode_logistic_regression,
    "decode_svm": decode_svm,
    "decode_random_forest": decode_random_forest,
    "decode_xgboost": decode_xgboost,
    "decode_with_holdout": decode_with_holdout,

    # Foundation models
    "decode_with_ndt3": decode_with_ndt3,
    "decode_with_mtm": decode_with_mtm,
    "decode_with_poyo": decode_with_poyo,
    "decode_with_labram": decode_with_labram,
    "decode_with_reve": decode_with_reve,
    "decode_with_cbramod": decode_with_cbramod,
    "decode_with_eegpt": decode_with_eegpt,
    "decode_with_eegmamba": decode_with_eegmamba,
    "decode_with_brainomni": decode_with_brainomni,

    # Feature extraction
    "compute_firing_rates": compute_firing_rates,
    "bin_spike_times": bin_spike_times,
    "extract_trial_features": extract_trial_features,
    "extract_spectral_features": extract_spectral_features,

    # PCA
    "pca_decomposition": pca_decomposition,
    
    # Preprocessing
    "normalize_array": normalize_array,

    # Plotting
    "plot_confusion_matrix": plot_confusion_matrix,
    "plot_raster": plot_raster,
}


def get_tool(name: str):
    """Get a tool function by name."""
    if name not in GENERIC_TOOLS:
        raise KeyError(f"Unknown tool: {name}. Available: {list(GENERIC_TOOLS.keys())}")
    return GENERIC_TOOLS[name]


def list_tools() -> List[str]:
    """List all available generic tools."""
    return list(GENERIC_TOOLS.keys())


def get_tool_signature(name: str) -> str:
    """Get the signature/docstring of a tool for LLM prompts."""
    tool = get_tool(name)
    import inspect
    sig = inspect.signature(tool)
    doc = tool.__doc__ or ""
    return f"{name}{sig}\n\n{doc}"


def get_all_tool_signatures() -> str:
    """Get all tool signatures for LLM prompts."""
    lines = ["# Available Tools\n"]
    for name in GENERIC_TOOLS:
        lines.append(f"## {name}")
        lines.append(get_tool_signature(name))
        lines.append("")
    return "\n".join(lines)
