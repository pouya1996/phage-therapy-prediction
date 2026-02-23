"""
Cross-Validation Utilities for Phage-Host Interaction Prediction.

Provides K-Fold CV with per-fold and aggregated metric reporting.

USAGE:
    from src.evaluation.cross_validation import cross_validate_baseline, cross_validate_cnn

    cv_results = cross_validate_baseline(model_factory, X, y, n_splits=10)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation.metrics import calculate_metrics
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def cross_validate_baseline(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run K-Fold cross-validation for a baseline (sklearn-style) model.

    Args:
        model_factory: Callable that returns a fresh BaselineModel instance.
        X: Feature matrix (n_samples, n_features).
        y: Binary labels (n_samples,).
        n_splits: Number of CV folds.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with per-fold metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        logger.info(f"CV Fold {fold_idx}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.train(X_train, y_train)
        y_pred, y_prob = model.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred, y_prob)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        logger.info(
            f"  Fold {fold_idx} — Acc: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    # Move fold column to front
    cols = ["fold"] + [c for c in results_df.columns if c != "fold"]
    results_df = results_df[cols]

    logger.info("CV complete — mean metrics:")
    for col in ["accuracy", "precision", "recall", "f1", "mcc", "auc", "specificity"]:
        logger.info(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    return results_df


def cross_validate_cnn(
    model_factory: Callable[[], Any],
    X_cnn: np.ndarray,
    X_mlp: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run K-Fold cross-validation for the MultiviewCNN model.

    Args:
        model_factory: Callable that returns a fresh MultiviewCNN (already built).
        X_cnn: CNN branch input (n_samples, 6, 27, 2).
        X_mlp: MLP branch input (n_samples, 4).
        y: Binary labels (n_samples,).
        n_splits: Number of CV folds.
        random_state: Random seed.

    Returns:
        DataFrame with per-fold metrics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cnn), start=1):
        logger.info(f"CNN CV Fold {fold_idx}/{n_splits}")

        X_cnn_train, X_cnn_val = X_cnn[train_idx], X_cnn[val_idx]
        X_mlp_train, X_mlp_val = X_mlp[train_idx], X_mlp[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.train(
            [X_cnn_train, X_mlp_train], y_train,
            [X_cnn_val, X_mlp_val], y_val,
        )
        y_pred, y_prob = model.predict([X_cnn_val, X_mlp_val])

        metrics = calculate_metrics(y_val, y_pred, y_prob)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        logger.info(
            f"  Fold {fold_idx} — Acc: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}"
        )

    results_df = pd.DataFrame(fold_results)
    cols = ["fold"] + [c for c in results_df.columns if c != "fold"]
    results_df = results_df[cols]

    logger.info("CNN CV complete — mean metrics:")
    for col in ["accuracy", "precision", "recall", "f1", "mcc", "auc", "specificity"]:
        logger.info(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")

    return results_df
