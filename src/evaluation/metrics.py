"""
Evaluation Metrics for Phage-Host Interaction Prediction.

Computes a comprehensive suite of binary classification metrics:
Accuracy, Precision, Recall, F1, MCC, AUC, Specificity, Confusion Matrix.

USAGE:
    from src.evaluation.metrics import calculate_metrics

    results = calculate_metrics(y_true, y_pred, y_prob)
    print(results)  # dict with all metrics
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, Union[float, int]]:
    """
    Calculate the full suite of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities (required for AUC).

    Returns:
        Dictionary with keys:
          accuracy, precision, recall, f1, mcc,
          auc, specificity, tn, fp, fn, tp.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = float("nan")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            logger.warning("AUC could not be computed (single class present)")

    metrics: Dict[str, Union[float, int]] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": auc,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    logger.debug(
        f"Metrics — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
        f"AUC: {metrics['auc']:.4f}, MCC: {metrics['mcc']:.4f}"
    )
    return metrics
