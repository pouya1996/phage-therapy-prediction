"""
Baseline Machine Learning Models for Phage-Host Interaction Prediction.

Implements traditional ML classifiers with a uniform interface:
KNN, SVM, Random Forest, XGBoost, AdaBoost, Logistic Regression.

Each model:
- Accepts flattened proteomic features + morphology/concentration features
- Supports train / predict / save / load

USAGE:
    from src.models.baseline_models import create_model

    model = create_model('rf', config.models['rf'])
    model.train(X_train, y_train)
    y_pred, y_prob = model.predict(X_test)
    model.save(Path('models/rf'))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaselineModel(ABC):
    """Abstract base class for all baseline classifiers."""

    def __init__(self, model_name: str, params: Dict[str, Any]):
        self.model_name = model_name
        self.params = params
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None
        self._build()
        logger.info(f"Initialized {self.model_name} with params: {self.params}")

    # ---- public API -------------------------------------------------------
    @abstractmethod
    def _build(self) -> None:
        """Instantiate the underlying sklearn estimator."""

    def train(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """
        Fit the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,).

        Returns:
            self for chaining.
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"Training {self.model_name} on {X_scaled.shape[0]:,} samples, "
                    f"{X_scaled.shape[1]} features")
        self.model.fit(X_scaled, y)
        logger.info(f"{self.model_name} training complete")
        return self

    # Maximum absolute z-score after scaling.  Features beyond this are
    # clipped to prevent out-of-distribution inputs (e.g. zero-variance
    # features in training) from saturating LR / SVM sigmoids.
    _CLIP_VALUE: float = 10.0

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and probability estimates.

        Applies StandardScaler transform, zeros out features that had
        zero variance during training, and clips the remainder to
        ±_CLIP_VALUE standard deviations.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Tuple of (predicted_labels, predicted_probabilities).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError(f"{self.model_name} has not been trained yet")
        X_scaled = self.scaler.transform(X)

        # Zero-variance features were constant (=0 after centering) in
        # training; force them to 0 for unseen data as well.
        zero_var = self.scaler.scale_ == 1.0
        if zero_var.any():
            X_scaled[:, zero_var] = 0.0

        X_scaled = np.clip(X_scaled, -self._CLIP_VALUE, self._CLIP_VALUE)
        y_pred = self.model.predict(X_scaled)
        y_prob = (
            self.model.predict_proba(X_scaled)[:, 1]
            if hasattr(self.model, "predict_proba")
            else self.model.decision_function(X_scaled)
        )
        return y_pred, y_prob

    def save(self, directory: Path) -> None:
        """Persist model and scaler to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / f"{self.model_name}_model.joblib")
        joblib.dump(self.scaler, directory / f"{self.model_name}_scaler.joblib")
        logger.info(f"Saved {self.model_name} to {directory}")

    def load(self, directory: Path) -> "BaselineModel":
        """Load a previously saved model and scaler from *directory*."""
        self.model = joblib.load(directory / f"{self.model_name}_model.joblib")
        self.scaler = joblib.load(directory / f"{self.model_name}_scaler.joblib")
        logger.info(f"Loaded {self.model_name} from {directory}")
        return self


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------
class KNNModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("knn", params)

    def _build(self) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=self.params.get("n_neighbors", 5),
            weights=self.params.get("weights", "distance"),
        )


class SVMModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("svm", params)

    def _build(self) -> None:
        self.model = SVC(
            kernel=self.params.get("kernel", "rbf"),
            C=self.params.get("C", 1.0),
            probability=True,
            random_state=42,
        )


class RFModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("rf", params)

    def _build(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            max_depth=self.params.get("max_depth", 10),
            random_state=42,
        )


class XGBoostModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("xgboost", params)

    def _build(self) -> None:
        from xgboost import XGBClassifier  # lazy import

        self.model = XGBClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            learning_rate=self.params.get("learning_rate", 0.1),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )


class AdaBoostModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("adaboost", params)

    def _build(self) -> None:
        self.model = AdaBoostClassifier(
            n_estimators=self.params.get("n_estimators", 50),
            random_state=42,
        )


class LogisticRegressionModel(BaselineModel):
    def __init__(self, params: Dict[str, Any]):
        super().__init__("lr", params)

    def _build(self) -> None:
        self.model = LogisticRegression(
            max_iter=self.params.get("max_iter", 1000),
            random_state=42,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, type] = {
    "knn": KNNModel,
    "svm": SVMModel,
    "rf": RFModel,
    "xgboost": XGBoostModel,
    "adaboost": AdaBoostModel,
    "lr": LogisticRegressionModel,
}


def create_model(name: str, params: Dict[str, Any]) -> BaselineModel:
    """
    Factory function to create a baseline model by name.

    Args:
        name: One of 'knn', 'svm', 'rf', 'xgboost', 'adaboost', 'lr'.
        params: Hyper-parameter dict (typically from config.yaml).

    Returns:
        An initialised BaselineModel subclass.

    Raises:
        ValueError: If *name* is not recognised.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key](params)
