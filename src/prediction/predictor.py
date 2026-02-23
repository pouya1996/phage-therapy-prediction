"""
Phage Predictor — load trained models and make predictions.

Supports:
  - Single-model prediction (any baseline or CNN)
  - All-model prediction (runs every loaded model independently)

USAGE:
    from src.prediction.predictor import PhagePredictor

    predictor = PhagePredictor()
    predictor.load_all_models()
    results = predictor.predict_all_models(X_cnn, X_mlp, X_baseline)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.models.baseline_models import create_model
from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

BASELINE_MODELS = ["knn", "svm", "rf", "xgboost", "adaboost", "lr"]


class PhagePredictor:
    """Load trained models and produce predictions."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
    ):
        """
        Args:
            models_dir: Root directory containing model subdirs.
        """
        self.models_dir = models_dir or Path(config.paths["models"])

        self._baseline_models: Dict[str, Any] = {}
        self._cnn_model = None
        self._cnn_scaler = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_all_models(self) -> None:
        """Load every available model from disk."""
        self._load_baselines()
        self._load_cnn()
        logger.info(
            f"Loaded {len(self._baseline_models)} baseline model(s) "
            f"+ CNN={'yes' if self._cnn_model else 'no'}"
        )

    def _load_baselines(self) -> None:
        """Load all baseline models that have saved artefacts."""
        for name in BASELINE_MODELS:
            model_dir = self.models_dir / name
            model_file = model_dir / f"{name}_model.joblib"
            if model_file.exists():
                try:
                    model_params = config.models.get(name, {})
                    model = create_model(name, model_params)
                    model.load(model_dir)
                    self._baseline_models[name] = model
                    logger.info(f"Loaded baseline model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

    def _load_cnn(self) -> None:
        """Load the MultiviewCNN model and its scaler."""
        cnn_dir = self.models_dir / "multiview_cnn"
        keras_file = cnn_dir / "multiview_cnn.keras"
        scaler_file = cnn_dir / "cnn_scaler.joblib"

        if not keras_file.exists():
            logger.info("No saved CNN model found — skipping")
            return

        try:
            from src.models.multiview_cnn import MultiviewCNN

            cnn_params = config.models.get("cnn", {})
            self._cnn_model = MultiviewCNN(cnn_params)
            result = self._cnn_model.load(cnn_dir)
            # load() returns (self, scaler)
            self._cnn_model, self._cnn_scaler = result
            if self._cnn_scaler is None and scaler_file.exists():
                self._cnn_scaler = joblib.load(scaler_file)
            logger.info("Loaded MultiviewCNN model + scaler")
        except Exception as e:
            logger.warning(f"Failed to load CNN: {e}")
            self._cnn_model = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_single_model(
        self,
        model_name: str,
        X_cnn: np.ndarray,
        X_mlp: np.ndarray,
        X_baseline: np.ndarray,
    ) -> np.ndarray:
        """
        Predict with a single model.

        Args:
            model_name: Model identifier ('rf', 'multiview_cnn', etc.).
            X_cnn: (N, 6, 27, 2) — used only for CNN.
            X_mlp: (N, 4) — used only for CNN.
            X_baseline: (N, 328) — used for baseline models.

        Returns:
            Predicted probabilities (N,).
        """
        if model_name == "multiview_cnn":
            return self._predict_cnn(X_cnn, X_mlp)

        if model_name not in self._baseline_models:
            raise ValueError(
                f"Model '{model_name}' not loaded. "
                f"Available: {list(self._baseline_models.keys())}"
            )
        _, y_prob = self._baseline_models[model_name].predict(X_baseline)
        return y_prob

    def predict_all_models(
        self,
        X_cnn: np.ndarray,
        X_mlp: np.ndarray,
        X_baseline: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run prediction with every loaded model independently.

        Returns:
            Dict mapping model_name → probabilities (N,).
        """
        results: Dict[str, np.ndarray] = {}

        for name in self._baseline_models:
            try:
                results[name] = self.predict_single_model(
                    name, X_cnn, X_mlp, X_baseline
                )
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")

        if self._cnn_model is not None:
            try:
                results["multiview_cnn"] = self.predict_single_model(
                    "multiview_cnn", X_cnn, X_mlp, X_baseline
                )
            except Exception as e:
                logger.warning(f"Prediction failed for multiview_cnn: {e}")

        if not results:
            raise RuntimeError("No models produced predictions")

        logger.info(f"Predictions from {len(results)} model(s)")
        return results

    # ------------------------------------------------------------------
    # CNN helper
    # ------------------------------------------------------------------
    # Same clip threshold used by BaselineModel.predict().
    _CLIP_VALUE: float = 10.0

    def _predict_cnn(
        self, X_cnn: np.ndarray, X_mlp: np.ndarray
    ) -> np.ndarray:
        """Scale CNN features, zero out zero-var dims, clip, and predict."""
        if self._cnn_model is None:
            raise RuntimeError("CNN model not loaded")

        if self._cnn_scaler is not None:
            shape = X_cnn.shape
            X_cnn_2d = X_cnn.reshape(-1, shape[-1])
            X_cnn_2d = self._cnn_scaler.transform(X_cnn_2d)

            zero_var = self._cnn_scaler.scale_ == 1.0
            if zero_var.any():
                X_cnn_2d[:, zero_var] = 0.0

            X_cnn = np.clip(
                X_cnn_2d, -self._CLIP_VALUE, self._CLIP_VALUE
            ).reshape(shape)

        _, y_prob = self._cnn_model.predict([X_cnn, X_mlp])
        return y_prob

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def loaded_model_names(self) -> List[str]:
        """Return names of all loaded models."""
        names = list(self._baseline_models.keys())
        if self._cnn_model is not None:
            names.append("multiview_cnn")
        return names
