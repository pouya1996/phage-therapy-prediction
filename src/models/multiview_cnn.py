"""
Multiview CNN-MLP Model for Phage-Host Interaction Prediction.

Two-branch architecture:
  CNN branch   processes proteomic features  (6, 27, 2)
  MLP branch   processes morphology + concentration features  (4,)

Branches are fused via concatenation → Dense layers → sigmoid output.

USAGE:
    from src.models.multiview_cnn import MultiviewCNN

    model = MultiviewCNN(config.models['cnn'])
    model.build()
    model.train([X_cnn_train, X_mlp_train], y_train,
                [X_cnn_val, X_mlp_val], y_val)
    y_pred, y_prob = model.predict([X_cnn_test, X_mlp_test])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy TensorFlow import — keeps the module importable even without TF.
# ---------------------------------------------------------------------------
_tf = None


def _get_tf():
    """Return the tensorflow module, importing it on first call."""
    global _tf
    if _tf is None:
        import tensorflow as tf_
        _tf = tf_
    return _tf


class MultiviewCNN:
    """Two-view CNN + MLP fusion model."""

    # Default architecture constants
    CNN_INPUT_SHAPE = (6, 27, 2)   # (rows, cols, channels=[phage, host])
    MLP_INPUT_DIM = 4              # 3 one-hot morphology + 1 log-concentration

    def __init__(self, params: Dict[str, Any]):
        """
        Args:
            params: Model hyper-parameters from config.yaml['models']['cnn'].
        """
        self.params = params
        self.model = None
        self.history = None
        logger.info(f"Initialized MultiviewCNN with params: {self.params}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, mlp_input_dim: int = MLP_INPUT_DIM) -> "Model":
        """
        Construct the two-branch model.

        Args:
            mlp_input_dim: Dimension of the MLP branch input.

        Returns:
            Compiled Keras Model.
        """
        tf = _get_tf()
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import (
            Activation, BatchNormalization, Conv2D, Dense,
            Dropout, Flatten, MaxPooling2D, concatenate,
        )
        from tensorflow.keras.optimizers import Adam

        dropout_rate = self.params.get("dropout", 0.3)
        learning_rate = self.params.get("learning_rate", 1e-4)

        # ---- CNN branch (proteomic features) ----
        cnn_input = Input(shape=self.CNN_INPUT_SHAPE, name="cnn_input")
        x = Conv2D(32, (3, 3), padding="same")(cnn_input)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        cnn_output = Dense(128, activation="relu", name="cnn_embedding")(x)

        # ---- MLP branch (morphology + concentration) ----
        mlp_input = Input(shape=(mlp_input_dim,), name="mlp_input")
        y = Dense(64, activation="relu")(mlp_input)
        y = Dense(32, activation="relu")(y)
        mlp_output = Dense(16, activation="relu", name="mlp_embedding")(y)

        # ---- Fusion ----
        merged = concatenate([cnn_output, mlp_output], name="fusion_layer")
        z = Dense(32, activation="relu")(merged)
        final_output = Dense(1, activation="sigmoid", name="final_output")(z)

        self.model = Model(
            inputs=[cnn_input, mlp_input],
            outputs=final_output,
            name="multiview_cnn",
        )
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Model built and compiled")
        self.model.summary(print_fn=logger.debug)
        return self.model

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: List[np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[List[np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        Train the model.

        Args:
            X_train: [cnn_features (N,6,27,2), mlp_features (N,4)].
            y_train: Binary labels (N,).
            X_val: Optional validation inputs (same structure).
            y_val: Optional validation labels.

        Returns:
            Keras History object.
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if self.model is None:
            raise RuntimeError("Call build() before train()")

        epochs = self.params.get("epochs", 100)
        batch_size = self.params.get("batch_size", 32)
        patience = self.params.get("patience", 20)

        callbacks = []
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=patience, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=patience // 2, min_lr=1e-6
                ),
            ]
        else:
            validation_data = None

        logger.info(
            f"Training MultiviewCNN — epochs={epochs}, batch_size={batch_size}, "
            f"samples={len(y_train):,}"
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("MultiviewCNN training complete")
        return self.history

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self, X: List[np.ndarray], threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Args:
            X: [cnn_features, mlp_features].
            threshold: Classification threshold.

        Returns:
            Tuple of (predicted_labels, predicted_probabilities).
        """
        if self.model is None:
            raise RuntimeError("Model has not been built/trained yet")
        y_prob = self.model.predict(X, verbose=0).ravel()
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, directory: Path, scaler=None) -> None:
        """Save the model (and optional scaler) to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "multiview_cnn.keras"
        self.model.save(model_path)
        logger.info(f"Saved MultiviewCNN to {model_path}")
        if scaler is not None:
            import joblib
            scaler_path = directory / "cnn_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved CNN scaler to {scaler_path}")

    def load(self, directory: Path):
        """Load a saved model (and scaler if present) from *directory*."""
        model_path = directory / "multiview_cnn.keras"
        tf = _get_tf()
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded MultiviewCNN from {model_path}")

        scaler_path = directory / "cnn_scaler.joblib"
        scaler = None
        if scaler_path.exists():
            import joblib
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded CNN scaler from {scaler_path}")
        return self, scaler
