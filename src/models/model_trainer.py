"""
Model Training Pipeline for Phage-Host Interaction Prediction.

Handles end-to-end training workflow:
1. Load interactions_data.csv (split by 'dataset' column)
2. Load 6×27 feature CSVs from data/processed/features/{phages,clinical_isolates}/
3. Look up phage morphology from data/phage_library/phage_metadata.json
4. Prepare CNN input (6,27,2) and MLP input (4,) for MultiviewCNN
5. Prepare flattened features for baseline models
6. Run 10-fold CV + final test evaluation
7. Save results to doc/results/

USAGE:
    python scripts/train_models.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.cross_validation import cross_validate_baseline, cross_validate_cnn
from src.evaluation.metrics import calculate_metrics
from src.models.baseline_models import create_model
from src.models.multiview_cnn import MultiviewCNN
from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# =========================================================================
# Data loading
# =========================================================================
class DataLoader:
    """
    Load and prepare data for model training.

    Reads interactions_data.csv, loads pre-computed 6×27 feature matrices
    from individual CSV files, and constructs phage morphology mappings.
    """

    MORPHOLOGIES = ["Myoviridae", "Podoviridae", "Siphoviridae"]
    FEATURE_SHAPE = (6, 27)

    def __init__(
        self,
        interactions_path: Path,
        features_dir: Path,
        metadata_path: Path,
    ):
        self.interactions_path = interactions_path
        self.phage_features_dir = features_dir / "phages"
        self.host_features_dir = features_dir / "clinical_isolates"
        self.metadata_path = metadata_path

        self._phage_cache: Dict[str, np.ndarray] = {}
        self._host_cache: Dict[str, np.ndarray] = {}
        self._morphology_map: Dict[str, str] = {}

    # ---- public ---------------------------------------------------------
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load interactions and split into train / test DataFrames.

        Returns:
            (train_df, test_df) with columns:
            phage, host, class, dataset, morphology, concentration.
        """
        logger.info(f"Loading interactions from {self.interactions_path}")
        df = pd.read_csv(self.interactions_path)
        logger.info(f"Total interactions: {len(df):,}")

        train_df = df[df["dataset"] == "train"].reset_index(drop=True)
        test_df = df[df["dataset"] == "test"].reset_index(drop=True)

        logger.info(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
        return train_df, test_df

    def load_morphology_map(self) -> Dict[str, str]:
        """Load phage → morphology mapping from JSON metadata."""
        logger.info(f"Loading phage metadata from {self.metadata_path}")
        with open(self.metadata_path) as f:
            metadata: List[Dict[str, str]] = json.load(f)
        self._morphology_map = {
            entry["name"]: entry["morphology"] for entry in metadata
        }
        logger.info(f"Loaded morphology for {len(self._morphology_map)} phages")
        return self._morphology_map

    def load_feature(self, name: str, entity: str) -> np.ndarray:
        """
        Load a single 6×27 feature matrix from CSV.

        Args:
            name: Accession ID.
            entity: 'phage' or 'host'.

        Returns:
            Feature matrix of shape (6, 27).
        """
        cache = self._phage_cache if entity == "phage" else self._host_cache
        if name in cache:
            return cache[name]

        base_dir = self.phage_features_dir if entity == "phage" else self.host_features_dir
        csv_path = base_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature file not found: {csv_path}")

        features = np.loadtxt(csv_path, delimiter=",")
        if features.shape != self.FEATURE_SHAPE:
            raise ValueError(
                f"Expected shape {self.FEATURE_SHAPE}, got {features.shape} "
                f"for {csv_path}"
            )
        cache[name] = features
        return features

    # ---- feature preparation --------------------------------------------
    def prepare_cnn_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build CNN + MLP inputs from a DataFrame of interactions.

        Args:
            df: DataFrame with columns phage, host, class, morphology, concentration.

        Returns:
            (X_cnn, X_mlp, y) where:
              X_cnn: (N, 6, 27, 2) – stacked phage/host features.
              X_mlp: (N, 4) – one-hot morphology + log concentration.
              y:     (N,) – binary labels.
        """
        if not self._morphology_map:
            self.load_morphology_map()

        cnn_list: List[np.ndarray] = []
        mlp_list: List[np.ndarray] = []
        labels: List[int] = []
        skipped = 0

        for _, row in df.iterrows():
            try:
                phage_feat = self.load_feature(row["phage"], "phage")
                host_feat = self.load_feature(row["host"], "host")
            except FileNotFoundError:
                skipped += 1
                continue

            # CNN input: stack phage and host as channels → (6, 27, 2)
            combined = np.stack([phage_feat, host_feat], axis=-1).astype(np.float32)
            cnn_list.append(combined)

            # MLP input: one-hot morphology + log10(concentration)
            morph = row["morphology"]
            morph_vec = [1.0 if m == morph else 0.0 for m in self.MORPHOLOGIES]
            conc_log = np.log10(row["concentration"] + 1e-9)
            mlp_list.append(morph_vec + [conc_log])

            labels.append(int(row["class"]))

        if skipped:
            logger.warning(f"Skipped {skipped} interactions (missing feature files)")

        X_cnn = np.array(cnn_list, dtype=np.float32)
        X_mlp = np.array(mlp_list, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        logger.info(
            f"Prepared CNN features: X_cnn={X_cnn.shape}, X_mlp={X_mlp.shape}, "
            f"y={y.shape} (pos={int(y.sum()):,}, neg={int((1-y).sum()):,})"
        )
        return X_cnn, X_mlp, y

    def prepare_baseline_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build flattened feature vectors for baseline models.

        Flattens proteomic features and appends morphology + concentration.

        Args:
            df: DataFrame with interaction data.

        Returns:
            (X, y) where X has shape (N, 6*27*2 + 4) = (N, 328).
        """
        X_cnn, X_mlp, y = self.prepare_cnn_features(df)
        # Flatten CNN features: (N, 6, 27, 2) → (N, 324)
        X_flat = X_cnn.reshape(X_cnn.shape[0], -1)
        # Concatenate with MLP features: (N, 324+4) = (N, 328)
        X = np.hstack([X_flat, X_mlp])
        logger.info(f"Prepared baseline features: X={X.shape}")
        return X, y


# =========================================================================
# Results
# =========================================================================
class ResultsManager:
    """Save training and evaluation results to doc/results/."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directory: {self.results_dir}")

    def save_cv_results(self, model_name: str, cv_df: pd.DataFrame) -> Path:
        """Save per-fold CV results."""
        path = self.results_dir / f"{model_name}_cv_results.csv"
        cv_df.to_csv(path, index=False, float_format="%.6f")
        logger.info(f"Saved CV results to {path}")
        return path

    def save_test_results(self, model_name: str, metrics: Dict[str, Any]) -> Path:
        """Save test-set evaluation metrics."""
        path = self.results_dir / f"{model_name}_test_results.csv"
        pd.DataFrame([metrics]).to_csv(path, index=False, float_format="%.6f")
        logger.info(f"Saved test results to {path}")
        return path

    def save_summary(self, all_results: List[Dict[str, Any]]) -> Path:
        """Save a summary table comparing all models."""
        path = self.results_dir / "model_comparison.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(path, index=False, float_format="%.6f")
        logger.info(f"Saved model comparison to {path}")
        return path


# =========================================================================
# Trainer
# =========================================================================
class ModelTrainer:
    """
    Orchestrates the full training pipeline.

    Steps:
      1. Load data via DataLoader
      2. For each baseline model: 10-fold CV → train on full train → evaluate on test
      3. For MultiviewCNN: 10-fold CV → train on full train → evaluate on test
      4. Save all results
    """

    BASELINE_MODELS = ["knn", "svm", "rf", "xgboost", "adaboost", "lr"]

    def __init__(self, data_loader: DataLoader, results_manager: ResultsManager):
        self.data_loader = data_loader
        self.results_manager = results_manager

    def run_baseline(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_splits: int = 10,
    ) -> Dict[str, Any]:
        """
        Train and evaluate a single baseline model.

        Pipeline:
          1. 10-fold CV on training data (evaluation only)
          2. Train on full training set → evaluate on test set
          3. Retrain on ALL data (train+test) → save production model

        Args:
            model_name: Model identifier (e.g. 'rf').
            X_train, y_train: Training data.
            X_test, y_test: Held-out test data.
            n_splits: Number of CV folds.

        Returns:
            Test-set metrics dictionary.
        """
        logger.info(f"{'='*80}")
        logger.info(f"Running baseline: {model_name.upper()}")
        logger.info(f"{'='*80}")

        model_params = config.models.get(model_name, {})

        # ---- 1. Cross-validation (evaluation only) ----
        def factory():
            return create_model(model_name, model_params)

        cv_df = cross_validate_baseline(factory, X_train, y_train, n_splits=n_splits)
        self.results_manager.save_cv_results(model_name, cv_df)

        # ---- 2. Train on full training set → test evaluation ----
        logger.info(f"Training {model_name} on full training set for test evaluation")
        eval_model = create_model(model_name, model_params)
        eval_model.train(X_train, y_train)

        y_pred, y_prob = eval_model.predict(X_test)
        test_metrics = calculate_metrics(y_test, y_pred, y_prob)
        test_metrics["model"] = model_name

        # ---- 3. Retrain on ALL data (train+test) → production model ----
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        logger.info(
            f"Retraining {model_name} on ALL data "
            f"({len(y_all):,} samples) for production"
        )
        prod_model = create_model(model_name, model_params)

        t0 = time.time()
        prod_model.train(X_all, y_all)
        training_time = time.time() - t0

        test_metrics["training_time_sec"] = round(training_time, 2)
        self.results_manager.save_test_results(model_name, test_metrics)

        # Save production model
        models_dir = Path(config.paths["models"]) / model_name
        prod_model.save(models_dir)

        logger.info(
            f"{model_name} test — Acc: {test_metrics['accuracy']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}, "
            f"Train time (prod): {training_time:.2f}s"
        )
        return test_metrics

    def run_cnn(
        self,
        X_cnn_train: np.ndarray,
        X_mlp_train: np.ndarray,
        y_train: np.ndarray,
        X_cnn_test: np.ndarray,
        X_mlp_test: np.ndarray,
        y_test: np.ndarray,
        n_splits: int = 10,
    ) -> Dict[str, Any]:
        """
        Train and evaluate the MultiviewCNN model.

        Pipeline:
          1. 10-fold CV on training data (evaluation only)
          2. Train on full training set → evaluate on test set
          3. Retrain on ALL data (train+test) → save production model

        Args:
            X_cnn_train, X_mlp_train, y_train: Training data.
            X_cnn_test, X_mlp_test, y_test: Test data.
            n_splits: Number of CV folds.

        Returns:
            Test-set metrics dictionary.
        """
        logger.info(f"{'='*80}")
        logger.info("Running MultiviewCNN")
        logger.info(f"{'='*80}")

        cnn_params = config.models.get("cnn", {})

        # ---- Normalize CNN features (fit on train only) ----
        scaler = StandardScaler()
        shape = X_cnn_train.shape
        X_cnn_train_2d = X_cnn_train.reshape(-1, shape[-1])
        X_cnn_train_scaled = scaler.fit_transform(X_cnn_train_2d).reshape(shape)

        shape_test = X_cnn_test.shape
        X_cnn_test_2d = X_cnn_test.reshape(-1, shape_test[-1])
        X_cnn_test_scaled = scaler.transform(X_cnn_test_2d).reshape(shape_test)

        logger.info("Applied StandardScaler to CNN features (fit on train)")

        # ---- 1. Cross-validation (evaluation only) ----
        def factory():
            m = MultiviewCNN(cnn_params)
            m.build(mlp_input_dim=X_mlp_train.shape[1])
            return m

        cv_df = cross_validate_cnn(
            factory, X_cnn_train_scaled, X_mlp_train, y_train, n_splits=n_splits
        )
        self.results_manager.save_cv_results("multiview_cnn", cv_df)

        # ---- 2. Train on full training set → test evaluation ----
        logger.info("Training MultiviewCNN on full training set for test evaluation")
        eval_model = MultiviewCNN(cnn_params)
        eval_model.build(mlp_input_dim=X_mlp_train.shape[1])

        n_val = int(len(y_train) * 0.1)
        indices = np.random.RandomState(42).permutation(len(y_train))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        eval_model.train(
            [X_cnn_train_scaled[train_idx], X_mlp_train[train_idx]],
            y_train[train_idx],
            [X_cnn_train_scaled[val_idx], X_mlp_train[val_idx]],
            y_train[val_idx],
        )

        y_pred, y_prob = eval_model.predict([X_cnn_test_scaled, X_mlp_test])
        test_metrics = calculate_metrics(y_test, y_pred, y_prob)
        test_metrics["model"] = "multiview_cnn"

        # ---- 3. Retrain on ALL data (train+test) → production model ----
        X_cnn_all = np.concatenate([X_cnn_train, X_cnn_test], axis=0)
        X_mlp_all = np.concatenate([X_mlp_train, X_mlp_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        # Re-fit scaler on ALL data for the production model
        prod_scaler = StandardScaler()
        shape_all = X_cnn_all.shape
        X_cnn_all_scaled = prod_scaler.fit_transform(
            X_cnn_all.reshape(-1, shape_all[-1])
        ).reshape(shape_all)

        logger.info(
            f"Retraining MultiviewCNN on ALL data "
            f"({len(y_all):,} samples) for production"
        )
        prod_model = MultiviewCNN(cnn_params)
        prod_model.build(mlp_input_dim=X_mlp_all.shape[1])

        t0 = time.time()
        prod_model.train(
            [X_cnn_all_scaled, X_mlp_all],
            y_all,
        )
        training_time = time.time() - t0

        test_metrics["training_time_sec"] = round(training_time, 2)
        self.results_manager.save_test_results("multiview_cnn", test_metrics)

        # Save production model + scaler
        models_dir = Path(config.paths["models"]) / "multiview_cnn"
        prod_model.save(models_dir, scaler=prod_scaler)

        logger.info(
            f"MultiviewCNN test — Acc: {test_metrics['accuracy']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}, "
            f"Train time (prod): {training_time:.2f}s"
        )
        return test_metrics

    def run_all(self, n_splits: int = 10) -> pd.DataFrame:
        """
        Run the complete training pipeline for all models.

        Args:
            n_splits: Number of CV folds.

        Returns:
            Summary DataFrame comparing all models.
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 80)

        # ---- Load data ----
        train_df, test_df = self.data_loader.load()

        # ---- Prepare baseline features ----
        logger.info("Preparing baseline features...")
        X_train_bl, y_train_bl = self.data_loader.prepare_baseline_features(train_df)
        X_test_bl, y_test_bl = self.data_loader.prepare_baseline_features(test_df)

        # ---- Prepare CNN features ----
        logger.info("Preparing CNN features...")
        X_cnn_train, X_mlp_train, y_train_cnn = self.data_loader.prepare_cnn_features(train_df)
        X_cnn_test, X_mlp_test, y_test_cnn = self.data_loader.prepare_cnn_features(test_df)

        # ---- Run all models ----
        all_results: List[Dict[str, Any]] = []

        # Baseline models
        for model_name in self.BASELINE_MODELS:
            try:
                metrics = self.run_baseline(
                    model_name, X_train_bl, y_train_bl, X_test_bl, y_test_bl,
                    n_splits=n_splits,
                )
                all_results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to run {model_name}: {e}")

        # MultiviewCNN
        try:
            cnn_metrics = self.run_cnn(
                X_cnn_train, X_mlp_train, y_train_cnn,
                X_cnn_test, X_mlp_test, y_test_cnn,
                n_splits=n_splits,
            )
            all_results.append(cnn_metrics)
        except Exception as e:
            logger.error(f"Failed to run MultiviewCNN: {e}")

        # ---- Save comparison summary ----
        summary_path = self.results_manager.save_summary(all_results)

        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(f"Results saved to {self.results_manager.results_dir}")
        logger.info("=" * 80)

        return pd.DataFrame(all_results)
