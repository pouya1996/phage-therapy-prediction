"""
Prediction Service — orchestrates preprocessor → predictor → ranker.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.prediction.preprocessor import PredictionPreprocessor
from src.prediction.predictor import PhagePredictor
from src.prediction.ranker import PhageRanker
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class PredictionService:
    """High-level orchestration consumed by the API layer."""

    def __init__(self) -> None:
        self.preprocessor = PredictionPreprocessor()
        self.predictor = PhagePredictor()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load all models into memory (call once at startup)."""
        if self._loaded:
            return
        logger.info("Loading prediction service…")
        self.predictor.load_all_models()
        self._loaded = True
        logger.info("Prediction service ready")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def available_models(self) -> List[str]:
        return self.predictor.loaded_model_names

    # ------------------------------------------------------------------
    # Main prediction entry point
    # ------------------------------------------------------------------
    def predict(
        self,
        fasta_path: Path,
        top_k: int = 10,
        view: str = "phage",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run full prediction pipeline on a FASTA file.

        Runs ALL loaded models independently and returns per-model
        rankings so the user can compare.

        Args:
            fasta_path: Path to the clinical-isolate FASTA.
            top_k: Number of top phages to return per model.
            view: ``"phage"`` (all phage × concentration combos) or
                  ``"interaction"`` (unique phages, best interaction).
            threshold: Feasibility cut-off for CI score.

        Returns:
            Dict with keys: models (per-model results), metadata.
        """
        if not self._loaded:
            self.load()

        t0 = time.time()

        # 1. Preprocess
        isolate_features = self.preprocessor.process_new_isolate(fasta_path)
        X_cnn, X_mlp, X_baseline, info = (
            self.preprocessor.prepare_prediction_input(isolate_features)
        )

        # Save extracted features alongside the uploaded file
        features_path = fasta_path.with_name(
            fasta_path.stem + "_features.csv"
        )
        np.savetxt(features_path, isolate_features, delimiter=",")
        logger.info(f"Saved features → {features_path}")

        # 2. Predict with every model
        all_predictions = self.predictor.predict_all_models(
            X_cnn, X_mlp, X_baseline
        )

        # 3. Rank per model
        per_model = {}
        for model_name, probabilities in all_predictions.items():
            ranker = PhageRanker(info, probabilities, model_name=model_name)
            rankings = ranker.to_dict(view=view, top_k=top_k)
            report = ranker.generate_recommendation_report(
                top_k=top_k, threshold=threshold
            )
            per_model[model_name] = {
                "rankings": rankings,
                "report": report,
            }

        elapsed = time.time() - t0
        logger.info(f"Prediction completed in {elapsed:.2f}s")

        return {
            "models": per_model,
            "metadata": {
                "model_names": list(all_predictions.keys()),
                "view": view,
                "top_k": top_k,
                "threshold": threshold,
                "total_interactions": len(info),
                "elapsed_seconds": round(elapsed, 3),
                "isolate_file": fasta_path.name,
            },
        }
