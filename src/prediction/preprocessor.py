"""
Prediction Preprocessor — process new FASTA files for model inference.

Reuses the existing preprocessing pipeline:
    fasta_parser.process_fasta()  → DNA FASTA → protein sequences
    feature_extractor.extract_and_aggregate_features() → 6×27 matrix

Then pairs the new isolate with every phage in the library to build
the full input tensors expected by the trained models.

USAGE:
    from src.prediction.preprocessor import PredictionPreprocessor

    pp = PredictionPreprocessor()
    isolate_features = pp.process_new_isolate(Path("upload/sample.fasta"))
    X_cnn, X_mlp, X_baseline, phage_ids = pp.prepare_prediction_input(
        isolate_features
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.fasta_parser import process_fasta
from src.preprocessing.feature_extractor import extract_and_aggregate_features
from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# Expected feature matrix shape (6 aggregation stats × 27 features)
FEATURE_SHAPE = (6, 27)

# Morphology classes (must match training order)
MORPHOLOGIES = ["Myoviridae", "Podoviridae", "Siphoviridae"]


class PredictionPreprocessor:
    """Process a clinical-isolate FASTA and pair it with the phage library."""

    def __init__(
        self,
        phage_features_dir: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        concentrations: Optional[List[float]] = None,
    ):
        self.phage_features_dir = phage_features_dir or (
            Path(config.paths["data_processed"]) / "features" / "phages"
        )
        self.metadata_path = metadata_path or (
            Path(config.paths["phage_library"]) / "phage_metadata.json"
        )
        self.concentrations = concentrations or [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        self._phage_cache: Dict[str, np.ndarray] = {}
        self._morphology_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Process new clinical isolate
    # ------------------------------------------------------------------
    def process_new_isolate(self, fasta_path: Path) -> np.ndarray:
        """
        Extract a 6×27 feature matrix from a clinical-isolate FASTA file.

        Pipeline: DNA FASTA → ORF finding → protein translation →
                  per-protein features (N, 27) → aggregate stats (6, 27).

        Args:
            fasta_path: Path to the uploaded .fasta file.

        Returns:
            np.ndarray of shape (6, 27).

        Raises:
            ValueError: If no proteins/ORFs are found.
        """
        logger.info(f"Processing new isolate: {fasta_path.name}")
        proteins = process_fasta(fasta_path)
        if not proteins:
            raise ValueError(
                f"No ORFs / proteins found in {fasta_path.name}. "
                "Ensure the file contains valid DNA sequences."
            )
        features = extract_and_aggregate_features(
            proteins, source_name=fasta_path.stem
        )
        self.validate_features(features, label=fasta_path.name)
        logger.info(
            f"Extracted features for {fasta_path.name}: shape={features.shape}, "
            f"proteins={len(proteins)}"
        )
        return features

    # ------------------------------------------------------------------
    # 2. Prepare prediction inputs
    # ------------------------------------------------------------------
    def prepare_prediction_input(
        self,
        isolate_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Pair isolate features with every phage × concentration combination.

        For each phage in the library and each concentration, creates:
          - CNN input  : (6, 27, 2) — stacked [phage, host] channels
          - MLP input  : (4,)       — one-hot morphology + log10(concentration)
          - Baseline   : (328,)     — flattened CNN + MLP

        Args:
            isolate_features: (6, 27) feature matrix of the new isolate.

        Returns:
            (X_cnn, X_mlp, X_baseline, interaction_info) where:
              X_cnn      : (N, 6, 27, 2)
              X_mlp      : (N, 4)
              X_baseline : (N, 328)
              interaction_info : list of dicts with phage_id, morphology,
                                 concentration for each row.
        """
        self._load_phage_library()

        cnn_list: List[np.ndarray] = []
        mlp_list: List[np.ndarray] = []
        info_list: List[Dict[str, Any]] = []

        for phage_id, phage_feat in self._phage_cache.items():
            morph = self._morphology_map.get(phage_id, "Siphoviridae")
            morph_vec = [1.0 if m == morph else 0.0 for m in MORPHOLOGIES]

            for conc in self.concentrations:
                combined = np.stack(
                    [phage_feat, isolate_features], axis=-1
                ).astype(np.float32)
                cnn_list.append(combined)

                conc_log = np.log10(conc + 1e-9)
                mlp_list.append(morph_vec + [conc_log])

                info_list.append({
                    "phage_id": phage_id,
                    "morphology": morph,
                    "concentration": conc,
                })

        X_cnn = np.array(cnn_list, dtype=np.float32)
        X_mlp = np.array(mlp_list, dtype=np.float32)

        # Baseline: flatten CNN + append MLP → (N, 328)
        X_flat = X_cnn.reshape(X_cnn.shape[0], -1)
        X_baseline = np.hstack([X_flat, X_mlp])

        logger.info(
            f"Prepared prediction input: {len(info_list):,} interactions "
            f"({len(self._phage_cache)} phages × "
            f"{len(self.concentrations)} concentrations)"
        )
        return X_cnn, X_mlp, X_baseline, info_list

    # ------------------------------------------------------------------
    # 3. Validation
    # ------------------------------------------------------------------
    def validate_features(
        self, features: np.ndarray, label: str = "input"
    ) -> None:
        """
        Validate that features are compatible with the trained models.

        Raises:
            ValueError: On shape mismatch, NaN, or Inf values.
        """
        if features.shape != FEATURE_SHAPE:
            raise ValueError(
                f"Feature shape mismatch for '{label}': "
                f"expected {FEATURE_SHAPE}, got {features.shape}"
            )
        if np.isnan(features).any():
            raise ValueError(f"Features contain NaN values for '{label}'")
        if np.isinf(features).any():
            raise ValueError(f"Features contain Inf values for '{label}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_phage_library(self) -> None:
        """Load all phage features and morphology map (cached)."""
        if self._phage_cache:
            return

        # Morphology map
        logger.info(f"Loading phage metadata from {self.metadata_path}")
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        self._morphology_map = {
            entry["name"]: entry["morphology"] for entry in metadata
        }

        # Feature CSVs
        csv_files = sorted(self.phage_features_dir.glob("*.csv"))
        for csv_path in csv_files:
            phage_id = csv_path.stem
            features = np.loadtxt(csv_path, delimiter=",")
            if features.shape == FEATURE_SHAPE:
                self._phage_cache[phage_id] = features

        logger.info(
            f"Loaded {len(self._phage_cache)} phage features, "
            f"{len(self._morphology_map)} morphology entries"
        )
