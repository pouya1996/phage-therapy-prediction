"""
Model Training Script

Runs the full training pipeline:
1. Load interactions_data.csv (train/test split via 'dataset' column)
2. Load 6×27 feature CSVs from data/processed/features/
3. Run 10-fold CV + final test evaluation for all models
4. Save results to doc/results/

USAGE:
    python scripts/train_models.py                # Run all models
    python scripts/train_models.py --model rf     # Run only Random Forest
    python scripts/train_models.py --model cnn    # Run only MultiviewCNN
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger
from src.models.model_trainer import DataLoader, ModelTrainer, ResultsManager

logger = setup_logger(__name__, level=config.logger.get("level"))


def main():
    parser = argparse.ArgumentParser(description="Train phage-host interaction models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run a specific model (knn, svm, rf, xgboost, adaboost, lr, cnn). "
             "If omitted, all models are run.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=config.training.get("cv_folds", 10),
        help="Number of cross-validation folds (default: 10)",
    )
    args = parser.parse_args()

    # ---- Set up paths ----
    interactions_path = Path(config.paths["interactions"])
    features_dir = Path(config.paths["data_processed"]) / "features"
    metadata_path = Path(config.paths["phage_library"]) / "phage_metadata.json"
    results_dir = Path(config.paths["results"])

    # ---- Initialise components ----
    data_loader = DataLoader(interactions_path, features_dir, metadata_path)
    results_manager = ResultsManager(results_dir)
    trainer = ModelTrainer(data_loader, results_manager)

    if args.model is None:
        # Run all models
        summary = trainer.run_all(n_splits=args.cv_folds)
        logger.info("\nFinal model comparison:")
        logger.info(f"\n{summary.to_string(index=False)}")
    elif args.model.lower() == "cnn":
        # Run CNN only
        train_df, test_df = data_loader.load()
        X_cnn_train, X_mlp_train, y_train = data_loader.prepare_cnn_features(train_df)
        X_cnn_test, X_mlp_test, y_test = data_loader.prepare_cnn_features(test_df)
        trainer.run_cnn(
            X_cnn_train, X_mlp_train, y_train,
            X_cnn_test, X_mlp_test, y_test,
            n_splits=args.cv_folds,
        )
    else:
        # Run a specific baseline model
        model_name = args.model.lower()
        train_df, test_df = data_loader.load()
        X_train, y_train = data_loader.prepare_baseline_features(train_df)
        X_test, y_test = data_loader.prepare_baseline_features(test_df)
        trainer.run_baseline(
            model_name, X_train, y_train, X_test, y_test,
            n_splits=args.cv_folds,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
