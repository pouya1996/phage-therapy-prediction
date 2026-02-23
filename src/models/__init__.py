"""Models module — baseline classifiers and MultiviewCNN."""

from src.models.baseline_models import BaselineModel, create_model
from src.models.model_trainer import DataLoader, ModelTrainer, ResultsManager

# Lazy import — requires tensorflow
def __getattr__(name: str):
    if name == "MultiviewCNN":
        from src.models.multiview_cnn import MultiviewCNN
        return MultiviewCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
