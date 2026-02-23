"""
Configuration Loader Module with singleton access.

This module loads `config.yaml` once at import time and exposes a singleton `config`
object for fast attribute-based access throughout the codebase.

USAGE:
    from src.utils.config_loader import config
    
    # Attribute-style access to sections
    config.models
    config.paths
    config.api
    config.logging
    config.logger  # alias to logging section

    # Dictionary access remains available
    config.get_all()
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Config object with attribute-based access to YAML contents."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config_data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as fh:
            self._config_data = yaml.safe_load(fh) or {}

        # Expose top-level keys as attributes for convenience
        for key, value in self._config_data.items():
            setattr(self, key, value)

        # Provide a convenient alias for logging configuration
        self.logger = self._config_data.get("logging", {})

    def get_all(self) -> Dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._config_data

    def reload(self) -> None:
        """Reload configuration from disk and refresh attributes."""
        self._load()

    def __repr__(self) -> str:
        keys = ", ".join(self._config_data.keys())
        return f"Config(path='{self.config_path}', sections=[{keys}])"


# Singleton instance loaded at import time
config = Config()