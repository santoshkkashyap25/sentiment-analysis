"""Centralized configuration management."""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Directory and file path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    raw_data: Path = None
    processed_data: Path = None
    models: Path = None
    logs: Path = None

    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = self.base_dir / "data" / "raw"
        if self.processed_data is None:
            self.processed_data = self.base_dir / "data" / "processed"
        if self.models is None:
            self.models = self.base_dir / "data" / "models"
        if self.logs is None:
            self.logs = self.base_dir / "logs"

    @property
    def best_model(self) -> Path:
        return self.models / "best_model.pkl"

    @property
    def feature_extractors(self) -> Path:
        return self.models / "feature_extractors.pkl"

    @property
    def features_tfidf(self) -> Path:
        return self.processed_data / "features_tfidf.pkl"

    @property
    def monitoring_db(self) -> Path:
        return self.base_dir / "monitoring.db"

    def ensure_dirs(self):
        for d in [self.raw_data, self.processed_data, self.models, self.logs]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model training and feature engineering configuration."""
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    max_features: int = 5000
    embedding_model: str = "all-MiniLM-L6-v2"
    resampling_strategy: str = "smote"
    tfidf_max_features: int = 500
    svd_components: int = 50
    max_rows: int = 5000


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    batch_max_size: int = 100
    batch_workers: int = 4
    model_version: str = "1.0"


@dataclass
class MonitoringConfig:
    """Monitoring and drift detection configuration."""
    drift_threshold: float = 0.05
    drift_percentage_threshold: float = 0.1
    performance_threshold: float = 0.05
    performance_alert_count: int = 3
    drift_alert_count: int = 5


@dataclass
class Config:
    """Root configuration object."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        config = cls()
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            config = config.update(data)
        return config

    def update(self, overrides: dict) -> "Config":
        for key, value in overrides.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            if hasattr(self.api, key):
                setattr(self.api, key, value)
            if hasattr(self.monitoring, key):
                setattr(self.monitoring, key, value)
        return self

    def to_dict(self) -> dict:
        return {
            "random_state": self.model.random_state,
            "test_size": self.model.test_size,
            "validation_size": self.model.validation_size,
            "max_features": self.model.max_features,
            "embedding_model": self.model.embedding_model,
            "resampling_strategy": self.model.resampling_strategy,
        }

    def get(self, key: str, default=None):
        return getattr(self.model, key, default)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return defaults."""
    config = Config()
    if config_path and os.path.exists(config_path):
        config = Config.from_file(config_path)
    return config
