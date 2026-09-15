import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
from src.models.threshold_calibration import ThresholdCalibrator


def test_threshold_calibrator_fit_and_predict():
    """Test that threshold calibration fits and improves negative recall."""
    calibrator = ThresholdCalibrator(target_negative_recall=0.80, min_negative_precision=0.60)

    # Simulated probability distribution for 3 classes [Neg, Neu, Pos]
    # Class 0: Negative, Class 1: Neutral, Class 2: Positive
    np.random.seed(42)
    n_samples = 300

    # True labels: 100 per class
    y_true = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # Simulated noisy probabilities
    probs = np.zeros((n_samples, 3))
    # Negative samples: slight tilt towards negative, but some uncertainty
    probs[:100, 0] = np.random.uniform(0.35, 0.65, 100)
    probs[:100, 1] = np.random.uniform(0.15, 0.35, 100)
    probs[:100, 2] = 1.0 - probs[:100, :2].sum(axis=1)

    # Neutral samples
    probs[100:200, 1] = np.random.uniform(0.40, 0.70, 100)
    probs[100:200, 0] = np.random.uniform(0.10, 0.30, 100)
    probs[100:200, 2] = 1.0 - probs[100:200, :2].sum(axis=1)

    # Positive samples
    probs[200:, 2] = np.random.uniform(0.45, 0.80, 100)
    probs[200:, 0] = np.random.uniform(0.05, 0.25, 100)
    probs[200:, 1] = 1.0 - probs[200:, :2].sum(axis=1)

    # Normalize to valid probabilities
    probs = np.clip(probs, 0.01, 0.98)
    probs = probs / probs.sum(axis=1, keepdims=True)

    result = calibrator.fit(probs, y_true)

    assert "thresholds" in result
    assert "negative_threshold" in result["thresholds"]
    assert 0.20 <= result["thresholds"]["negative_threshold"] <= 0.50

    # Test batch prediction
    preds = calibrator.predict(probs)
    assert len(preds) == n_samples
    assert set(preds).issubset({0, 1, 2})

    # Test single prediction
    sentiment, conf = calibrator.predict_single({"Negative": 0.38, "Neutral": 0.32, "Positive": 0.30})
    assert sentiment in ["Negative", "Neutral", "Positive"]
    assert 0.0 <= conf <= 1.0


def test_threshold_calibrator_save_and_load(tmp_path):
    """Test saving and loading calibrator configuration."""
    calibrator = ThresholdCalibrator()
    calibrator.thresholds = {"negative_threshold": 0.38}
    calibrator.baseline_metrics = {"accuracy": 0.85}
    calibrator.calibrated_metrics = {"accuracy": 0.86}

    save_file = str(tmp_path / "calibration.json")
    calibrator.save(save_file)

    loaded = ThresholdCalibrator.load(save_file)
    assert loaded.thresholds["negative_threshold"] == 0.38
    assert loaded.baseline_metrics["accuracy"] == 0.85
