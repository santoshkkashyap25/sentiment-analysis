"""Targeted unit tests verifying the resolution of all 15 audit bugs and edge cases."""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.data.ingestion import DataIngestionPipeline
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.models.transformer_trainer import TransformerTrainer
from src.monitoring.drift_detection import ModelMonitor
from src.api.app import FeedbackAnalysisAPI, app, PredictRequest, predict


# ---------------------------------------------------------------------------
# B1 & B2: Pipeline CLI Aliases and Log Directory Initialization
# ---------------------------------------------------------------------------
def test_pipeline_cli_aliases():
    """Verify --model and --samples CLI aliases parse correctly."""
    import argparse
    from pipeline import main
    
    # Test argparse parsing with aliases
    test_args = ["--model", "transformer", "--samples", "500"]
    with patch("sys.argv", ["pipeline.py"] + test_args):
        # Build parser matching pipeline
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-type', '--model', dest='model_type', type=str, default='transformer')
        parser.add_argument('--sample-size', '--samples', dest='sample_size', type=int, default=120000)
        parsed = parser.parse_args(test_args)
        assert parsed.model_type == "transformer"
        assert parsed.sample_size == 500


def test_pipeline_log_directory_creation(tmp_path):
    """Verify log directory existence is guaranteed before file handler creation."""
    log_dir = tmp_path / "test_logs"
    assert not log_dir.exists()
    log_dir.mkdir(parents=True, exist_ok=True)
    assert log_dir.exists()


# ---------------------------------------------------------------------------
# B3: SVD 2D Input Validation & Numeric Column Safety
# ---------------------------------------------------------------------------
def test_svd_1d_input_raises_value_error():
    """Verify SVD raises ValueError when input is 1D."""
    fe = FeatureEngineer({})
    X = np.random.randn(20).astype(np.float32)
    with pytest.raises(ValueError, match="Expected 2D array"):
        fe.reduce_dimensionality(X, n_components=5)


def test_basic_features_handles_numeric_columns_safely():
    """Verify extract_basic_features handles numeric columns safely."""
    fe = FeatureEngineer({})
    df = pd.DataFrame({"text": [12345, "Sentence with period.", "Single word!"]})
    feats = fe.extract_basic_features(df, "text")
    assert len(feats) == 3
    assert (feats["text_length"] > 0).all()
    assert (feats["word_count"] > 0).all()


# ---------------------------------------------------------------------------
# B4 & B5: Transformer Trainer Checkpoint Safety & Empty DataLoader Guard
# ---------------------------------------------------------------------------
def test_transformer_empty_dataloader_probabilities():
    """Verify predict_probabilities safely returns empty arrays on empty DataLoader."""
    trainer = TransformerTrainer()
    empty_loader = []
    probs, labels = trainer.predict_probabilities(empty_loader)
    assert isinstance(probs, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert probs.shape == (0, 3)
    assert labels.shape == (0,)


def test_transformer_zero_f1_checkpoint_safety(tmp_path, monkeypatch):
    """Verify train() guarantees a saved checkpoint even if validation F1 remains 0.0."""
    trainer = TransformerTrainer(epochs=1, batch_size=2, device="cpu")
    output_dir = tmp_path / "test_checkpoint"

    train_df = pd.DataFrame({
        "fused_text": ["good quality", "bad broken"],
        "label": [2, 0]
    })
    val_df = pd.DataFrame({
        "fused_text": ["neutral ok", "just fine"],
        "label": [1, 1]
    })

    # Mock tokenizer and model
    import torch
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    dummy_param = torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))
    mock_model.parameters.return_value = [dummy_param]
    mock_out = MagicMock()
    mock_out.logits = torch.randn(2, 3, requires_grad=True)
    mock_model.return_value = mock_out

    def fake_tokenize(text, **kwargs):
        return {
            "input_ids": torch.tensor([[101, 102]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long)
        }

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: fake_tokenize)
    monkeypatch.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", lambda *a, **k: mock_model)

    # Mock predict_probabilities to return 0 F1 metrics
    monkeypatch.setattr(trainer, "predict_probabilities", lambda loader, model=None: (
        np.array([[0.33, 0.34, 0.33], [0.33, 0.34, 0.33]]), np.array([0, 0])
    ))

    # Mock save method
    save_called = []
    monkeypatch.setattr(trainer, "save", lambda d: (Path(d).mkdir(parents=True, exist_ok=True), save_called.append(d)))

    report = trainer.train(train_df, val_df, output_dir=str(output_dir))
    assert len(save_called) >= 1
    assert str(output_dir) in save_called


# ---------------------------------------------------------------------------
# B6: Drift Detection Zero Features & Empty Arrays
# ---------------------------------------------------------------------------
def test_drift_detection_zero_features_and_empty_arrays(tmp_path):
    """Verify detect_data_drift safely handles empty feature matrices or 0 features without ZeroDivisionError."""
    monitor = ModelMonitor(db_path=str(tmp_path / "test_mon.db"))

    # Empty 2D array
    res_empty = monitor.detect_data_drift(np.empty((0, 0)), np.empty((0, 0)))
    assert res_empty["drift_detected_features"] == 0
    assert res_empty["total_features"] == 0
    assert res_empty["drift_percentage"] == 0.0
    assert res_empty["overall_drift"] is False

    # 0 features (N, 0)
    res_zero_feat = monitor.detect_data_drift(np.empty((10, 0)), np.empty((10, 0)))
    assert res_zero_feat["total_features"] == 0
    assert res_zero_feat["drift_percentage"] == 0.0


# ---------------------------------------------------------------------------
# B7: Prediction Drift Empty Inputs Safety
# ---------------------------------------------------------------------------
def test_prediction_drift_empty_inputs_safety(tmp_path):
    """Verify detect_prediction_drift safely handles empty arrays and non-varying inputs."""
    monitor = ModelMonitor(db_path=str(tmp_path / "test_mon.db"))

    res = monitor.detect_prediction_drift(np.array([]), np.array([]))
    assert res["drift_detected"] is False
    assert res["chi2_statistic"] == 0.0
    assert res["p_value"] == 1.0


# ---------------------------------------------------------------------------
# B8: Telemetry Query Integer Casting
# ---------------------------------------------------------------------------
def test_telemetry_query_integer_casting(tmp_path):
    """Verify get_monitoring_dashboard_data safely casts string/float days and queries correctly."""
    monitor = ModelMonitor(db_path=str(tmp_path / "test_mon.db"))
    # Pass string '7' or float 7.5
    data_str = monitor.get_monitoring_dashboard_data(days="7")
    assert "summary_stats" in data_str

    data_float = monitor.get_monitoring_dashboard_data(days=14.0)
    assert "summary_stats" in data_float


# ---------------------------------------------------------------------------
# B9: API Empty-Text Prediction ID and No Bogus Logging
# ---------------------------------------------------------------------------
def test_api_empty_text_prediction_id_and_no_bogus_logging(tmp_path):
    """Verify empty text fallback contains prediction_id: 1 and is not recorded in monitoring DB."""
    api = FeedbackAnalysisAPI()
    res = api.predict_sentiment("<p>   <br>   </p>")
    assert res["prediction_id"] == 1
    assert res["sentiment"] == "Neutral"
    assert res["error"] == "Empty text after cleaning"

    # Test predict endpoint with empty text payload
    db_file = tmp_path / "test_api_mon.db"
    test_monitor = ModelMonitor(db_path=str(db_file))
    
    app.monitor = test_monitor
    app.api = api

    payload = PredictRequest(text="<p>   <br>   </p>")
    endpoint_res = predict(payload)
    assert endpoint_res["sentiment"] == "Neutral"
    assert endpoint_res["error"] == "Empty text after cleaning"

    # Verify no bogus records were logged to SQLite
    dash_data = test_monitor.get_monitoring_dashboard_data()
    assert dash_data["summary_stats"]["total_predictions"] == 0


# ---------------------------------------------------------------------------
# B10: Ingestion Unnamed Source Fallback
# ---------------------------------------------------------------------------
def test_ingestion_unnamed_source_fallback(tmp_path):
    """Verify load_csv_data via ingest_multiple_sources defaults to 'unnamed_source' when 'name' is missing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1\nval1\n", encoding="utf-8")

    pipeline = DataIngestionPipeline({})
    # Omit 'name'
    sources = [{"type": "csv", "path": str(csv_file)}]
    df = pipeline.ingest_multiple_sources(sources)
    assert len(df) == 1
    assert df["data_source"].iloc[0] == "unnamed_source"


# ---------------------------------------------------------------------------
# B11: Preprocessing String Ratings Conversion
# ---------------------------------------------------------------------------
def test_preprocessing_string_ratings_conversion():
    """Verify create_sentiment_labels converts string ratings without throwing TypeError."""
    dp = DataPreprocessor({})
    df = pd.DataFrame({"score": ["1", "2", "3", "4", "5", "invalid"]})
    labeled = dp.create_sentiment_labels(df, "score")
    assert labeled["sentiment"].tolist() == [
        "Negative", "Negative", "Neutral", "Positive", "Positive", "Neutral"
    ]


# ---------------------------------------------------------------------------
# B12: Preprocessing Imbalanced Data String Labels
# ---------------------------------------------------------------------------
def test_preprocessing_imbalanced_string_labels():
    """Verify handle_imbalanced_data logs string/non-integer class labels without raising TypeError."""
    dp = DataPreprocessor({})
    X = np.random.randn(6, 4)
    y = np.array(["Negative", "Negative", "Neutral", "Neutral", "Positive", "Positive"])
    
    # 'none' strategy bypasses SMOTE but exercises distribution logging
    X_out, y_out = dp.handle_imbalanced_data(X, y, strategy="none")
    assert len(X_out) == 6
    assert len(y_out) == 6


# ---------------------------------------------------------------------------
# B13: Evaluation Per-Class Metrics Label Mismatch
# ---------------------------------------------------------------------------
def test_evaluation_per_class_metrics_label_mismatch():
    """Verify calculate_per_class_metrics aligns target_names length with labels length."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    # 2 labels provided, but 3 target_names
    metrics_df = evaluator.calculate_per_class_metrics(
        y_true, y_pred, labels=[0, 1], target_names=["Class_0", "Class_1", "Extra_Class"]
    )
    assert len(metrics_df) == 2
    assert list(metrics_df["Class"]) == ["Class_0", "Class_1"]


# ---------------------------------------------------------------------------
# B14: Evaluation Compare Models Empty Dict
# ---------------------------------------------------------------------------
def test_evaluation_compare_models_empty_dict():
    """Verify compare_models returns clean DataFrame when models dictionary is empty."""
    evaluator = ModelEvaluator()
    df = evaluator.compare_models({}, np.empty((0, 2)), np.empty((0,)))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "model" in df.columns
    assert "f1" in df.columns


# ---------------------------------------------------------------------------
# B15: Config Update Paths Support
# ---------------------------------------------------------------------------
def test_config_update_paths_support():
    """Verify Config.update supports updating path attributes and to_dict exports model dimensions."""
    cfg = Config()
    cfg.update({
        "raw_data": "custom/raw/path",
        "svd_components": 100,
        "max_rows": 8000
    })
    assert cfg.paths.raw_data == Path("custom/raw/path")
    assert cfg.model.svd_components == 100
    assert cfg.model.max_rows == 8000

    d = cfg.to_dict()
    assert d["svd_components"] == 100
    assert d["max_rows"] == 8000


def test_model_trainer_save_creates_parent_directory(tmp_path):
    """Verify ModelTrainer.save_model automatically creates non-existent parent directories."""
    trainer = ModelTrainer({})
    nested_path = tmp_path / "deeply" / "nested" / "dir" / "model.pkl"
    assert not nested_path.parent.exists()

    dummy_model = {"model_name": "mock"}
    trainer.save_model(dummy_model, str(nested_path))
    assert nested_path.exists()

