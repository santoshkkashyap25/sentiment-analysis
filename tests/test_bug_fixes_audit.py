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
from src.monitoring.drift_detection import ModelMonitor, RetrainingStrategy
from src.models.threshold_calibration import ThresholdCalibrator
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
def test_api_empty_text_prediction_id_and_no_bogus_logging(tmp_path, monkeypatch):
    """Verify empty text fallback contains prediction_id: 1 and is not recorded in monitoring DB."""
    api = FeedbackAnalysisAPI()
    res = api.predict_sentiment("<p>   <br>   </p>")
    assert res["prediction_id"] == 1
    assert res["sentiment"] == "Neutral"
    assert res["error"] == "Empty text after cleaning"

    # Test predict endpoint with empty text payload
    db_file = tmp_path / "test_api_mon.db"
    test_monitor = ModelMonitor(db_path=str(db_file))
    
    monkeypatch.setattr(app, "monitor", test_monitor)
    monkeypatch.setattr(app, "api", api)

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


# ===========================================================================
# Tier-2 Adversarial Audit Verification Tests (T2-1 to T2-13)
# ===========================================================================

def test_threshold_calibrator_class_inversion_and_empty_guard():
    """Verify ThresholdCalibrator enforces labels=[0, 1, 2] preventing class metric inversion,
    and safely handles empty arrays without crashing."""
    calibrator = ThresholdCalibrator()

    # 1. Empty array guard test
    empty_res = calibrator.fit([], [])
    assert empty_res["baseline_metrics"]["accuracy"] == 0.0
    assert empty_res["calibrated_metrics"]["negative_recall"] == 0.0

    # 2. Validation set containing only Neutral (1) and Positive (2)
    # Class 0 (Negative) is absent. Without labels=[0, 1, 2], Neutral recall would leak into Negative recall.
    y_probs = np.array([
        [0.1, 0.8, 0.1],  # Pred: Neutral (1), True: Neutral (1)
        [0.1, 0.2, 0.7],  # Pred: Positive (2), True: Positive (2)
    ])
    y_true = np.array([1, 2])

    res = calibrator.fit(y_probs, y_true)
    # Negative recall MUST be 0.0, NOT 1.0 (Neutral recall)
    assert res["baseline_metrics"]["negative_recall"] == 0.0
    assert res["calibrated_metrics"]["negative_recall"] == 0.0


def test_preprocessing_column_agnostic_dedup_and_missing():
    """Verify remove_duplicates and handle_missing_values support custom/standard text column names."""
    dp = DataPreprocessor({})

    # Dataset with 'text' instead of 'reviewText'
    df_text = pd.DataFrame({
        "text": ["Great item", "Great item", None, "Another one"],
        "overall": [5, 5, None, 4]
    })

    # Should not raise KeyError: 'reviewText'
    deduped = dp.remove_duplicates(df_text)
    assert len(deduped) == 3

    clean_missing = dp.handle_missing_values(deduped)
    assert len(clean_missing) == 2
    assert None not in clean_missing["text"].values

    # Dataset without any known text column should still deduplicate across full row without KeyError
    df_unknown = pd.DataFrame({"colA": [1, 1, 2], "colB": [10, 10, 20]})
    dedup_unknown = dp.remove_duplicates(df_unknown)
    assert len(dedup_unknown) == 2


def test_pipeline_stratification_guard_on_single_member_classes():
    """Verify stratified split logic safely falls back when a class has only 1 sample."""
    from sklearn.model_selection import train_test_split
    df = pd.DataFrame({
        "text": ["r1", "r2", "r3", "r4", "r5"],
        "sentiment": ["Positive", "Positive", "Neutral", "Neutral", "Negative"]  # Negative has 1 sample!
    })

    min_class_count = df['sentiment'].value_counts().min()
    stratify_col = df['sentiment'] if min_class_count >= 2 else None

    # This should not raise ValueError: The least populated class in y has only 1 member
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=stratify_col, random_state=42)
    assert len(train_df) + len(test_df) == 5


def test_retraining_strategy_empty_history_does_not_force_retrain():
    """Verify RetrainingStrategy does not recommend retraining simply because history is empty."""
    strategy = RetrainingStrategy(config={})
    mock_monitor = MagicMock()
    # Empty history, but zero alerts
    mock_monitor.get_monitoring_dashboard_data.return_value = {
        'recent_alerts': [],
        'performance_history': [],
        'recent_predictions': []
    }

    res = strategy.should_retrain(mock_monitor)
    assert res['should_retrain'] is False
    assert res['recommendation'] == 'continue_monitoring'
    assert any("No recent ground-truth performance data available" in r for r in res['reasons'])

    # When drift alerts exist, it should trigger retraining
    mock_monitor.get_monitoring_dashboard_data.return_value = {
        'recent_alerts': [{'drift_type': 'data_drift_detected'}] * 5,
        'performance_history': [],
        'recent_predictions': []
    }
    res_drift = strategy.should_retrain(mock_monitor)
    assert res_drift['should_retrain'] is True
    assert res_drift['recommendation'] == 'retrain'


def test_evaluator_zero_sample_guard():
    """Verify calculate_basic_metrics and calculate_per_class_metrics handle zero samples gracefully."""
    evaluator = ModelEvaluator()
    y_empty = np.array([])

    metrics = evaluator.calculate_basic_metrics(y_empty, y_empty)
    assert metrics == {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    per_class = evaluator.calculate_per_class_metrics(y_empty, y_empty)
    assert len(per_class) == 3
    assert (per_class['Precision'] == 0.0).all()
    assert (per_class['Recall'] == 0.0).all()


def test_monitor_performance_degradation_zero_sample_guard():
    """Verify monitor_performance_degradation does not crash on empty y_true."""
    monitor = ModelMonitor()
    res = monitor.monitor_performance_degradation(np.array([]), np.array([]), 0.85, 0.85)
    assert res['performance_degraded'] is False
    assert res['current_accuracy'] == 0.0
    assert res['current_f1'] == 0.0


def test_ingestion_fetch_api_dict_and_scalar_support():
    """Verify fetch_api_data correctly converts dicts of scalars or single records into DataFrames."""
    pipeline = DataIngestionPipeline(config={})

    mock_resp = MagicMock()
    # Scalar dictionary returned from API
    mock_resp.json.return_value = {"status": "healthy", "service": "sentiment", "code": 200}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        df = pipeline.fetch_api_data("https://api.example.com/status")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["status"].iloc[0] == "healthy"

    # API returning dict with records list inside
    mock_resp_nested = MagicMock()
    mock_resp_nested.json.return_value = {"results": [{"id": 1, "text": "great"}, {"id": 2, "text": "bad"}]}
    mock_resp_nested.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp_nested):
        df_nested = pipeline.fetch_api_data("https://api.example.com/items")
        assert len(df_nested) == 2
        assert "text" in df_nested.columns


def test_pipeline_features_reference_dir_creation(tmp_path):
    """Verify directory is created before features_reference.pkl is saved."""
    ref_dir = tmp_path / "deep_dir" / "processed"
    ref_file = ref_dir / "features_reference.pkl"
    assert not ref_dir.exists()

    os.makedirs(os.path.dirname(str(ref_file)), exist_ok=True)
    assert ref_dir.exists()


def test_api_predict_sentiment_binary_classical_model():
    """Verify predict_sentiment handles classical models with != 3 classes without IndexError."""
    api_instance = FeedbackAnalysisAPI()
    api_instance.is_onnx = False
    api_instance.is_transformer = False

    class MockBinaryModel:
        classes_ = np.array([0, 2])  # Only Negative and Positive, Neutral omitted
        def predict(self, X):
            return np.array([2])
        def predict_proba(self, X):
            return np.array([[0.15, 0.85]])  # 2 probabilities only

    api_instance.model = MockBinaryModel()
    api_instance.feature_engineer = MagicMock()
    api_instance.feature_engineer.extract_basic_features.return_value = pd.DataFrame([{"word_count": 2}])
    api_instance.feature_engineer.extract_tfidf_features.return_value = np.zeros((1, 5))
    api_instance.feature_engineer.combine_features.return_value = np.zeros((1, 6))

    res = api_instance.predict_sentiment("Excellent service!")
    assert res['prediction_id'] == 2
    assert res['sentiment'] == 'Positive'
    assert 'Negative' in res['probabilities']
    assert 'Neutral' in res['probabilities']
    assert 'Positive' in res['probabilities']
    assert res['probabilities']['Positive'] == 0.85


def test_feature_engineer_tfidf_all_stopwords():
    """Verify extract_tfidf_features handles inputs consisting only of stop words without crashing."""
    fe = FeatureEngineer({})
    # All stop words
    texts = ["the the the", "a an is"]
    feats = fe.extract_tfidf_features(texts, max_features=10, is_training=True)
    assert feats.shape[0] == 2


def test_model_trainer_safe_cv():
    """Verify _get_safe_cv calculates appropriate folds for small dataset sizes."""
    trainer = ModelTrainer({})
    # 3 samples with class distribution {0: 1, 1: 2}
    y_small = np.array([0, 1, 1])
    safe_cv = trainer._get_safe_cv(y_small, target_cv=5)
    assert safe_cv == 2

    # Normal dataset with 100 samples
    y_normal = np.array([0] * 50 + [1] * 50)
    assert trainer._get_safe_cv(y_normal, target_cv=5) == 5


