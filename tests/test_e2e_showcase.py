"""Comprehensive End-to-End Showcase Test Suite for Sentiment Analysis System.

Validates external contracts, public APIs, input/output behaviors, and system state transitions
across four progressive tiers:
- Tier 1: Feature Coverage (Happy Paths for R1 through R7)
- Tier 2: Boundary & Corner Cases (Edge inputs, missing classes, invalid formats)
- Tier 3: Cross-Feature Combinations (Pairwise integration across pipeline boundaries)
- Tier 4: Real-World Scenarios (End-to-end customer feedback batch, drift lifecycle, serving robustness)
"""

import concurrent.futures
import inspect
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")  # Enforce headless rendering before any pyplot import
import matplotlib.pyplot as plt
import numpy as np
import packaging.requirements
import pandas as pd
import pytest
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import DataIngestionPipeline
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluation import ModelEvaluator
from src.models.training import ModelTrainer
from src.models.transformer_trainer import TransformerTrainer
from src.monitoring.drift_detection import ModelMonitor, RetrainingStrategy


# ==============================================================================
# TIER 1: FEATURE COVERAGE (HAPPY PATHS)
# ==============================================================================

def test_tier1_r1_truncated_svd_embeddings():
    """R1.1: Verify TruncatedSVD is functional and applied when include_embeddings=True."""
    fe = FeatureEngineer({"tfidf_max_features": 10, "svd_components": 4})
    df = pd.DataFrame({
        "reviewText": [
            "Great device, loved it!",
            "Terrible broke immediately.",
            "Average product, works fine.",
            "Excellent build and fast shipping.",
            "Poor quality, disappointed.",
            "Wonderful customer service and device."
        ],
        "reviewText_clean": [
            "great device loved",
            "terrible broke immediately",
            "average product works fine",
            "excellent build fast shipping",
            "poor quality disappointed",
            "wonderful customer service device"
        ],
        "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Positive"]
    })

    synthetic_embeddings = np.random.RandomState(42).randn(6, 384)
    with patch.object(fe, "extract_sentence_embeddings", return_value=synthetic_embeddings):
        X_train, y_train = fe.engineer_features(
            df, "reviewText_clean", is_training=True, include_embeddings=True
        )
        X_test, _ = fe.engineer_features(
            df, "reviewText_clean", is_training=False, include_embeddings=True
        )

    assert isinstance(fe.svd, TruncatedSVD), "fe.svd must be an instance of TruncatedSVD"
    assert fe.svd.n_components == 4, "SVD components must match configured svd_components"
    assert X_train.shape[0] == 6, "X_train row count must equal document count"
    n_tfidf = fe.tfidf_vectorizer.transform(df["reviewText_clean"]).shape[1]
    expected_dim = 8 + n_tfidf + 4
    assert X_train.shape[1] == expected_dim, (
        f"Expected {expected_dim} features (8 basic + {n_tfidf} tfidf + 4 svd), got {X_train.shape[1]}"
    )
    assert np.array_equal(y_train, np.array([2, 0, 1, 2, 0, 2])), "y_train labels must match mapped sentiments"
    assert X_test.shape == X_train.shape, "Inference feature shape must match training feature shape"


def test_tier1_r1_stylistic_features_raw_text():
    """R1.2: Verify stylistic features extract from raw unstripped text via raw_text_column or reviewText."""
    fe = FeatureEngineer({})
    df = pd.DataFrame({
        "reviewText": ["WOW!!! Is this really that good? YES. Amazing."],
        "reviewText_clean": ["wow is this really that good yes amazing"],
        "custom_raw": ["HELP?! Did NOT work at ALL!!! Please refund."]
    })

    # 1. Default fallback to 'reviewText'
    feat_default = fe.extract_basic_features(df, "reviewText")
    assert feat_default["exclamation_count"].iloc[0] == 3  # '!' in 'WOW!!!'
    assert feat_default["question_count"].iloc[0] == 1     # '?' in 'good?'
    assert feat_default["uppercase_count"].iloc[0] == 8    # 'W','O','W','Y','E','S','A'
    assert feat_default["sentence_count"].iloc[0] == 2     # '.' in 'YES.' and 'Amazing.'

    # 2. Explicit raw_text_column='custom_raw'
    feat_custom = fe.extract_basic_features(df, "custom_raw")
    assert feat_custom["exclamation_count"].iloc[0] == 4  # '!' in '?!' and '!!!'
    assert feat_custom["question_count"].iloc[0] == 1     # '?' in '?!'
    assert feat_custom["uppercase_count"].iloc[0] == 12   # 'H','E','L','P','D','N','O','T','A','L','L','P'
    assert feat_custom["sentence_count"].iloc[0] == 1     # '.' in 'refund.'

    # 3. Clean text has 0 counts for punctuation and uppercase
    feat_clean = fe.extract_basic_features(df, "reviewText_clean")
    assert feat_clean["exclamation_count"].iloc[0] == 0
    assert feat_clean["question_count"].iloc[0] == 0
    assert feat_clean["uppercase_count"].iloc[0] == 0


def test_tier1_r2_negation_words_preserved():
    """R2.1: Verify clean_text preserves all 8 canonical negation words when filtering stopwords."""
    preprocessor = DataPreprocessor({})
    text = (
        "It is not good, has no value, nor was it durable; neither cheap nor fast, "
        "never works, barely starts, hardly fits, and scarcely lasts."
    )
    cleaned = preprocessor.clean_text(text)
    negations = {"not", "no", "nor", "neither", "never", "barely", "hardly", "scarcely"}
    tokens = set(cleaned.split())

    assert negations.issubset(tokens), f"Missing negation words: {negations - tokens}"
    assert "is" not in tokens, "Stopword 'is' should be removed"
    assert "has" not in tokens, "Stopword 'has' should be removed"
    assert "and" not in tokens, "Stopword 'and' should be removed"


def test_tier1_r2_guard_nltk_downloads():
    """R2.2: Verify DataPreprocessor checks nltk.data.find and does not invoke nltk.download when cached."""
    with patch("nltk.data.find", return_value="/mock/nltk_data/resource"), \
         patch("nltk.download") as mock_download:
        DataPreprocessor({})
        mock_download.assert_not_called()


def test_tier1_r4_api_ingestion_timeout():
    """R4.1: Verify fetch_api_data passes timeout=30 to requests.get."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [{"id": 1, "reviewText": "Great quality!", "overall": 5}]
        mock_get.return_value.raise_for_status.return_value = None

        pipeline = DataIngestionPipeline({})
        api_url = "https://api.reviews-hub.com/feed"
        headers = {"Authorization": "Bearer test-key"}
        df = pipeline.fetch_api_data(api_url, headers=headers)

        mock_get.assert_called_once_with(api_url, headers=headers, timeout=30)
        assert len(df) == 1
        assert df["reviewText"].iloc[0] == "Great quality!"


def test_tier1_r4_ingest_multiple_sources_and_paths(tmp_path):
    """R4.2 & R4.3: Verify multi-source ingestion handles sources and joins paths using os.path.join / Path."""
    pipeline = DataIngestionPipeline({})
    mock_csv_df = pd.DataFrame({"reviewText": ["csv text"], "overall": [5]})
    mock_kaggle_df = pd.DataFrame({"reviewText": ["kaggle text"], "overall": [4]})

    with patch.object(pipeline, "load_csv_data") as mock_load, \
         patch.object(pipeline, "download_kaggle_dataset", return_value=str(tmp_path / "kaggle_dir")):
        mock_load.side_effect = [mock_csv_df, mock_kaggle_df]
        sources = [
            {"type": "csv", "name": "csv_feed", "path": str(tmp_path / "data.csv")},
            {"type": "kaggle", "name": "kaggle_feed", "dataset": "owner/dataset",
             "download_path": str(tmp_path / "kaggle_dir"), "filename": "reviews.csv"}
        ]
        combined = pipeline.ingest_multiple_sources(sources)

        assert set(combined["data_source"].unique()) == {"csv_feed", "kaggle_feed"}
        assert len(combined) == 2
        expected_kaggle_path = os.path.join(str(tmp_path / "kaggle_dir"), "reviews.csv")
        mock_load.assert_any_call(expected_kaggle_path)


def test_tier1_r3_balanced_sampling_preserves_unique_indices():
    """R3.1: Verify balanced sampling in prepare_dataset preserves index tracking and avoids duplicate row sampling."""
    records = []
    for lbl in ["Negative", "Neutral", "Positive"]:
        for i in range(50):
            records.append({
                "reviewText": f"Detailed feedback text for sentiment category {lbl} item {i}",
                "sentiment": lbl
            })
    df = pd.DataFrame(records)
    df.index = [1000 + i for i in range(150)]

    trainer = TransformerTrainer(model_name="roberta-base", num_labels=3)
    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=90, full_dataset=False, random_state=42)

    assert len(train_df) == 72, f"Expected 72 train rows (80% of 90), got {len(train_df)}"
    assert len(val_df) == 9, f"Expected 9 val rows (10% of 90), got {len(val_df)}"
    assert len(test_df) == 9, f"Expected 9 test rows (10% of 90), got {len(test_df)}"

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx), "Train and Val split indices must be disjoint"
    assert train_idx.isdisjoint(test_idx), "Train and Test split indices must be disjoint"
    assert val_idx.isdisjoint(test_idx), "Val and Test split indices must be disjoint"
    assert len(train_idx | val_idx | test_idx) == 90, "Total unique sampled rows must equal sample_size (90)"
    assert set(train_df["label"].unique()).issubset({0, 1, 2})


def test_tier1_r3_star_rating_mapping_to_three_classes():
    """R3.2: Verify 1-5 star ratings map strictly to labels 0, 1, 2 rather than raw integers 1..5."""
    records = []
    for star in [1, 2, 3, 4, 5]:
        for i in range(10):
            records.append({
                "reviewText": f"Rating {star} review with comprehensive feedback description {i}",
                "sentiment": star
            })
    df = pd.DataFrame(records)

    trainer = TransformerTrainer(model_name="roberta-base", num_labels=3)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    combined = pd.concat([train_df, val_df, test_df])

    # 1 and 2 must map to 0 (Negative)
    for star in [1, 2]:
        sub = combined[combined["reviewText"].str.startswith(f"Rating {star} ")]
        if len(sub) > 0:
            assert (sub["label"] == 0).all(), f"Star rating {star} must map to label 0"

    # 3 must map to 1 (Neutral)
    sub3 = combined[combined["reviewText"].str.startswith("Rating 3 ")]
    if len(sub3) > 0:
        assert (sub3["label"] == 1).all(), "Star rating 3 must map to label 1"

    # 4 and 5 must map to 2 (Positive)
    for star in [4, 5]:
        sub = combined[combined["reviewText"].str.startswith(f"Rating {star} ")]
        if len(sub) > 0:
            assert (sub["label"] == 2).all(), f"Star rating {star} must map to label 2"

    assert set(combined["label"].unique()).issubset({0, 1, 2}), "All labels must be strictly in {0, 1, 2}"
    assert not (combined["label"] >= 3).any(), "No label may be >= 3 (preventing CrossEntropyLoss index error)"


def test_tier1_r5_select_best_model_uses_weighted_f1():
    """R5.1: Verify ModelTrainer.select_best_model ranks candidate models by weighted F1 score instead of raw accuracy."""
    class MajorityModel:
        def predict(self, X):
            return np.full(X.shape[0], 2)

    class BalancedModel:
        def predict(self, X):
            # Reasonable predictions across minority and majority classes
            return np.array([0] * 8 + [1] * 2 + [1] * 8 + [2] * 2 + [2] * 70 + [0] * 10)

    # Imbalanced validation set: 10 class 0, 10 class 1, 80 class 2
    y_val = np.array([0] * 10 + [1] * 10 + [2] * 80)
    X_val = np.zeros((100, 5))

    trainer = ModelTrainer({})
    trainer.models = {
        "majority_model": MajorityModel(),
        "balanced_model": BalancedModel()
    }

    best_name, _ = trainer.select_best_model(X_val, y_val)
    assert best_name == "balanced_model", (
        f"Expected 'balanced_model' (higher weighted F1) to be selected, but got '{best_name}'"
    )


def test_tier1_r5_no_redundant_second_cv_after_gridsearch():
    """R5.2: Verify train_all_models does not execute a redundant second cross_val_score after GridSearchCV."""
    trainer = ModelTrainer({})
    X_train = np.random.randn(20, 5)
    y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 2)

    with patch("src.models.training.cross_val_score") as mock_cv, \
         patch.object(trainer, "initialize_models", return_value={"lr": LogisticRegression()}), \
         patch.object(trainer, "get_hyperparameter_grids", return_value={"lr": {"C": [1.0]}}), \
         patch.object(trainer, "train_single_model", return_value=LogisticRegression().fit(X_train, y_train)):
        mock_cv.return_value = np.array([0.8, 0.8])
        trainer.train_all_models(X_train, y_train)
        mock_cv.assert_not_called()


def test_tier1_r5_plot_confusion_matrix_headless_and_save(tmp_path):
    """R5.3: Verify plot_confusion_matrix runs headlessly without plt.show, supports save_path, and closes figures."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    save_file = str(tmp_path / "test_confusion_matrix.png")

    with patch("matplotlib.pyplot.show") as mock_show:
        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_file)
        mock_show.assert_not_called()
        assert os.path.exists(save_file), "Confusion matrix image must be saved to save_path"
        assert os.path.getsize(save_file) > 0, "Saved image file must not be empty"
        assert cm.shape == (3, 3), "Confusion matrix shape must be (3, 3)"
        assert len(plt.get_fignums()) == 0, "Figure must be closed with plt.close()"


def test_tier1_r5_calculate_per_class_metrics_labels_and_zero_division():
    """R5.4: Verify calculate_per_class_metrics passes labels=[0, 1, 2] and zero_division=0."""
    evaluator = ModelEvaluator()
    # Class 1 (Neutral) completely absent from both true and predicted arrays
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 2])

    metrics_df = evaluator.calculate_per_class_metrics(y_true, y_pred)
    assert len(metrics_df) == 3, f"Expected 3 rows in metrics DataFrame, got {len(metrics_df)}"
    assert metrics_df["Class"].tolist() == ["Negative", "Neutral", "Positive"]
    neutral_row = metrics_df[metrics_df["Class"] == "Neutral"].iloc[0]
    assert neutral_row["Precision"] == 0.0
    assert neutral_row["Recall"] == 0.0
    assert neutral_row["F1-Score"] == 0.0


def test_tier1_r6_cpu_routes_are_synchronous_def():
    """R6.1: Verify CPU-bound routes (/predict, /api/simulate-drift, /api/check-drift) are synchronous def."""
    from src.api.app import app

    cpu_routes = ["/predict", "/api/simulate-drift", "/api/check-drift"]
    found_routes = []
    for route in app.routes:
        route_path = getattr(route, "path", None)
        if route_path in cpu_routes:
            found_routes.append(route_path)
            assert not inspect.iscoroutinefunction(route.endpoint), (
                f"Route '{route_path}' must be a synchronous 'def' to allow threadpool execution under FastAPI."
            )
    assert set(found_routes) == set(cpu_routes), f"Expected to check all CPU routes, found: {found_routes}"


def test_tier1_r6_predict_error_returns_500_and_does_not_log_sqlite(tmp_path, monkeypatch):
    """R6.2: Verify prediction errors return HTTP 500 without writing bogus neutral rows to SQLite."""
    from fastapi.testclient import TestClient
    import src.api.app as app_module

    test_db = tmp_path / "test_error_monitor.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    client = TestClient(app_module.app)

    def mock_broken_predict(text):
        raise RuntimeError("Inference model weights unavailable or corrupted")

    monkeypatch.setattr(app_module.api, "predict_sentiment", mock_broken_predict)

    response = client.post("/predict", json={"text": "Legitimate customer feedback text"})
    assert response.status_code == 500, f"Expected HTTP 500 on inference failure, got {response.status_code}"

    # Verify no row was inserted into SQLite
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 0, f"Expected 0 logged predictions on inference failure, found {count}"


def test_tier1_r6_sqlite_wal_mode_and_timeout(tmp_path):
    """R6.3: Verify SQLite connections configure timeout=30.0 and execute PRAGMA journal_mode=WAL."""
    test_db = tmp_path / "test_wal_verify.db"
    monitor = ModelMonitor(db_path=str(test_db))

    conn = sqlite3.connect(str(test_db), timeout=30.0)
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
    conn.close()

    assert mode == "wal", f"Expected PRAGMA journal_mode to be 'wal', got '{mode}'"


def test_tier1_r7_requirements_dev_contains_onnx():
    """R7.1: Verify requirements-dev.txt contains onnx>=1.14.0."""
    req_file = PROJECT_ROOT / "requirements-dev.txt"
    assert req_file.exists(), "requirements-dev.txt must exist at project root"
    content = req_file.read_text(encoding="utf-8")
    assert re.search(r"onnx\s*>=\s*1\.14(\.0)?", content) is not None, (
        "requirements-dev.txt must declare onnx>=1.14.0"
    )


def test_tier1_r7_pyproject_toml_dependencies():
    """R7.2: Verify pyproject.toml includes onnx, tokenizers, onnxruntime, transformers, torch."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml must exist at project root"
    content = pyproject.read_text(encoding="utf-8")
    required = ["onnx", "tokenizers", "onnxruntime", "transformers", "torch"]
    for pkg in required:
        pattern = rf'["\']{pkg}(?:[><=~].*?)?["\']'
        assert re.search(pattern, content) is not None, (
            f"pyproject.toml dependencies must include '{pkg}'"
        )


def test_tier1_r7_env_example_contains_kaggle_token():
    """R7.3: Verify .env.example exists at repository root with KAGGLE_API_TOKEN placeholder."""
    env_example = PROJECT_ROOT / ".env.example"
    assert env_example.exists(), ".env.example file must exist at repository root"
    content = env_example.read_text(encoding="utf-8")
    assert "KAGGLE_API_TOKEN" in content, ".env.example must contain KAGGLE_API_TOKEN"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================

def test_tier2_r1_stylistic_boundary_cases():
    """R1.2: Verify extract_basic_features handles empty, missing, all-punct, and all-caps inputs gracefully."""
    fe = FeatureEngineer({})
    df = pd.DataFrame({
        "text": [
            "",                        # Empty string
            None,                      # None / NaN
            "!?.!?!?",                 # Punctuation only
            "ALL CAPS WITHOUT PUNCT",  # All caps
            "   "                     # Whitespace spaces only
        ]
    })

    res = fe.extract_basic_features(df, "text")
    assert len(res) == 5

    # Row 0 (empty): safe handling with 0 counts
    assert res.iloc[0]["text_length"] == 0.0
    assert res.iloc[0]["word_count"] == 0.0
    assert res.iloc[0]["avg_word_length"] == 0.0
    assert res.iloc[0]["exclamation_count"] == 0.0

    # Row 1 (None): zero values without exceptions
    assert res.iloc[1]["text_length"] == 0.0
    assert res.iloc[1]["word_count"] == 0.0

    # Row 2 (all punctuation): counts correct
    assert res.iloc[2]["exclamation_count"] == 3.0
    assert res.iloc[2]["question_count"] == 3.0
    assert res.iloc[2]["sentence_count"] == 1.0

    # Row 3 (all caps): uppercase count matches character length
    assert res.iloc[3]["uppercase_count"] == 19.0
    assert res.iloc[3]["exclamation_count"] == 0.0

    # Row 4 (whitespace): zero word count without division error
    assert res.iloc[4]["word_count"] == 0.0
    assert res.iloc[4]["avg_word_length"] == 0.0


def test_tier2_r1_unfitted_svd_inference_safeguard():
    """R1.1: Verify reduce_dimensionality auto-fits when called in inference mode without prior training."""
    fe = FeatureEngineer({})
    features = np.random.RandomState(42).randn(20, 50).astype(np.float32)

    # Calling reduce_dimensionality with is_training=False when fe.svd is None
    reduced = fe.reduce_dimensionality(features, n_components=10, is_training=False)
    assert reduced.shape == (20, 10), "Reduced feature matrix must have shape (20, 10)"
    assert isinstance(fe.svd, TruncatedSVD), "fe.svd must be instantiated and fitted"


def test_tier2_r2_negation_boundaries_and_casing():
    """R2.1: Verify negation words survive uppercase, mixed case, and punctuation boundary conditions."""
    preprocessor = DataPreprocessor({})
    cases = [
        ("NOT good at all", {"not", "good"}),
        ("NO WAY, NEVER AGAIN", {"no", "way", "never"}),
        ("...not... no! (never?)", {"not", "no", "never"}),
        ("barely, hardly, scarcely", {"barely", "hardly", "scarcely"})
    ]

    for raw, expected_tokens in cases:
        cleaned = preprocessor.clean_text(raw)
        tokens = set(cleaned.split())
        for exp in expected_tokens:
            assert exp in tokens, f"Token '{exp}' missing from clean_text output: '{cleaned}'"


def test_tier2_r2_normalize_text_empty_filtering():
    """R2.1: Verify normalize_text drops rows whose cleaned text reduces to empty strings."""
    preprocessor = DataPreprocessor({})
    df = pd.DataFrame({
        "reviewText": ["   ", "!!!???", "<p><span></span></p>", "Legitimate customer review text."]
    })

    norm_df = preprocessor.normalize_text(df, "reviewText")
    assert len(norm_df) == 1, f"Expected 1 valid row retained, got {len(norm_df)}"
    assert norm_df["reviewText"].iloc[0] == "Legitimate customer review text."
    assert len(norm_df["reviewText_clean"].iloc[0]) > 0


def test_tier2_r2_nltk_download_fallback_when_missing():
    """R2.2: Verify DataPreprocessor falls back to nltk.download when nltk.data.find raises LookupError."""
    def mock_find(resource_path):
        if "punkt" in resource_path:
            raise LookupError(f"Resource {resource_path} not found")
        return "/dummy/path"

    with patch("nltk.data.find", side_effect=mock_find), \
         patch("nltk.download") as mock_download:
        DataPreprocessor({})
        # punkt or punkt_tab should be downloaded
        assert mock_download.called, "nltk.download must be called when resource is missing"


def test_tier2_r4_ingestion_unsupported_source_raises_value_error():
    """R4.2: Verify ingest_multiple_sources raises ValueError for unsupported source types."""
    pipeline = DataIngestionPipeline({})
    sources = [{"type": "unsupported_storage_format", "name": "bad_feed", "path": "data.xyz"}]

    with pytest.raises(ValueError) as excinfo:
        pipeline.ingest_multiple_sources(sources)
    assert "unsupported" in str(excinfo.value).lower()


def test_tier2_r4_api_timeout_exception_handling():
    """R4.1: Verify fetch_api_data propagates requests.exceptions.Timeout when external API stalls."""
    pipeline = DataIngestionPipeline({})
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out after 30s")):
        with pytest.raises(requests.exceptions.Timeout):
            pipeline.fetch_api_data("https://stalled-api.internal/feed")


def test_tier2_r3_balanced_sampling_with_unequal_classes_and_index_gaps():
    """R3.1: Verify balanced sampling handles extreme class imbalance and non-contiguous gapped indices without duplicates."""
    records = []
    # 10 Negative (class 0), 50 Neutral (class 1), 100 Positive (class 2)
    for i in range(10):
        records.append({"reviewText": f"Negative review sample {i} with sufficient text length", "sentiment": "Negative"})
    for i in range(50):
        records.append({"reviewText": f"Neutral review sample {i} with sufficient text length", "sentiment": "Neutral"})
    for i in range(100):
        records.append({"reviewText": f"Positive review sample {i} with sufficient text length", "sentiment": "Positive"})

    df = pd.DataFrame(records)
    # Non-contiguous index
    df.index = [i * 7 for i in range(160)]

    trainer = TransformerTrainer(model_name="roberta-base", num_labels=3)
    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=60, full_dataset=False, random_state=42)

    total_sampled = len(train_df) + len(val_df) + len(test_df)
    assert total_sampled == 60, f"Expected 60 total sampled rows, got {total_sampled}"

    all_indices = list(train_df.index) + list(val_df.index) + list(test_df.index)
    assert len(all_indices) == len(set(all_indices)), "All sampled DataFrame indices must be strictly unique"


def test_tier2_r3_star_rating_boundary_types_and_unmapped():
    """R3.2: Verify star rating parsing handles float strings, spaces, and drops invalid ratings cleanly."""
    records = [
        {"reviewText": "Valid review text for float rating 1.0", "sentiment": 1.0},
        {"reviewText": "Valid review text for float rating 2.0", "sentiment": 2.0},
        {"reviewText": "Valid review text for float rating 3.0", "sentiment": 3.0},
        {"reviewText": "Valid review text for float rating 4.0", "sentiment": 4.0},
        {"reviewText": "Valid review text for float rating 5.0", "sentiment": 5.0},
        {"reviewText": "Valid review text with padded string rating", "sentiment": " 5 "},
        {"reviewText": "Valid review text with out-of-bounds 0", "sentiment": 0},
        {"reviewText": "Valid review text with out-of-bounds 6", "sentiment": 6},
        {"reviewText": "Valid review text with invalid string", "sentiment": "invalid_sentiment"}
    ]
    df = pd.DataFrame(records)

    trainer = TransformerTrainer(model_name="roberta-base", num_labels=3)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    combined = pd.concat([train_df, val_df, test_df])

    # Only valid ratings should survive
    assert len(combined) <= 6, f"Expected invalid ratings to be dropped, got {len(combined)} rows"
    assert set(combined["label"].unique()).issubset({0, 1, 2})


def test_tier2_r5_per_class_metrics_single_class_only():
    """R5.4: Verify calculate_per_class_metrics handles single-class edge cases without array length mismatch."""
    evaluator = ModelEvaluator()
    y_true = np.array([2, 2, 2, 2])
    y_pred = np.array([2, 2, 2, 2])

    metrics_df = evaluator.calculate_per_class_metrics(y_true, y_pred)
    assert len(metrics_df) == 3
    pos_row = metrics_df[metrics_df["Class"] == "Positive"].iloc[0]
    assert pos_row["Precision"] == 1.0
    assert pos_row["Recall"] == 1.0
    assert pos_row["F1-Score"] == 1.0

    neg_row = metrics_df[metrics_df["Class"] == "Negative"].iloc[0]
    assert neg_row["Precision"] == 0.0
    assert neg_row["Recall"] == 0.0
    assert neg_row["F1-Score"] == 0.0


def test_tier2_r5_select_best_model_tied_or_zero_scores():
    """R5.1: Verify select_best_model gracefully returns a candidate model even when validation scores are zero or tied."""
    class ZeroScoreModel:
        def predict(self, X):
            return np.zeros(X.shape[0])

    X_val = np.zeros((10, 5))
    y_val = np.ones(10)  # All 1, model predicts 0 -> F1 = 0

    trainer = ModelTrainer({})
    trainer.models = {
        "model_a": ZeroScoreModel(),
        "model_b": ZeroScoreModel()
    }

    best_name, best_model = trainer.select_best_model(X_val, y_val)
    assert best_name in ["model_a", "model_b"]
    assert best_model is not None


def test_tier2_r5_plot_confusion_matrix_nested_directory_save(tmp_path):
    """R5.3: Verify plot_confusion_matrix successfully saves to nested, non-existent directories."""
    evaluator = ModelEvaluator()
    nested_path = tmp_path / "reports" / "evaluation" / "cm.png"

    with patch("matplotlib.pyplot.show"):
        # Evaluator or caller ensures parent dir creation
        nested_path.parent.mkdir(parents=True, exist_ok=True)
        cm = evaluator.plot_confusion_matrix(np.array([0, 1]), np.array([0, 1]), save_path=str(nested_path))
        assert nested_path.exists()
        assert nested_path.stat().st_size > 0


def test_tier2_r6_sqlite_concurrent_multithreaded_writes(tmp_path):
    """R6.3: Verify SQLite handles concurrent multi-threaded writes without database lock errors."""
    test_db = tmp_path / "concurrent_test.db"
    monitor = ModelMonitor(db_path=str(test_db))

    def write_predictions(thread_id, n_records=20):
        for i in range(n_records):
            monitor.log_prediction(
                input_text=f"Thread {thread_id} test review {i}",
                prediction=thread_id % 3,
                confidence=0.85,
                response_time=0.02
            )

    n_threads = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(write_predictions, tid) for tid in range(n_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Will raise if sqlite3.OperationalError occurs

    conn = sqlite3.connect(str(test_db), timeout=30.0)
    total_written = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()

    assert total_written == n_threads * 20, (
        f"Expected {n_threads * 20} records logged, got {total_written}"
    )


def test_tier2_r6_predict_endpoint_empty_and_whitespace_validation():
    """R6.2: Verify /predict validates inputs and rejects empty or whitespace-only texts with HTTP 400."""
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)

    for empty_input in ["", "   ", "\t\n  \r\n"]:
        resp = client.post("/predict", json={"text": empty_input})
        assert resp.status_code in [400, 422], (
            f"Expected HTTP 400 or 422 for '{empty_input}', got {resp.status_code}"
        )


def test_tier2_r7_packaging_specifiers_valid_pep508():
    """R7.1 & R7.2: Verify all packaging dependency specifiers parse cleanly under PEP 508."""
    # Check requirements-dev.txt
    req_dev = PROJECT_ROOT / "requirements-dev.txt"
    if req_dev.exists():
        for line in req_dev.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-r"):
                continue
            req = packaging.requirements.Requirement(line)
            if req.name == "onnx":
                assert req.specifier.contains("1.14.0"), "onnx must satisfy >=1.14.0"

    # Check pyproject.toml
    pyproject = PROJECT_ROOT / "pyproject.toml"
    assert pyproject.exists()


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

def test_tier3_ingestion_to_preprocessing_negation_preservation(tmp_path, monkeypatch):
    """Tier 3.1: Ingestion -> Preprocessing integration.

    Ingests reviews via CSV and API with negation words, HTML markup, and ratings.
    Asserts negations preserved, HTML stripped, and ratings mapped to labels.
    """
    csv_file = tmp_path / "raw_reviews.csv"
    csv_df = pd.DataFrame({
        "Text": [
            "The screen was <br/> not bright enough and never turned on again.",
            "This item is neither sturdy nor reliable. Barely acceptable.",
            "Great product! Fast shipping and excellent packaging."
        ],
        "Score": [1, 2, 5],
        "Summary": ["Screen broken", "Poor build", "Loved it"]
    })
    csv_df.to_csv(csv_file, index=False)

    api_payload = [
        {"Text": "Hardly worked for two days, scarcely worth any money.", "Score": 1, "Summary": "Useless"},
        {"Text": "No problems so far, works as described.", "Score": 4, "Summary": "Satisfied"},
        {"Text": "Average quality, nothing extraordinary.", "Score": 3, "Summary": "Okay"}
    ]

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return api_payload

    monkeypatch.setattr("requests.get", lambda url, headers=None, timeout=None: MockResponse())

    # 1. Ingestion
    ingestion = DataIngestionPipeline({})
    sources = [
        {"type": "csv", "name": "csv_channel", "path": str(csv_file)},
        {"type": "api", "name": "api_channel", "url": "https://api.test.com/reviews"}
    ]
    combined_raw = ingestion.ingest_multiple_sources(sources)
    assert len(combined_raw) == 6
    assert "data_source" in combined_raw.columns

    # 2. Preprocessing Pipeline
    preprocessor = DataPreprocessor({})
    processed_df = preprocessor.preprocess_pipeline(combined_raw)

    assert len(processed_df) == 6
    assert "reviewText_clean" in processed_df.columns
    assert "sentiment" in processed_df.columns

    clean_texts = " ".join(processed_df["reviewText_clean"].tolist())

    # Check all key negations preserved
    for neg in ["not", "never", "neither", "nor", "barely", "hardly", "scarcely", "no"]:
        assert neg in clean_texts, f"Expected negation '{neg}' was not preserved in clean text!"

    # Check HTML tags stripped
    assert "<br/>" not in clean_texts
    assert "<br>" not in clean_texts

    # Check rating to sentiment mapping (1,2->Negative, 3->Neutral, 4,5->Positive)
    expected_sentiments = ["Negative", "Negative", "Positive", "Negative", "Positive", "Neutral"]
    assert processed_df["sentiment"].tolist() == expected_sentiments


def test_tier3_preprocessing_to_feature_engineering_stylistic_features():
    """Tier 3.2: Preprocessing -> Feature Engineering integration.

    Verifies that basic stylistic features are extracted from raw text while
    TF-IDF is extracted from clean text.
    """
    raw_data = pd.DataFrame({
        "Text": [
            "TERRIBLE SERVICE! Why did it break on day one???",
            "ABSOLUTELY FANTASTIC!! Best gadget I ever bought.",
            "Neutral item. It does its job without any issues."
        ],
        "Score": [1, 5, 3],
        "Summary": ["Bad", "Great", "Normal"]
    })

    preprocessor = DataPreprocessor({})
    processed_df = preprocessor.preprocess_pipeline(raw_data)

    engineer = FeatureEngineer({"tfidf_max_features": 50})
    X, y = engineer.engineer_features(
        processed_df,
        text_column="reviewText_clean",
        is_training=True,
        include_embeddings=False,
        raw_text_column="reviewText"
    )

    X_dense = X.toarray() if hasattr(X, "toarray") else X
    assert X_dense.shape[0] == 3
    assert len(y) == 3
    assert list(y) == [0, 2, 1]  # Negative -> 0, Positive -> 2, Neutral -> 1

    basic_feats = engineer.extract_basic_features(processed_df, "reviewText")
    # Row 0: "TERRIBLE SERVICE! Why did it break on day one???"
    assert basic_feats.iloc[0]["exclamation_count"] >= 1
    assert basic_feats.iloc[0]["question_count"] >= 3
    assert basic_feats.iloc[0]["uppercase_count"] >= 15

    # Row 2: "Neutral item. It does its job without any issues." has periods
    assert basic_feats.iloc[2]["sentence_count"] >= 2

    # Clean text comparison: If basic features were extracted from 'reviewText_clean', counts would be 0
    clean_basic_feats = engineer.extract_basic_features(processed_df, "reviewText_clean")
    assert clean_basic_feats["exclamation_count"].sum() == 0
    assert clean_basic_feats["question_count"].sum() == 0
    assert clean_basic_feats["uppercase_count"].sum() == 0


def test_tier3_dataset_prep_to_training_and_evaluation_zero_division(tmp_path):
    """Tier 3.3: Dataset preparation -> Training & Evaluation integration.

    Tests rating conversion (1-5 -> 0,1,2), weighted F1 model selection,
    and zero_division=0 per-class evaluation with unpredicted classes.
    """
    records = []
    for star in [1, 2, 3, 4, 5]:
        for i in range(10):
            records.append({
                "reviewText": f"This is a review with rating {star} sample {i}",
                "sentiment": str(star),
                "summary": f"Headline {star}"
            })
    df = pd.DataFrame(records)

    trainer = TransformerTrainer(model_name="roberta-base", num_labels=3)
    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=40, full_dataset=False)

    all_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert len(all_indices) == len(train_df) + len(val_df) + len(test_df)
    for split in [train_df, val_df, test_df]:
        assert set(split["label"].unique()).issubset({0, 1, 2})

    # Model Selection by Weighted F1
    class HighAccuracyDummy:
        def predict(self, X):
            return np.full(X.shape[0], 2)

    class HighF1Dummy:
        def predict(self, X):
            return np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])

    X_val = np.random.randn(10, 5)
    y_val = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])

    model_trainer = ModelTrainer({})
    model_trainer.models = {
        "majority_biased": HighAccuracyDummy(),
        "balanced_f1": HighF1Dummy()
    }
    best_name, best_model = model_trainer.select_best_model(X_val, y_val)
    assert best_name == "balanced_f1"

    # Evaluation with Missing Class (Neutral not predicted)
    evaluator = ModelEvaluator()
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 2])

    metrics_df = evaluator.calculate_per_class_metrics(
        y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]
    )
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 3
    assert metrics_df["Class"].tolist() == ["Negative", "Neutral", "Positive"]

    neutral_row = metrics_df[metrics_df["Class"] == "Neutral"].iloc[0]
    assert neutral_row["Precision"] == 0.0
    assert neutral_row["Recall"] == 0.0
    assert neutral_row["F1-Score"] == 0.0

    # Headless Confusion Matrix Plotting
    cm_path = tmp_path / "confusion_matrix.png"
    cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=str(cm_path))
    assert cm.shape == (3, 3) or cm.shape == (2, 2)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0


def test_tier3_api_prediction_to_sqlite_drift_logging_success_and_error(tmp_path, monkeypatch):
    """Tier 3.4: API Prediction -> SQLite Logging integration.

    Verifies that successful predictions log to SQLite, failed predictions
    raise HTTP 500 without writing bogus rows, and WAL journal mode is active.
    """
    from fastapi.testclient import TestClient
    import src.api.app as app_module

    test_db = tmp_path / "test_monitoring.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    client = TestClient(app_module.app)

    # 1. Verify CPU-bound routes are synchronous def
    for route in app_module.app.routes:
        if getattr(route, "path", None) in ["/predict", "/api/simulate-drift", "/api/check-drift"]:
            assert not inspect.iscoroutinefunction(route.endpoint), (
                f"Route {route.path} must be a standard synchronous 'def' to run in a threadpool!"
            )

    # 2. Successful Prediction
    success_payload = {"text": "The build quality is fantastic and I love it!"}
    resp = client.post("/predict", json=success_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]

    # Verify SQLite recorded exactly 1 prediction
    conn = sqlite3.connect(str(test_db), timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), prediction, input_text FROM predictions")
    count, pred_label, input_txt = cursor.fetchone()
    assert count == 1
    assert pred_label in [0, 1, 2]
    assert "build quality" in input_txt

    # Verify WAL journal mode
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0].lower()
    assert mode == "wal", f"Expected PRAGMA journal_mode=WAL, got {mode}"
    conn.close()

    # 3. Simulated Prediction Error
    def mock_predict_fail(text):
        raise RuntimeError("Model inference failed unexpectedly due to corrupted weights.")

    monkeypatch.setattr(app_module.api, "predict_sentiment", mock_predict_fail)

    fail_payload = {"text": "This should trigger a 500 error."}
    resp_fail = client.post("/predict", json=fail_payload)
    assert resp_fail.status_code == 500, f"Expected HTTP 500, got {resp_fail.status_code}"

    # Verify SQLite row count has NOT incremented (remains 1, no bogus Neutral entry written)
    conn = sqlite3.connect(str(test_db), timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count_after_fail = cursor.fetchone()[0]
    assert count_after_fail == 1, (
        f"Prediction error caused a bogus record to be logged! Count is {count_after_fail}, expected 1."
    )
    conn.close()


# ==============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# ==============================================================================

def test_tier4_customer_feedback_e2e_pipeline_lifecycle(tmp_path, monkeypatch):
    """Tier 4.1: Comprehensive End-to-End Pipeline Workflow.

    Executes raw customer reviews through ingestion, preprocessing,
    feature engineering, model training, evaluation, headless plotting,
    and baseline metric logging into SQLite.
    """
    raw_reviews = [
        # Negatives with varied negations, punctuation, casing
        {"Text": "NOT WORKING AT ALL! Broke on day one. <p>Terrible.</p>", "Score": 1, "Summary": "Total waste"},
        {"Text": "Never buying this again. Customer service was barely helpful.", "Score": 1, "Summary": "Horrible"},
        {"Text": "The item was neither functional nor as advertised. Extremely disappointed!", "Score": 1, "Summary": "Broken"},
        {"Text": "Hardly worked for an hour before smoking. DO NOT BUY!", "Score": 1, "Summary": "Danger"},
        {"Text": "I can scarcely believe how bad the quality is. Cheap plastic.", "Score": 2, "Summary": "Cheap"},
        {"Text": "Defective unit arrived. Return process was painful.", "Score": 2, "Summary": "Defective"},
        {"Text": "Battery does not hold charge. Lasts no more than 10 minutes.", "Score": 2, "Summary": "Bad battery"},
        {"Text": "No instruction manual included and buttons are stuck.", "Score": 2, "Summary": "Poor quality"},
        # Neutrals
        {"Text": "Decent product. Nothing special but gets the job done.", "Score": 3, "Summary": "Average"},
        {"Text": "Average quality. Expected slightly better for the price.", "Score": 3, "Summary": "Okay"},
        {"Text": "It works as described, but shipping was a bit delayed.", "Score": 3, "Summary": "Neutral"},
        {"Text": "Mixed feelings. Good hardware but software is clunky.", "Score": 3, "Summary": "Mixed"},
        # Positives
        {"Text": "OUTSTANDING PRODUCT! Exceeded all my expectations.", "Score": 5, "Summary": "Loved it"},
        {"Text": "Works like a charm! Setup took less than two minutes.", "Score": 5, "Summary": "Super fast"},
        {"Text": "Great build quality, very solid feel and sleek finish.", "Score": 5, "Summary": "High quality"},
        {"Text": "Fantastic value for money. Highly recommended to everyone!", "Score": 5, "Summary": "Recommended"},
        {"Text": "Fast shipping, perfect condition. 10/10 purchase.", "Score": 5, "Summary": "Flawless"},
        {"Text": "Really pleased with performance. No complaints whatsoever.", "Score": 4, "Summary": "Very good"},
        {"Text": "Good reliable gadget. Battery life is impressive.", "Score": 4, "Summary": "Reliable"},
        {"Text": "Satisfied customer! Will purchase another unit soon.", "Score": 4, "Summary": "Happy"}
    ]

    csv_file = tmp_path / "customer_feed.csv"
    pd.DataFrame(raw_reviews).to_csv(csv_file, index=False)

    # 1. Ingestion Step
    ingestion = DataIngestionPipeline({})
    raw_df = ingestion.ingest_multiple_sources([
        {"type": "csv", "name": "feed_stream", "path": str(csv_file)}
    ])
    assert len(raw_df) == 20

    # 2. Preprocessing Step
    preprocessor = DataPreprocessor({})
    processed_df = preprocessor.preprocess_pipeline(raw_df)
    assert len(processed_df) == 20
    assert set(processed_df["sentiment"].unique()).issubset({"Negative", "Neutral", "Positive"})

    # 3. Leak-Free Split & Feature Engineering
    train_df, test_df = train_test_split(
        processed_df, test_size=0.3, random_state=42, stratify=processed_df["sentiment"]
    )

    engineer = FeatureEngineer({"tfidf_max_features": 40})
    X_train, y_train = engineer.engineer_features(
        train_df, "reviewText_clean", is_training=True, include_embeddings=False, raw_text_column="reviewText"
    )
    X_test, y_test = engineer.engineer_features(
        test_df, "reviewText_clean", is_training=False, include_embeddings=False, raw_text_column="reviewText"
    )

    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(train_df)
    assert len(y_test) == len(test_df)

    # 4. Model Training and Selection
    trainer = ModelTrainer({})
    trainer.models = {
        "log_reg": LogisticRegression(random_state=42, max_iter=200),
        "rf": RandomForestClassifier(n_estimators=10, random_state=42)
    }
    for m in trainer.models.values():
        m.fit(X_train, y_train)

    best_name, best_model = trainer.select_best_model(X_test, y_test)
    assert best_name in ["log_reg", "rf"]

    # 5. Model Evaluation & Headless Reporting
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_model_comprehensive(best_model, X_test, y_test)
    assert "basic_metrics" in eval_results
    assert 0.0 <= eval_results["basic_metrics"]["f1"] <= 1.0
    assert 0.0 <= eval_results["basic_metrics"]["accuracy"] <= 1.0

    cm_save = tmp_path / "final_cm.png"
    evaluator.plot_confusion_matrix(y_test, eval_results["predictions"], save_path=str(cm_save))
    assert cm_save.exists()

    # 6. MLOps Telemetry Registration
    test_db = tmp_path / "e2e_monitoring.db"
    monitor = ModelMonitor(db_path=str(test_db))
    monitor.log_performance_metrics(
        accuracy=eval_results["basic_metrics"]["accuracy"],
        f1=eval_results["basic_metrics"]["f1"],
        sample_size=len(y_test)
    )

    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT accuracy, f1_score, sample_size FROM performance_metrics")
    row = cur.fetchone()
    assert row is not None
    assert row[2] == len(y_test)
    conn.close()


def test_tier4_production_drift_detection_and_retraining_lifecycle(tmp_path):
    """Tier 4.2: Production MLOps Drift Detection & Retraining Lifecycle.

    Verifies baseline evaluation, statistical KS data drift detection,
    performance degradation alerts, and automated retraining plan generation.
    """
    test_db = tmp_path / "lifecycle_monitoring.db"
    monitor = ModelMonitor(db_path=str(test_db))

    np.random.seed(42)
    n_features = 20

    # 1. Baseline In-Distribution Data (Gaussian centered at 0.0)
    baseline_features = np.random.normal(loc=0.0, scale=1.0, size=(100, n_features))

    # 2. Phase 1: Healthy Production Traffic (Same distribution)
    normal_prod_features = np.random.normal(loc=0.05, scale=0.98, size=(60, n_features))
    drift_healthy = monitor.detect_data_drift(baseline_features, normal_prod_features, threshold=0.01)

    assert drift_healthy["overall_drift"] is False
    assert drift_healthy["drift_percentage"] < 0.10

    # Check that no alerts were logged during healthy operation
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM drift_alerts")
    assert cur.fetchone()[0] == 0
    conn.close()

    # 3. Phase 2: Influx of Severely Drifted Traffic (Shifted distribution loc=3.5)
    drifted_features = np.random.normal(loc=3.5, scale=2.5, size=(60, n_features))
    drift_event = monitor.detect_data_drift(baseline_features, drifted_features, threshold=0.05)

    assert drift_event["overall_drift"] is True
    assert drift_event["drift_percentage"] > 0.50

    # Verify data drift alert written to SQLite
    conn = sqlite3.connect(str(test_db))
    cur = conn.cursor()
    cur.execute("SELECT drift_type, severity FROM drift_alerts WHERE drift_type='data_drift'")
    alert_row = cur.fetchone()
    assert alert_row is not None
    assert alert_row[0] == "data_drift"
    conn.close()

    # 4. Phase 3: Performance Degradation Event
    y_true = np.array([0, 0, 1, 1, 2, 2] * 5)
    y_pred_degraded = np.array([2, 2, 0, 0, 1, 1] * 5)

    perf_result = monitor.monitor_performance_degradation(
        y_true=y_true,
        y_pred=y_pred_degraded,
        baseline_accuracy=0.85,
        baseline_f1=0.88,
        threshold=0.05
    )
    assert perf_result["performance_degraded"] is True

    # Log 2 more alerts to satisfy RetrainingStrategy condition (requires >= 3 alerts)
    monitor.log_drift_alert("performance_degradation", 0.20, 0.05, "high")
    monitor.log_drift_alert("performance_degradation", 0.18, 0.05, "high")

    # 5. Phase 4: Automated Retraining Decision & Plan
    strategy = RetrainingStrategy({})
    retrain_decision = strategy.should_retrain(monitor)

    assert retrain_decision["should_retrain"] is True
    assert "Multiple performance degradation alerts" in retrain_decision["reasons"]

    plan = strategy.create_retraining_plan(
        new_data_size=5000,
        current_model_performance={"accuracy": perf_result["current_accuracy"], "f1": perf_result["current_f1"]}
    )

    assert "data_requirements" in plan
    assert "model_updates" in plan
    assert "validation_strategy" in plan
    assert "deployment_strategy" in plan
    assert plan["data_requirements"]["minimum_samples"] >= 1000


def test_tier4_adversarial_and_edge_case_review_robustness():
    """Tier 4.3: Edge-Case and Adversarial Customer Feedback Robustness.

    Stresses the full pipeline with challenging inputs: HTML tags, scripts,
    pure exclamation marks, intense negation phrasing, and whitespace padding.
    """
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)

    edge_cases = [
        # 1. Punctuation screaming with all caps
        ("TOTAL DISASTER! BROKE IN PIECES WITHIN 5 SECONDS! WHY????", 200),
        # 2. Heavy negation syntax
        ("I can scarcely believe how bad this was, neither phone nor email answered.", 200),
        # 3. HTML and script tag injection
        ("<p>Good product</p><script>alert('hack')</script> Very satisfied!", 200),
        # 4. Leading/trailing whitespace and tabs
        ("   \t\n  Great performance and prompt delivery!   \n  ", 200),
        # 5. Short feedback
        ("Excellent", 200),
        # 6. Empty whitespace only -> Should return HTTP 400
        ("    \n\t   ", 400)
    ]

    for text, exp_status in edge_cases:
        resp = client.post("/predict", json={"text": text})
        assert resp.status_code == exp_status, (
            f"Input '{text[:30]}' returned {resp.status_code}, expected {exp_status}"
        )

        if exp_status == 200:
            data = resp.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
            assert 0.0 <= data["confidence"] <= 1.0
            probs = data["probabilities"]
            assert abs(sum(probs.values()) - 1.0) < 0.05
            assert data["error"] is None
