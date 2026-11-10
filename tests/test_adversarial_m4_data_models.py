"""Adversarial stress and edge-case test suite for data processing, feature engineering, models, and evaluation.
Executed by Challenger M4-1 for Milestone 4.

Covers:
1. Data Preprocessing:
   - Multi-lingual text (Arabic, Cyrillic, CJK), emojis, Unicode zero-width characters.
   - Extreme text lengths, whitespace variations, and XSS/HTML tag stripping.
   - Robustness with missing values and negation preservation.
2. Feature Engineering:
   - Stylistic feature extraction under zero-variance features and raw text column routing.
   - SVD dimensionality reduction rank clipping and sparse representation stability.
   - CSR sparse matrix preservation and memory safety in combine_features.
3. Transformer Trainer & Ingestion:
   - Label mapping across 1-5 star ratings and pre-encoded {0, 1, 2} labels.
   - Balanced sampling row deduplication and index tracking.
   - Stratified dataset splitting and small dataset boundary conditions.
4. Model Evaluation & Selection:
   - Per-class metrics zero-division robustness under single-class or missing predictions.
   - Model selection weighted F1 prioritization over raw accuracy on imbalanced distributions.
   - Zero-score and tied-candidate model selection stability.
   - Headless confusion matrix plotting with nested directory creation and no GUI blocking.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.data.preprocessing import DataPreprocessor, NEGATION_WORDS
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluation import ModelEvaluator
from src.models.training import ModelTrainer
from src.models.transformer_trainer import TransformerTrainer


# ============================================================================
# Suite 1: Data Preprocessing Adversarial Robustness
# ============================================================================

class TestM4DataPreprocessingRobustness:
    """Stress tests for text normalization, negation preservation, and missing value handling."""

    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor({})

    def test_unicode_multilingual_emojis_negation_preservation(self, preprocessor):
        """Negation words must survive alongside Arabic, Cyrillic, CJK, and emoji tokens."""
        text = (
            "هذا ليس جيدا but not acceptable! "
            "это не хорошо nevertheless scarcely adequate. "
            "非常糟糕 never buying again! 🚀🔥"
        )
        cleaned = preprocessor.clean_text(text)
        tokens = set(cleaned.split())

        for target in ["not", "scarcely", "never"]:
            assert target in tokens, f"Expected negation '{target}' in cleaned tokens: {tokens}"

    def test_extreme_text_length_and_whitespace(self, preprocessor):
        """100,000+ character review with mixed control characters processes without memory or regex recursion errors."""
        long_review = ("Super product not bad at all! \r\n\t\b " * 3000)
        cleaned = preprocessor.clean_text(long_review)
        tokens = cleaned.split()

        assert "not" in tokens
        assert "bad" in tokens
        assert len(cleaned) > 50000

    def test_html_stripping_and_xss_safety(self, preprocessor):
        """Standard HTML markup and script tags are cleanly stripped without deleting negation context."""
        html_text = (
            "<div><p>Customer feedback: <script>alert('xss')</script>"
            "This item is <strong>not</strong> durable at all!</p></div>"
        )
        cleaned = preprocessor.clean_text(html_text)
        tokens = cleaned.split()

        assert "alert" not in tokens
        assert "script" not in tokens
        assert "not" in tokens
        assert "durable" in tokens

    def test_missing_and_nan_in_dataframe_pipeline(self, preprocessor):
        """preprocess_pipeline handles NaN reviewText, NaN summary, and duplicate entries seamlessly."""
        raw_df = pd.DataFrame({
            "Text": [
                "Legitimate positive review one.",
                None,
                np.nan,
                "Legitimate positive review one.",  # duplicate
                "Average neutral review two.",
                "Terrible broken product three."
            ],
            "Score": [5, 4, 3, 5, 3, 1],
            "Summary": ["Good", None, "Neutral", "Good", np.nan, "Bad"]
        })

        processed = preprocessor.preprocess_pipeline(raw_df)

        assert len(processed) == 3, f"Expected 3 unique non-null rows, got {len(processed)}"
        assert "reviewText_clean" in processed.columns
        assert "sentiment" in processed.columns
        assert set(processed["sentiment"].unique()) == {"Positive", "Neutral", "Negative"}


# ============================================================================
# Suite 2: Feature Engineering Adversarial Robustness
# ============================================================================

class TestM4FeatureEngineeringRobustness:
    """Stress tests for stylistic feature extraction, SVD dimensionality reduction, and feature combination."""

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer({"tfidf_max_features": 100, "svd_components": 10})

    def test_stylistic_features_zero_variance(self, engineer):
        """Reviews with identical lowercase text and zero punctuation produce valid numeric features without division by zero."""
        df = pd.DataFrame({
            "reviewText": ["item ok", "item ok", "item ok"],
            "reviewText_clean": ["item ok", "item ok", "item ok"]
        })

        features = engineer.extract_basic_features(df, "reviewText")

        assert (features["exclamation_count"] == 0).all()
        assert (features["question_count"] == 0).all()
        assert (features["uppercase_count"] == 0).all()
        assert not features.isna().any().any(), "Features dataframe contains NaN values!"
        assert (features["avg_word_length"] > 0).all()

    def test_stylistic_features_raw_text_vs_clean_text(self, engineer):
        """Stylistic counts (exclamations, questions, uppercase) are preserved from raw_text_column rather than stripped clean text."""
        df = pd.DataFrame({
            "rawText": ["AMAZING BUY!!! Why wait?", "TERRIBLE QUALITY! Do NOT buy."],
            "cleanedText": ["amazing buy why wait", "terrible quality do not buy"]
        })

        features = engineer.extract_basic_features(df, "rawText")

        assert features["exclamation_count"].iloc[0] == 3
        assert features["question_count"].iloc[0] == 1
        assert features["uppercase_count"].iloc[0] == 11

    def test_svd_rank_clipping_and_sparse_preservation(self, engineer):
        """TruncatedSVD clips components cleanly when sample count < components, and output remains float32."""
        features = np.random.randn(6, 25).astype(np.float32)
        reduced = engineer.reduce_dimensionality(features, n_components=10, is_training=True)

        assert reduced.shape == (6, 6), f"Expected clipped shape (6, 6), got {reduced.shape}"
        assert reduced.dtype == np.float32

    def test_combine_features_memory_safety(self, engineer):
        """combine_features outputs valid CSR sparse matrix without memory explosion or dense conversion."""
        basic_df = pd.DataFrame({
            "text_length": [10, 20],
            "word_count": [2, 4],
            "sentence_count": [1, 2],
            "avg_word_length": [4.0, 4.5],
            "exclamation_count": [1, 0],
            "question_count": [0, 1],
            "uppercase_count": [2, 3],
            "subjectivity": [0.5, 0.7]
        })
        tfidf_sparse = engineer.extract_tfidf_features(
            ["great product indeed", "terrible product failed"],
            max_features=20,
            is_training=True
        )

        combined = engineer.combine_features(basic_df, tfidf_sparse, embeddings=None)

        assert issparse(combined), "Output of combine_features must be a sparse matrix"
        assert combined.shape[0] == 2
        assert combined.shape[1] == 8 + tfidf_sparse.shape[1]


# ============================================================================
# Suite 3: Transformer Trainer & Ingestion Adversarial Robustness
# ============================================================================

class TestM4TransformerTrainerRobustness:
    """Stress tests for TransformerTrainer dataset preparation, label mapping, and split disjointness."""

    @pytest.fixture
    def trainer(self):
        return TransformerTrainer(model_name="roberta-base")

    def test_star_rating_mapping_and_split_disjointness(self, trainer):
        """1-5 star ratings map accurately to labels {0, 1, 2} and produce strictly disjoint train/val/test splits."""
        n_samples = 45
        df = pd.DataFrame({
            "reviewText": [f"Customer feedback review number {i} for robustness check." for i in range(n_samples)],
            "sentiment": ([1, 2, 3, 4, 5] * 9)
        })

        train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=45, random_state=42)

        total_rows = len(train_df) + len(val_df) + len(test_df)
        assert total_rows == n_samples

        # Verify strict index disjointness
        assert set(train_df.index).isdisjoint(set(val_df.index))
        assert set(train_df.index).isdisjoint(set(test_df.index))
        assert set(val_df.index).isdisjoint(set(test_df.index))

        # Verify label space strictly in {0, 1, 2}
        all_labels = set(train_df["label"].unique()) | set(val_df["label"].unique()) | set(test_df["label"].unique())
        assert all_labels.issubset({0, 1, 2})

    def test_balanced_sampling_row_deduplication(self, trainer):
        """Balanced sampling correctly drops sampled indices from remaining pool without duplicate row sampling."""
        # Highly imbalanced: 100 positive (rating 5), 10 neutral (rating 3), 10 negative (rating 1)
        reviews = [f"Positive review text {i}" for i in range(100)] + \
                  [f"Neutral review text {i}" for i in range(10)] + \
                  [f"Negative review text {i}" for i in range(10)]
        sentiments = [5] * 100 + [3] * 10 + [1] * 10
        df = pd.DataFrame({"reviewText": reviews, "sentiment": sentiments})

        # Request balanced sample of 45 items (15 per class targeted, then fill from majority)
        train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=45, random_state=42)

        combined = pd.concat([train_df, val_df, test_df])
        assert len(combined) == 45
        assert len(combined.index.unique()) == 45, "Duplicate indices detected in balanced sampling!"

    def test_dataset_boundary_minimum_sample_size(self, trainer):
        """Datasets meeting minimum stratifiable size (N >= 11) split cleanly without ValueError."""
        n = 15
        df = pd.DataFrame({
            "reviewText": [f"Review sample {i} for boundary testing." for i in range(n)],
            "sentiment": ([1, 3, 5] * (n // 3 + 1))[:n]
        })

        train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0


# ============================================================================
# Suite 4: Evaluation & Model Selection Adversarial Robustness
# ============================================================================

class TestM4EvaluationAndModelSelectionRobustness:
    """Stress tests for ModelEvaluator metrics and ModelTrainer model selection."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    @pytest.fixture
    def trainer(self):
        return ModelTrainer({})

    def test_per_class_metrics_single_class_zero_division(self, evaluator):
        """calculate_per_class_metrics succeeds with 3-class dataframe when true/pred contain only 1 class."""
        y_true = np.array([2, 2, 2, 2, 2])
        y_pred = np.array([2, 2, 2, 2, 2])

        metrics_df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert len(metrics_df) == 3
        assert list(metrics_df["Class"]) == ["Negative", "Neutral", "Positive"]
        assert metrics_df["Precision"].iloc[2] == 1.0
        assert metrics_df["Precision"].iloc[0] == 0.0
        assert metrics_df["Precision"].iloc[1] == 0.0
        assert not metrics_df.isna().any().any()

    def test_select_best_model_weighted_f1_over_accuracy(self, trainer):
        """select_best_model selects balanced classifier with higher weighted F1 over majority classifier with higher accuracy."""
        # 90 class 2, 10 class 0
        y_val = np.array([2] * 90 + [0] * 10)
        X_val = np.zeros((100, 2))

        class MajorityModel:
            def predict(self, X):
                return np.full(len(X), 2, dtype=int)  # 90% accuracy, ~0.85 weighted F1

        class BalancedModel:
            def predict(self, X):
                # 82 class 2 correct, 8 class 0 correct -> 90% balanced F1
                preds = np.zeros(len(X), dtype=int)
                preds[:82] = 2
                preds[82:90] = 0
                preds[90:98] = 0
                preds[98:100] = 2
                return preds

        trainer.models = {
            "majority_model": MajorityModel(),
            "balanced_model": BalancedModel()
        }

        best_name, best_model = trainer.select_best_model(X_val, y_val)

        assert best_name == "balanced_model", (
            f"Expected 'balanced_model' with higher weighted F1, got '{best_name}'"
        )
        assert best_model is trainer.models["balanced_model"]

    def test_select_best_model_tied_and_zero_scores(self, trainer):
        """select_best_model returns a valid registered model when all models achieve 0.0 F1 score."""
        y_val = np.full(30, 2, dtype=int)
        X_val = np.zeros((30, 2))

        class ZeroModel0:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        class ZeroModel1:
            def predict(self, X):
                return np.ones(len(X), dtype=int)

        trainer.models = {
            "zero_0": ZeroModel0(),
            "zero_1": ZeroModel1()
        }

        best_name, best_model = trainer.select_best_model(X_val, y_val)

        assert best_name in trainer.models
        assert best_model is trainer.models[best_name]
        assert trainer.best_model is best_model

    def test_plot_confusion_matrix_headless_and_nested_directory(self, evaluator, tmp_path):
        """plot_confusion_matrix creates missing nested directories, saves image, closes figure, and returns matrix."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])

        out_dir = tmp_path / "deeply" / "nested" / "reports"
        save_path = str(out_dir / "cm.png")

        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)

        assert cm.shape == (3, 3)
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0
