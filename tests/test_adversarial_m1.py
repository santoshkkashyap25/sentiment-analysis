"""Empirical Adversarial Stress Tests for Milestone 1 (Data Pipeline & Feature Engineering).

Challenger 1 Adversarial Test Suite.
Covers:
1. Adversarial negation stress testing (double negations, mixed casing, punctuation,
   contractions, complex sentences, all 8 words, lemmatizer invariance).
2. Adversarial NLTK resource guard testing (isolated missing resource triggers,
   all-cached bypass, offline network error initialization and fallback).
3. Adversarial TruncatedSVD & embeddings testing (n_samples < n_components rank clipping,
   train/val dimension consistency, 0-sample/1D/single-col edge cases,
   and stylistic feature extraction with NaN, None, empty strings, pure numbers, and emojis).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import nltk
from nltk.stem import WordNetLemmatizer

from src.data.preprocessing import DataPreprocessor, NEGATION_WORDS
from src.features.feature_engineering import FeatureEngineer


# ============================================================================
# Area 1: Adversarial Negation Stress Testing
# ============================================================================

class TestAdversarialNegation:
    """Adversarial stress testing of negation word preservation."""

    TARGET_WORDS = {'not', 'no', 'nor', 'neither', 'never', 'barely', 'hardly', 'scarcely'}

    def test_target_words_set_equality(self):
        """Assert all 8 target words match NEGATION_WORDS constant exactly."""
        assert self.TARGET_WORDS == NEGATION_WORDS

    @pytest.mark.parametrize("word", list(TARGET_WORDS))
    def test_single_negation_word_survives(self, word):
        """Each negation word must survive clean_text as a standalone token."""
        dp = DataPreprocessor({})
        cleaned = dp.clean_text(f"This is {word} good")
        assert word in cleaned.split()

    def test_all_eight_negation_words_in_complex_sentence(self):
        """All 8 target words survive clean_text under a complex compound sentence structure."""
        dp = DataPreprocessor({})
        complex_text = (
            "Although the phone was not bad, it was definitely not not good either; "
            "neither the screen nor the battery lasted well, barely lasting through noon, "
            "scarcely reaching 2 hours of screen-on time, hardly acceptable in 2026, "
            "and customer support never answered and offered no refund whatsoever!"
        )
        cleaned = dp.clean_text(complex_text)
        tokens = set(cleaned.split())

        for word in self.TARGET_WORDS:
            assert word in tokens, f"Target negation word '{word}' was stripped from complex text: {cleaned}"

    def test_double_negations_preserved(self):
        """Double negations and consecutive negation pairs must retain both tokens without deduplication."""
        dp = DataPreprocessor({})
        cases = [
            ("It is not not good", ["not", "not", "good"]),
            ("We not never said that", ["not", "never", "said"]),
            ("never not satisfied", ["never", "not", "satisfied"]),
            ("hardly not acceptable", ["hardly", "not", "acceptable"]),
            ("no not ever", ["no", "not", "ever"]),
        ]
        for text, expected_sequence in cases:
            cleaned = dp.clean_text(text)
            tokens = cleaned.split()
            for exp in expected_sequence:
                assert exp in tokens, f"Expected '{exp}' in cleaned output '{cleaned}' for input '{text}'"
            # Verify repetition count for double tokens
            if "not not" in text:
                assert tokens.count("not") == 2, f"Expected 2 'not' tokens in: {tokens}"

    def test_mixed_casing_negations(self):
        """Adversarial casing variations of negation words must all normalize and survive."""
        dp = DataPreprocessor({})
        cased_inputs = [
            ("NoT acceptable", "not"),
            ("nOt bad", "not"),
            ("NOT good", "not"),
            ("NEVER again", "never"),
            ("nEvEr arrived", "never"),
            ("bArElY functional", "barely"),
            ("HaRdLy visible", "hardly"),
            ("sCaRcElY enough", "scarcely"),
            ("nEiThEr option", "neither"),
            ("NoR did that", "nor"),
            ("nO refund", "no"),
        ]
        for text, expected_token in cased_inputs:
            cleaned = dp.clean_text(text)
            assert expected_token in cleaned.split(), f"Failed to preserve '{expected_token}' from '{text}'"

    def test_punctuation_attached_negations(self):
        """Punctuation attached directly to individual negation words must not alter or strip the word."""
        dp = DataPreprocessor({})
        punct_cases = [
            ("not!", "not"),
            ("hardly?", "hardly"),
            ("never...", "never"),
            ("(no)", "no"),
            ("barely,", "barely"),
            ("scarcely;", "scarcely"),
            ("neither / nor", ["neither", "nor"]),
            ("[not]", "not"),
            ("{never}!", "never"),
            ("\"scarcely\"", "scarcely"),
        ]
        for text, expected in punct_cases:
            cleaned = dp.clean_text(text)
            tokens = cleaned.split()
            if isinstance(expected, list):
                for exp in expected:
                    assert exp in tokens, f"Expected '{exp}' in '{cleaned}' from '{text}'"
            else:
                assert expected in tokens, f"Expected '{expected}' in '{cleaned}' from '{text}'"

    def test_hyphenated_negation_compound_behavior(self):
        """Document and verify behavior when negation words are connected by unspaced hyphens.
        Because clean_text applies re.sub(r'[^a-zA-Z\\s]', '', text), unspaced hyphens/dashes
        concatenate tokens (e.g. 'neither--nor' -> 'neithernor', 'not-bad' -> 'notbad').
        """
        dp = DataPreprocessor({})
        # Unspaced hyphens concatenate
        assert dp.clean_text("neither--nor") == "neithernor"
        assert dp.clean_text("not-bad") == "notbad"
        # Spaced hyphens or dashes preserve individual tokens
        assert set(dp.clean_text("neither - nor").split()) == {"neither", "nor"}
        assert set(dp.clean_text("not - bad").split()) == {"not", "bad"}

    def test_contractions_and_informal_negations(self):
        """Contractions with apostrophes stripped retain their negation-bearing stems."""
        dp = DataPreprocessor({})
        text = "don't can't won't shouldn't isn't aren't didn't couldn't"
        cleaned = dp.clean_text(text)
        tokens = cleaned.split()
        expected = ["dont", "cant", "wont", "shouldnt", "isnt", "arent", "didnt", "couldnt"]
        for exp in expected:
            assert exp in tokens, f"Contraction '{exp}' was unexpectedly stripped: {cleaned}"

    def test_negation_words_invariant_under_lemmatization(self):
        """Verify that WordNetLemmatizer does not mutate any of the 8 negation words across all POS tags."""
        lemmatizer = WordNetLemmatizer()
        for word in self.TARGET_WORDS:
            for pos in ['n', 'v', 'a', 'r', 's']:
                assert lemmatizer.lemmatize(word, pos=pos) == word, (
                    f"Lemmatizer mutated '{word}' (pos={pos})"
                )


# ============================================================================
# Area 2: Adversarial NLTK Resource Guard Testing
# ============================================================================

class TestAdversarialNLTKResourceGuard:
    """Stress testing of NLTK resource finding, targeted downloading, and offline tolerance."""

    NLTK_RESOURCES = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
    }

    @pytest.mark.parametrize("target_name", ['punkt', 'punkt_tab', 'stopwords', 'wordnet'])
    def test_missing_single_resource_triggers_only_that_download(self, target_name, monkeypatch):
        """When exactly one resource is missing locally, only that resource triggers nltk.download."""
        target_path = self.NLTK_RESOURCES[target_name]
        downloaded = []
        original_find = nltk.data.find

        def mock_find(res_path):
            if res_path == target_path:
                raise LookupError(f"Simulated missing: {res_path}")
            return original_find(res_path)

        def mock_download(res_name, quiet=True):
            downloaded.append(res_name)

        monkeypatch.setattr(nltk.data, 'find', mock_find)
        monkeypatch.setattr(nltk, 'download', mock_download)

        dp = DataPreprocessor({})

        assert downloaded == [target_name], (
            f"Expected only '{target_name}' to trigger download, got: {downloaded}"
        )

    def test_all_resources_cached_triggers_zero_downloads(self, monkeypatch):
        """When all resources exist locally, zero downloads should be attempted."""
        downloaded = []

        def spy_download(res_name, quiet=True):
            downloaded.append(res_name)

        monkeypatch.setattr(nltk, 'download', spy_download)
        dp = DataPreprocessor({})

        assert len(downloaded) == 0, f"Unexpected downloads occurred: {downloaded}"

    def test_offline_environment_initialization_no_crash(self, monkeypatch):
        """Simulate an offline environment where all downloads fail with network errors.
        Initialization must complete gracefully without uncaught exceptions.
        """
        def mock_find_missing(path):
            raise LookupError(f"Offline: {path} not found")

        def mock_download_network_error(res_name, quiet=True):
            raise ConnectionError(f"Network is unreachable: failed to download {res_name}")

        monkeypatch.setattr(nltk.data, 'find', mock_find_missing)
        monkeypatch.setattr(nltk, 'download', mock_download_network_error)

        # Must initialize without raising ConnectionError or LookupError
        dp = DataPreprocessor({})
        assert isinstance(dp.stop_words, set)
        assert dp.NEGATION_WORDS == self.NLTK_RESOURCES and False or len(dp.NEGATION_WORDS) == 8

    def test_offline_environment_clean_text_fallback_without_punkt(self, monkeypatch):
        """When punkt is missing offline, clean_text falls back gracefully to str.split()."""
        original_find = nltk.data.find

        def mock_find_no_punkt(path):
            if 'punkt' in path:
                raise LookupError(f"Offline: {path}")
            return original_find(path)

        def mock_download_fail(name, quiet=True):
            raise ConnectionError("Offline")

        monkeypatch.setattr(nltk.data, 'find', mock_find_no_punkt)
        monkeypatch.setattr(nltk, 'download', mock_download_fail)

        dp = DataPreprocessor({})
        cleaned = dp.clean_text("The fast brown fox jumps over the lazy dog!")
        # Splitting and stopword filtering should still function
        assert "fox" in cleaned.split()
        assert "dog" in cleaned.split()


# ============================================================================
# Area 3: Adversarial TruncatedSVD & Stylistic Feature Extraction Testing
# ============================================================================

class TestAdversarialSVDAndStylisticFeatures:
    """Stress testing of SVD matrix dimensionality constraints and raw text edge cases."""

    def test_svd_rank_clipping_when_samples_less_than_components(self):
        """When n_samples < n_components, scikit-learn SVD rank is bounded by n_samples without error."""
        fe = FeatureEngineer({})
        n_samples, n_features, n_components = 4, 30, 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        reduced = fe.reduce_dimensionality(X, n_components=n_components, is_training=True)

        # Scikit-learn TruncatedSVD bounds components to min(n_samples, n_components)
        assert reduced.shape == (n_samples, n_samples)
        assert reduced.dtype == np.float32

    def test_svd_train_val_dimension_consistency_low_samples(self):
        """Pipeline preserves feature dimension consistency between train (low samples) and inference."""
        fe = FeatureEngineer({'tfidf_max_features': 50, 'svd_components': 10})
        fe.extract_sentence_embeddings = lambda texts: np.random.randn(len(texts), 32).astype(np.float32)

        train_df = pd.DataFrame({
            'reviewText': ['GREAT product!', 'TERRIBLE item.', 'OK average.', 'SUPER fast!'],
            'reviewText_clean': ['great product', 'terrible item', 'ok average', 'super fast'],
            'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive']
        })
        val_df = pd.DataFrame({
            'reviewText': ['AWESOME quality!'],
            'reviewText_clean': ['awesome quality'],
            'sentiment': ['Positive']
        })

        X_train, y_train = fe.engineer_features(
            train_df, text_column='reviewText_clean', is_training=True, include_embeddings=True
        )
        X_val, y_val = fe.engineer_features(
            val_df, text_column='reviewText_clean', is_training=False, include_embeddings=True
        )

        assert issparse(X_train) and issparse(X_val)
        assert X_train.shape[1] == X_val.shape[1], (
            f"Train cols ({X_train.shape[1]}) != Val cols ({X_val.shape[1]})"
        )
        assert X_val.shape[0] == 1

    def test_svd_empty_input_raises_value_error(self):
        """TruncatedSVD on empty input (0 samples) raises ValueError."""
        fe = FeatureEngineer({})
        X_empty = np.empty((0, 30), dtype=np.float32)
        with pytest.raises(ValueError, match=r"minimum of 1 is required"):
            fe.reduce_dimensionality(X_empty, n_components=5, is_training=True)

    def test_svd_1d_input_raises_value_error(self):
        """TruncatedSVD on 1D input raises ValueError requiring 2D array."""
        fe = FeatureEngineer({})
        X_1d = np.random.randn(30).astype(np.float32)
        with pytest.raises(ValueError, match=r"Expected 2D array"):
            fe.reduce_dimensionality(X_1d, n_components=5, is_training=True)

    def test_svd_single_column_raises_value_error(self):
        """TruncatedSVD requires at least 2 features, raising ValueError on single column."""
        fe = FeatureEngineer({})
        X_1col = np.random.randn(10, 1).astype(np.float32)
        with pytest.raises(ValueError, match=r"minimum of 2 is required"):
            fe.reduce_dimensionality(X_1col, n_components=1, is_training=True)

    def test_raw_text_stylistic_features_adversarial_types(self):
        """extract_basic_features must handle NaN, None, empty strings, pure numbers, pure whitespace, and emojis."""
        fe = FeatureEngineer({})
        adversarial_df = pd.DataFrame({
            'raw_text': [
                np.nan,
                None,
                "",
                "   \t\n   ",
                12345,
                3.14159,
                "😊🔥🚀🎉💯",
                "FANTASTIC!!! Really? YES! 100% OK.",
            ]
        })

        features = fe.extract_basic_features(adversarial_df, 'raw_text')

        # Check required columns
        expected_cols = [
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_count', 'subjectivity'
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing column '{col}' in extracted features"

        # Check that NO NaNs exist in the entire output DataFrame
        nan_counts = features.isna().sum().to_dict()
        assert all(c == 0 for c in nan_counts.values()), f"Found NaNs in features: {nan_counts}"

        # Row 0 (np.nan): all zeros except subjectivity
        assert features.loc[0, 'text_length'] == 0.0
        assert features.loc[0, 'word_count'] == 0.0
        assert features.loc[0, 'exclamation_count'] == 0.0

        # Row 2 (""): all zeros
        assert features.loc[2, 'text_length'] == 0.0

        # Row 6 (emoji): length is 5 emojis, no crash
        assert features.loc[6, 'text_length'] == 5.0

        # Row 7 (valid rich text): uppercase and punctuation captured
        assert features.loc[7, 'exclamation_count'] == 4.0
        assert features.loc[7, 'question_count'] == 1.0
        assert features.loc[7, 'uppercase_count'] >= 15.0

    def test_engineer_features_end_to_end_with_adversarial_raw_text(self):
        """Complete engineer_features pipeline execution with adversarial raw_text data and embeddings."""
        fe = FeatureEngineer({'tfidf_max_features': 50, 'svd_components': 4})
        fe.extract_sentence_embeddings = lambda texts: np.random.randn(len(texts), 32).astype(np.float32)

        df = pd.DataFrame({
            'raw_text': [
                np.nan, None, "", "   \t\n  ", 12345, 3.14159, "😊🔥🚀🎉💯", "Hello World! WHAT??? 100% OK."
            ],
            'reviewText_clean': [
                "placeholder item", "placeholder item", "placeholder item", "placeholder item",
                "hello world", "hello world", "hello ok", "hello world what ok"
            ],
            'sentiment': ['Negative', 'Negative', 'Negative', 'Negative', 'Neutral', 'Neutral', 'Positive', 'Positive']
        })

        X, y = fe.engineer_features(
            df,
            text_column='reviewText_clean',
            raw_text_column='raw_text',
            is_training=True,
            include_embeddings=True
        )

        assert issparse(X)
        assert X.shape[0] == 8
        assert len(y) == 8
        assert not np.isnan(X.toarray()).any(), "Resulting sparse feature matrix contains NaNs!"
