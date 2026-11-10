"""Unit tests for feature engineering pipeline and stylistic feature extraction."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance with test-sized parameters."""
    return FeatureEngineer({
        'tfidf_max_features': 50,
        'svd_components': 8
    })


@pytest.fixture
def sample_reviews_df():
    """Create sample DataFrame with raw text, clean text, and sentiment labels."""
    return pd.DataFrame({
        'reviewText': [
            "FANTASTIC product! Really loved it? Absolutely YES! A++",
            "TERRIBLE quality! Broken on arrival, why sell this?! Never again.",
            "Average item. It works as expected. Nothing special."
        ],
        'reviewText_clean': [
            "fantastic product really loved it absolutely yes a",
            "terrible quality broken on arrival why sell this never again",
            "average item it works as expected nothing special"
        ],
        'sentiment': ['Positive', 'Negative', 'Neutral']
    })


# ============================================================================
# 1. TruncatedSVD & Dimensionality Reduction Tests (R1.1)
# ============================================================================

def test_truncated_svd_imported_and_available():
    """Verify TruncatedSVD is imported and available in feature_engineering namespace."""
    import src.features.feature_engineering as fe_module
    assert hasattr(fe_module, 'TruncatedSVD')
    assert fe_module.TruncatedSVD is TruncatedSVD


def test_reduce_dimensionality_training_mode(feature_engineer):
    """Test reduce_dimensionality fits SVD and reduces dimensions during training."""
    n_samples, n_features, n_components = 40, 50, 10
    features = np.random.randn(n_samples, n_features).astype(np.float32)

    reduced = feature_engineer.reduce_dimensionality(
        features, n_components=n_components, is_training=True
    )

    assert feature_engineer.svd is not None
    assert isinstance(feature_engineer.svd, TruncatedSVD)
    assert reduced.shape == (n_samples, n_components)
    assert reduced.dtype == np.float32


def test_reduce_dimensionality_inference_reuses_fitted_svd(feature_engineer):
    """Test reduce_dimensionality reuses pre-fitted SVD in inference mode without re-fitting."""
    n_features, n_components = 50, 10
    X_train = np.random.randn(40, n_features).astype(np.float32)
    feature_engineer.reduce_dimensionality(X_train, n_components=n_components, is_training=True)

    original_svd = feature_engineer.svd

    X_test = np.random.randn(5, n_features).astype(np.float32)
    reduced_test = feature_engineer.reduce_dimensionality(
        X_test, n_components=n_components, is_training=False
    )

    assert reduced_test.shape == (5, n_components)
    assert feature_engineer.svd is original_svd  # No re-instantiation / data leakage


def test_reduce_dimensionality_inference_without_training_auto_fits(feature_engineer):
    """Test inference mode auto-fits SVD if self.svd is None, preventing crash."""
    assert feature_engineer.svd is None

    X_infer = np.random.randn(30, 40).astype(np.float32)
    reduced = feature_engineer.reduce_dimensionality(
        X_infer, n_components=8, is_training=False
    )

    assert feature_engineer.svd is not None
    assert reduced.shape == (30, 8)


def test_reduce_dimensionality_n_components_exceeds_features_raises(feature_engineer):
    """Test that requesting n_components > n_features raises ValueError."""
    X = np.random.randn(20, 5).astype(np.float32)
    with pytest.raises(ValueError, match=r"n_components.*must be <= n_features"):
        feature_engineer.reduce_dimensionality(X, n_components=10, is_training=True)


def test_reduce_dimensionality_component_count_vs_sample_rank(feature_engineer):
    """Test SVD behavior when sample count is less than n_components."""
    # When n_samples < n_components, scikit-learn SVD rank is bounded by n_samples
    n_samples, n_features, n_components = 4, 30, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    reduced = feature_engineer.reduce_dimensionality(
        X, n_components=n_components, is_training=True
    )
    # Output components equal min(n_samples, n_components)
    assert reduced.shape == (n_samples, min(n_samples, n_components))


# ============================================================================
# 2. Sentence Embeddings Integration & Mocking Tests (R1.1)
# ============================================================================

def test_engineer_features_with_mocked_embeddings_training(feature_engineer, sample_reviews_df, monkeypatch):
    """Test full engineer_features pipeline with mocked sentence embeddings in training mode."""
    embedding_dim = 64
    mock_called = False

    def mock_extract(texts, model_name=None):
        nonlocal mock_called
        mock_called = True
        return np.random.randn(len(texts), embedding_dim).astype(np.float32)

    monkeypatch.setattr(feature_engineer, 'extract_sentence_embeddings', mock_extract)

    X, y = feature_engineer.engineer_features(
        sample_reviews_df,
        text_column='reviewText_clean',
        is_training=True,
        include_embeddings=True
    )

    assert mock_called
    assert issparse(X)
    assert feature_engineer.svd is not None
    assert X.shape[0] == len(sample_reviews_df)
    # Label mapping check: Positive -> 2, Negative -> 0, Neutral -> 1
    assert list(y) == [2, 0, 1]


def test_engineer_features_with_mocked_embeddings_inference(feature_engineer, sample_reviews_df, monkeypatch):
    """Test engineer_features with embeddings handles train/inference split without re-fitting."""
    embedding_dim = 64

    def mock_extract(texts, model_name=None):
        return np.random.randn(len(texts), embedding_dim).astype(np.float32)

    monkeypatch.setattr(feature_engineer, 'extract_sentence_embeddings', mock_extract)

    # 1. Fit on training split
    X_train, y_train = feature_engineer.engineer_features(
        sample_reviews_df,
        text_column='reviewText_clean',
        is_training=True,
        include_embeddings=True
    )
    svd_instance = feature_engineer.svd
    tfidf_instance = feature_engineer.tfidf_vectorizer

    # 2. Transform validation split
    val_df = pd.DataFrame({
        'reviewText': ["LOVED IT! Best thing ever."],
        'reviewText_clean': ["loved it best thing ever"],
        'sentiment': ['Positive']
    })
    X_val, y_val = feature_engineer.engineer_features(
        val_df,
        text_column='reviewText_clean',
        is_training=False,
        include_embeddings=True
    )

    assert feature_engineer.svd is svd_instance
    assert feature_engineer.tfidf_vectorizer is tfidf_instance
    assert X_val.shape[0] == 1
    assert X_val.shape[1] == X_train.shape[1]
    assert list(y_val) == [2]


def test_engineer_features_without_embeddings(feature_engineer, sample_reviews_df):
    """Test engineer_features when include_embeddings=False does not invoke SVD."""
    X, y = feature_engineer.engineer_features(
        sample_reviews_df,
        text_column='reviewText_clean',
        is_training=True,
        include_embeddings=False
    )
    assert feature_engineer.svd is None
    assert issparse(X)
    assert X.shape[0] == len(sample_reviews_df)


# ============================================================================
# 3. Stylistic Feature Extraction: Raw Text vs Clean Text Tests (R1.2)
# ============================================================================

def test_stylistic_features_raw_text_preserves_signals(feature_engineer):
    """Verify stylistic features extract punctuation and uppercase counts from unstripped raw text."""
    raw_df = pd.DataFrame({
        'reviewText': [
            "THIS IS INCREDIBLE!!! Really? YES! Absolutely GREAT.",
            "terrible, never buy."
        ]
    })
    features = feature_engineer.extract_basic_features(raw_df, 'reviewText')

    # Row 0: 4 '!', 1 '?', 28 uppercase letters, 2 '.'
    assert features.loc[0, 'exclamation_count'] == 4
    assert features.loc[0, 'question_count'] == 1
    assert features.loc[0, 'uppercase_count'] >= 20
    assert features.loc[0, 'sentence_count'] >= 1

    # Row 1: 0 '!', 0 '?', 0 uppercase letters
    assert features.loc[1, 'exclamation_count'] == 0
    assert features.loc[1, 'question_count'] == 0
    assert features.loc[1, 'uppercase_count'] == 0


def test_stylistic_features_clean_text_destroys_signals(feature_engineer):
    """Verify that normalized/cleaned text loses all punctuation and casing signals (all zero)."""
    clean_df = pd.DataFrame({
        'reviewText_clean': [
            "this is incredible really yes absolutely great",
            "terrible never buy"
        ]
    })
    features = feature_engineer.extract_basic_features(clean_df, 'reviewText_clean')

    # Clean text strips all punctuation and lowercases all text
    assert (features['exclamation_count'] == 0).all()
    assert (features['question_count'] == 0).all()
    assert (features['uppercase_count'] == 0).all()
    assert (features['sentence_count'] == 0).all()


def test_engineer_features_raw_text_column_explicit(feature_engineer):
    """Test engineer_features uses explicit raw_text_column for stylistic signals while tfidf uses clean."""
    df = pd.DataFrame({
        'raw_review': [
            "EXTRAORDINARY!!! Best quality? DEFINITELY YES!",
            "AWFUL! Worst product ever!"
        ],
        'reviewText_clean': [
            "extraordinary best quality definitely yes",
            "awful worst product ever"
        ],
        'sentiment': ['Positive', 'Negative']
    })

    # Call with raw_text_column='raw_review'
    X, y = feature_engineer.engineer_features(
        df,
        text_column='reviewText_clean',
        is_training=True,
        raw_text_column='raw_review'
    )

    # Basic features are the first 8 columns of combined sparse matrix
    basic_dense = X[:, :8].toarray()
    # Indices in FEATURE_COLS:
    # 0: text_length, 1: word_count, 2: sentence_count, 3: avg_word_length,
    # 4: exclamation_count, 5: question_count, 6: uppercase_count, 7: subjectivity
    exclamation_col = 4
    question_col = 5
    uppercase_col = 6

    # Verify stylistic signals are extracted from raw_review (non-zero)
    assert basic_dense[0, exclamation_col] == 4  # 3 from '!!!' + 1 from '!'
    assert basic_dense[0, question_col] == 1     # '?'
    assert basic_dense[0, uppercase_col] > 0    # ALL CAPS words


def test_engineer_features_fallback_to_reviewText_column(feature_engineer):
    """Test engineer_features falls back to 'reviewText' if raw_text_column is not provided."""
    df = pd.DataFrame({
        'reviewText': [
            "AMAZING! Does it work? YES!",
            "BAD! Return now!"
        ],
        'reviewText_clean': [
            "amazing does it work yes",
            "bad return now"
        ],
        'sentiment': ['Positive', 'Negative']
    })

    # Do not pass raw_text_column; should detect 'reviewText'
    X, y = feature_engineer.engineer_features(
        df,
        text_column='reviewText_clean',
        is_training=True
    )

    basic_dense = X[:, :8].toarray()
    exclamation_col = 4
    question_col = 5
    uppercase_col = 6

    assert basic_dense[0, exclamation_col] == 2
    assert basic_dense[0, question_col] == 1
    assert basic_dense[0, uppercase_col] > 0


def test_engineer_features_fallback_when_no_raw_text_available(feature_engineer):
    """Test engineer_features operates safely when only cleaned text column exists."""
    df = pd.DataFrame({
        'reviewText_clean': [
            "simple clean review text without raw",
            "another clean review"
        ],
        'sentiment': ['Neutral', 'Neutral']
    })

    X, y = feature_engineer.engineer_features(
        df,
        text_column='reviewText_clean',
        is_training=True
    )

    assert issparse(X)
    assert X.shape[0] == 2
    # Signals will be 0, but no KeyError or exception is raised
    basic_dense = X[:, :8].toarray()
    exclamation_col = 4
    assert basic_dense[:, exclamation_col].sum() == 0


def test_tfidf_always_uses_clean_text_column(feature_engineer):
    """Verify that TF-IDF vectorizer receives cleaned text even when raw text is supplied."""
    df = pd.DataFrame({
        'reviewText': [
            "<b>FANTASTIC</b>!!! Quality is 100% GREAT.",
            "TERRIBLE... Worst purchase ever."
        ],
        'reviewText_clean': [
            "fantastic quality is great",
            "terrible worst purchase ever"
        ],
        'sentiment': ['Positive', 'Negative']
    })

    feature_engineer.engineer_features(
        df,
        text_column='reviewText_clean',
        is_training=True,
        raw_text_column='reviewText'
    )

    vocab = feature_engineer.tfidf_vectorizer.vocabulary_
    # Verify HTML tags and punctuation did not leak into TF-IDF vocabulary
    assert '<b>fantastic</b>' not in vocab
    assert 'fantastic' in vocab


# ============================================================================
# 4. Serialization & Combiner Tests
# ============================================================================

def test_save_and_load_feature_extractors_with_svd(feature_engineer, sample_reviews_df, tmp_path, monkeypatch):
    """Test that save_feature_extractors and load_feature_extractors properly persist SVD."""
    embedding_dim = 32

    def mock_extract(texts, model_name=None):
        return np.random.randn(len(texts), embedding_dim).astype(np.float32)

    monkeypatch.setattr(feature_engineer, 'extract_sentence_embeddings', mock_extract)

    feature_engineer.engineer_features(
        sample_reviews_df,
        text_column='reviewText_clean',
        is_training=True,
        include_embeddings=True
    )

    save_path = str(tmp_path / 'extractors.pkl')
    feature_engineer.save_feature_extractors(save_path)

    # Load into fresh instance
    new_fe = FeatureEngineer({})
    new_fe.load_feature_extractors(save_path)

    assert new_fe.svd is not None
    assert isinstance(new_fe.svd, TruncatedSVD)
    assert new_fe.tfidf_vectorizer is not None

    # Test that restored SVD can transform new data
    dummy_emb = np.random.randn(2, embedding_dim).astype(np.float32)
    transformed = new_fe.reduce_dimensionality(dummy_emb, is_training=False)
    # Rank is min(n_samples, n_components) = min(3, 8) = 3
    assert transformed.shape == (2, new_fe.svd.components_.shape[0])
