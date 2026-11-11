import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
import nltk
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def preprocessor():
    return DataPreprocessor({})


@pytest.fixture
def feature_engineer():
    return FeatureEngineer({'tfidf_max_features': 100})


def test_clean_text_removes_html(preprocessor):
    raw_html = "<p>This is a <b>fantastic</b> product!</p>"
    cleaned = preprocessor.clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "fantastic" in cleaned
    assert "product" in cleaned


def test_clean_text_removes_punctuation_and_lowercases(preprocessor):
    raw_text = "EXCELLENT quality, 100% recommended!!!"
    cleaned = preprocessor.clean_text(raw_text)
    assert "!" not in cleaned
    assert "%" not in cleaned
    assert "excellent" in cleaned
    assert "recommended" in cleaned


def test_sentiment_label_mapping(preprocessor):
    df = pd.DataFrame({'overall': [1, 2, 3, 4, 5]})
    labeled = preprocessor.create_sentiment_labels(df, 'overall')
    assert labeled['sentiment'].tolist() == ['Negative', 'Negative', 'Neutral', 'Positive', 'Positive']


def test_handle_missing_values(preprocessor):
    df = pd.DataFrame({
        'reviewText': ['Great product', None, np.nan, 'Terrible service'],
        'overall': [5, 1, 3, 1]
    })
    cleaned = preprocessor.handle_missing_values(df)
    assert len(cleaned) == 2


def test_extract_basic_features_no_target_leakage(feature_engineer):
    df = pd.DataFrame({
        'reviewText_clean': ['great product high quality', 'bad broken item']
    })
    features = feature_engineer.extract_basic_features(df, 'reviewText_clean')

    # Polarity should NOT be present (prevents target leakage)
    assert 'polarity' not in features.columns
    # Linguistic structural features should be present
    assert 'word_count' in features.columns
    assert 'text_length' in features.columns
    assert 'subjectivity' in features.columns


def test_tfidf_feature_shape(feature_engineer):
    texts = [
        "great fast shipping and good quality",
        "terrible broken defective item customer support",
        "average item nothing special"
    ]
    # Training fit
    tfidf_train = feature_engineer.extract_tfidf_features(texts, max_features=50, is_training=True)
    assert tfidf_train.shape[0] == 3
    assert tfidf_train.shape[1] <= 50

    # Inference transform
    new_text = ["fast shipping"]
    tfidf_test = feature_engineer.extract_tfidf_features(new_text, is_training=False)
    assert tfidf_test.shape[0] == 1
    assert tfidf_test.shape[1] == tfidf_train.shape[1]


# ============================================================================
# New Preprocessing Tests: Negation Preservation & Guarded Downloads (R2.1, R2.2)
# ============================================================================

def test_clean_text_preserves_all_eight_negation_words(preprocessor):
    """Verify all 8 negation words survive clean_text while stopwords are removed."""
    input_text = (
        "This product is not good and has no value, neither works nor lasts. "
        "It never arrived and barely functioned, hardly usable and scarcely worth anything at all."
    )
    cleaned = preprocessor.clean_text(input_text)
    tokens = set(cleaned.split())

    expected_negations = {'not', 'no', 'nor', 'neither', 'never', 'barely', 'hardly', 'scarcely'}
    for neg in expected_negations:
        assert neg in tokens, f"Negation word '{neg}' was removed from cleaned text: {cleaned}"

    removed_stopwords = {'this', 'is', 'and', 'has', 'it', 'at', 'all'}
    for sw in removed_stopwords:
        assert sw not in tokens, f"Stopword '{sw}' was unexpectedly preserved in: {cleaned}"


def test_clean_text_preserves_negation_sentiment_phrases(preprocessor):
    """Verify common sentiment-bearing negative phrases retain their negation markers."""
    test_cases = [
        ("not happy with the purchase", "not", "happy"),
        ("no defect found", "no", "defect"),
        ("never arrived on time", "never", "arrived"),
        ("neither durable nor reliable", "neither", "reliable"),
        ("barely worked out of box", "barely", "worked"),
        ("hardly noticeable scratch", "hardly", "noticeable"),
        ("scarcely usable interface", "scarcely", "usable"),
    ]
    for raw, neg_word, content_word in test_cases:
        cleaned = preprocessor.clean_text(raw)
        assert neg_word in cleaned.split(), f"Expected '{neg_word}' in '{cleaned}'"


def test_preprocessor_nltk_downloads_guarded(monkeypatch):
    """Verify nltk.download is NOT invoked when resources exist locally."""
    download_calls = []

    def spy_download(resource, *args, **kwargs):
        download_calls.append(resource)

    monkeypatch.setattr(nltk, 'download', spy_download)
    dp = DataPreprocessor({})

    assert len(download_calls) == 0, f"nltk.download called unexpectedly for: {download_calls}"


def test_preprocessor_nltk_downloads_called_when_resource_missing(monkeypatch):
    """Verify nltk.download IS invoked for a missing resource."""
    download_calls = []
    original_find = nltk.data.find

    def mock_find(resource_path, *args, **kwargs):
        if 'punkt_tab' in resource_path:
            raise LookupError("Resource punkt_tab missing")
        return original_find(resource_path, *args, **kwargs)

    def spy_download(resource, *args, **kwargs):
        download_calls.append(resource)

    monkeypatch.setattr(nltk.data, 'find', mock_find)
    monkeypatch.setattr(nltk, 'download', spy_download)

    dp = DataPreprocessor({})

    assert 'punkt_tab' in download_calls, "Expected punkt_tab to be downloaded when missing"
    assert 'punkt' not in download_calls, "punkt should not be downloaded when found"
    assert 'stopwords' not in download_calls, "stopwords should not be downloaded when found"
    assert 'wordnet' not in download_calls, "wordnet should not be downloaded when found"


def test_clean_text_edge_cases(preprocessor):
    """Verify clean_text edge cases: empty strings, None/NaN, uppercase negations, punctuation."""
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text(np.nan) == ""

    # Uppercase negations
    cleaned_upper = preprocessor.clean_text("NOT GOOD, NEVER USED, NO WAY")
    assert "not" in cleaned_upper.split()
    assert "never" in cleaned_upper.split()
    assert "no" in cleaned_upper.split()

    # Negation with attached punctuation
    cleaned_punct = preprocessor.clean_text("It is not, I repeat NOT, acceptable!")
    assert "not" in cleaned_punct.split()


if __name__ == '__main__':
    print("Running preprocessing tests directly...", flush=True)
    prep = DataPreprocessor({})
    feat = FeatureEngineer({'tfidf_max_features': 100})
    test_clean_text_removes_html(prep)
    print("test_clean_text_removes_html: PASSED", flush=True)
    test_clean_text_removes_punctuation_and_lowercases(prep)
    print("test_clean_text_removes_punctuation_and_lowercases: PASSED", flush=True)
    test_sentiment_label_mapping(prep)
    print("test_sentiment_label_mapping: PASSED", flush=True)
    test_handle_missing_values(prep)
    print("test_handle_missing_values: PASSED", flush=True)
    test_extract_basic_features_no_target_leakage(feat)
    print("test_extract_basic_features_no_target_leakage: PASSED", flush=True)
    test_tfidf_feature_shape(feat)
    print("test_tfidf_feature_shape: PASSED", flush=True)
    test_clean_text_preserves_all_eight_negation_words(prep)
    print("test_clean_text_preserves_all_eight_negation_words: PASSED", flush=True)
    test_clean_text_preserves_negation_sentiment_phrases(prep)
    print("test_clean_text_preserves_negation_sentiment_phrases: PASSED", flush=True)
    test_clean_text_edge_cases(prep)
    print("test_clean_text_edge_cases: PASSED", flush=True)
    print("ALL PREPROCESSING TESTS PASSED!", flush=True)
