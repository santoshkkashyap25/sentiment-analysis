import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
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
    print("ALL PREPROCESSING TESTS PASSED!", flush=True)
