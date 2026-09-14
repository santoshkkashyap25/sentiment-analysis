"""Feature extraction from text data."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
import pickle
from typing import Dict, Tuple, List
import logging
import os
import warnings
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Handles feature extraction from text data"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.svd = None

    def extract_basic_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract basic text features without sentiment target leakage"""
        feature_df = pd.DataFrame(index=df.index)

        feature_df['text_length'] = df[text_column].str.len().fillna(0)
        feature_df['word_count'] = df[text_column].str.split().str.len().fillna(0)
        feature_df['sentence_count'] = df[text_column].str.count(r'\.').fillna(0)

        # Avoid zero division
        safe_word_count = feature_df['word_count'].replace(0, 1)
        feature_df['avg_word_length'] = (
            df[text_column].str.replace(' ', '').str.len().fillna(0) / safe_word_count
        )

        feature_df['exclamation_count'] = df[text_column].str.count('!').fillna(0)
        feature_df['question_count'] = df[text_column].str.count(r'\?').fillna(0)
        feature_df['uppercase_count'] = df[text_column].str.count(r'[A-Z]').fillna(0)

        # Subjectivity measures factual vs subjective tone (0 to 1) without leaking label direction
        feature_df['subjectivity'] = df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )

        self.logger.info(f"Extracted {len(feature_df.columns)} basic features")
        return feature_df

    def extract_tfidf_features(self, texts: List[str], max_features: int = 500, is_training: bool = True):
        """Extract TF-IDF features with strict train/inference separation"""
        if is_training or self.tfidf_vectorizer is None:
            min_df_val = 1 if len(texts) < 5 else 2
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=min_df_val,
                max_df=0.95,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)

        self.logger.info(f"Extracted TF-IDF features: {tfidf_features.shape}")
        return tfidf_features

    def extract_sentence_embeddings(self, texts: List[str],
                                  model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Extract sentence embeddings using pre-trained models (lazy loaded)"""
        if self.sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(model_name)

        embeddings = self.sentence_transformer.encode(texts, batch_size=16, show_progress_bar=False)
        self.logger.info(f"Extracted sentence embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def reduce_dimensionality(self, features: np.ndarray,
                            n_components: int = 50, is_training: bool = True) -> np.ndarray:
        """Reduce feature dimensionality using SVD"""
        if is_training or self.svd is None:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_features = self.svd.fit_transform(features)
        else:
            reduced_features = self.svd.transform(features)

        self.logger.info(f"Reduced dimensions from {features.shape[1]} to {n_components}")
        return reduced_features.astype(np.float32)

    def combine_features(self, basic_features: pd.DataFrame,
                        tfidf_features,
                        embeddings: np.ndarray = None):
        """Combine features in a memory-safe way (no dense conversion)"""
        FEATURE_COLS = [
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_count', 'subjectivity'
        ]
        cols = [c for c in FEATURE_COLS if c in basic_features.columns]
        basic_sparse = csr_matrix(basic_features[cols].fillna(0).values.astype(np.float32))

        if embeddings is not None:
            embeddings_sparse = csr_matrix(embeddings)
            combined = hstack([basic_sparse, tfidf_features, embeddings_sparse]).tocsr()
            self.logger.info(f"Combined sparse features (with embeddings): {combined.shape}")
        else:
            combined = hstack([basic_sparse, tfidf_features]).tocsr()
            self.logger.info(f"Combined sparse features: {combined.shape}")

        return combined

    def engineer_features(self, df: pd.DataFrame, text_column: str,
                         is_training: bool = True,
                         include_embeddings: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Complete feature engineering pipeline without data leakage"""
        self.logger.info(f"Starting feature engineering (is_training={is_training})...")

        feature_df = self.extract_basic_features(df, text_column)
        tfidf_features = self.extract_tfidf_features(
            df[text_column].tolist(),
            max_features=self.config.get('tfidf_max_features', 500),
            is_training=is_training
        )

        embeddings = None
        if include_embeddings:
            embeddings = self.extract_sentence_embeddings(df[text_column].tolist())
            embeddings = self.reduce_dimensionality(
                embeddings,
                n_components=self.config.get('svd_components', 50),
                is_training=is_training
            )

        X = self.combine_features(feature_df, tfidf_features, embeddings)

        y = None
        if 'sentiment' in df.columns:
            label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            y = df['sentiment'].map(label_mapping).values

        self.logger.info("Feature engineering complete")
        return X, y

    def save_feature_extractors(self, filepath: str):
        """Save feature extractors for later use"""
        extractors = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'sentence_transformer': self.sentence_transformer,
            'svd': self.svd
        }
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        self.logger.info(f"Feature extractors saved to {filepath}")

    def load_feature_extractors(self, filepath: str):
        """Load pre-trained feature extractors"""
        with open(filepath, 'rb') as f:
            extractors = pickle.load(f)
        self.tfidf_vectorizer = extractors['tfidf_vectorizer']
        self.sentence_transformer = extractors['sentence_transformer']
        self.svd = extractors['svd']
        self.logger.info(f"Feature extractors loaded from {filepath}")
