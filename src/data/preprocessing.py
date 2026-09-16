"""Data cleaning and preprocessing."""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None
from typing import Tuple, Dict
import logging
import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    """Handles data cleaning and preprocessing"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Ensure essential NLTK resources are available
        for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=['reviewText'], keep='first')
        removed_count = initial_count - len(df_clean)
        self.logger.info(f"Removed {removed_count} duplicate records")
        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df_clean = df.dropna(subset=['reviewText'])

        if 'overall' in df_clean.columns:
            df_clean = df_clean.assign(overall=df_clean['overall'].fillna(df_clean['overall'].median()))

        if 'summary' in df_clean.columns:
            df_clean = df_clean.assign(summary=df_clean['summary'].fillna(''))

        self.logger.info(f"Handled missing values, {len(df_clean)} records remaining")
        return df_clean

    def clean_text(self, text: str) -> str:
        """Clean individual text review"""
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        tokens = [token for token in tokens if token not in self.stop_words]
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def normalize_text(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Normalize text data"""
        self.logger.info("Starting text normalization...")
        df[f'{text_column}_clean'] = df[text_column].apply(self.clean_text)
        df = df[df[f'{text_column}_clean'].str.len() > 0]
        self.logger.info(f"Text normalization complete, {len(df)} records remaining")
        return df

    def create_sentiment_labels(self, df: pd.DataFrame, rating_column: str) -> pd.DataFrame:
        """Create sentiment labels from ratings"""
        def rating_to_sentiment(rating):
            if rating <= 2:
                return 'Negative'
            elif rating >= 4:
                return 'Positive'
            else:
                return 'Neutral'

        df['sentiment'] = df[rating_column].apply(rating_to_sentiment)
        return df

    def handle_imbalanced_data(self, X: np.ndarray, y: np.ndarray,
                              strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle imbalanced dataset using various techniques"""

        self.logger.info(f"Original class distribution: {np.bincount(y)}")

        if strategy == 'smote':
            if SMOTE is None:
                raise ImportError("imblearn is required for SMOTE resampling. Install with: pip install imblearn")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            if RandomUnderSampler is None:
                raise ImportError("imblearn is required for undersampling. Install with: pip install imblearn")
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        elif strategy == 'combined':
            if SMOTE is None or RandomUnderSampler is None:
                raise ImportError("imblearn is required for combined resampling. Install with: pip install imblearn")
            smote = SMOTE(random_state=42)
            X_temp, y_temp = smote.fit_resample(X, y)
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_temp, y_temp)
        else:
            X_resampled, y_resampled = X, y

        self.logger.info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled

    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        self.logger.info("Starting preprocessing pipeline...")

        df = df.rename(columns={
            'Text': 'reviewText',
            'Score': 'overall',
            'Summary': 'summary'
        })

        if 'Time' in df.columns:
            df['review_time'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.normalize_text(df, 'reviewText')

        if 'overall' in df.columns:
            df = self.create_sentiment_labels(df, 'overall')

        self.logger.info("Preprocessing pipeline complete")
        return df
