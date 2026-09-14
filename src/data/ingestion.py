"""Data ingestion from multiple sources."""

import pandas as pd
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings("ignore")


class DataIngestionPipeline:
    """Handles data ingestion from multiple sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _get_kaggle(self):
        import kaggle
        return kaggle

    def download_kaggle_dataset(self, dataset_name: str, download_path: str) -> str:
        """Download dataset from Kaggle"""
        try:
            kaggle = self._get_kaggle()
            kaggle.api.dataset_download_files(
                dataset_name,
                path=download_path,
                unzip=True
            )
            self.logger.info(f"Downloaded {dataset_name} to {download_path}")
            return download_path
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            raise

    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise

    def fetch_api_data(self, api_url: str, headers: Dict = None) -> pd.DataFrame:
        """Fetch data from API endpoint"""
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            self.logger.info(f"Fetched {len(df)} records from API")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching API data: {e}")
            raise

    def ingest_multiple_sources(self, sources: List[Dict]) -> pd.DataFrame:
        """Ingest data from multiple sources and combine"""
        dataframes = []

        for source in sources:
            if source['type'] == 'csv':
                df = self.load_csv_data(source['path'])
            elif source['type'] == 'api':
                df = self.fetch_api_data(source['url'], source.get('headers'))
            elif source['type'] == 'kaggle':
                path = self.download_kaggle_dataset(source['dataset'], source['download_path'])
                df = self.load_csv_data(f"{path}/{source['filename']}")

            df['data_source'] = source['name']
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df)} records from {len(sources)} sources")
        return combined_df
