"""Data ingestion from multiple sources."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import pandas as pd
import requests

warnings.filterwarnings("ignore")


class DataIngestionPipeline:
    """Handles data ingestion from multiple sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _get_kaggle(self):
        import kaggle
        return kaggle

    def download_kaggle_dataset(self, dataset_name: str, download_path: Union[str, Path]) -> str:
        """Download dataset from Kaggle"""
        try:
            kaggle = self._get_kaggle()
            download_path_str = str(download_path)
            kaggle.api.dataset_download_files(
                dataset_name,
                path=download_path_str,
                unzip=True
            )
            self.logger.info(f"Downloaded {dataset_name} to {download_path_str}")
            return download_path_str
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {e}")
            raise

    def load_csv_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file with cross-platform path normalization"""
        try:
            if isinstance(file_path, str) and "\\" in file_path and os.name != "nt":
                file_path = file_path.replace("\\", "/")
            path_obj = Path(file_path)
            df = pd.read_csv(path_obj)
            self.logger.info(f"Loaded {len(df)} records from {path_obj}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise

    def fetch_api_data(self, api_url: str, headers: Optional[Dict] = None, timeout: int = 30) -> pd.DataFrame:
        """Fetch data from API endpoint with timeout"""
        try:
            response = requests.get(api_url, headers=headers, timeout=timeout)
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
            source_type = source.get('type')
            if source_type == 'csv':
                df = self.load_csv_data(source['path'])
            elif source_type == 'api':
                timeout = source.get('timeout', 30)
                df = self.fetch_api_data(source['url'], headers=source.get('headers'), timeout=timeout)
            elif source_type == 'kaggle':
                path = self.download_kaggle_dataset(source['dataset'], source['download_path'])
                file_path = os.path.join(path, source['filename'])
                df = self.load_csv_data(file_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            df['data_source'] = source['name']
            dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df)} records from {len(sources)} sources")
        return combined_df
