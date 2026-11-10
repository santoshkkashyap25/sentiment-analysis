"""Unit tests for DataIngestionPipeline in src/data/ingestion.py."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import DataIngestionPipeline


@pytest.fixture
def ingestion_pipeline():
    """Fixture providing a default DataIngestionPipeline instance."""
    return DataIngestionPipeline(config={})


# ============================================================================
# Tests for fetch_api_data and timeout handling (R4.1)
# ============================================================================

def test_fetch_api_data_default_timeout(ingestion_pipeline):
    """Verify that fetch_api_data specifies timeout=30 by default when calling requests.get."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 1, "reviewText": "Great quality!", "overall": 5},
        {"id": 2, "reviewText": "Not bad", "overall": 3},
    ]
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        api_url = "https://api.example.com/reviews"
        df = ingestion_pipeline.fetch_api_data(api_url)

        # Assert requests.get was invoked with timeout=30
        mock_get.assert_called_once_with(api_url, headers=None, timeout=30)
        assert len(df) == 2
        assert "reviewText" in df.columns
        assert df["reviewText"].tolist() == ["Great quality!", "Not bad"]


def test_fetch_api_data_custom_timeout(ingestion_pipeline):
    """Verify that fetch_api_data allows callers to override the default timeout."""
    mock_response = MagicMock()
    mock_response.json.return_value = [{"id": 10, "text": "Quick test"}]
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        api_url = "https://api.example.com/fast-endpoint"
        custom_headers = {"Authorization": "Bearer test-key"}
        df = ingestion_pipeline.fetch_api_data(api_url, headers=custom_headers, timeout=10)

        mock_get.assert_called_once_with(api_url, headers=custom_headers, timeout=10)
        assert len(df) == 1


def test_fetch_api_data_http_error_propagation(ingestion_pipeline):
    """Verify that HTTP errors (e.g., 404, 500) from requests.raise_for_status are logged and raised."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.exceptions.HTTPError, match="404 Client Error"):
            ingestion_pipeline.fetch_api_data("https://api.example.com/nonexistent")


def test_fetch_api_data_timeout_exception(ingestion_pipeline):
    """Verify that requests.exceptions.Timeout is properly logged and re-raised."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with pytest.raises(requests.exceptions.Timeout, match="Connection timed out"):
            ingestion_pipeline.fetch_api_data("https://api.example.com/slow-endpoint")


# ============================================================================
# Tests for ingest_multiple_sources and unsupported source types (R4.2)
# ============================================================================

@pytest.mark.parametrize("unsupported_type", ["parquet", "sql", "xml", "kafka", "unknown_format"])
def test_ingest_multiple_sources_unsupported_type_raises_value_error(ingestion_pipeline, unsupported_type):
    """Verify that ingest_multiple_sources raises ValueError with exact message for unsupported source types."""
    bad_source = {
        "type": unsupported_type,
        "name": f"test_{unsupported_type}",
        "path": "data/sample.file",
    }
    with pytest.raises(ValueError, match=f"Unsupported source type: {unsupported_type}"):
        ingestion_pipeline.ingest_multiple_sources([bad_source])


def test_ingest_multiple_sources_missing_type_raises_value_error(ingestion_pipeline):
    """Verify that a source dictionary lacking a 'type' key raises ValueError(Unsupported source type: None)."""
    invalid_source = {
        "name": "no_type_source",
        "path": "data/sample.csv",
    }
    with pytest.raises(ValueError, match="Unsupported source type: None"):
        ingestion_pipeline.ingest_multiple_sources([invalid_source])


def test_ingest_multiple_sources_halts_on_subsequent_unsupported_source(ingestion_pipeline, tmp_path):
    """Verify that an unsupported source after a valid one does not reuse or pollute the prior DataFrame."""
    # Create a valid CSV file
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("id,reviewText\n1,Good\n", encoding="utf-8")

    sources = [
        {"type": "csv", "name": "valid_source", "path": str(csv_file)},
        {"type": "mongodb", "name": "bad_source"},
    ]
    # Must immediately raise ValueError, not return or duplicate data
    with pytest.raises(ValueError, match="Unsupported source type: mongodb"):
        ingestion_pipeline.ingest_multiple_sources(sources)


def test_ingest_multiple_sources_empty_list(ingestion_pipeline):
    """Verify that passing an empty list of sources returns an empty DataFrame without raising errors."""
    result = ingestion_pipeline.ingest_multiple_sources([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ============================================================================
# Tests for cross-platform path handling and Kaggle / CSV loading (R4.3)
# ============================================================================

def test_load_csv_data_with_string_path(ingestion_pipeline, tmp_path):
    """Verify load_csv_data loads CSV when passed a string path."""
    csv_path = tmp_path / "test_str.csv"
    csv_path.write_text("col_a,col_b\nval1,val2\nval3,val4\n", encoding="utf-8")

    df = ingestion_pipeline.load_csv_data(str(csv_path))
    assert len(df) == 2
    assert list(df.columns) == ["col_a", "col_b"]


def test_load_csv_data_with_pathlib_path(ingestion_pipeline, tmp_path):
    """Verify load_csv_data loads CSV when passed a pathlib.Path object."""
    csv_path = tmp_path / "test_pathlib.csv"
    csv_path.write_text("item,rating\nPhone,5\nLaptop,4\n", encoding="utf-8")

    df = ingestion_pipeline.load_csv_data(csv_path)
    assert len(df) == 2
    assert "item" in df.columns
    assert "rating" in df.columns


def test_load_csv_data_missing_file_raises(ingestion_pipeline, tmp_path):
    """Verify load_csv_data raises FileNotFoundError for missing files."""
    missing_file = tmp_path / "non_existent_file.csv"
    with pytest.raises(FileNotFoundError):
        ingestion_pipeline.load_csv_data(missing_file)


def test_ingest_multiple_sources_kaggle_cross_platform_path(ingestion_pipeline, tmp_path, monkeypatch):
    """Verify that kaggle dataset ingestion handles cross-platform directory paths using os.path.join or Path."""
    # Create mock downloaded directory structure
    download_dir = tmp_path / "nested" / "kaggle_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    csv_file = download_dir / "Reviews.csv"
    csv_file.write_text("review_id,text\n101,Excellent service\n102,Terrible delay\n", encoding="utf-8")

    # Mock download_kaggle_dataset to simulate Kaggle CLI output
    def mock_download(dataset_name, download_path):
        assert dataset_name == "sample/sentiment-dataset"
        return str(download_path)

    monkeypatch.setattr(ingestion_pipeline, "download_kaggle_dataset", mock_download)

    source = {
        "type": "kaggle",
        "name": "kaggle_feed",
        "dataset": "sample/sentiment-dataset",
        "download_path": str(download_dir),
        "filename": "Reviews.csv",
    }

    df = ingestion_pipeline.ingest_multiple_sources([source])
    assert len(df) == 2
    assert df["data_source"].iloc[0] == "kaggle_feed"
    assert "text" in df.columns


def test_ingest_multiple_sources_combined_integration(ingestion_pipeline, tmp_path, monkeypatch):
    """Verify combining multiple disparate sources (csv, api, kaggle) in a single ingestion call."""
    # 1. Setup CSV source
    csv_file = tmp_path / "local.csv"
    csv_file.write_text("review,score\nCSV item 1,5\n", encoding="utf-8")

    # 2. Setup API mock
    mock_response = MagicMock()
    mock_response.json.return_value = [{"review": "API item 1", "score": 4}]
    mock_response.raise_for_status.return_value = None

    # 3. Setup Kaggle mock
    kaggle_dir = tmp_path / "kaggle_raw"
    kaggle_dir.mkdir()
    kaggle_csv = kaggle_dir / "feed.csv"
    kaggle_csv.write_text("review,score\nKaggle item 1,2\n", encoding="utf-8")

    monkeypatch.setattr(ingestion_pipeline, "download_kaggle_dataset", lambda d, p: str(kaggle_dir))

    sources = [
        {"type": "csv", "name": "source_csv", "path": str(csv_file)},
        {"type": "api", "name": "source_api", "url": "https://api.test/data"},
        {"type": "kaggle", "name": "source_kaggle", "dataset": "d/set", "download_path": str(kaggle_dir), "filename": "feed.csv"},
    ]

    with patch("requests.get", return_value=mock_response) as mock_get:
        combined = ingestion_pipeline.ingest_multiple_sources(sources)

        # Assert requests.get was called with timeout=30
        mock_get.assert_called_once_with("https://api.test/data", headers=None, timeout=30)

        assert len(combined) == 3
        assert list(combined["data_source"]) == ["source_csv", "source_api", "source_kaggle"]
        assert list(combined["review"]) == ["CSV item 1", "API item 1", "Kaggle item 1"]
        assert list(combined["score"]) == [5, 4, 2]


def test_download_kaggle_dataset_invokes_kaggle_api(ingestion_pipeline, monkeypatch, tmp_path):
    """Verify download_kaggle_dataset calls kaggle.api.dataset_download_files correctly."""
    mock_kaggle = MagicMock()
    monkeypatch.setattr(ingestion_pipeline, "_get_kaggle", lambda: mock_kaggle)

    target_dir = str(tmp_path / "raw_downloads")
    result = ingestion_pipeline.download_kaggle_dataset("owner/reviews-dataset", target_dir)

    mock_kaggle.api.dataset_download_files.assert_called_once_with(
        "owner/reviews-dataset",
        path=target_dir,
        unzip=True
    )
    assert result == target_dir


def test_download_kaggle_dataset_error_propagation(ingestion_pipeline, monkeypatch):
    """Verify download_kaggle_dataset logs and re-raises exceptions from Kaggle API."""
    mock_kaggle = MagicMock()
    mock_kaggle.api.dataset_download_files.side_effect = RuntimeError("Kaggle 401 Unauthorized")
    monkeypatch.setattr(ingestion_pipeline, "_get_kaggle", lambda: mock_kaggle)

    with pytest.raises(RuntimeError, match="Kaggle 401 Unauthorized"):
        ingestion_pipeline.download_kaggle_dataset("owner/reviews-dataset", "data/raw")
