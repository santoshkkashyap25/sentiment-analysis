"""Adversarial stress-test suite for DataIngestionPipeline (src/data/ingestion.py).
Executed by Challenger 2 for Milestone 1.
"""

import inspect
import os
from pathlib import Path
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.ingestion import DataIngestionPipeline


@pytest.fixture
def pipeline():
    return DataIngestionPipeline(config={"app_name": "sentiment_analysis"})


# ============================================================================
# Category 1: Adversarial Source Type Testing
# ============================================================================

def test_source_empty_dict_raises_value_error(pipeline):
    """Ingest list with empty dictionary {} raises ValueError('Unsupported source type: None')."""
    with pytest.raises(ValueError) as exc_info:
        pipeline.ingest_multiple_sources([{}])
    assert "Unsupported source type: None" in str(exc_info.value)


def test_source_missing_type_raises_value_error(pipeline):
    """Ingest list with dictionary missing 'type' key raises ValueError."""
    bad_source = {"name": "missing_type_source", "path": "some/path.csv"}
    with pytest.raises(ValueError) as exc_info:
        pipeline.ingest_multiple_sources([bad_source])
    assert "Unsupported source type: None" in str(exc_info.value)


def test_source_explicit_none_type_raises_value_error(pipeline):
    """Ingest list with {'type': None} raises ValueError."""
    bad_source = {"type": None, "name": "none_source"}
    with pytest.raises(ValueError) as exc_info:
        pipeline.ingest_multiple_sources([bad_source])
    assert "Unsupported source type: None" in str(exc_info.value)


@pytest.mark.parametrize("unsupported_type", [
    "parquet", "database", "ftp", "s3", "azure_blob", "gcs", "snowflake", "avro", 123, 3.14, False
])
def test_unsupported_source_types_raise_value_error(pipeline, unsupported_type):
    """Ingest list with unsupported types ('parquet', 'database', 'ftp', 's3', etc.) raises ValueError."""
    bad_source = {"type": unsupported_type, "name": f"source_{unsupported_type}"}
    with pytest.raises(ValueError) as exc_info:
        pipeline.ingest_multiple_sources([bad_source])
    assert f"Unsupported source type: {unsupported_type}" in str(exc_info.value)


def test_unsupported_source_no_data_duplication_or_state_corruption(pipeline, tmp_path):
    """Verify that when an unsupported source follows a valid source:
    1. ValueError is raised.
    2. Pipeline instance state remains uncorrupted.
    3. Re-running with valid sources succeeds without data duplication from failed runs.
    """
    valid_csv = tmp_path / "valid1.csv"
    valid_csv.write_text("id,reviewText\n1,Hello world\n", encoding="utf-8")

    initial_config = dict(pipeline.config)

    # First attempt: valid followed by unsupported 'ftp'
    sources_fail = [
        {"type": "csv", "name": "src1", "path": str(valid_csv)},
        {"type": "ftp", "name": "src_ftp", "path": "ftp://example.com/data.csv"}
    ]
    with pytest.raises(ValueError, match="Unsupported source type: ftp"):
        pipeline.ingest_multiple_sources(sources_fail)

    # Verify pipeline instance state is completely intact
    assert pipeline.config == initial_config
    assert not hasattr(pipeline, "dataframes")  # no leaky accumulators on self

    # Second attempt: purely valid sources
    sources_pass = [
        {"type": "csv", "name": "src1", "path": str(valid_csv)},
    ]
    df = pipeline.ingest_multiple_sources(sources_pass)
    # Verify no duplication occurred
    assert len(df) == 1
    assert df["reviewText"].iloc[0] == "Hello world"
    assert df["data_source"].iloc[0] == "src1"


def test_ingest_empty_list_returns_clean_empty_dataframe(pipeline):
    """Ingest empty list [] returns clean empty DataFrame."""
    result = pipeline.ingest_multiple_sources([])
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert len(result) == 0
    assert len(result.columns) == 0


# ============================================================================
# Category 2: Adversarial API Timeout Testing
# ============================================================================

def test_fetch_api_data_signature_default_timeout(pipeline):
    """Verify that fetch_api_data default parameter is exactly 30."""
    sig = inspect.signature(pipeline.fetch_api_data)
    assert "timeout" in sig.parameters
    assert sig.parameters["timeout"].default == 30


def test_ingest_multiple_sources_api_default_timeout(pipeline):
    """Verify ingest_multiple_sources passes timeout=30 if unspecified in API source dict."""
    with patch.object(pipeline, "fetch_api_data", return_value=pd.DataFrame([{"review": "text"}])) as mock_fetch:
        sources = [{"type": "api", "name": "api_source", "url": "https://api.test/data"}]
        df = pipeline.ingest_multiple_sources(sources)
        mock_fetch.assert_called_once_with("https://api.test/data", headers=None, timeout=30)
        assert len(df) == 1
        assert df["data_source"].iloc[0] == "api_source"


def test_ingest_multiple_sources_api_custom_timeout_override(pipeline):
    """Verify ingest_multiple_sources forwards custom timeout override from source dict."""
    with patch.object(pipeline, "fetch_api_data", return_value=pd.DataFrame([{"review": "text"}])) as mock_fetch:
        sources = [{"type": "api", "name": "api_source", "url": "https://api.test/data", "timeout": 45}]
        df = pipeline.ingest_multiple_sources(sources)
        mock_fetch.assert_called_once_with("https://api.test/data", headers=None, timeout=45)


def test_fetch_api_data_immediate_timeout_live_server(pipeline):
    """Adversarial live network timeout testing:
    Spin up a local HTTP server that sleeps before responding.
    Verify requests.exceptions.Timeout is raised when timeout is small.
    """
    class SlowHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Sleep longer than the client's timeout
            time.sleep(0.3)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'[{"status": "ok"}]')

        def log_message(self, format, *args):
            pass  # Suppress server logs during test

    server = HTTPServer(("127.0.0.1", 0), SlowHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        url = f"http://127.0.0.1:{port}/slow"
        start_time = time.time()
        with pytest.raises(requests.exceptions.Timeout):
            # timeout=0.05s should expire long before 0.3s server response
            pipeline.fetch_api_data(url, timeout=0.05)
        elapsed = time.time() - start_time
        assert elapsed < 2.0, f"Timeout took {elapsed}s, expected < 2.0s"
    finally:
        server.server_close()


def test_fetch_api_data_timeout_fractional_value(pipeline):
    """Verify timeout=0.001 with requests raises requests.exceptions.Timeout."""
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with pytest.raises(requests.exceptions.Timeout):
            pipeline.fetch_api_data("https://httpbin.org/delay/5", timeout=0.001)


# ============================================================================
# Category 3: Adversarial Path Handling
# ============================================================================

def test_load_csv_data_windows_backslash_string_path(pipeline, tmp_path):
    """Test load_csv_data with Windows backslash string path."""
    sub_dir = tmp_path / "win_subdir"
    sub_dir.mkdir()
    csv_file = sub_dir / "win_data.csv"
    csv_file.write_text("col1,col2\nval_a,val_b\n", encoding="utf-8")

    # Construct backslash string path explicitly
    backslash_path = str(csv_file).replace("/", "\\")
    assert "\\" in backslash_path or os.name != "nt"

    df = pipeline.load_csv_data(backslash_path)
    assert len(df) == 1
    assert df["col1"].iloc[0] == "val_a"
    assert df["col2"].iloc[0] == "val_b"


def test_load_csv_data_posix_forward_slash_string_path(pipeline, tmp_path):
    """Test load_csv_data with POSIX forward slash string path."""
    sub_dir = tmp_path / "posix_subdir"
    sub_dir.mkdir()
    csv_file = sub_dir / "posix_data.csv"
    csv_file.write_text("x,y\n10,20\n", encoding="utf-8")

    # Construct forward slash string path explicitly
    posix_path = str(csv_file).replace("\\", "/")
    assert "/" in posix_path

    df = pipeline.load_csv_data(posix_path)
    assert len(df) == 1
    assert df["x"].iloc[0] == 10
    assert df["y"].iloc[0] == 20


def test_load_csv_data_pathlib_path_object(pipeline, tmp_path):
    """Test load_csv_data with pathlib.Path objects directly."""
    csv_file = tmp_path / "pathlib_file.csv"
    csv_file.write_text("name,score\nalice,95\n", encoding="utf-8")

    assert isinstance(csv_file, Path)
    df = pipeline.load_csv_data(csv_file)
    assert len(df) == 1
    assert df["name"].iloc[0] == "alice"
    assert df["score"].iloc[0] == 95


def test_load_csv_data_path_with_spaces_and_special_chars(pipeline, tmp_path):
    """Test load_csv_data with paths containing spaces and non-standard characters."""
    special_dir = tmp_path / "special folder with spaces & (symbols)"
    special_dir.mkdir()
    csv_file = special_dir / "data#1 [2026].csv"
    csv_file.write_text("a,b\n1,2\n", encoding="utf-8")

    df_str = pipeline.load_csv_data(str(csv_file))
    assert len(df_str) == 1

    df_path = pipeline.load_csv_data(csv_file)
    assert len(df_path) == 1


def test_download_kaggle_dataset_pathlib_input(pipeline, monkeypatch, tmp_path):
    """Verify download_kaggle_dataset converts pathlib.Path input to str for kaggle API."""
    mock_kaggle = MagicMock()
    monkeypatch.setattr(pipeline, "_get_kaggle", lambda: mock_kaggle)

    target_path = tmp_path / "kaggle_target_folder"
    res = pipeline.download_kaggle_dataset("sentiment/dataset", target_path)

    # Kaggle API expects path parameter as str
    mock_kaggle.api.dataset_download_files.assert_called_once_with(
        "sentiment/dataset",
        path=str(target_path),
        unzip=True
    )
    assert res == str(target_path)
    assert isinstance(res, str)


def test_ingest_multiple_sources_kaggle_os_path_join_behavior(pipeline, tmp_path, monkeypatch):
    """Verify ingest_multiple_sources uses os.path.join properly on Windows with Kaggle paths."""
    kaggle_dir = tmp_path / "raw_kaggle_dir"
    kaggle_dir.mkdir()
    data_file = kaggle_dir / "sentiment_stream.csv"
    data_file.write_text("id,sentiment\n1,Positive\n2,Negative\n", encoding="utf-8")

    # Simulate download_kaggle_dataset returning kaggle_dir as str (Windows style backslashes)
    win_dir_str = str(kaggle_dir).replace("/", "\\")
    monkeypatch.setattr(pipeline, "download_kaggle_dataset", lambda d, p: win_dir_str)

    source = {
        "type": "kaggle",
        "name": "kaggle_feed",
        "dataset": "org/sentiment-feed",
        "download_path": win_dir_str,
        "filename": "sentiment_stream.csv"
    }

    df = pipeline.ingest_multiple_sources([source])
    assert len(df) == 2
    assert list(df["sentiment"]) == ["Positive", "Negative"]
    assert list(df["data_source"]) == ["kaggle_feed", "kaggle_feed"]


def test_empty_csv_and_header_only_csv(pipeline, tmp_path):
    """Verify behavior on empty CSV file and header-only CSV file."""
    header_only_csv = tmp_path / "header_only.csv"
    header_only_csv.write_text("col1,col2,col3\n", encoding="utf-8")

    df = pipeline.load_csv_data(header_only_csv)
    assert len(df) == 0
    assert list(df.columns) == ["col1", "col2", "col3"]

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(pd.errors.EmptyDataError):
        pipeline.load_csv_data(empty_csv)


# ============================================================================
# Category 4: Deep Adversarial Stress Scenarios
# ============================================================================

def test_api_invalid_json_response_raises(pipeline):
    """Verify that if API returns non-JSON body, JSONDecodeError is propagated."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "doc", 0)

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.exceptions.JSONDecodeError):
            pipeline.fetch_api_data("https://api.test/bad_json")


def test_api_http_500_propagates_in_ingest_multiple_sources(pipeline):
    """Verify HTTP 500 error from API halts ingest_multiple_sources cleanly without state leaks."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")

    with patch("requests.get", return_value=mock_response):
        sources = [
            {"type": "api", "name": "failing_api", "url": "https://api.test/fail"}
        ]
        with pytest.raises(requests.exceptions.HTTPError, match="500 Internal Server Error"):
            pipeline.ingest_multiple_sources(sources)


def test_kaggle_trailing_slashes_path_resolution(pipeline, tmp_path, monkeypatch):
    """Verify os.path.join handles directory paths with trailing slashes correctly."""
    target_dir = tmp_path / "trailing_dir"
    target_dir.mkdir()
    feed_file = target_dir / "target_feed.csv"
    feed_file.write_text("id,text\n99,Success\n", encoding="utf-8")

    # Path with trailing slash
    trailing_path_str = str(target_dir) + os.sep
    monkeypatch.setattr(pipeline, "download_kaggle_dataset", lambda d, p: trailing_path_str)

    source = {
        "type": "kaggle",
        "name": "trailing_kaggle",
        "dataset": "owner/ds",
        "download_path": trailing_path_str,
        "filename": "target_feed.csv"
    }

    df = pipeline.ingest_multiple_sources([source])
    assert len(df) == 1
    assert df["text"].iloc[0] == "Success"
    assert df["data_source"].iloc[0] == "trailing_kaggle"


def test_heterogeneous_schema_ingestion(pipeline, tmp_path):
    """Verify ingest_multiple_sources combines sources with non-identical schemas cleanly."""
    csv1 = tmp_path / "schema1.csv"
    csv1.write_text("id,reviewText,rating\n1,Good,5\n", encoding="utf-8")

    csv2 = tmp_path / "schema2.csv"
    csv2.write_text("id,text,verified\n2,Bad,True\n", encoding="utf-8")

    sources = [
        {"type": "csv", "name": "source1", "path": str(csv1)},
        {"type": "csv", "name": "source2", "path": str(csv2)},
    ]

    combined = pipeline.ingest_multiple_sources(sources)
    assert len(combined) == 2
    assert "data_source" in combined.columns
    assert set(combined["data_source"]) == {"source1", "source2"}
    # Verify both columns exist with appropriate NaNs
    assert "reviewText" in combined.columns
    assert "text" in combined.columns
    assert pd.isna(combined.loc[combined["data_source"] == "source2", "reviewText"].iloc[0])
    assert pd.isna(combined.loc[combined["data_source"] == "source1", "text"].iloc[0])


def test_high_volume_sources_stress(pipeline, tmp_path):
    """Stress test with 100 individual sources to verify scalability and memory handling."""
    csv_file = tmp_path / "batch.csv"
    csv_file.write_text("id,value\n1,val\n", encoding="utf-8")

    sources = [
        {"type": "csv", "name": f"source_{i}", "path": str(csv_file)}
        for i in range(100)
    ]

    combined = pipeline.ingest_multiple_sources(sources)
    assert len(combined) == 100
    assert len(combined["data_source"].unique()) == 100
