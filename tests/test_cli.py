"""Unit tests for SentiPulse CLI interface."""

import json
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sentipulse.cli import main


def test_cli_analyze_command(capsys):
    """Verify 'sentipulse analyze' formats and outputs sentiment data."""
    mock_res = {
        "sentiment": "Positive",
        "confidence": 0.92,
        "probabilities": {"Negative": 0.03, "Neutral": 0.05, "Positive": 0.92},
        "escalation_recommended": False,
        "response_time_ms": 32.1
    }

    with patch("sentipulse.client.SentiPulse.analyze", return_value=mock_res):
        main(["analyze", "Wonderful customer service!"])
        captured = capsys.readouterr()
        assert "SentiPulse Analysis" in captured.out
        assert "Sentiment:   Positive" in captured.out
        assert "Confidence:  92.0%" in captured.out
        assert "Latency:     32.1 ms" in captured.out


def test_cli_analyze_json_flag(capsys):
    """Verify 'sentipulse analyze --json' outputs raw JSON parseable string."""
    mock_res = {
        "sentiment": "Negative",
        "confidence": 0.96,
        "probabilities": {"Negative": 0.96, "Neutral": 0.02, "Positive": 0.02},
        "escalation_recommended": True,
        "response_time_ms": 25.0
    }

    with patch("sentipulse.client.SentiPulse.analyze", return_value=mock_res):
        main(["analyze", "Awful product!", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["sentiment"] == "Negative"
        assert data["escalation_recommended"] is True


def test_cli_batch_csv_processing(tmp_path):
    """Verify 'sentipulse batch' processes CSV rows and saves output CSV."""
    input_csv = tmp_path / "reviews.csv"
    output_csv = tmp_path / "reviews_analyzed.csv"

    df_in = pd.DataFrame({
        "reviewText": ["Terrific shoes!", "Broken buckle arrived."]
    })
    df_in.to_csv(input_csv, index=False)

    def mock_analyze(text):
        if "Terrific" in text:
            return {"sentiment": "Positive", "confidence": 0.91, "escalation_recommended": False}
        return {"sentiment": "Negative", "confidence": 0.95, "escalation_recommended": True}

    with patch("sentipulse.client.SentiPulse.analyze", side_effect=mock_analyze):
        main(["batch", str(input_csv), "-o", str(output_csv)])

    assert output_csv.exists()
    df_out = pd.read_csv(output_csv)
    assert len(df_out) == 2
    assert "sentiment" in df_out.columns
    assert "confidence" in df_out.columns
    assert "escalate" in df_out.columns
    assert df_out["sentiment"].iloc[0] == "Positive"
    assert df_out["sentiment"].iloc[1] == "Negative"
    assert df_out["escalate"].iloc[1] == True
