"""Unit tests for ONNX model export, INT8 quantization, and API ONNX runtime."""

import os
from pathlib import Path
import pytest
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.onnx_exporter import OnnxExporter
from src.api.app import FeedbackAnalysisAPI


@pytest.fixture(scope="module")
def transformer_model_dir():
    """Ensure transformer sentiment model directory exists with trained model."""
    model_dir = Path("data/models/transformer_sentiment")
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        pytest.skip("No trained transformer model found at data/models/transformer_sentiment")
    return str(model_dir)


def test_onnx_exporter_init(transformer_model_dir):
    """Test OnnxExporter initialization."""
    exporter = OnnxExporter(transformer_model_dir)
    assert exporter.model_dir == Path(transformer_model_dir)
    assert exporter.tokenizer is None
    assert exporter.model is None


def test_onnx_export_and_quantize(transformer_model_dir, tmp_path):
    """Test end-to-end ONNX export and INT8 dynamic quantization."""
    exporter = OnnxExporter(transformer_model_dir)
    exporter.load_model()
    assert exporter.tokenizer is not None
    assert exporter.model is not None

    test_onnx = str(tmp_path / "test_model.onnx")
    test_quant = str(tmp_path / "test_model_quantized.onnx")

    # 1. Export
    exported_path = exporter.export_to_onnx(output_onnx_path=test_onnx, max_length=64)
    assert os.path.exists(exported_path)
    assert os.path.getsize(exported_path) > 10 * 1024 * 1024  # At least 10 MB

    # 2. Quantize
    quantized_path = exporter.quantize_int8(input_onnx_path=test_onnx, output_quantized_path=test_quant)
    assert os.path.exists(quantized_path)
    assert os.path.getsize(quantized_path) < os.path.getsize(exported_path)

    # 3. Parity Check
    parity = exporter.verify_parity(
        quantized_onnx_path=test_quant,
        test_samples=[
            "This product exceeded all my expectations, wonderful quality!",
            "Terrible experience, broken on arrival and support never answered."
        ]
    )
    assert parity["verified"] is True
    assert parity["max_abs_delta"] < 0.25  # Quantization tolerance


def test_api_tiered_loading_and_predict(transformer_model_dir):
    """Test that FeedbackAnalysisAPI correctly prioritizes ONNX or PyTorch transformer."""
    api = FeedbackAnalysisAPI()
    assert api.is_onnx or api.is_transformer

    # Perform a live prediction
    result = api.predict_sentiment("The build quality is outstanding and durable!")
    assert result["error"] is None
    assert result["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert "probabilities" in result
    assert result["probabilities"]["Positive"] > 0.0
