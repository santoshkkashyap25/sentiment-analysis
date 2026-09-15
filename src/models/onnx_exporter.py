"""ONNX Model Exporter and 8-bit Dynamic Quantizer for Lean Cloud Deployment.

Converts fine-tuned Hugging Face transformer models into optimized ONNX format
and applies INT8 dynamic quantization to fit comfortably within 512 MB RAM environments.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fix Windows console charmap encoding issues for torch.onnx output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


class OnnxExporter:
    """Exports PyTorch Transformers to ONNX and creates INT8 quantized models."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the PyTorch model and tokenizer from model directory."""
        logger.info(f"Loading model and tokenizer from {self.model_dir} for ONNX export...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.model.to("cpu")  # ONNX export on CPU for clean graph trace
        return self

    def export_to_onnx(
        self,
        output_onnx_path: Optional[str] = None,
        max_length: int = 256
    ) -> str:
        """Export PyTorch transformer model to ONNX with dynamic batch and sequence axes."""
        if self.model is None:
            self.load_model()

        if output_onnx_path is None:
            output_onnx_path = str(self.model_dir / "model.onnx")

        logger.info(f"Tracing and exporting PyTorch model to ONNX: {output_onnx_path}")

        # Dummy inputs for graph tracing
        dummy_text = "This is a great product with wonderful quality!"
        inputs = self.tokenizer(
            dummy_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_names = ["input_ids", "attention_mask"]
        dummy_inputs = (inputs["input_ids"], inputs["attention_mask"])

        # Check if model takes token_type_ids (e.g. BERT vs RoBERTa/DistilBERT)
        forward_varnames = getattr(getattr(self.model, "forward", None), "__code__", None)
        if "token_type_ids" in inputs and forward_varnames and "token_type_ids" in forward_varnames.co_varnames:
            input_names.append("token_type_ids")
            dummy_inputs = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

        output_names = ["logits"]

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"}
        }
        if "token_type_ids" in input_names:
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

        # Perform ONNX export with external_data=False for single-file artifact
        torch.onnx.export(
            self.model,
            dummy_inputs,
            output_onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
            external_data=False,
            do_constant_folding=True
        )

        file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
        logger.info(f"ONNX model exported successfully: {output_onnx_path} ({file_size_mb:.2f} MB)")
        return output_onnx_path

    def quantize_int8(
        self,
        input_onnx_path: Optional[str] = None,
        output_quantized_path: Optional[str] = None
    ) -> str:
        """Apply INT8 Dynamic Quantization to ONNX model, shrinking footprint by ~75%."""
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError as e:
            raise ImportError(
                "onnx and onnxruntime are required for INT8 quantization. Install via 'pip install onnx onnxruntime'."
            ) from e

        if input_onnx_path is None:
            input_onnx_path = str(self.model_dir / "model.onnx")

        if output_quantized_path is None:
            output_quantized_path = str(self.model_dir / "model_quantized.onnx")

        logger.info(f"Applying INT8 dynamic quantization: {input_onnx_path} -> {output_quantized_path}")

        # Clear existing value_info to allow clean shape inference during Gemm-to-MatMul replacement
        model_proto = onnx.load(input_onnx_path)
        model_proto.graph.ClearField("value_info")
        clean_temp_path = input_onnx_path + ".clean.onnx"
        onnx.save(model_proto, clean_temp_path)

        try:
            quantize_dynamic(
                model_input=clean_temp_path,
                model_output=output_quantized_path,
                weight_type=QuantType.QInt8
            )
        finally:
            if os.path.exists(clean_temp_path):
                try:
                    os.remove(clean_temp_path)
                except Exception:
                    pass

        orig_mb = os.path.getsize(input_onnx_path) / (1024 * 1024)
        quant_mb = os.path.getsize(output_quantized_path) / (1024 * 1024)
        reduction_pct = (1 - quant_mb / orig_mb) * 100

        logger.info(
            f"INT8 Quantization complete! Size reduced from {orig_mb:.2f} MB to {quant_mb:.2f} MB "
            f"(-{reduction_pct:.1f}% reduction, ideal for 512 MB RAM environments)."
        )
        return output_quantized_path

    def verify_parity(
        self,
        quantized_onnx_path: Optional[str] = None,
        test_samples: Optional[list] = None
    ) -> Dict[str, Any]:
        """Verify inference probability parity between PyTorch and Quantized ONNX."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, skipping parity verification.")
            return {"verified": False, "reason": "onnxruntime missing"}

        if self.model is None:
            self.load_model()

        if quantized_onnx_path is None:
            quantized_onnx_path = str(self.model_dir / "model_quantized.onnx")

        if test_samples is None:
            test_samples = [
                "This product is wonderful, exceeded all my expectations!",
                "Terrible experience, arrived damaged and support never replied.",
                "It works fine, nothing extraordinary but serves the purpose."
            ]

        session = ort.InferenceSession(quantized_onnx_path, providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in session.get_inputs()]

        results = []
        max_delta = 0.0

        for text in test_samples:
            encoded = self.tokenizer(text, max_length=256, truncation=True, return_tensors="pt")

            # 1. PyTorch Logits & Probs
            with torch.no_grad():
                pt_out = self.model(**encoded)
                pt_probs = torch.softmax(pt_out.logits, dim=-1).cpu().numpy()[0]

            # 2. ONNX Logits & Probs
            onnx_inputs = {
                "input_ids": encoded["input_ids"].numpy(),
                "attention_mask": encoded["attention_mask"].numpy()
            }
            if "token_type_ids" in input_names and "token_type_ids" in encoded:
                onnx_inputs["token_type_ids"] = encoded["token_type_ids"].numpy()

            onnx_out = session.run(None, onnx_inputs)[0]
            # Softmax calculation using numpy
            exp_logits = np.exp(onnx_out[0] - np.max(onnx_out[0]))
            onnx_probs = exp_logits / np.sum(exp_logits)

            delta = float(max(abs(pt_probs - onnx_probs)))
            max_delta = max(max_delta, delta)

            results.append({
                "text": text,
                "pytorch_probs": [float(p) for p in pt_probs],
                "onnx_probs": [float(p) for p in onnx_probs],
                "max_abs_delta": delta,
                "pytorch_pred": int(pt_probs.argmax()),
                "onnx_pred": int(onnx_probs.argmax())
            })

        logger.info(f"Parity verification complete across {len(test_samples)} samples. Max prob delta: {max_delta:.4f}")
        return {
            "verified": True,
            "max_abs_delta": max_delta,
            "predictions_match": all(r["pytorch_pred"] == r["onnx_pred"] for r in results),
            "samples": results
        }
