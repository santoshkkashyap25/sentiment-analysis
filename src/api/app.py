"""FastAPI application for sentiment analysis predictions with embedded MLOps dashboard."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
import logging
import os
import sqlite3
from datetime import datetime
import time
import threading
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from src.features.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.monitoring.drift_detection import ModelMonitor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Customer Feedback Analysis & MLOps API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request schemas
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer feedback text to analyze")

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str
    model_version: str = "1.0"
    processed_text: Optional[str] = None
    response_time_ms: float = 0.0
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str = "1.0"

class StatsResponse(BaseModel):
    total_requests: int
    uptime: str
    model_info: Dict[str, Any]

request_count = 0
request_lock = threading.Lock()
monitor = ModelMonitor(db_path=str(BASE_DIR / 'monitoring.db'))


class FeedbackAnalysisAPI:
    """Main API class for feedback analysis inference with tiered model loading."""

    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.preprocessor = DataPreprocessor({})
        self.onnx_session = None
        self.onnx_tokenizer = None
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.calibrator = None
        self.device = None
        self.is_onnx = False
        self.is_transformer = False
        self.load_model_and_extractors()

    def load_model_and_extractors(self):
        """Tiered model loading: 1) ONNX INT8 Runtime, 2) PyTorch Transformer, 3) Classical GBDT."""
        transformer_dir = BASE_DIR / 'data' / 'models' / 'transformer_sentiment'

        # Tier 1: Check for Quantized INT8 ONNX Model (Ultra-Lean CPU Execution, ~230 MB RAM)
        quantized_path = transformer_dir / 'model_quantized.onnx'
        if quantized_path.exists() and (transformer_dir / 'tokenizer_config.json').exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                from src.models.threshold_calibration import ThresholdCalibrator

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.onnx_session = ort.InferenceSession(
                    str(quantized_path),
                    sess_options=opts,
                    providers=['CPUExecutionProvider']
                )
                self.onnx_tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))

                calib_path = transformer_dir / 'calibration.json'
                if calib_path.exists():
                    self.calibrator = ThresholdCalibrator.load(str(calib_path))
                    logger.info("Loaded decision threshold calibration for ONNX Runtime")
                else:
                    self.calibrator = ThresholdCalibrator()

                self.is_onnx = True
                logger.info(f"Quantized INT8 ONNX Model loaded successfully from {quantized_path} (Ultra-Lean Runtime)")
            except Exception as e:
                logger.warning(f"Error loading ONNX model: {e}. Checking PyTorch checkpoint.")

        # Tier 2: Check for fine-tuned PyTorch Transformer model
        if not self.is_onnx and transformer_dir.exists() and (transformer_dir / 'config.json').exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from src.models.threshold_calibration import ThresholdCalibrator

                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(str(transformer_dir))
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(str(transformer_dir)).to(self.device)
                self.transformer_model.eval()

                calib_path = transformer_dir / 'calibration.json'
                if calib_path.exists():
                    self.calibrator = ThresholdCalibrator.load(str(calib_path))
                    logger.info("Loaded decision threshold calibration config")
                else:
                    self.calibrator = ThresholdCalibrator()

                self.is_transformer = True
                dev_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'
                logger.info(f"PyTorch Transformer loaded successfully on {dev_name} (Calibrated)")
            except Exception as e:
                logger.warning(f"Error loading Transformer model: {e}. Falling back to classical model.")

        # Tier 3: Classical Scikit-Learn model & feature extractors
        try:
            if not self.is_onnx and not self.is_transformer:
                model_path = os.getenv('MODEL_PATH', str(BASE_DIR / 'data' / 'models' / 'best_model.pkl'))
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"Model loaded successfully from {model_path}")
                else:
                    logger.warning(f"Model file not found at {model_path}")

            # Always load feature extractors if available (for drift detection & simulation)
            extractors_path = os.getenv('EXTRACTORS_PATH', str(BASE_DIR / 'data' / 'models' / 'feature_extractors.pkl'))
            if os.path.exists(extractors_path):
                with open(extractors_path, 'rb') as f:
                    extractors = pickle.load(f)

                self.feature_engineer = FeatureEngineer({})
                self.feature_engineer.tfidf_vectorizer = extractors.get('tfidf_vectorizer')
                self.feature_engineer.sentence_transformer = extractors.get('sentence_transformer')
                self.feature_engineer.svd = extractors.get('svd')
                logger.info("Feature extractors loaded successfully")
            else:
                logger.warning(f"Extractors file not found at {extractors_path}")

        except Exception as e:
            logger.error(f"Error loading model/extractors: {e}")

    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        return self.preprocessor.clean_text(text)

    def extract_features(self, text: str) -> np.ndarray:
        """Extract features from preprocessed text without target leakage"""
        df = pd.DataFrame({'reviewText_clean': [text]})

        feature_df = self.feature_engineer.extract_basic_features(df, 'reviewText_clean')
        tfidf_features = self.feature_engineer.extract_tfidf_features([text], is_training=False)

        embeddings = None
        if self.feature_engineer.svd is not None and self.feature_engineer.sentence_transformer is not None:
            try:
                raw_emb = self.feature_engineer.extract_sentence_embeddings([text])
                embeddings = self.feature_engineer.reduce_dimensionality(raw_emb, n_components=50, is_training=False)
            except Exception as e:
                logger.warning(f"Embeddings generation skipped: {e}")

        features = self.feature_engineer.combine_features(feature_df, tfidf_features, embeddings)
        return features

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for given feedback text using ONNX, Transformer or Classical model"""
        try:
            if not self.is_onnx and not self.is_transformer and (self.model is None or self.feature_engineer is None):
                self.load_model_and_extractors()

            clean_text = self.preprocess_text(text)

            if not clean_text.strip():
                return {
                    'sentiment': 'Neutral',
                    'confidence': 0.33,
                    'probabilities': {'Negative': 0.33, 'Neutral': 0.34, 'Positive': 0.33},
                    'processed_text': '',
                    'error': 'Empty text after cleaning'
                }

            # A. ONNX Runtime INT8 Branch (Ultra-Lean CPU Execution, ~230 MB RAM)
            if self.is_onnx and self.onnx_session is not None:
                encoded = self.onnx_tokenizer(
                    text,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors='np'
                )
                input_names = [inp.name for inp in self.onnx_session.get_inputs()]
                onnx_inputs = {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"]
                }
                if "token_type_ids" in input_names and "token_type_ids" in encoded:
                    onnx_inputs["token_type_ids"] = encoded["token_type_ids"]

                raw_out = self.onnx_session.run(None, onnx_inputs)[0][0]
                exp_logits = np.exp(raw_out - np.max(raw_out))
                probs = exp_logits / np.sum(exp_logits)

                prob_dict = {
                    'Negative': round(float(probs[0]), 4),
                    'Neutral': round(float(probs[1]), 4),
                    'Positive': round(float(probs[2]), 4)
                }

                if self.calibrator:
                    sentiment, confidence = self.calibrator.predict_single(prob_dict)
                else:
                    pred_idx = int(np.argmax(probs))
                    sentiment = ['Negative', 'Neutral', 'Positive'][pred_idx]
                    confidence = round(float(np.max(probs)), 4)

                pred_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
                return {
                    'prediction_id': pred_map.get(sentiment, 1),
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': prob_dict,
                    'processed_text': clean_text,
                    'runtime': 'onnx-int8-quantized',
                    'error': None
                }

            # B. Transformer Inference Branch (GPU Accelerated)
            if self.is_transformer and self.transformer_model is not None:
                import torch
                with torch.no_grad():
                    inputs = self.transformer_tokenizer(
                        text,
                        truncation=True,
                        max_length=128,
                        padding=True,
                        return_tensors='pt'
                    ).to(self.device)
                    outputs = self.transformer_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                prob_dict = {
                    'Negative': round(float(probs[0]), 4),
                    'Neutral': round(float(probs[1]), 4),
                    'Positive': round(float(probs[2]), 4)
                }

                if self.calibrator:
                    sentiment, confidence = self.calibrator.predict_single(prob_dict)
                else:
                    pred_idx = int(np.argmax(probs))
                    sentiment = ['Negative', 'Neutral', 'Positive'][pred_idx]
                    confidence = round(float(np.max(probs)), 4)

                pred_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
                return {
                    'prediction_id': pred_map.get(sentiment, 1),
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': prob_dict,
                    'processed_text': clean_text,
                    'runtime': 'pytorch-transformer',
                    'error': None
                }

            # C. Classical ML Branch (Gradient Boosting / Logistic Regression)
            if self.model is None:
                raise RuntimeError("Model is not loaded. Run pipeline.py first.")

            features = self.extract_features(clean_text)
            prediction = int(self.model.predict(features)[0])

            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                prob_dict = {
                    'Negative': round(float(probs[0]), 4),
                    'Neutral': round(float(probs[1]), 4),
                    'Positive': round(float(probs[2]), 4)
                }
                confidence = round(float(np.max(probs)), 4)
            else:
                prob_dict = {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 0.0}
                prob_dict[['Negative', 'Neutral', 'Positive'][prediction]] = 1.0
                confidence = 1.0

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = sentiment_map.get(prediction, 'Neutral')

            return {
                'prediction_id': prediction,
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': prob_dict,
                'processed_text': clean_text,
                'error': None
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'prediction_id': 1,
                'sentiment': 'Unknown',
                'confidence': 0.0,
                'probabilities': {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 0.0},
                'processed_text': text,
                'error': str(e)
            }


api = FeedbackAnalysisAPI()
start_time_server = datetime.now()


@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    """Serve embedded modern web dashboard"""
    index_file = TEMPLATE_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard loading...</h1>")
    return FileResponse(str(index_file))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_loaded = (api.onnx_session is not None or api.transformer_model is not None or api.model is not None)
    version = '3.0-ONNX-INT8' if api.is_onnx else ('2.0-Transformer' if api.is_transformer else '1.0-Classical')
    return {
        'status': 'healthy' if is_loaded else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': is_loaded,
        'model_version': version
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """Real-time single feedback sentiment prediction"""
    global request_count

    with request_lock:
        request_count += 1

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must be a non-empty string")

    t0 = time.time()
    result = api.predict_sentiment(payload.text)
    latency_sec = time.time() - t0
    latency_ms = round(latency_sec * 1000, 2)

    # Log to SQLite monitoring database
    try:
        prediction_int = result.get('prediction_id', 1)
        monitor.log_prediction(
            input_text=payload.text[:500],
            prediction=prediction_int,
            confidence=result.get('confidence', 0.0),
            response_time=latency_sec
        )
    except Exception as e:
        logger.warning(f"Could not log prediction to monitoring DB: {e}")

    result['timestamp'] = datetime.now().isoformat()
    result['model_version'] = '3.0-ONNX-INT8' if api.is_onnx else ('2.0-Transformer' if api.is_transformer else '1.0')
    result['response_time_ms'] = latency_ms

    return result


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics"""
    model_type = 'ONNX-INT8-Quantized' if api.is_onnx else ('PyTorch-Transformer' if api.is_transformer else (type(api.model).__name__ if api.model else 'None'))
    version = '3.0-ONNX-INT8' if api.is_onnx else ('2.0-Transformer' if api.is_transformer else '1.0-Classical')
    return {
        'total_requests': request_count,
        'uptime': str(datetime.now() - start_time_server),
        'model_info': {
            'version': version,
            'type': model_type,
            'classes': ['Negative', 'Neutral', 'Positive']
        }
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Query monitoring.db to return live MLOps telemetry and drift metrics"""
    db_path = BASE_DIR / "monitoring.db"
    if not db_path.exists():
        return {
            "total_predictions": 0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "sentiment_counts": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "recent_predictions": [],
            "drift_alerts": []
        }

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query recent predictions
        cursor.execute('''
            SELECT id, timestamp, input_text, prediction, confidence, response_time
            FROM predictions
            ORDER BY id DESC LIMIT 20
        ''')
        rows = cursor.fetchall()

        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        recent = []
        latencies = []
        counts = {"Negative": 0, "Neutral": 0, "Positive": 0}

        for r in rows:
            lbl = sentiment_labels.get(r["prediction"], "Neutral")
            counts[lbl] = counts.get(lbl, 0) + 1
            lat = round(float(r["response_time"] or 0) * 1000, 2)
            latencies.append(lat)
            recent.append({
                "id": r["id"],
                "timestamp": str(r["timestamp"]),
                "text": r["input_text"],
                "sentiment": lbl,
                "confidence": round(float(r["confidence"] or 0), 4),
                "latency_ms": lat
            })

        # Overall count and latencies
        cursor.execute("SELECT COUNT(*), AVG(response_time) FROM predictions")
        tot_count, avg_resp = cursor.fetchone()
        avg_latency_ms = round((avg_resp or 0) * 1000, 2)

        # Percentiles
        cursor.execute("SELECT response_time FROM predictions WHERE response_time IS NOT NULL ORDER BY response_time")
        all_lats = [float(row[0]) * 1000 for row in cursor.fetchall()]
        p50 = round(float(np.percentile(all_lats, 50)), 2) if all_lats else 0.0
        p95 = round(float(np.percentile(all_lats, 95)), 2) if all_lats else 0.0

        # Query drift alerts
        cursor.execute('''
            SELECT id, timestamp, drift_type, metric_value, threshold, severity
            FROM drift_alerts
            ORDER BY id DESC LIMIT 10
        ''')
        alerts = [
            {
                "id": a["id"],
                "timestamp": str(a["timestamp"]),
                "type": a["drift_type"],
                "value": round(float(a["metric_value"]), 4),
                "threshold": round(float(a["threshold"]), 4),
                "severity": a["severity"]
            }
            for a in cursor.fetchall()
        ]

        conn.close()

        return {
            "total_predictions": tot_count or len(recent),
            "avg_latency_ms": avg_latency_ms,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "sentiment_counts": counts,
            "recent_predictions": recent,
            "drift_alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error fetching telemetry: {e}")
        return {"error": str(e)}


@app.post("/api/simulate-drift")
async def simulate_drift():
    """Simulate out-of-distribution drift traffic to test the MLOps detector"""
    try:
        # Generate synthetic out-of-distribution reviews (anomalous machine crash telemetry / foreign text)
        ood_samples = [
            "FATAL ERR: 0x8849F buffer overflow at segment 12 offset 992",
            "WARNING device voltage spike 4.88V on bus controller #04",
            "NullPointerException at com.system.kernel.Driver.init() line 44",
            "Hardware interrupt 0xEE received without valid handler",
            "SYNTAX ERROR invalid token near line 1 position 482",
            "Thermal threshold exceeded: CPU temp 98C cooling fan inactive",
            "Segmentation fault core dumped /usr/bin/daemon_worker",
            "Invalid checksum in header block 0xFA39B92 expected 0x0000",
            "Connection timeout on socket 192.168.1.1:9092 after 30000ms",
            "Stack overflow in recursive subroutine handler index 9"
        ] * 4

        # Extract features for OOD samples
        if api.feature_engineer is None:
            api.load_model_and_extractors()

        ood_clean = [api.preprocess_text(s) for s in ood_samples]
        ood_features = api.feature_engineer.extract_tfidf_features(ood_clean, is_training=False)

        # Load reference baseline features
        ref_path = BASE_DIR / "data" / "processed" / "features_reference.pkl"
        if ref_path.exists():
            with open(str(ref_path), "rb") as f:
                ref_features = pickle.load(f)
        else:
            # Fallback baseline
            ref_clean = ["great product high quality fast shipping wonderful experience love it"] * 40
            ref_features = api.feature_engineer.extract_tfidf_features(ref_clean, is_training=False)

        # Run statistical drift detection
        drift_result = monitor.detect_data_drift(ref_features, ood_features, threshold=0.05)

        return {
            "status": "simulation_complete",
            "drift_detected": drift_result["overall_drift"],
            "drift_percentage": round(drift_result["drift_percentage"] * 100, 2),
            "drifted_features_count": drift_result["drift_detected_features"],
            "total_features_tested": drift_result["total_features"],
            "message": "Data drift detected! Statistical alert logged." if drift_result["overall_drift"] else "No drift detected."
        }
    except Exception as e:
        logger.error(f"Error simulating drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
