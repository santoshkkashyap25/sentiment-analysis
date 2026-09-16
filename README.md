# SentiPulse

> **Production-grade sentiment classification engine & real-time MLOps drift monitor powered by INT8 quantized RoBERTa with calibrated negative recall (>90%).**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![RoBERTa](https://img.shields.io/badge/Model-RoBERTa--base-yellow.svg)](https://huggingface.co/roberta-base)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](tests/)

---

## Overview

Most open-source sentiment models suffer from two critical limitations in real-world customer support pipelines:

1. **Standard Argmax Bias & Missed Complaints**: Conventional cross-entropy classifiers optimize for overall accuracy, frequently missing subtle, sarcastic, or low-confidence negative feedback. In customer operations, a false negative (failing to detect an unhappy customer) is significantly costlier than a false positive.
2. **Excessive Resource Footprint**: Full-precision transformer models (`torch` + `roberta-base`) demand 500 MB to 1.2 GB of RAM and dedicated GPUs, making them expensive or impossible to run on free/low-cost cloud tiers (e.g., Render's 512 MB RAM free tier).

**SentiPulse** solves both problems:
- **Fine-Tuned RoBERTa-base**: Trained across **120,000 customer reviews** with Cosine Annealing and Stochastic Weight Averaging (SWA).
- **Post-Training Decision Calibration**: Empirically calibrated decision boundaries ($\tau_{\text{neg}} = 0.200$) lift negative class recall from **76.40% to 90.34%** without destabilizing positive precision.
- **Dynamic INT8 Quantization**: Exported to ONNX and dynamically quantized (`QuantType.QInt8`), slashing model size by **75%** (120 MB uncompressed, compressed to **85 MB** in Git to bypass GitHub's 100 MB limit) and running in **~230 MB RAM on CPU** at ~20–35 ms inference latency.
- **Real-Time MLOps Telemetry & Drift Monitoring**: Integrated two-sample **Kolmogorov-Smirnov (KS) test** ($p < 0.05$) continuously testing live feedback query windows against training feature distributions to automatically alert before concept drift degrades downstream systems.

---

## System Architecture

```text
[ Incoming Customer Feedback ]
               │
               ▼
   [ Data Preprocessing Pipeline ]
   (HTML strip, lowercasing, stop-words, lemmatization)
               │
               ▼
 ┌────────────────────────────────────────┐
 │   SentiPulse Inference Engine (FastAPI)│
 │                                        │
 │   1. Tokenizer (RoBERTa Fast BPE)      │
 │   2. ONNX Runtime (INT8 CPU Session)   │
 │   3. Raw Logits ──► Calibrated Softmax │
 │      (τ_neg = 0.200, τ_pos = 0.450)    │
 └───────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[ Prediction Output ]   [ SQLite Audit Log ]
(Sentiment, Conf,       (Latency, Query Text,
 Probabilities, Tokens)  Timestamps)
                                 │
                                 ▼
                    [ Live KS Drift Detector ]
                    (2-Sample KS Test: p < 0.05
                     Live vs. Training Corpus)
                                 │
                                 ▼
                    [ Dual-Tab Web Dashboard ]
                    (Analyzer UI + MLOps Telemetry)
```

---

## 📊 Validated Benchmarks

All metrics evaluated on a held-out test set of **12,000 Amazon customer reviews** using an NVIDIA GeForce RTX 5050 Laptop GPU for training and CPU for INT8 ONNX deployment:

| Metric | Baseline (DistilBERT) | Uncalibrated RoBERTa | SentiPulse (RoBERTa INT8 + Calibrated) | Production Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 81.70% | 84.80% | **84.80%** | +3.10% overall accuracy |
| **Weighted F1** | 81.50% | 84.58% | **84.58%** | Reliable multi-class balance |
| **Negative Recall** | 76.40% | 79.10% | **90.34%** | **+13.94% customer complaints caught** |
| **Negative F1** | 73.20% | 76.80% | **81.16%** | High-precision triage for churn prevention |
| **Artifact Size** | 268 MB | 498 MB | **85.3 MB (.gz) / 120.6 MB** | Under GitHub 100 MB limit |
| **Memory Footprint** | ~550 MB | ~1,100 MB | **~230 MB RAM** | **Fits Render 512 MB Free Tier** |
| **CPU Latency (P50)** | ~45 ms | ~85 ms | **~25–35 ms** | Real-time interactive response |

### Quantization & Decision Calibration Tradeoffs
* **Latency vs. Resource Tradeoff**: INT8 CPU execution runs in ~25–35 ms per query on standard cloud vCPUs. While GPU batched inference is faster (<5 ms), CPU ONNX requires no GPU drivers, zero CUDA overhead, and consumes ~230 MB RAM.
* **Calibration Tradeoff**: Calibrating $\tau_{\text{neg}} = 0.200$ intentionally accepts a minor drop in raw negative precision (from 84.2% to 73.7%) in exchange for boosting Negative Recall to **90.34%**. In support ticketing, catching 9 out of 10 negative complaints far outweighs the cost of reviewing a neutral feedback.

---

## 🚀 Deployment: Render.com (Free Tier Verified)

SentiPulse is pre-configured to deploy on Render's **Free Tier (0.1 CPU, 512 MB RAM)** without memory exhaustion or external paid object storage.

### Option 1: Native Python Web Service (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy SentiPulse to Render"
   git push origin main
   ```
   *The ONNX model is stored as `data/models/transformer_sentiment/model_quantized.onnx.gz` (85.35 MB). The application decompresses it on first boot in ~2 seconds.*

2. **Create Web Service**:
   - Go to [dashboard.render.com](https://dashboard.render.com) $\rightarrow$ **New +** $\rightarrow$ **Web Service**.
   - Connect your repository.
   - Configure:
     - **Runtime**: `Python 3`
     - **Build Command**:
       ```bash
       pip install --upgrade pip && pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
       ```
     - **Start Command**:
       ```bash
       uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
       ```
     - **Instance Type**: `Free` (512 MB RAM)
     - **Environment Variable**: `PYTHON_VERSION` = `3.11.9`
3. Click **Create Web Service**. Your service will boot and be reachable at `https://<service-name>.onrender.com`.

---

### Option 2: Docker Deployment

The included [`Dockerfile`](Dockerfile) dynamically binds to Render's assigned `$PORT`:

1. On Render, select **New +** $\rightarrow$ **Web Service**.
2. Select your repository and choose **Runtime**: `Docker`.
3. Choose **Instance Type**: `Free`.
4. Click **Create Web Service**.

---

### Option 3: Render Blueprint (`render.yaml`)

1. Go to **Blueprints** $\rightarrow$ **New Blueprint Instance**.
2. Connect your repository.
3. Render reads [`render.yaml`](render.yaml) and configures all commands, caching, and health check endpoints automatically.

---

## 💻 Local Quickstart

### 1. Clone & Setup
```bash
git clone https://github.com/<your-username>/sentipulse.git
cd sentipulse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK tokenizers
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
```

### 2. Start the Server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload
```

- **Interactive Web Dashboard**: [http://localhost:5000](http://localhost:5000)
  - Click on the top header title **SentiPulse** at any time to return directly to the Sentiment Analyzer.
- **Swagger / OpenAPI Documentation**: [http://localhost:5000/docs](http://localhost:5000/docs)
- **Health Check**: [http://localhost:5000/health](http://localhost:5000/health)

### 3. Run Automated Tests
```bash
pytest tests/ -v
```
*Validates API responses, input preprocessing edge cases, ONNX export parity, and threshold calibrator logic (17 tests passing).*

---

## 🔌 API Reference

### 1. Predict Sentiment
`POST /predict`

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The build quality is superb, delivery was fast and customer care was helpful!"}'
```

**Response**:
```json
{
  "sentiment": "Positive",
  "confidence": 0.903,
  "probabilities": {
    "Negative": 0.0385,
    "Neutral": 0.0585,
    "Positive": 0.903
  },
  "timestamp": "2026-09-16T14:28:00.123456",
  "model_version": "3.0-ONNX-INT8",
  "processed_text": "build quality superb delivery fast customer care helpful",
  "response_time_ms": 26.4,
  "error": null
}
```

### 2. Real-Time MLOps Telemetry
`GET /api/telemetry`

Returns cumulative request counters, average latency, P50/P95 latency percentiles, sentiment class distributions, and the latest 20 inferences from SQLite.

### 3. Kolmogorov-Smirnov Drift Monitor
`POST /api/check-drift`

Extracts features from the last 20–50 live customer queries recorded in SQLite and calculates the two-sample Kolmogorov-Smirnov statistic against the training reference baseline (`data/processed/features_reference.pkl`). If $>10\%$ of features shift with $p < 0.05$, an automated retraining alert is flagged.

### 4. Health & Model Diagnostics
`GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3.0-ONNX-INT8",
  "model_name": "RoBERTa INT8",
  "benchmark": {
    "f1_score": 0.8458,
    "accuracy": 0.8480,
    "negative_recall": 0.9034
  }
}
```

---

## 🛠️ Offline Model Training & Reproduction

To train or reproduce the pipeline from scratch on your own dataset:

```bash
# Run transformer pipeline with Cosine Annealing, SWA & ONNX quantization
python pipeline.py --model transformer --samples 120000 --epochs 3 --batch-size 16
```

Pipeline execution steps:
1. Stratified data splitting with zero target leakage (`train=96k`, `val=12k`, `test=12k`).
2. RoBERTa-base fine-tuning with Hugging Face `Trainer`, FP16 mixed precision, and SWA.
3. Decision threshold calibration optimizing for Negative Recall $>90\%$.
4. PyTorch graph export to ONNX followed by dynamic INT8 quantization.
5. Verification of output logits parity between PyTorch FP32 and ONNX INT8.

---

## 📌 Repository Metadata

<details>
<summary><b>Click to expand GitHub repository settings</b></summary>

* **Repository Name**: `sentipulse` (or `sentipulse-sentiment-analysis`)
* **Description**: `Production-ready customer sentiment classification engine powered by INT8 quantized RoBERTa with calibrated negative recall (>90%) and real-time Kolmogorov-Smirnov drift monitoring.`
* **Website**: `https://<your-service-name>.onrender.com`
* **Topics / Tags**: `sentiment-analysis`, `roberta`, `onnx`, `int8-quantization`, `fastapi`, `mlops`, `drift-detection`, `customer-feedback`, `render-deployment`, `python`

</details>

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.