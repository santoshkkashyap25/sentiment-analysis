# SentiPulse | Intelligent Customer Feedback Analytics & MLOps Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![RoBERTa](https://img.shields.io/badge/Model-RoBERTa--base-yellow.svg)](https://huggingface.co/roberta-base)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade sentiment classification and MLOps monitoring platform engineered for real-world customer feedback streams.

Fine-tuned on **120,000 customer reviews** using **RoBERTa-base** with Cosine Annealing, Stochastic Weight Averaging (SWA), and threshold calibration, then exported into an **INT8 dynamically quantized ONNX model** running in **~230 MB RAM on CPU**—fitting comfortably within Render's 512 MB free tier.

---

## GitHub Repository Setup

When creating or configuring the GitHub repository, use these settings:

* **Repository Name**: `sentipulse` (or `sentipulse-sentiment-analysis`)
* **Description**: `Production-ready customer sentiment classification engine powered by INT8 quantized RoBERTa with calibrated negative recall (>90%) and real-time Kolmogorov-Smirnov drift monitoring.`
* **Website**: `https://<your-service-name>.onrender.com`
* **Topics / Tags**:
  `sentiment-analysis`, `roberta`, `onnx`, `int8-quantization`, `fastapi`, `mlops`, `drift-detection`, `customer-feedback`, `render-deployment`, `python`

---

## 🎯 Key Capabilities

* **Calibrated Negative Recall (>90%)**: Standard argmax sentiment classifiers frequently miss subtle or passive-aggressive customer complaints. We apply post-training threshold calibration ($\tau_{\text{neg}} = 0.200$), lifting Negative Recall from 76.4% to **90.34%**.
* **Ultra-Lean Production Runtime**: The unquantized 477 MB PyTorch model is converted to a single-file ONNX graph and dynamic INT8 quantized (`QuantType.QInt8`), shrinking disk footprint to **120.6 MB** (-75% reduction) and runtime memory to **~230 MB RAM** on CPU.
* **Dual-Mode Interactive Interface**:
  * **Tab 1 (Sentiment Analyzer)**: Clean, business-focused feedback review tester with preset reviews (Positive, Negative, Neutral), calibrated confidence scoring, Softmax probability distribution bars, and lemmatized NLP token extraction.
  * **Tab 2 (MLOps Telemetry & Drift)**: Operations console with real-time latency percentiles (P50/P95), live sentiment distribution chart, and SQLite inference audit log.
* **Real Live Query KS Drift Detection**: Computes the two-sample **Kolmogorov-Smirnov (KS) statistic** ($p < 0.05$) comparing the last 20–50 actual live customer queries in SQLite against the training corpus baseline (`features_reference.pkl`). If $>10\%$ of vocabulary features shift, an automated retraining alert is flagged.

---

## 📊 Validated Model Benchmarks

Fine-tuned on **120,000 reviews** (96,000 Train, 12,000 Validation, 12,000 Held-Out Test) on an NVIDIA GeForce RTX 5050 Laptop GPU (3 Epochs with FP16, Cosine Annealing, and SWA):

| Metric | Baseline (DistilBERT) | RoBERTa-base + SWA (Current) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 81.70% | **84.80%** | **+3.10%** |
| **Weighted F1-Score** | 81.50% | **84.58%** | **+3.08%** |
| **Negative Class Recall** | 76.40% | **90.34%** | **+13.94%** |
| **Negative Class F1** | 73.20% | **81.16%** | **+7.96%** |
| **Model Size** | 268 MB | **120.61 MB** (INT8 ONNX) | **-55.0%** |
| **Runtime Memory (RAM)** | ~550 MB | **~230 MB** (ONNX CPU) | **Fits 512 MB Free Tier** |

---

## 🚀 Deployment Guide: Render.com

Deploying SentiPulse on [Render.com](https://render.com) takes under 5 minutes on the **Free Tier (512 MB RAM)**.

### Method 1: Web Service (Python Native) — Recommended

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy SentiPulse to Render"
   git push origin main
   ```
   *(Note: The repository includes `data/models/transformer_sentiment/model_quantized.onnx.gz` which is 85 MB, well below GitHub's 100 MB limit. The application automatically decompresses it on startup in 2 seconds).*

2. **Create New Web Service on Render**:
   * Log in to [dashboard.render.com](https://dashboard.render.com).
   * Click **New +** $\rightarrow$ **Web Service**.
   * Connect your GitHub repository (`sentipulse`).

3. **Configure Service Settings**:
   * **Name**: `sentipulse` (or your preferred name)
   * **Region**: *Oregon (US West)* or nearest to you
   * **Branch**: `main`
   * **Runtime**: `Python 3`
   * **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
     ```
   * **Start Command**:
     ```bash
     uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
     ```
   * **Instance Type**: `Free` (0.1 CPU, 512 MB RAM)

4. **Environment Variables**:
   Under **Advanced**, add:
   * `PYTHON_VERSION` = `3.11.9`

5. **Deploy**:
   * Click **Create Web Service**. Render will install dependencies and start Uvicorn.
   * Once deployed, your web application and interactive API will be live at:
     `https://<your-service-name>.onrender.com`

---

### Method 2: Docker on Render

If you prefer containerized deployment, the repository includes a multi-stage, production-ready `Dockerfile`:

1. On Render, select **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository and select **Runtime**: `Docker`.
3. Instance Type: `Free`.
4. Click **Create Web Service**. Render will automatically build the image and bind Uvicorn to `$PORT`.

---

### Method 3: One-Click Blueprint (`render.yaml`)

This repository includes a [`render.yaml`](render.yaml) blueprint specification. On Render:
1. Click **Blueprints** $\rightarrow$ **New Blueprint Instance**.
2. Connect your repository. Render automatically reads `render.yaml` and provisions the web service with all build commands, environment variables, and health check paths pre-configured.

---

## 💻 Local Development

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/<your-username>/sentipulse.git
cd sentipulse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
```

### 2. Launch Local Server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload
```
* **Web Application & MLOps Dashboard**: [http://localhost:5000](http://localhost:5000)
* **Interactive OpenAPI Docs**: [http://localhost:5000/docs](http://localhost:5000/docs)

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 🔌 API Endpoints Reference

### 1. Predict Sentiment
`POST /predict`
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "The build quality is exceptional and shipping was prompt!"}'
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
  "timestamp": "2026-09-15T22:47:34.963869",
  "model_version": "3.0-ONNX-INT8",
  "processed_text": "build quality exceptional shipping prompt",
  "response_time_ms": 25.1,
  "error": null
}
```

### 2. Live Telemetry
`GET /api/telemetry`
Returns aggregate inference counters, P50/P95 latency percentiles, sentiment distribution counts, and recent SQLite audit records.

### 3. Real Query KS Drift Check
`POST /api/check-drift`
Extracts TF-IDF features from the last 20–50 real customer feedback entries in SQLite, runs the two-sample Kolmogorov-Smirnov test against baseline, and records alerts if $>10\%$ of features shift ($p < 0.05$).

### 4. Health Check
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

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.