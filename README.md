# SentiPulse

> **Production-grade sentiment classification engine & real-time MLOps drift monitor powered by INT8 quantized RoBERTa with calibrated negative recall (>90%).**

[![CI Pipeline](https://github.com/santoshkkashyap25/sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/santoshkkashyap25/sentiment-analysis/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7.svg?logo=render&logoColor=white)](https://sentipulse-q0mx.onrender.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![RoBERTa](https://img.shields.io/badge/Model-RoBERTa--base-yellow.svg)](https://huggingface.co/roberta-base)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live Application**: [https://sentipulse-q0mx.onrender.com/](https://sentipulse-q0mx.onrender.com/)

---

## Live Demo

The production application is deployed and accessible at:  
**[https://sentipulse-q0mx.onrender.com/](https://sentipulse-q0mx.onrender.com/)**

- **Interactive Sentiment Analyzer**: Real-time feedback classification with confidence scoring and calibrated negative recall.
- **MLOps Telemetry & Drift Console**: Live Kolmogorov-Smirnov statistical drift detection, query telemetry, and SQLite-backed metrics.

---

## Overview

**SentiPulse** is an end-to-end sentiment classification engine and real-time MLOps monitoring platform built for customer feedback streams:

- **Fine-Tuned RoBERTa-base**: Trained across **120,000 customer reviews** with Cosine Annealing and Stochastic Weight Averaging (SWA).
- **Post-Training Decision Calibration**: Empirically calibrated decision boundaries ($\tau_{\text{neg}} = 0.200$) achieve a Negative Class Recall of **90.34%** on held-out test data.
- **Dynamic INT8 Quantization**: Exported to ONNX and dynamically quantized (`QuantType.QInt8`), shrinking model disk footprint to **120.6 MB** (compressed to **85.3 MB** in git) and running in **~230 MB RAM on CPU** (~25–35 ms latency).
- **Statistical Drift Detection**: Integrated two-sample **Kolmogorov-Smirnov (KS) test** ($p < 0.05$) continuously monitoring live feedback query windows in SQLite against the training feature distribution.

---

## Validated Benchmarks

All metrics evaluated on a held-out test set of **12,000 Amazon customer reviews** using an NVIDIA GeForce RTX 5050 Laptop GPU for training and CPU for INT8 ONNX deployment:

| Metric | Baseline (DistilBERT) | Uncalibrated RoBERTa | SentiPulse (RoBERTa INT8 + Calibrated) | Production Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 81.70% | 84.80% | **84.80%** | +3.10% overall accuracy |
| **Weighted F1** | 81.50% | 84.58% | **84.58%** | Reliable multi-class balance |
| **Negative Recall** | 76.40% | 79.10% | **90.34%** | **+13.94% customer complaints caught** |
| **Negative F1** | 73.20% | 76.80% | **81.16%** | High-precision triage for churn prevention |
| **Artifact Size** | 268 MB | 498 MB | **85.3 MB (.gz) / 120.6 MB** | -75% disk footprint |
| **Memory Footprint** | ~550 MB | ~1,100 MB | **~230 MB RAM** | -58% memory usage |
| **CPU Latency (P50)** | ~45 ms | ~85 ms | **~25–35 ms** | Real-time interactive response |

### Quantization & Decision Calibration Tradeoffs
* **Latency vs. Resource Tradeoff**: INT8 CPU execution runs in ~25–35 ms per query on standard cloud vCPUs. While GPU batched inference is faster (<5 ms), CPU ONNX requires no GPU drivers, zero CUDA overhead, and consumes ~230 MB RAM.
* **Calibration Tradeoff**: Calibrating $\tau_{\text{neg}} = 0.200$ intentionally accepts a minor drop in raw negative precision (from 84.2% to 73.7%) in exchange for boosting Negative Recall to **90.34%**. In support ticketing, catching 9 out of 10 negative complaints far outweighs the cost of reviewing a neutral feedback.

---

## Offline Model Training & Reproduction

To train or reproduce the pipeline on custom feedback data (requires a CSV with text and rating columns):

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

## Python Developer SDK & CLI (`sentipulse`)

SentiPulse provides a clean, dual-mode Python library and command-line utility.

### 1. Python SDK Usage
```python
from sentipulse import SentiPulse

# Mode A: Local embedded CPU inference (~25-35 ms, zero-torch runtime)
sp = SentiPulse()

# Mode B: Remote client (connect to your deployed microservice)
# sp = SentiPulse(api_url="https://sentipulse-q0mx.onrender.com")

result = sp.analyze("Arrived damaged, customer support ignored my refund request.")
print(result)
# {
#   "sentiment": "Negative",
#   "confidence": 0.909,
#   "probabilities": {"Negative": 0.854, "Neutral": 0.116, "Positive": 0.030},
#   "escalation_recommended": True,
#   "response_time_ms": 28.5
# }
```

### 2. Command-Line Interface (CLI)
```bash
# Analyze a single review string
sentipulse analyze "The shoes fit comfortably and the color matches the pictures."

# Raw JSON output
sentipulse analyze "Awful experience, never buying again." --json

# High-throughput batch classification for entire CSV datasets
sentipulse batch reviews.csv --text-col reviewText --output analyzed_reviews.csv

# Launch the FastAPI server locally
sentipulse serve --host 0.0.0.0 --port 5000
```

---

## Support Desk & Webhook Escalations

SentiPulse can automatically dispatch high-priority escalation alerts to **Slack, Discord, Zendesk, or custom webhooks** when customer feedback exhibits high-confidence negative sentiment ($\ge 85\%$):

```bash
# Configure destination webhook via API
curl -X POST http://localhost:5000/api/webhook/config \
  -H "Content-Type: application/json" \
  -d '{"webhook_url": "https://hooks.slack.com/services/...", "min_negative_conf": 0.85, "enabled": true}'
```

---

## Chrome & Chromium Browser Extension

A Manifest V3 extension (`extension/`) for e-commerce operators to inspect and triage live reviews on **Amazon, Trustpilot, and Google Reviews**:

1. Navigate to `chrome://extensions/` and enable **Developer mode**.
2. Click **Load unpacked** and select the [`extension/`](extension/) directory.
3. Open any Amazon or Trustpilot product page and click **Inspect Page Reviews** to view instant sentiment distribution and highlighted critical complaints.

---

## Production Docker Appliance

Run SentiPulse as an on-premises, privacy-compliant microservice with persistent SQLite telemetry:

```bash
# Launch with Docker Compose
docker compose up -d

# Check container health status
curl http://localhost:5000/health
```

---

## License
Distributed under the MIT License. See `LICENSE` for more information.