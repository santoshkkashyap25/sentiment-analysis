# Production Dockerfile for SentiPulse Sentiment Analysis & MLOps API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

# Install system build dependencies and cleanup in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK corpora
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy application source code
COPY . .

# Ensure data and logs directories exist
RUN mkdir -p data/raw data/processed data/models logs

# Decompress quantized ONNX model during image build for zero-latency startup
RUN python -c "import os, gzip, shutil; gz='data/models/transformer_sentiment/model_quantized.onnx.gz'; onnx='data/models/transformer_sentiment/model_quantized.onnx'; (not os.path.exists(onnx) and os.path.exists(gz)) and [shutil.copyfileobj(gzip.open(gz, 'rb'), open(onnx, 'wb')), os.remove(gz)]"

# Expose API port
EXPOSE 5000

# Health check with dynamic PORT fallback
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Start FastAPI server via Uvicorn with dynamic Render PORT support
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-5000}"]
