# Customer Feedback Analysis System

A machine learning pipeline for analyzing customer feedback and extracting actionable insights through sentiment classification.


<!-- ## Model Performance

Our best performing model achieves:
- **Accuracy**: 89.2%
- **F1-Score**: 89.1% (weighted)
- **Precision**: 89.3% (weighted)
- **Recall**: 89.2% (weighted)

### Model Comparison Results:
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| XGBoost | **0.892** | **0.891** | 45s |
| Random Forest | 0.887 | 0.886 | 32s |
| Logistic Regression | 0.881 | 0.880 | 8s |
| SVM | 0.876 | 0.875 | 120s | -->

## API Usage

### Start the Server
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 5000
```

### API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! Great quality."}'
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.94,
  "probabilities": {
    "Negative": 0.02,
    "Neutral": 0.04,
    "Positive": 0.94
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "model_version": "1.0"
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product, highly recommend!",
      "Poor quality, very disappointed.",
      "Average item, nothing special."
    ]
  }'
```

## Performance Monitoring

### Real-time Metrics
- Request latency (P50, P95, P99)
- Throughput (requests/second)
- Error rates
- Model confidence distribution

### Drift Detection
- **Data Drift**: Statistical tests on input features
- **Prediction Drift**: Distribution changes in predictions
- **Performance Drift**: Accuracy degradation over time

### Alerting Thresholds
- Response time > 1 second
- Error rate > 5%
- Accuracy drop > 5%
- Drift detection p-value < 0.05

## Retraining Strategy

### Automatic Retraining Triggers:
1. **Performance degradation**: Accuracy drops >5%
2. **Data drift detected**: >10% of features show drift
3. **Time-based**: Monthly retraining schedule
4. **Data volume**: Every 10,000 new samples

### Retraining Process:
1. **Data Collection**: Gather new labeled samples
2. **Quality Validation**: Check data quality and balance
3. **Model Training**: Retrain with combined dataset
4. **A/B Testing**: Gradual rollout with performance comparison
5. **Deployment**: Replace model if performance improves


## Configuration

### Environment Variables
```bash
MODEL_PATH=/app/data/models/best_model.pkl
EXTRACTORS_PATH=/app/data/models/feature_extractors.pkl
LOG_LEVEL=INFO
```

### Model Configuration
```json
{
  "random_state": 42,
  "test_size": 0.2,
  "max_features": 5000,
  "embedding_model": "all-MiniLM-L6-v2",
  "resampling_strategy": "smote"
}
```