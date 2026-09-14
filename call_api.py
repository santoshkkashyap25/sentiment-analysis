"""Client test script for real-time customer feedback sentiment analysis API."""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"


def check_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("=== 1. Health Check ===")
    print(json.dumps(response.json(), indent=2), "\n")


def single_prediction(text):
    url = f"{BASE_URL}/predict"
    payload = {"text": text}
    t0 = time.time()
    response = requests.post(url, json=payload)
    elapsed = round((time.time() - t0) * 1000, 2)
    print(f"=== Single Prediction ({elapsed}ms) ===")
    print(f"Input: \"{text}\"")
    print(json.dumps(response.json(), indent=2), "\n")


def get_telemetry():
    url = f"{BASE_URL}/api/telemetry"
    response = requests.get(url)
    print("=== Live MLOps Telemetry ===")
    data = response.json()
    print(f"Total Predictions: {data.get('total_predictions')}")
    print(f"Avg Latency: {data.get('avg_latency_ms')} ms (P95: {data.get('p95_latency_ms')} ms)")
    print(f"Sentiment Counts: {data.get('sentiment_counts')}")
    print(f"Recent Drift Alerts: {len(data.get('drift_alerts', []))}\n")


def test_drift_simulation():
    url = f"{BASE_URL}/api/simulate-drift"
    print("=== Testing Statistical Drift Simulator ===")
    response = requests.post(url)
    print(json.dumps(response.json(), indent=2), "\n")


if __name__ == "__main__":
    check_health()

    # Test distinct sentiments in real time
    single_prediction("The product quality is fantastic and delivery was super fast!")
    single_prediction("Terrible experience, arrived broken and support was unhelpful.")
    single_prediction("Average item, nothing special but does the job.")

    # Check telemetry
    get_telemetry()

    # Test drift simulation
    test_drift_simulation()
