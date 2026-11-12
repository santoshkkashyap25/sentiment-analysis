"""Unified Client SDK for SentiPulse Sentiment Intelligence & MLOps Engine."""

import time
from typing import Dict, Any, Optional, List, Union
import requests


class SentiPulse:
    """Unified client for SentiPulse sentiment classification and customer triage.
    
    Supports two modes of operation:
    1. **Remote Client Mode** (recommended for thin clients): Provide `api_url` (e.g. `http://localhost:5000`
       or a production cloud endpoint). Dispatches HTTP requests without loading model weights into local memory.
    2. **Local Embedded Mode**: Omit `api_url`. Executes directly in-process on CPU via ONNX INT8 runtime (~25-35 ms).
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
        escalation_threshold: float = 0.85,
    ):
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_key = api_key.strip() if api_key else None
        self.timeout = timeout
        self.escalation_threshold = escalation_threshold
        self._local_api = None

        if not self.api_url:
            from src.api.app import FeedbackAnalysisAPI
            self._local_api = FeedbackAnalysisAPI()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment for a single piece of customer feedback text.
        
        Args:
            text: Customer review or feedback string.
            
        Returns:
            Dict containing sentiment, confidence, probability breakdown, and escalation flag.
        """
        if not text or not str(text).strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.33,
                "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
                "escalation_recommended": False,
                "processed_text": "",
                "error": "Empty text input"
            }

        if len(text) > 25000:
            raise ValueError(f"Input text exceeds maximum allowed length of 25,000 characters (received {len(text)}).")

        t0 = time.time()

        if self.api_url:
            post_kwargs = {"json": {"text": text}, "timeout": self.timeout}
            if self.api_key:
                post_kwargs["headers"] = {"X-API-Key": self.api_key}
            resp = requests.post(
                f"{self.api_url}/predict",
                **post_kwargs
            )
            resp.raise_for_status()
            data = resp.json()
        else:
            data = self._local_api.predict_sentiment(text)
            data["response_time_ms"] = round((time.time() - t0) * 1000, 2)

        sentiment = data.get("sentiment", "Neutral")
        confidence = float(data.get("confidence", 0.0))
        data["text"] = text
        data["escalation_recommended"] = (
            sentiment == "Negative" and confidence >= self.escalation_threshold
        )
        return data

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of customer feedback strings."""
        return [self.analyze(t) for t in texts]

    def health(self) -> Dict[str, Any]:
        """Check operational status of the SentiPulse inference engine."""
        if self.api_url:
            get_kwargs = {"timeout": self.timeout}
            if self.api_key:
                get_kwargs["headers"] = {"X-API-Key": self.api_key}
            resp = requests.get(f"{self.api_url}/health", **get_kwargs)
            resp.raise_for_status()
            return resp.json()
        
        is_loaded = (
            getattr(self._local_api, "onnx_session", None) is not None or
            getattr(self._local_api, "transformer_model", None) is not None or
            getattr(self._local_api, "model", None) is not None
        )
        return {
            "status": "healthy" if is_loaded else "degraded",
            "model_loaded": is_loaded,
            "runtime": "local-in-process",
            "model_name": "RoBERTa INT8" if getattr(self._local_api, "is_onnx", False) else "Classical"
        }

    def check_drift(self) -> Dict[str, Any]:
        """Evaluate statistical feature drift on recent live queries."""
        if self.api_url:
            post_kwargs = {"timeout": self.timeout}
            if self.api_key:
                post_kwargs["headers"] = {"X-API-Key": self.api_key}
            resp = requests.post(f"{self.api_url}/api/check-drift", **post_kwargs)
            resp.raise_for_status()
            return resp.json()
        
        from src.api.app import check_live_drift
        return check_live_drift()

