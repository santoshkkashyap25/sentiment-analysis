"""Automated Webhook Dispatcher for Customer Triage and Escalations."""

import logging
import threading
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """Dispatches asynchronous webhooks to Slack, Discord, Zendesk, or generic endpoints.
    
    Triggered when an incoming feedback exhibits high-confidence Negative sentiment
    (exceeding `min_negative_conf`), recommending immediate support triage.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        min_negative_conf: float = 0.85,
        timeout: float = 5.0
    ):
        self.webhook_url = webhook_url
        self.min_negative_conf = min_negative_conf
        self.timeout = timeout

    def should_escalate(self, sentiment: str, confidence: float) -> bool:
        """Evaluate if sentiment and confidence meet escalation threshold."""
        return sentiment == "Negative" and confidence >= self.min_negative_conf

    def format_payload(self, text: str, sentiment: str, confidence: float, probabilities: Optional[Dict] = None) -> Dict[str, Any]:
        """Construct structured escalation payload supporting generic and Slack webhook standards."""
        is_slack = self.webhook_url and ("hooks.slack.com" in self.webhook_url or "discord.com" in self.webhook_url)

        if is_slack:
            # Formatted markdown attachment for Slack / Discord channels
            return {
                "text": f"🚨 *SentiPulse Customer Escalation Alert* ({confidence * 100:.1f}% Negative)",
                "attachments": [
                    {
                        "color": "#e11d48",
                        "title": "Immediate Support Follow-up Recommended",
                        "text": text[:600],
                        "fields": [
                            {"title": "Sentiment", "value": sentiment, "short": True},
                            {"title": "Confidence", "value": f"{confidence * 100:.1f}%", "short": True},
                            {"title": "Action", "value": "Customer Churn Risk Detected", "short": False}
                        ],
                        "footer": "SentiPulse Automated Triage"
                    }
                ]
            }

        # Generic webhook JSON schema for Zapier / Make / Internal microservices
        return {
            "event": "customer_escalation",
            "source": "sentipulse-engine",
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "text": text[:1000],
            "severity": "high" if confidence >= 0.90 else "medium",
            "probabilities": probabilities or {},
            "recommendation": "immediate_support_followup"
        }

    def dispatch_escalation(
        self,
        text: str,
        sentiment: str,
        confidence: float,
        probabilities: Optional[Dict] = None
    ) -> bool:
        """Asynchronously dispatch escalation alert in a background thread."""
        if not self.webhook_url or not self.should_escalate(sentiment, confidence):
            return False

        payload = self.format_payload(text, sentiment, confidence, probabilities)

        # Non-blocking async dispatch so caller latency is unaffected
        t = threading.Thread(
            target=self._send,
            args=(self.webhook_url, payload),
            daemon=True
        )
        t.start()
        return True

    def _send(self, url: str, payload: Dict[str, Any]):
        """Send HTTP POST request to webhook destination."""
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                logger.warning(f"Webhook dispatch failed with HTTP {resp.status_code}: {resp.text}")
            else:
                logger.info(f"Successfully dispatched customer escalation alert to webhook ({resp.status_code})")
        except Exception as e:
            logger.warning(f"Webhook connection error: {e}")
