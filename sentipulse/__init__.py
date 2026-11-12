"""SentiPulse: Production-Grade Sentiment Classification & Real-Time MLOps Engine.

Zero-GPU inference powered by 8-bit quantized RoBERTa with calibrated
negative recall (>90%) and built-in Kolmogorov-Smirnov drift monitoring.
"""

from sentipulse.client import SentiPulse

__version__ = "1.0.0"
__all__ = ["SentiPulse"]

