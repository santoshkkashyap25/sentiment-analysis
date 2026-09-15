"""Decision threshold calibration for customer feedback sentiment classification."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

logger = logging.getLogger(__name__)


class ThresholdCalibrator:
    """Calibrates multi-class probability decision boundaries.
    
    In customer feedback sentiment analysis, false negatives (missing an angry customer)
    are significantly more detrimental to business retention than false positives.
    This calibrator optimizes thresholds (tau_neg, tau_neu, tau_pos) to maximize
    negative sentiment recall while maintaining a high overall F1 score.
    """

    def __init__(
        self,
        target_negative_recall: float = 0.85,
        min_negative_precision: float = 0.70,
        class_names: Optional[list] = None
    ):
        self.target_negative_recall = target_negative_recall
        self.min_negative_precision = min_negative_precision
        self.class_names = class_names or ["Negative", "Neutral", "Positive"]
        self.thresholds = {
            "negative_threshold": 0.333,
            "neutral_threshold": 0.333,
            "positive_threshold": 0.333
        }
        self.baseline_metrics = {}
        self.calibrated_metrics = {}

    def fit(self, y_probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Find optimal decision thresholds on validation set probabilities.
        
        Args:
            y_probs: Predicted probabilities of shape (N, 3) for [Negative, Neutral, Positive].
            y_true: True integer labels in {0, 1, 2}.
            
        Returns:
            Dict containing optimal thresholds and performance comparisons.
        """
        y_probs = np.asarray(y_probs)
        y_true = np.asarray(y_true)

        # 1. Baseline uncalibrated argmax performance
        y_pred_baseline = np.argmax(y_probs, axis=1)
        p_base, r_base, f1_base, _ = precision_recall_fscore_support(
            y_true, y_pred_baseline, average=None, zero_division=0
        )
        self.baseline_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_baseline)),
            "weighted_f1": float(f1_score(y_true, y_pred_baseline, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred_baseline, average="macro", zero_division=0)),
            "negative_recall": float(r_base[0]),
            "negative_precision": float(p_base[0]),
            "negative_f1": float(f1_base[0]),
        }

        # 2. Grid search over negative decision threshold tau_neg in [0.20, 0.50]
        best_threshold_neg = 0.333
        best_objective_score = -1.0
        best_preds = y_pred_baseline

        candidate_tau_negs = np.linspace(0.20, 0.48, 29)

        for tau_neg in candidate_tau_negs:
            preds = self._predict_with_thresholds(y_probs, tau_neg=tau_neg)
            p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
                y_true, preds, average=None, zero_division=0
            )
            neg_prec = p_cls[0]
            neg_rec = r_cls[0]
            macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

            # Optimization objective: Reward negative recall, penalize if precision drops below minimum
            if neg_prec >= self.min_negative_precision:
                objective = neg_rec * 0.6 + macro_f1 * 0.4
            else:
                objective = (neg_rec * 0.6 + macro_f1 * 0.4) * (neg_prec / max(self.min_negative_precision, 1e-5))

            if objective > best_objective_score:
                best_objective_score = objective
                best_threshold_neg = float(tau_neg)
                best_preds = preds

        self.thresholds["negative_threshold"] = round(best_threshold_neg, 4)

        # 3. Calculate calibrated metrics
        p_cal, r_cal, f1_cal, _ = precision_recall_fscore_support(
            y_true, best_preds, average=None, zero_division=0
        )
        self.calibrated_metrics = {
            "accuracy": float(accuracy_score(y_true, best_preds)),
            "weighted_f1": float(f1_score(y_true, best_preds, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, best_preds, average="macro", zero_division=0)),
            "negative_recall": float(r_cal[0]),
            "negative_precision": float(p_cal[0]),
            "negative_f1": float(f1_cal[0]),
        }

        logger.info(
            f"Calibrated negative threshold: {best_threshold_neg:.3f} | "
            f"Negative Recall: {self.baseline_metrics['negative_recall']:.3f} -> {self.calibrated_metrics['negative_recall']:.3f} | "
            f"Macro F1: {self.baseline_metrics['macro_f1']:.3f} -> {self.calibrated_metrics['macro_f1']:.3f}"
        )

        return {
            "thresholds": self.thresholds,
            "baseline_metrics": self.baseline_metrics,
            "calibrated_metrics": self.calibrated_metrics,
        }

    def _predict_with_thresholds(
        self, y_probs: np.ndarray, tau_neg: float = 0.333
    ) -> np.ndarray:
        """Apply decision thresholds to probability array."""
        preds = np.zeros(len(y_probs), dtype=int)
        for i, prob in enumerate(y_probs):
            p_neg, p_neu, p_pos = prob[0], prob[1], prob[2]
            if p_neg >= tau_neg:
                preds[i] = 0  # Negative
            else:
                preds[i] = 1 if p_neu > p_pos else 2
        return preds

    def predict(self, y_probs: np.ndarray) -> np.ndarray:
        """Predict labels using fitted calibrated thresholds."""
        tau_neg = self.thresholds.get("negative_threshold", 0.333)
        return self._predict_with_thresholds(y_probs, tau_neg=tau_neg)

    def predict_single(self, prob_dict: Dict[str, float]) -> Tuple[str, float]:
        """Apply calibrated threshold to a single inference probability dict."""
        p_neg = prob_dict.get("Negative", 0.0)
        p_neu = prob_dict.get("Neutral", 0.0)
        p_pos = prob_dict.get("Positive", 0.0)

        tau_neg = self.thresholds.get("negative_threshold", 0.333)

        if p_neg >= tau_neg:
            conf = min(1.0, 0.5 + (p_neg - tau_neg) / (1.0 - tau_neg + 1e-6) * 0.5)
            return "Negative", round(conf, 4)
        elif p_pos >= p_neu:
            conf = min(1.0, p_pos)
            return "Positive", round(conf, 4)
        else:
            conf = min(1.0, p_neu)
            return "Neutral", round(conf, 4)

    def save(self, filepath: str):
        """Save threshold config to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "thresholds": self.thresholds,
            "baseline_metrics": self.baseline_metrics,
            "calibrated_metrics": self.calibrated_metrics,
            "class_names": self.class_names,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved threshold calibration to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ThresholdCalibrator":
        """Load threshold config from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        calibrator = cls(class_names=data.get("class_names"))
        calibrator.thresholds = data.get("thresholds", {})
        calibrator.baseline_metrics = data.get("baseline_metrics", {})
        calibrator.calibrated_metrics = data.get("calibrated_metrics", {})
        return calibrator
