"""Model evaluation and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from typing import Dict, Any, Tuple
import logging
import warnings
warnings.filterwarnings("ignore")


class ModelEvaluator:
    """Handles model evaluation and metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              average: str = 'weighted') -> Dict:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        return metrics

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      target_names: list = None,
                                      labels: list = None) -> str:
        """Generate detailed classification report"""
        if labels is not None and target_names is None:
            target_names = [f"Class_{lbl}" for lbl in labels]
        elif target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']
        if labels is None:
            labels = [0, 1, 2] if len(target_names) == 3 else list(range(len(target_names)))

        report = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
        return report

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: list = None, figsize: Tuple = (8, 6),
                            save_path: str = None, labels: list = None):
        """Plot confusion matrix and optionally save to file without blocking"""
        if labels is not None and target_names is None:
            target_names = [f"Class_{lbl}" for lbl in labels]
        elif target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']
        if labels is None:
            labels = [0, 1, 2] if len(target_names) == 3 else list(range(len(target_names)))

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.warning("matplotlib and seaborn are required to plot confusion matrix")
            return confusion_matrix(y_true, y_pred, labels=labels)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        try:
            plt.figure(figsize=figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            if save_path:
                import os
                parent_dir = os.path.dirname(os.path.abspath(save_path))
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                self.logger.info(f"Confusion matrix plot saved to {save_path}")
        finally:
            plt.close()

        return cm

    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: list = None,
                                  labels: list = None) -> pd.DataFrame:
        """Calculate per-class metrics"""
        if labels is not None and target_names is None:
            target_names = [f"Class_{lbl}" for lbl in labels]
        elif target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']
        if labels is None:
            labels = [0, 1, 2] if len(target_names) == 3 else list(range(len(target_names)))

        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

        # Align target_names length strictly with labels length
        if len(target_names) > len(labels):
            target_names = target_names[:len(labels)]
        elif len(target_names) < len(labels):
            target_names = list(target_names) + [f"Class_{lbl}" for lbl in labels[len(target_names):]]

        metrics_df = pd.DataFrame({
            'Class': target_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        return metrics_df

    def evaluate_model_comprehensive(self, model: Any, X_test: np.ndarray,
                                   y_test: np.ndarray, target_names: list = None) -> Dict:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = None

        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            self.logger.warning("Model does not support probability predictions")

        if target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']
        labels = [0, 1, 2] if len(target_names) == 3 else list(range(len(target_names)))

        basic_metrics = self.calculate_basic_metrics(y_test, y_pred)
        report = self.generate_classification_report(y_test, y_pred, target_names, labels=labels)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        per_class_metrics = self.calculate_per_class_metrics(y_test, y_pred, target_names, labels=labels)

        results = {
            'basic_metrics': basic_metrics,
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        self.logger.info("Model Evaluation Results:")
        self.logger.info(f"Accuracy: {basic_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {basic_metrics['precision']:.4f}")
        self.logger.info(f"Recall: {basic_metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {basic_metrics['f1']:.4f}")

        return results

    def compare_models(self, models: Dict, X_test: np.ndarray,
                      y_test: np.ndarray) -> pd.DataFrame:
        """Compare multiple models"""
        results = []

        for name, model in models.items():
            y_pred = model.predict(X_test)
            metrics = self.calculate_basic_metrics(y_test, y_pred)
            metrics['model'] = name
            results.append(metrics)

        if not results:
            return pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        return comparison_df
