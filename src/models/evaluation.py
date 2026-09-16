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
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average)
        }

        return metrics

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     target_names: list = None) -> str:
        """Generate detailed classification report"""
        if target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']

        report = classification_report(y_true, y_pred, target_names=target_names)
        return report

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: list = None, figsize: Tuple = (8, 6)):
        """Plot confusion matrix"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.warning("matplotlib and seaborn are required to plot confusion matrix")
            return confusion_matrix(y_true, y_pred)

        if target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        return cm

    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: list = None) -> pd.DataFrame:
        """Calculate per-class metrics"""
        if target_names is None:
            target_names = ['Negative', 'Neutral', 'Positive']

        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)

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

        basic_metrics = self.calculate_basic_metrics(y_test, y_pred)
        report = self.generate_classification_report(y_test, y_pred, target_names)
        cm = confusion_matrix(y_test, y_pred)
        per_class_metrics = self.calculate_per_class_metrics(y_test, y_pred, target_names)

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

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        return comparison_df
