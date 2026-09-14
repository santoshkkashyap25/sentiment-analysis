"""Model performance monitoring and drift detection."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import pickle
import sqlite3
import warnings
warnings.filterwarnings("ignore")


class ModelMonitor:
    """Handles model performance monitoring and drift detection"""

    def __init__(self, db_path: str = 'monitoring.db'):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Setup monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                input_text TEXT,
                prediction INTEGER,
                confidence REAL,
                response_time REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                accuracy REAL,
                f1_score REAL,
                sample_size INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                drift_type TEXT,
                metric_value REAL,
                threshold REAL,
                severity TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def log_prediction(self, input_text: str, prediction: int,
                      confidence: float, response_time: float):
        """Log prediction for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (timestamp, input_text, prediction, confidence, response_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), input_text, prediction, confidence, response_time))

        conn.commit()
        conn.close()

    def detect_data_drift(self, reference_features: np.ndarray,
                          current_features: np.ndarray,
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        drift_results = {}

        if hasattr(reference_features, 'toarray'):
            reference_features = reference_features.toarray()
        elif not isinstance(reference_features, np.ndarray):
            reference_features = np.asarray(reference_features)

        if hasattr(current_features, 'toarray'):
            current_features = current_features.toarray()
        elif not isinstance(current_features, np.ndarray):
            current_features = np.asarray(current_features)

        min_features = min(50, reference_features.shape[1], current_features.shape[1])
        reference_features = reference_features[:, :min_features]
        current_features = current_features[:, :min_features]

        for i in range(min_features):
            ref_feature = reference_features[:, i]
            curr_feature = current_features[:, i]

            if np.all(ref_feature == ref_feature[0]) and np.all(curr_feature == curr_feature[0]):
                ks_stat, p_value = 0.0, 1.0
            else:
                ks_stat, p_value = ks_2samp(ref_feature, curr_feature)

            drift_results[f'feature_{i}'] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': p_value < threshold
            }

        drift_count = sum(1 for result in drift_results.values()
                          if result['drift_detected'])
        drift_percentage = drift_count / len(drift_results)

        overall_result = {
            'drift_detected_features': drift_count,
            'total_features': len(drift_results),
            'drift_percentage': drift_percentage,
            'overall_drift': drift_percentage > 0.1,
            'feature_results': drift_results
        }

        if overall_result['overall_drift']:
            self.log_drift_alert('data_drift', drift_percentage, 0.1, 'medium')

        return overall_result

    def detect_prediction_drift(self, reference_predictions: np.ndarray,
                               current_predictions: np.ndarray,
                               threshold: float = 0.05) -> Dict[str, Any]:
        """Detect prediction drift using distribution comparison"""
        ref_counts = np.bincount(reference_predictions, minlength=3)
        curr_counts = np.bincount(current_predictions, minlength=3)

        chi2_stat, p_value = chi2_contingency([ref_counts, curr_counts])[:2]

        result = {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'drift_detected': p_value < threshold,
            'reference_distribution': ref_counts / len(reference_predictions),
            'current_distribution': curr_counts / len(current_predictions)
        }

        if result['drift_detected']:
            self.log_drift_alert('prediction_drift', chi2_stat, threshold, 'high')

        return result

    def monitor_performance_degradation(self, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      baseline_accuracy: float,
                                      baseline_f1: float,
                                      threshold: float = 0.05) -> Dict[str, Any]:
        """Monitor for performance degradation"""
        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred, average='weighted')

        accuracy_drop = baseline_accuracy - current_accuracy
        f1_drop = baseline_f1 - current_f1

        result = {
            'current_accuracy': current_accuracy,
            'current_f1': current_f1,
            'baseline_accuracy': baseline_accuracy,
            'baseline_f1': baseline_f1,
            'accuracy_drop': accuracy_drop,
            'f1_drop': f1_drop,
            'performance_degraded': (accuracy_drop > threshold) or (f1_drop > threshold)
        }

        self.log_performance_metrics(current_accuracy, current_f1, len(y_true))

        if result['performance_degraded']:
            severity = 'high' if max(accuracy_drop, f1_drop) > threshold * 2 else 'medium'
            self.log_drift_alert('performance_degradation',
                               max(accuracy_drop, f1_drop), threshold, severity)

        return result

    def log_performance_metrics(self, accuracy: float, f1: float, sample_size: int):
        """Log performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO performance_metrics (timestamp, accuracy, f1_score, sample_size)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now(), accuracy, f1, sample_size))

        conn.commit()
        conn.close()

    def log_drift_alert(self, drift_type: str, metric_value: float,
                       threshold: float, severity: str):
        """Log drift alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO drift_alerts (timestamp, drift_type, metric_value, threshold, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), drift_type, metric_value, threshold, severity))

        conn.commit()
        conn.close()

        self.logger.warning(f"Drift alert: {drift_type} - Value: {metric_value}, "
                          f"Threshold: {threshold}, Severity: {severity}")

    def get_monitoring_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        conn = sqlite3.connect(self.db_path)

        performance_df = pd.read_sql_query(f'''
            SELECT * FROM performance_metrics
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
        ''', conn)

        predictions_df = pd.read_sql_query(f'''
            SELECT * FROM predictions
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
            LIMIT 1000
        ''', conn)

        alerts_df = pd.read_sql_query(f'''
            SELECT * FROM drift_alerts
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp DESC
        ''', conn)

        conn.close()

        if not predictions_df.empty:
            avg_response_time = predictions_df['response_time'].mean()
            prediction_distribution = predictions_df['prediction'].value_counts().to_dict()
            avg_confidence = predictions_df['confidence'].mean()
        else:
            avg_response_time = 0
            prediction_distribution = {}
            avg_confidence = 0

        dashboard_data = {
            'performance_history': performance_df.to_dict('records'),
            'recent_predictions': predictions_df.to_dict('records'),
            'recent_alerts': alerts_df.to_dict('records'),
            'summary_stats': {
                'avg_response_time': avg_response_time,
                'prediction_distribution': prediction_distribution,
                'avg_confidence': avg_confidence,
                'total_predictions': len(predictions_df),
                'total_alerts': len(alerts_df)
            }
        }

        return dashboard_data


class RetrainingStrategy:
    """Handles model retraining strategy"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def should_retrain(self, monitor: ModelMonitor,
                      performance_threshold: float = 0.05,
                      drift_threshold: float = 0.1) -> Dict[str, Any]:
        """Determine if model should be retrained"""
        dashboard_data = monitor.get_monitoring_dashboard_data(days=7)

        recent_alerts = dashboard_data['recent_alerts']

        performance_alerts = sum(1 for alert in recent_alerts
                               if alert['drift_type'] == 'performance_degradation')
        drift_alerts = sum(1 for alert in recent_alerts
                          if 'drift' in alert['drift_type'])

        should_retrain = False
        reasons = []

        if performance_alerts >= 3:
            should_retrain = True
            reasons.append("Multiple performance degradation alerts")

        if drift_alerts >= 5:
            should_retrain = True
            reasons.append("Multiple drift alerts detected")

        performance_history = dashboard_data['performance_history']
        if not performance_history:
            should_retrain = True
            reasons.append("No recent performance data available")

        result = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'performance_alerts': performance_alerts,
            'drift_alerts': drift_alerts,
            'recommendation': 'retrain' if should_retrain else 'continue_monitoring'
        }

        return result

    def create_retraining_plan(self, new_data_size: int,
                             current_model_performance: Dict) -> Dict[str, Any]:
        """Create a retraining plan"""
        plan = {
            'data_requirements': {
                'minimum_samples': max(1000, new_data_size),
                'class_balance_check': True,
                'data_quality_validation': True
            },
            'model_updates': {
                'hyperparameter_tuning': True,
                'feature_engineering_review': True,
                'model_architecture_review': new_data_size > 10000
            },
            'validation_strategy': {
                'cross_validation_folds': 5,
                'holdout_test_size': 0.2,
                'performance_baseline': current_model_performance
            },
            'deployment_strategy': {
                'gradual_rollout': True,
                'a_b_testing': True,
                'rollback_plan': True
            },
            'estimated_timeline': {
                'data_preparation': '1-2 days',
                'model_training': '2-3 days',
                'validation_testing': '1 day',
                'deployment': '1 day'
            }
        }

        return plan
