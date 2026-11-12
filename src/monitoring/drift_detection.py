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

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection with 30s timeout and WAL journal mode enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def setup_database(self):
        """Setup monitoring database with multi-tenant partitioning support and schema migrations."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    input_text TEXT,
                    prediction INTEGER,
                    confidence REAL,
                    response_time REAL,
                    tenant_id TEXT DEFAULT 'default'
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    accuracy REAL,
                    f1_score REAL,
                    sample_size INTEGER,
                    tenant_id TEXT DEFAULT 'default'
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    drift_type TEXT,
                    metric_value REAL,
                    threshold REAL,
                    severity TEXT,
                    tenant_id TEXT DEFAULT 'default'
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT UNIQUE NOT NULL,
                    tenant_id TEXT NOT NULL,
                    tenant_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Automatic column migrations for existing databases
            for table in ['predictions', 'drift_alerts', 'performance_metrics']:
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [c[1] for c in cursor.fetchall()]
                if 'tenant_id' not in cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT DEFAULT 'default'")

            # Indexes for tenant-partitioned queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_tenant ON predictions(tenant_id, id DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_tenant ON drift_alerts(tenant_id, id DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_lookup ON api_keys(api_key)")

            conn.commit()
        finally:
            conn.close()

    def log_prediction(self, input_text: str, prediction: int,
                      confidence: float, response_time: float,
                      tenant_id: str = "default"):
        """Log prediction for monitoring with tenant partitioning."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO predictions (timestamp, input_text, prediction, confidence, response_time, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), input_text, prediction, confidence, response_time, tenant_id or "default"))

            conn.commit()
        finally:
            conn.close()

    def create_api_key(self, tenant_id: str, tenant_name: str = None) -> str:
        """Generate and store a new secure API key for a tenant."""
        import secrets
        clean_tenant = tenant_id.strip().lower()
        api_key = f"sp_live_{clean_tenant}_{secrets.token_hex(16)}"
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_keys (api_key, tenant_id, tenant_name)
                VALUES (?, ?, ?)
            ''', (api_key, clean_tenant, tenant_name or clean_tenant))
            conn.commit()
            return api_key
        finally:
            conn.close()

    def validate_api_key(self, api_key: str) -> str:
        """Validate API key and return corresponding tenant_id, or None if invalid."""
        if not api_key:
            return None
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT tenant_id FROM api_keys WHERE api_key = ?', (api_key.strip(),))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List active API keys and their tenants."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT key_id, api_key, tenant_id, tenant_name, created_at FROM api_keys ORDER BY key_id DESC')
            rows = cursor.fetchall()
            return [
                {
                    "key_id": r[0],
                    "api_key_masked": f"{r[1][:12]}...{r[1][-4:]}" if len(r[1]) > 16 else "***",
                    "tenant_id": r[2],
                    "tenant_name": r[3],
                    "created_at": str(r[4])
                }
                for r in rows
            ]
        finally:
            conn.close()

    def detect_data_drift(self, reference_features: np.ndarray,
                          current_features: np.ndarray,
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Detect data drift using statistical tests with empty matrix safety guards"""
        drift_results = {}

        if hasattr(reference_features, 'toarray'):
            reference_features = reference_features.toarray()
        elif not isinstance(reference_features, np.ndarray):
            reference_features = np.asarray(reference_features)

        if hasattr(current_features, 'toarray'):
            current_features = current_features.toarray()
        elif not isinstance(current_features, np.ndarray):
            current_features = np.asarray(current_features)

        if reference_features.size == 0 or current_features.size == 0:
            return {
                'drift_detected_features': 0,
                'total_features': 0,
                'drift_percentage': 0.0,
                'overall_drift': False,
                'feature_results': {}
            }

        if reference_features.ndim != 2 or current_features.ndim != 2:
            return {
                'drift_detected_features': 0,
                'total_features': 0,
                'drift_percentage': 0.0,
                'overall_drift': False,
                'feature_results': {}
            }

        min_features = min(50, reference_features.shape[1], current_features.shape[1])
        if min_features == 0:
            return {
                'drift_detected_features': 0,
                'total_features': 0,
                'drift_percentage': 0.0,
                'overall_drift': False,
                'feature_results': {}
            }

        reference_features = reference_features[:, :min_features]
        current_features = current_features[:, :min_features]

        for i in range(min_features):
            ref_feature = reference_features[:, i]
            curr_feature = current_features[:, i]

            if len(ref_feature) == 0 or len(curr_feature) == 0:
                ks_stat, p_value = 0.0, 1.0
            elif np.all(ref_feature == ref_feature[0]) and np.all(curr_feature == curr_feature[0]):
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
        drift_percentage = drift_count / max(len(drift_results), 1)

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
        """Detect prediction drift using distribution comparison with empty array safety"""
        if len(reference_predictions) == 0 or len(current_predictions) == 0:
            return {
                'chi2_statistic': 0.0,
                'p_value': 1.0,
                'drift_detected': False,
                'reference_distribution': np.zeros(3),
                'current_distribution': np.zeros(3)
            }

        ref_counts = np.bincount(reference_predictions, minlength=3)
        curr_counts = np.bincount(current_predictions, minlength=3)

        if np.sum(ref_counts) == 0 or np.sum(curr_counts) == 0:
            chi2_stat, p_value = 0.0, 1.0
        else:
            chi2_stat, p_value = chi2_contingency([ref_counts, curr_counts])[:2]

        result = {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < threshold,
            'reference_distribution': ref_counts / max(len(reference_predictions), 1),
            'current_distribution': curr_counts / max(len(current_predictions), 1)
        }

        if result['drift_detected']:
            self.log_drift_alert('prediction_drift', chi2_stat, threshold, 'high')

        return result

    def monitor_performance_degradation(self, y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      baseline_accuracy: float,
                                      baseline_f1: float,
                                      threshold: float = 0.05) -> Dict[str, Any]:
        """Monitor for performance degradation with zero-sample guard"""
        if len(y_true) == 0:
            return {
                'current_accuracy': 0.0,
                'current_f1': 0.0,
                'baseline_accuracy': baseline_accuracy,
                'baseline_f1': baseline_f1,
                'accuracy_drop': 0.0,
                'f1_drop': 0.0,
                'performance_degraded': False
            }

        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

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
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, accuracy, f1_score, sample_size)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now(), accuracy, f1, sample_size))

            conn.commit()
        finally:
            conn.close()

    def log_drift_alert(self, drift_type: str, metric_value: float,
                       threshold: float, severity: str,
                       tenant_id: str = "default"):
        """Log drift alert with tenant partitioning."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO drift_alerts (timestamp, drift_type, metric_value, threshold, severity, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), drift_type, metric_value, threshold, severity, tenant_id or "default"))

            conn.commit()
        finally:
            conn.close()

        self.logger.warning(f"Drift alert [{tenant_id}]: {drift_type} - Value: {metric_value}, "
                          f"Threshold: {threshold}, Severity: {severity}")

    def get_monitoring_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Get data for monitoring dashboard with sanitized query days"""
        safe_days = max(1, int(days))
        conn = self._get_connection()
        try:
            performance_df = pd.read_sql_query(f'''
                SELECT * FROM performance_metrics
                WHERE timestamp >= datetime('now', '-{safe_days} days')
                ORDER BY timestamp DESC
            ''', conn)

            predictions_df = pd.read_sql_query(f'''
                SELECT * FROM predictions
                WHERE timestamp >= datetime('now', '-{safe_days} days')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', conn)

            alerts_df = pd.read_sql_query(f'''
                SELECT * FROM drift_alerts
                WHERE timestamp >= datetime('now', '-{safe_days} days')
                ORDER BY timestamp DESC
            ''', conn)
        finally:
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

    def get_recent_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get recent metrics / monitoring dashboard data."""
        return self.get_monitoring_dashboard_data(days=days)


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
            if drift_alerts > 0 or performance_alerts > 0:
                should_retrain = True
                reasons.append("No recent performance data available and drift/alerts detected")
            else:
                reasons.append("No recent ground-truth performance data available")

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
