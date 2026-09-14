"""Model training and selection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
from typing import Dict, Tuple, Any
import logging
import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    """Handles model training and selection"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_model = None

    def initialize_models(self) -> Dict:
        """Initialize different ML models"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'mlp': MLPClassifier(random_state=42, max_iter=500),
            'xgboost': XGBClassifier(random_state=42, eval_metric='mlogloss')
        }

        self.logger.info(f"Initialized {len(models)} models")
        return models

    def get_hyperparameter_grids(self) -> Dict:
        """Get hyperparameter grids for GridSearch"""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 6]
            }
        }

        return param_grids

    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          param_grid: Dict = None) -> Any:
        """Train a single model with optional hyperparameter tuning"""
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5,
                scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train all models and return trained models"""
        models = self.initialize_models()
        param_grids = self.get_hyperparameter_grids()
        trained_models = {}

        for name, model in models.items():
            self.logger.info(f"Training {name}...")

            param_grid = param_grids.get(name)
            trained_model = self.train_single_model(model, X_train, y_train, param_grid)
            trained_models[name] = trained_model

            cv_scores = cross_val_score(trained_model, X_train, y_train,
                                      cv=5, scoring='f1_weighted')
            self.logger.info(f"{name} CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.models = trained_models
        return trained_models

    def select_best_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[str, Any]:
        """Select best model based on validation performance"""
        best_score = 0
        best_model_name = None

        performance_results = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            performance_results[name] = score

            if score > best_score:
                best_score = score
                best_model_name = name

        self.logger.info("Model Performance on Validation Set:")
        for name, score in performance_results.items():
            self.logger.info(f"{name}: {score:.4f}")

        self.best_model = self.models[best_model_name]
        self.logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")

        return best_model_name, self.best_model

    def save_model(self, model: Any, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> Any:
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        self.logger.info(f"Model loaded from {filepath}")
        return model
