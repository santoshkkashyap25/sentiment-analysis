import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator


# =====================================================================
# Tests for ModelTrainer (src/models/training.py)
# =====================================================================

def test_select_best_model_weighted_f1_over_accuracy():
    """Verify select_best_model prefers higher weighted F1 over higher raw accuracy."""
    trainer = ModelTrainer({})

    # Highly imbalanced validation data: 90 Positive (label 2), 10 Negative (label 0)
    y_val = np.array([2] * 90 + [0] * 10)
    X_val = np.zeros((100, 2))

    # Model A: Naive majority predictor
    # Always predicts class 2.
    # Accuracy = 90 / 100 = 0.90 (90%)
    # Weighted F1: 0.1 * 0.0 + 0.9 * (2 * 0.9 * 1.0 / 1.9) = 0.8526 (85.26%)
    class MajorityModel:
        def predict(self, X):
            return np.full(len(X), 2)

    # Model B: Balanced predictor
    # Predicts 80 class 2 correctly, 10 class 2 misclassified as class 0
    # Predicts 8 class 0 correctly, 2 class 0 misclassified as class 2
    # Accuracy = (80 + 8) / 100 = 0.88 (88%, LOWER than Model A's 90%)
    # Class 0: TP=8, FP=10, FN=2 -> P=8/18=0.4444, R=8/10=0.80, F1=0.5714
    # Class 2: TP=80, FP=2, FN=10 -> P=80/82=0.9756, R=80/90=0.8889, F1=0.9302
    # Weighted F1: 0.1 * 0.5714 + 0.9 * 0.9302 = 0.8943 (89.43%, HIGHER than Model A's 85.26%)
    class BalancedModel:
        def predict(self, X):
            preds = np.zeros(len(X), dtype=int)
            # First 90 true class 2: 80 predicted as 2, 10 predicted as 0
            preds[:80] = 2
            preds[80:90] = 0
            # Last 10 true class 0: 8 predicted as 0, 2 predicted as 2
            preds[90:98] = 0
            preds[98:100] = 2
            return preds

    trainer.models = {
        "model_majority": MajorityModel(),
        "model_balanced": BalancedModel()
    }

    best_name, best_model = trainer.select_best_model(X_val, y_val)
    # Under raw accuracy, model_majority would win (0.90 vs 0.88).
    # Under weighted F1, model_balanced MUST win (0.8943 vs 0.8526)!
    assert best_name == "model_balanced"
    assert best_model is trainer.models["model_balanced"]


def test_select_best_model_empty_models_raises_value_error():
    """Verify select_best_model raises ValueError when self.models is empty."""
    trainer = ModelTrainer({})
    trainer.models = {}
    with pytest.raises(ValueError, match="No models available to select from"):
        trainer.select_best_model(np.zeros((5, 2)), np.zeros(5))


def test_train_all_models_skips_redundant_cv_when_param_grid():
    """Verify train_all_models skips redundant cross_val_score when param_grid is present."""
    trainer = ModelTrainer({})

    dummy_model_with_grid = MagicMock()
    dummy_model_without_grid = MagicMock()

    with patch.object(trainer, "initialize_models", return_value={
        "model_with_grid": dummy_model_with_grid,
        "model_without_grid": dummy_model_without_grid
    }), patch.object(trainer, "get_hyperparameter_grids", return_value={
        "model_with_grid": {"param": [1, 2]}
        # model_without_grid has no entry in param_grids
    }), patch.object(trainer, "train_single_model", side_effect=lambda m, X, y, grid: m), \
       patch("src.models.training.cross_val_score", return_value=np.array([0.8, 0.85, 0.82])) as mock_cv:

        X_train = np.zeros((20, 2))
        y_train = np.array([0] * 10 + [1] * 10)

        trained_models = trainer.train_all_models(X_train, y_train)

        assert "model_with_grid" in trained_models
        assert "model_without_grid" in trained_models

        # cross_val_score should ONLY be called once (for model_without_grid)
        assert mock_cv.call_count == 1
        mock_cv.assert_called_once_with(
            dummy_model_without_grid, X_train, y_train, cv=5, scoring="f1_weighted"
        )


def test_train_single_model_logs_grid_search_score(caplog):
    """Verify train_single_model logs grid_search.best_score_ when param_grid is present."""
    import logging
    from sklearn.linear_model import LogisticRegression

    trainer = ModelTrainer({})
    model = LogisticRegression(max_iter=100)
    param_grid = {"C": [0.1, 1.0]}
    X_train = np.random.randn(30, 4)
    y_train = np.array([0] * 15 + [1] * 15)

    with caplog.at_level(logging.INFO):
        best_est = trainer.train_single_model(model, X_train, y_train, param_grid=param_grid)

    assert best_est is not None
    assert "GridSearch CV F1 Score:" in caplog.text
    assert "Best parameters:" in caplog.text


def test_train_single_model_without_param_grid():
    """Verify train_single_model fits directly when param_grid is None."""
    trainer = ModelTrainer({})
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=100)
    X_train = np.random.randn(20, 2)
    y_train = np.array([0] * 10 + [1] * 10)

    fitted_model = trainer.train_single_model(model, X_train, y_train, param_grid=None)
    assert fitted_model is model
    assert hasattr(fitted_model, "classes_")


def test_save_and_load_model(tmp_path):
    """Verify save_model and load_model serialize and deserialize models correctly."""
    trainer = ModelTrainer({})
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)

    filepath = str(tmp_path / "model.pkl")
    trainer.save_model(model, filepath)

    loaded_model = trainer.load_model(filepath)
    np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))


# =====================================================================
# Tests for ModelEvaluator (src/models/evaluation.py)
# =====================================================================

def test_plot_confusion_matrix_save_and_close(tmp_path):
    """Verify plot_confusion_matrix saves to save_path, does not call plt.show, and closes figures."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    save_file = str(tmp_path / "nested" / "output_plots" / "cm.png")

    plt.close("all")
    with patch("matplotlib.pyplot.show") as mock_show:
        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_file)
        mock_show.assert_not_called()

    # File must exist and have non-zero size
    assert Path(save_file).exists()
    assert Path(save_file).stat().st_size > 0

    # Figure must be closed (no open figures in matplotlib)
    assert len(plt.get_fignums()) == 0

    # Returned confusion matrix matches dimensions
    assert cm.shape == (3, 3)


def test_plot_confusion_matrix_without_save_path():
    """Verify plot_confusion_matrix works without save_path and closes figures."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    plt.close("all")
    with patch("matplotlib.pyplot.show") as mock_show:
        cm = evaluator.plot_confusion_matrix(y_true, y_pred)
        mock_show.assert_not_called()

    assert len(plt.get_fignums()) == 0
    assert cm.shape == (3, 3)


def test_plot_confusion_matrix_import_error_fallback(monkeypatch):
    """Verify plot_confusion_matrix falls back to confusion_matrix(..., labels=labels) on ImportError."""
    import builtins
    evaluator = ModelEvaluator()
    y_true = np.array([0, 2, 0])
    y_pred = np.array([0, 0, 2])

    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name in ("matplotlib", "seaborn"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    cm = evaluator.plot_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)


def test_calculate_per_class_metrics_missing_classes():
    """Verify calculate_per_class_metrics returns 3 rows with 0.0 metrics when a class is absent."""
    evaluator = ModelEvaluator()
    # Class 1 (Neutral) is missing from both true and predicted
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 2])

    metrics_df = evaluator.calculate_per_class_metrics(y_true, y_pred)
    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 3
    assert list(metrics_df["Class"]) == ["Negative", "Neutral", "Positive"]

    # Active classes should have 1.0
    neg_row = metrics_df[metrics_df["Class"] == "Negative"].iloc[0]
    pos_row = metrics_df[metrics_df["Class"] == "Positive"].iloc[0]
    assert neg_row["Precision"] == 1.0
    assert neg_row["Recall"] == 1.0
    assert neg_row["F1-Score"] == 1.0
    assert pos_row["Precision"] == 1.0
    assert pos_row["Recall"] == 1.0
    assert pos_row["F1-Score"] == 1.0

    # Missing class 1 (Neutral) should produce 0.0 with zero_division=0
    neu_row = metrics_df[metrics_df["Class"] == "Neutral"].iloc[0]
    assert neu_row["Precision"] == 0.0
    assert neu_row["Recall"] == 0.0
    assert neu_row["F1-Score"] == 0.0


def test_calculate_per_class_metrics_custom_labels_and_names():
    """Verify calculate_per_class_metrics handles custom classes and labels."""
    evaluator = ModelEvaluator()
    y_true = np.array([10, 20, 10])
    y_pred = np.array([10, 10, 10])

    metrics_df = evaluator.calculate_per_class_metrics(
        y_true, y_pred, target_names=["ClassA", "ClassB"], labels=[10, 20]
    )
    assert len(metrics_df) == 2
    assert list(metrics_df["Class"]) == ["ClassA", "ClassB"]


def test_calculate_per_class_metrics_labels_without_target_names():
    """Verify calculate_per_class_metrics autogenerates target_names when labels are supplied without target_names."""
    evaluator = ModelEvaluator()
    y_true = np.array([10, 20, 10])
    y_pred = np.array([10, 10, 10])

    metrics_df = evaluator.calculate_per_class_metrics(y_true, y_pred, labels=[10, 20])
    assert len(metrics_df) == 2
    assert list(metrics_df["Class"]) == ["Class_10", "Class_20"]


def test_generate_classification_report_missing_classes():
    """Verify generate_classification_report handles missing classes without crashing."""
    evaluator = ModelEvaluator()
    # Only classes 0 and 2 present
    y_true = np.array([0, 2, 0, 2])
    y_pred = np.array([0, 2, 2, 2])

    report = evaluator.generate_classification_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "Negative" in report
    assert "Neutral" in report
    assert "Positive" in report


def test_calculate_basic_metrics_zero_division():
    """Verify calculate_basic_metrics computes cleanly with zero_division."""
    evaluator = ModelEvaluator()
    y_true = np.array([0, 2])
    y_pred = np.array([0, 0])

    metrics = evaluator.calculate_basic_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert isinstance(metrics["f1"], float)


def test_evaluate_model_comprehensive():
    """Verify evaluate_model_comprehensive returns full results dictionary."""
    evaluator = ModelEvaluator()

    class DummyModel:
        def predict(self, X):
            return np.array([0, 1, 2])

        def predict_proba(self, X):
            return np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])

    model = DummyModel()
    X_test = np.zeros((3, 2))
    y_test = np.array([0, 1, 2])

    results = evaluator.evaluate_model_comprehensive(model, X_test, y_test)
    assert "basic_metrics" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results
    assert "per_class_metrics" in results
    assert "predictions" in results
    assert "probabilities" in results
    assert results["confusion_matrix"].shape == (3, 3)


def test_compare_models():
    """Verify compare_models sorts models by F1 descending."""
    evaluator = ModelEvaluator()

    class HighPerfModel:
        def predict(self, X):
            return np.array([0, 1, 2])

    class LowPerfModel:
        def predict(self, X):
            return np.array([2, 2, 2])

    models = {"low_perf": LowPerfModel(), "high_perf": HighPerfModel()}
    X_test = np.zeros((3, 2))
    y_test = np.array([0, 1, 2])

    df = evaluator.compare_models(models, X_test, y_test)
    assert list(df["model"]) == ["high_perf", "low_perf"]
    assert df["f1"].iloc[0] >= df["f1"].iloc[1]
