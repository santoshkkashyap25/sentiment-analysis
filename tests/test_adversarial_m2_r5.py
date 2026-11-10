"""Adversarial stress testing suite for Milestone 2 (R5.1, R5.2, R5.3, R5.4).

Empirical challenger tests for:
- ModelTrainer.select_best_model
- ModelTrainer.train_all_models
- ModelEvaluator.plot_confusion_matrix
- ModelEvaluator.calculate_per_class_metrics
"""

import os
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.models.evaluation import ModelEvaluator
from src.models.training import ModelTrainer


# =====================================================================
# Task 1: Adversarial Stress Testing of ModelTrainer.select_best_model
# =====================================================================

class TestSelectBestModelAdversarial:
    """Stress tests for ModelTrainer.select_best_model."""

    def test_all_candidate_models_predict_completely_wrong_labels_zero_score(self):
        """When all models achieve score = 0.0, verify no KeyError: None occurs and a valid model is returned."""
        class InverseModel0:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)  # predicts 0 for all

        class InverseModel1:
            def predict(self, X):
                return np.ones(len(X), dtype=int)   # predicts 1 for all

        trainer = ModelTrainer({})
        trainer.models = {
            "model_pred_0": InverseModel0(),
            "model_pred_1": InverseModel1()
        }

        # True labels are all 2 (Positive). Any prediction of 0 or 1 gets score 0.0.
        X_val = np.zeros((30, 4))
        y_val = np.full(30, 2, dtype=int)

        best_name, best_model = trainer.select_best_model(X_val, y_val)

        # Must not raise KeyError: None, must return a valid registered model
        assert best_name in trainer.models, f"Returned name '{best_name}' not in trainer.models"
        assert best_model is trainer.models[best_name]
        assert trainer.best_model is best_model

    def test_empty_models_raises_value_error(self):
        """When trainer.models = {}, verify ValueError is raised cleanly."""
        trainer = ModelTrainer({})
        trainer.models = {}

        X_val = np.zeros((10, 2))
        y_val = np.zeros(10)

        with pytest.raises(ValueError, match="No models available to select from"):
            trainer.select_best_model(X_val, y_val)

    def test_imbalanced_dataset_majority_accuracy_vs_balanced_weighted_f1(self):
        """Verify model selection strictly prioritizes weighted F1 over raw accuracy.

        Constructs an imbalanced dataset where:
        - Model A (Majority Classifier) achieves 90.00% accuracy but 0.8526 weighted F1.
        - Model B (Balanced Classifier) achieves 88.00% accuracy (LOWER) but 0.8944 weighted F1 (HIGHER).

        Select_best_model must select Model B.
        """
        # Ground truth: 90 Positive (2), 10 Negative (0)
        y_val = np.array([2] * 90 + [0] * 10)
        X_val = np.zeros((100, 3))

        class MajorityModel:
            def predict(self, X):
                return np.full(len(X), 2, dtype=int)

        class BalancedModel:
            def predict(self, X):
                # 80 class 2 correct, 10 class 2 predicted as 0
                # 8 class 0 correct, 2 class 0 predicted as 2
                preds = np.zeros(len(X), dtype=int)
                preds[:80] = 2
                preds[80:90] = 0
                preds[90:98] = 0
                preds[98:100] = 2
                return preds

        acc_maj = accuracy_score(y_val, MajorityModel().predict(X_val))
        f1_maj = f1_score(y_val, MajorityModel().predict(X_val), average='weighted', zero_division=0)

        acc_bal = accuracy_score(y_val, BalancedModel().predict(X_val))
        f1_bal = f1_score(y_val, BalancedModel().predict(X_val), average='weighted', zero_division=0)

        assert acc_maj > acc_bal, f"Setup verification: acc_maj ({acc_maj}) must exceed acc_bal ({acc_bal})"
        assert f1_bal > f1_maj, f"Setup verification: f1_bal ({f1_bal}) must exceed f1_maj ({f1_maj})"

        trainer = ModelTrainer({})
        trainer.models = {
            "model_majority_high_acc": MajorityModel(),
            "model_balanced_high_f1": BalancedModel()
        }

        best_name, best_model = trainer.select_best_model(X_val, y_val)
        assert best_name == "model_balanced_high_f1", (
            f"Expected 'model_balanced_high_f1' to be selected based on weighted F1 ({f1_bal:.4f} vs {f1_maj:.4f}), "
            f"but selected '{best_name}'"
        )
        assert best_model is trainer.models["model_balanced_high_f1"]

    def test_imbalanced_3class_majority_vs_balanced(self):
        """Verify 3-class imbalanced dataset where majority model has higher accuracy than balanced model."""
        y_val = np.array([2] * 90 + [1] * 5 + [0] * 5)
        X_val = np.zeros((100, 2))

        class Majority3ClassModel:
            def predict(self, X):
                return np.full(len(X), 2, dtype=int)

        class Balanced3ClassModel:
            def predict(self, X):
                preds = np.zeros(len(X), dtype=int)
                preds[:80] = 2
                preds[80:90] = 1
                preds[90:95] = 1
                preds[95:99] = 0
                preds[99:100] = 2
                return preds

        acc_a = accuracy_score(y_val, Majority3ClassModel().predict(X_val))
        f1_a = f1_score(y_val, Majority3ClassModel().predict(X_val), average='weighted', zero_division=0)

        acc_b = accuracy_score(y_val, Balanced3ClassModel().predict(X_val))
        f1_b = f1_score(y_val, Balanced3ClassModel().predict(X_val), average='weighted', zero_division=0)

        assert acc_a > acc_b  # 90% vs 89%
        assert f1_b > f1_a    # 0.9115 vs 0.8526

        trainer = ModelTrainer({})
        trainer.models = {
            "majority": Majority3ClassModel(),
            "balanced": Balanced3ClassModel()
        }

        best_name, _ = trainer.select_best_model(X_val, y_val)
        assert best_name == "balanced"

    def test_single_model_in_models(self):
        """Verify select_best_model works when only 1 model is available."""
        class SingleModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        trainer = ModelTrainer({})
        trainer.models = {"single": SingleModel()}
        name, model = trainer.select_best_model(np.zeros((5, 2)), np.zeros(5, dtype=int))
        assert name == "single"
        assert model is trainer.models["single"]

    def test_all_models_tied_positive_scores(self):
        """When multiple models have identical non-zero scores, verify a valid model is selected deterministically."""
        class FixedModel:
            def predict(self, X):
                return np.array([0, 1, 2])

        trainer = ModelTrainer({})
        trainer.models = {
            "first_model": FixedModel(),
            "second_model": FixedModel()
        }
        name, model = trainer.select_best_model(np.zeros((3, 2)), np.array([0, 1, 2]))
        assert name == "first_model"  # First model with maximum score is retained
        assert model is trainer.models["first_model"]


# =====================================================================
# Task 2: Adversarial Stress Testing of ModelTrainer.train_all_models
# =====================================================================

class TestTrainAllModelsAdversarial:
    """Stress tests for ModelTrainer.train_all_models."""

    def test_cv_never_invoked_when_param_grid_present(self):
        """Verify with mock inspection that cross_val_score is NOT invoked when param_grid is present."""
        trainer = ModelTrainer({})

        m1 = MagicMock(name="model_with_grid_1")
        m2 = MagicMock(name="model_with_grid_2")

        with patch.object(trainer, "initialize_models", return_value={"m1": m1, "m2": m2}), \
             patch.object(trainer, "get_hyperparameter_grids", return_value={"m1": {"alpha": [0.1]}, "m2": {"beta": [1]}}), \
             patch.object(trainer, "train_single_model", side_effect=lambda model, X, y, grid: model), \
             patch("src.models.training.cross_val_score") as mock_cv:

            X_train = np.zeros((15, 2))
            y_train = np.array([0, 1, 2] * 5)

            trained = trainer.train_all_models(X_train, y_train)

            mock_cv.assert_not_called()
            assert len(trained) == 2

    def test_cv_invoked_only_when_param_grid_is_none_or_empty(self):
        """Verify cross_val_score is invoked ONLY for models whose param_grid is None or empty."""
        trainer = ModelTrainer({})

        m_grid = MagicMock(name="model_with_grid")
        m_none = MagicMock(name="model_none_grid")
        m_empty = MagicMock(name="model_empty_grid")

        with patch.object(trainer, "initialize_models", return_value={
            "grid_model": m_grid,
            "none_model": m_none,
            "empty_model": m_empty
        }), patch.object(trainer, "get_hyperparameter_grids", return_value={
            "grid_model": {"C": [0.1, 1.0]},
            "empty_model": {}  # empty dict evaluates to falsy in `if not param_grid:`
        }), patch.object(trainer, "train_single_model", side_effect=lambda model, X, y, grid: model), \
           patch("src.models.training.cross_val_score", return_value=np.array([0.9, 0.9])) as mock_cv:

            X_train = np.zeros((15, 2))
            y_train = np.array([0, 1, 2] * 5)

            trained = trainer.train_all_models(X_train, y_train)

            assert len(trained) == 3
            # cross_val_score must be called for none_model and empty_model, but NOT grid_model
            assert mock_cv.call_count == 2
            called_models = [call.args[0] for call in mock_cv.call_args_list]
            assert m_none in called_models
            assert m_empty in called_models
            assert m_grid not in called_models

    def test_mixed_model_suite_cv_counts(self):
        """5 models: 3 with param_grids, 2 without. Verify exactly 2 calls to cross_val_score."""
        trainer = ModelTrainer({})
        models = {f"m_{i}": MagicMock(name=f"m_{i}") for i in range(5)}
        grids = {
            "m_0": {"param": [1]},
            "m_1": {"param": [2]},
            "m_2": {"param": [3]},
            # m_3 and m_4 have no grid
        }

        with patch.object(trainer, "initialize_models", return_value=models), \
             patch.object(trainer, "get_hyperparameter_grids", return_value=grids), \
             patch.object(trainer, "train_single_model", side_effect=lambda m, X, y, g: m), \
             patch("src.models.training.cross_val_score", return_value=np.array([0.85])) as mock_cv:

            X_train = np.zeros((10, 2))
            y_train = np.array([0, 1] * 5)

            trained = trainer.train_all_models(X_train, y_train)
            assert len(trained) == 5
            assert mock_cv.call_count == 2


# =====================================================================
# Task 3: Adversarial Stress Testing of ModelEvaluator.plot_confusion_matrix
# =====================================================================

class TestPlotConfusionMatrixAdversarial:
    """Stress tests for ModelEvaluator.plot_confusion_matrix."""

    def test_plt_show_is_never_called_in_headless_environment(self):
        """In headless/CI environment, verify plt.show() is never called under any circumstances."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 2, 1, 0])

        with patch("matplotlib.pyplot.show") as mock_show:
            evaluator.plot_confusion_matrix(y_true, y_pred)
            mock_show.assert_not_called()

        with patch("matplotlib.pyplot.show") as mock_show:
            evaluator.plot_confusion_matrix(y_true, y_pred, save_path="dummy.png")
            mock_show.assert_not_called()

    def test_deeply_nested_nonexistent_directory_creation_and_save(self, tmp_path):
        """Verify deeply nested path 'tmp/deep/a/b/c/cm.png' is automatically created and image saved."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        nested_save_path = tmp_path / "deep" / "a" / "b" / "c" / "cm.png"
        assert not nested_save_path.parent.exists(), "Parent dir should not exist prior to test"

        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=str(nested_save_path))

        assert nested_save_path.exists(), f"Image was not saved to {nested_save_path}"
        assert nested_save_path.stat().st_size > 0, "Saved image file is empty"
        assert cm.shape == (3, 3)

    def test_resource_leak_check_all_figures_closed_even_on_error(self, tmp_path):
        """Verify len(matplotlib.pyplot.get_fignums()) == 0 before and after, even if savefig raises error."""
        plt.close("all")
        assert len(plt.get_fignums()) == 0, "Precondition failed: open figures exist"

        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])

        # Normal run
        evaluator.plot_confusion_matrix(y_true, y_pred)
        assert len(plt.get_fignums()) == 0, "Figure was leaked after plot_confusion_matrix without save_path"

        save_path = str(tmp_path / "leak_check.png")
        evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        assert len(plt.get_fignums()) == 0, "Figure was leaked after plot_confusion_matrix with save_path"

        # Adversarial run: simulate error during savefig
        with patch("matplotlib.pyplot.savefig", side_effect=IOError("Disk full or permission denied")):
            with pytest.raises(IOError):
                evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)

        # Even on error, finally: plt.close() MUST prevent figure leak!
        assert len(plt.get_fignums()) == 0, "Figure was leaked when savefig raised an exception"


# =====================================================================
# Task 4: Adversarial Stress Testing of ModelEvaluator.calculate_per_class_metrics
# =====================================================================

class TestCalculatePerClassMetricsAdversarial:
    """Stress tests for ModelEvaluator.calculate_per_class_metrics."""

    def test_single_class_only_class_0(self):
        """Test with y_true = [0, 0, 0] and y_pred = [0, 0, 0].

        Verify DataFrame has exactly 3 rows ('Negative', 'Neutral', 'Positive')
        and no array length mismatch or zero division errors occur.
        """
        evaluator = ModelEvaluator()
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any warning (e.g. UndefinedMetricWarning) fails test
            df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert list(df["Class"]) == ["Negative", "Neutral", "Positive"]
        assert list(df.columns) == ["Class", "Precision", "Recall", "F1-Score"]

        # Class 0 (Negative): TP=3, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        neg = df.loc[df["Class"] == "Negative"].iloc[0]
        assert neg["Precision"] == 1.0
        assert neg["Recall"] == 1.0
        assert neg["F1-Score"] == 1.0

        # Classes 1 and 2: Absent -> zero_division=0 gives 0.0
        for cls_name in ["Neutral", "Positive"]:
            row = df.loc[df["Class"] == cls_name].iloc[0]
            assert row["Precision"] == 0.0
            assert row["Recall"] == 0.0
            assert row["F1-Score"] == 0.0

    def test_single_class_only_class_1(self):
        """Test with y_true = [1, 1, 1] and y_pred = [1, 1, 1]."""
        evaluator = ModelEvaluator()
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert len(df) == 3
        neu = df.loc[df["Class"] == "Neutral"].iloc[0]
        assert neu["Precision"] == 1.0
        assert neu["Recall"] == 1.0
        assert neu["F1-Score"] == 1.0

        for cls_name in ["Negative", "Positive"]:
            row = df.loc[df["Class"] == cls_name].iloc[0]
            assert row["Precision"] == 0.0
            assert row["Recall"] == 0.0
            assert row["F1-Score"] == 0.0

    def test_single_class_only_class_2(self):
        """Test with y_true = [2, 2, 2] and y_pred = [2, 2, 2]."""
        evaluator = ModelEvaluator()
        y_true = np.array([2, 2, 2])
        y_pred = np.array([2, 2, 2])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert len(df) == 3
        pos = df.loc[df["Class"] == "Positive"].iloc[0]
        assert pos["Precision"] == 1.0
        assert pos["Recall"] == 1.0
        assert pos["F1-Score"] == 1.0

        for cls_name in ["Negative", "Neutral"]:
            row = df.loc[df["Class"] == cls_name].iloc[0]
            assert row["Precision"] == 0.0
            assert row["Recall"] == 0.0
            assert row["F1-Score"] == 0.0

    def test_missing_middle_class_1(self):
        """Classes 0 and 2 present, class 1 completely absent in true and pred."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 2, 0, 2])
        y_pred = np.array([0, 2, 2, 0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert len(df) == 3
        assert list(df["Class"]) == ["Negative", "Neutral", "Positive"]

        neu = df.loc[df["Class"] == "Neutral"].iloc[0]
        assert neu["Precision"] == 0.0
        assert neu["Recall"] == 0.0
        assert neu["F1-Score"] == 0.0

    def test_completely_disjoint_predictions(self):
        """True labels are all 0, but model predicts all 2."""
        evaluator = ModelEvaluator()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([2, 2, 2, 2])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = evaluator.calculate_per_class_metrics(y_true, y_pred)

        assert len(df) == 3
        # Class 0: Recall=0, Precision=0, F1=0
        neg = df.loc[df["Class"] == "Negative"].iloc[0]
        assert neg["Precision"] == 0.0
        assert neg["Recall"] == 0.0
        assert neg["F1-Score"] == 0.0

        # Class 1: Absent
        neu = df.loc[df["Class"] == "Neutral"].iloc[0]
        assert neu["Precision"] == 0.0
        assert neu["Recall"] == 0.0

        # Class 2: Precision=0, Recall=0, F1=0
        pos = df.loc[df["Class"] == "Positive"].iloc[0]
        assert pos["Precision"] == 0.0
        assert pos["Recall"] == 0.0
        assert pos["F1-Score"] == 0.0

    def test_single_element_arrays(self):
        """Single element array y_true=[1], y_pred=[1]."""
        evaluator = ModelEvaluator()
        y_true = np.array([1])
        y_pred = np.array([1])

        df = evaluator.calculate_per_class_metrics(y_true, y_pred)
        assert len(df) == 3
        neu = df.loc[df["Class"] == "Neutral"].iloc[0]
        assert neu["Precision"] == 1.0
        assert neu["Recall"] == 1.0
        assert neu["F1-Score"] == 1.0

    def test_generate_classification_report_single_class_no_crash(self):
        """Verify generate_classification_report also succeeds with single class without raising exceptions."""
        evaluator = ModelEvaluator()
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = evaluator.generate_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert "Negative" in report
        assert "Neutral" in report
        assert "Positive" in report
