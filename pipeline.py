import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

from src.data.ingestion import DataIngestionPipeline
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.monitoring.drift_detection import ModelMonitor, RetrainingStrategy
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FeedbackAnalysisPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_data_ingestion(self):
        """Run data ingestion step"""
        self.logger.info("Starting data ingestion...")

        ingestion = DataIngestionPipeline(self.config)

        sources = [
            {
                'type': 'kaggle',
                'name': 'amazon_reviews',
                'dataset': 'arhamrumi/amazon-product-reviews',
                'download_path': 'data/raw',
                'filename': 'Reviews.csv'
            }
        ]

        raw_data = ingestion.ingest_multiple_sources(sources)
        raw_data.to_csv('data/raw/Reviews.csv', index=False)
        self.logger.info(f"Raw data saved: {len(raw_data)} records")

        return raw_data

    def run_preprocessing(self, raw_data):
        """Run data preprocessing step"""
        self.logger.info("Starting data preprocessing...")

        preprocessor = DataPreprocessor(self.config)
        processed_data = preprocessor.preprocess_pipeline(raw_data)

        processed_data.to_csv('data/processed/processed_reviews.csv', index=False)
        self.logger.info(f"Processed data saved: {len(processed_data)} records")

        return processed_data

    def run_feature_engineering_splits(self, train_df, val_df, test_df):
        """Run feature engineering on splits without data leakage"""
        self.logger.info("Starting leak-free feature engineering...")

        feature_engineer = FeatureEngineer(self.config)

        # Fit extractors strictly on training set
        X_train, y_train = feature_engineer.engineer_features(
            train_df, 'reviewText_clean', is_training=True, include_embeddings=False
        )

        # Transform validation and test sets
        X_val, y_val = feature_engineer.engineer_features(
            val_df, 'reviewText_clean', is_training=False, include_embeddings=False
        )
        X_test, y_test = feature_engineer.engineer_features(
            test_df, 'reviewText_clean', is_training=False, include_embeddings=False
        )

        feature_engineer.save_feature_extractors('data/models/feature_extractors.pkl')
        return (X_train, X_val, X_test, y_train, y_val, y_test), feature_engineer

    def run_model_training(self, X_train, y_train, X_val, y_val):
        """Run model training and selection on balanced training data"""
        self.logger.info("Starting model training on balanced training split...")

        preprocessor = DataPreprocessor(self.config)
        X_train_balanced, y_train_balanced = preprocessor.handle_imbalanced_data(
            X_train, y_train, strategy='smote'
        )

        trainer = ModelTrainer(self.config)
        trained_models = trainer.train_all_models(X_train_balanced, y_train_balanced)

        best_model_name, best_model = trainer.select_best_model(X_val, y_val)
        trainer.save_model(best_model, 'data/models/best_model.pkl')

        return best_model, trained_models

    def run_model_evaluation(self, model, data_splits, trained_models):
        """Run model evaluation step"""
        self.logger.info("Starting model evaluation...")

        X_train, X_val, X_test, y_train, y_val, y_test = data_splits

        evaluator = ModelEvaluator()

        test_results = evaluator.evaluate_model_comprehensive(
            model, X_test, y_test
        )

        model_comparison = evaluator.compare_models(trained_models, X_test, y_test)

        self.logger.info("Model Comparison Results:")
        self.logger.info(f"\n{model_comparison}")

        return test_results, model_comparison

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        try:
            os.makedirs('data/raw', exist_ok=True)
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('data/models', exist_ok=True)
            os.makedirs('logs', exist_ok=True)

            raw_data_path = "data/raw/Reviews.csv"
            processed_data_path = "data/processed/processed_reviews.csv"

            if os.path.exists(raw_data_path):
                self.logger.info(f"Found existing raw dataset at {raw_data_path}, skipping ingestion.")
                raw_data = pd.read_csv(raw_data_path)
            else:
                raw_data = self.run_data_ingestion()

            if os.path.exists(processed_data_path):
                self.logger.info(f"Found processed dataset at {processed_data_path}, skipping preprocessing.")
                processed_data = pd.read_csv(processed_data_path)
            else:
                processed_data = self.run_preprocessing(raw_data)

            MAX_ROWS = 5000
            if len(processed_data) > MAX_ROWS:
                self.logger.info(f"Limiting dataset from {len(processed_data)} to {MAX_ROWS} rows for memory efficiency.")
                samples_per_class = int(MAX_ROWS / 3)
                sampled = []
                for _, group in processed_data.groupby('sentiment'):
                    sampled.append(group.sample(min(len(group), samples_per_class), random_state=self.config.get('random_state', 42)))
                processed_data = pd.concat(sampled, ignore_index=True)

            # Split raw text records first to eliminate data leakage
            df_temp, test_df = train_test_split(
                processed_data, test_size=0.2, stratify=processed_data['sentiment'], random_state=42
            )
            train_df, val_df = train_test_split(
                df_temp, test_size=0.25, stratify=df_temp['sentiment'], random_state=42
            )
            self.logger.info(f"Dataset split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

            data_splits, feature_engineer = self.run_feature_engineering_splits(train_df, val_df, test_df)
            X_train, X_val, X_test, y_train, y_val, y_test = data_splits

            best_model, trained_models = self.run_model_training(X_train, y_train, X_val, y_val)

            test_results, model_comparison = self.run_model_evaluation(
                best_model, data_splits, trained_models
            )

            # --- Monitoring ---
            monitor = ModelMonitor()
            baseline_accuracy = test_results['basic_metrics']['accuracy']
            baseline_f1 = test_results['basic_metrics']['f1']

            monitor.log_performance_metrics(
                accuracy=baseline_accuracy,
                f1=baseline_f1,
                sample_size=len(y_test)
            )

            self.logger.info(f"Logged performance: acc={baseline_accuracy:.4f}, f1={baseline_f1:.4f}")

            # --- Save Reference Features for Drift Monitoring ---
            ref_path = "data/processed/features_reference.pkl"
            import pickle
            with open(ref_path, "wb") as f:
                pickle.dump(X_train[:1000], f)
            self.logger.info("Saved reference baseline features for future drift comparison.")

            drift_result = monitor.detect_data_drift(X_train[:1000], X_test)
            self.logger.info(f"Baseline drift check (Train vs Test): drift_detected={drift_result['overall_drift']}")

            # --- Retraining Decision ---
            strategy = RetrainingStrategy(config=self.config)
            retrain_decision = strategy.should_retrain(monitor)

            if retrain_decision["should_retrain"]:
                self.logger.warning(f"Retraining triggered: {retrain_decision['reasons']}")

                retraining_plan = strategy.create_retraining_plan(
                    new_data_size=len(processed_data),
                    current_model_performance={'accuracy': baseline_accuracy, 'f1': baseline_f1}
                )

                self.logger.info("Starting retraining process...")
                best_model, trained_models = self.run_model_training(X_train, y_train, X_val, y_val)
                test_results, model_comparison = self.run_model_evaluation(best_model, data_splits, trained_models)

                model_path = "data/models/best_model_retrained.pkl"
                import joblib
                joblib.dump(best_model, model_path)
                self.logger.info(f"Retrained model saved at {model_path}")

                import pickle
                with open(ref_path, "wb") as f:
                    pickle.dump(X_train[:1000], f)

                monitor.log_performance_metrics(
                    accuracy=test_results['basic_metrics']['accuracy'],
                    f1=test_results['basic_metrics']['f1'],
                    sample_size=len(y_test)
                )

                self.logger.info("Retraining completed successfully.")

            else:
                self.logger.info("Model healthy — no retraining required.")

            self.logger.info("Pipeline completed successfully!")

            return {
                'model': best_model,
                'feature_engineer': feature_engineer,
                'test_results': test_results,
                'model_comparison': model_comparison,
                'monitor': monitor,
                'retrain_decision': retrain_decision
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

    def run_transformer_pipeline(
        self,
        sample_size: int = 120000,
        full_dataset: bool = False,
        epochs: int = 3,
        batch_size: int = 16,
        model_name: str = "roberta-base",
        use_cosine: bool = True,
        use_swa: bool = True,
        export_onnx: bool = True
    ):
        """Run end-to-end Pretrained Transformer Fine-Tuning with GPU, SWA, Cosine LR, and ONNX INT8 export."""
        import json
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
        from torch.utils.data import DataLoader
        from src.models.transformer_trainer import TransformerTrainer, FeedbackDataset
        from src.models.threshold_calibration import ThresholdCalibrator

        self.logger.info("=" * 60)
        self.logger.info(f"STARTING TRANSFORMER GPU PIPELINE ({model_name.upper()} NVIDIA ACCELERATED)")
        self.logger.info("=" * 60)

        # Prevent Windows from entering modern standby / suspending GPU during background training
        if sys.platform == "win32":
            try:
                import ctypes
                # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001) | ES_AWAYMODE_REQUIRED (0x00000040)
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
                self.logger.info("Windows sleep prevention activated (System & AwayMode enabled for uninterrupted GPU training).")
            except Exception as e:
                self.logger.warning(f"Could not set thread execution state: {e}")

        processed_data_path = "data/processed/processed_reviews.csv"
        raw_data_path = "data/raw/Reviews.csv"

        if os.path.exists(processed_data_path):
            self.logger.info(f"Loading processed reviews from {processed_data_path}...")
            df = pd.read_csv(processed_data_path)
        elif os.path.exists(raw_data_path):
            self.logger.info(f"Loading raw reviews from {raw_data_path}...")
            df = pd.read_csv(raw_data_path)
            preprocessor = DataPreprocessor(self.config)
            df = preprocessor.preprocess_pipeline(df)
        else:
            df = self.run_data_ingestion()
            df = self.run_preprocessing(df)

        trainer = TransformerTrainer(
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            use_cosine=use_cosine,
            use_swa=use_swa
        )

        train_df, val_df, test_df = trainer.prepare_dataset(
            df=df,
            sample_size=sample_size,
            full_dataset=full_dataset
        )

        output_dir = "data/models/transformer_sentiment"
        training_report = trainer.train(train_df, val_df, output_dir=output_dir)

        # 1. Validation Probabilities & Threshold Calibration
        self.logger.info("Running Decision Threshold Calibration on Validation Set...")
        text_col = "fused_text" if "fused_text" in val_df.columns else ("reviewText" if "reviewText" in val_df.columns else "reviewText_clean")
        val_dataset = FeedbackDataset(val_df[text_col].values, val_df["label"].values, trainer.tokenizer, trainer.max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_probs, val_labels = trainer.predict_probabilities(val_loader)

        calibrator = ThresholdCalibrator()
        calibration_results = calibrator.fit(val_probs, val_labels)
        calibrator.save(os.path.join(output_dir, "calibration.json"))

        # 2. Test Set Evaluation (Uncalibrated vs Calibrated)
        self.logger.info("Evaluating on Untouched Held-Out Test Set...")
        test_probs, test_labels = trainer.evaluate_test_set(test_df, text_column=text_col)

        raw_test_preds = np.argmax(test_probs, axis=1)
        calibrated_test_preds = calibrator.predict(test_probs)

        p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(test_labels, raw_test_preds, average=None, zero_division=0)
        p_cal, r_cal, f1_cal, _ = precision_recall_fscore_support(test_labels, calibrated_test_preds, average=None, zero_division=0)

        test_metrics = {
            "uncalibrated": {
                "accuracy": float(accuracy_score(test_labels, raw_test_preds)),
                "weighted_f1": float(f1_score(test_labels, raw_test_preds, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(test_labels, raw_test_preds, average="macro", zero_division=0)),
                "negative_recall": float(r_raw[0]),
                "negative_precision": float(p_raw[0]),
            },
            "calibrated": {
                "accuracy": float(accuracy_score(test_labels, calibrated_test_preds)),
                "weighted_f1": float(f1_score(test_labels, calibrated_test_preds, average="weighted", zero_division=0)),
                "macro_f1": float(f1_score(test_labels, calibrated_test_preds, average="macro", zero_division=0)),
                "negative_recall": float(r_cal[0]),
                "negative_precision": float(p_cal[0]),
            }
        }

        self.logger.info("=== HELD-OUT TEST SET RESULTS ===")
        self.logger.info(
            f"Uncalibrated (argmax): Accuracy={test_metrics['uncalibrated']['accuracy']:.4f} | "
            f"Weighted F1={test_metrics['uncalibrated']['weighted_f1']:.4f} | "
            f"Negative Recall={test_metrics['uncalibrated']['negative_recall']:.4f}"
        )
        self.logger.info(
            f"Calibrated Thresholds: Accuracy={test_metrics['calibrated']['accuracy']:.4f} | "
            f"Weighted F1={test_metrics['calibrated']['weighted_f1']:.4f} | "
            f"Negative Recall={test_metrics['calibrated']['negative_recall']:.4f}"
        )

        # Save test evaluation report
        with open(os.path.join(output_dir, "test_evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # 3. Automatic ONNX & INT8 Quantization Export
        parity_results = None
        if export_onnx:
            self.logger.info("Starting ONNX and INT8 Dynamic Quantization Export for lean deployment...")
            try:
                from src.models.onnx_exporter import OnnxExporter
                exporter = OnnxExporter(output_dir)
                onnx_path = exporter.export_to_onnx()
                quant_path = exporter.quantize_int8(onnx_path)
                parity_results = exporter.verify_parity(quant_path)
                self.logger.info(
                    f"INT8 ONNX model export successful: {quant_path} (Parity verified: {parity_results.get('verified')})"
                )
            except Exception as e:
                self.logger.error(f"ONNX export encountered error: {e}", exc_info=True)

        # Log to MLOps monitor
        monitor = ModelMonitor()
        monitor.log_performance_metrics(
            accuracy=test_metrics["calibrated"]["accuracy"],
            f1=test_metrics["calibrated"]["weighted_f1"],
            sample_size=len(test_labels)
        )

        self.logger.info("Transformer GPU Pipeline completed successfully!")

        return {
            "trainer": trainer,
            "calibrator": calibrator,
            "training_report": training_report,
            "calibration_results": calibration_results,
            "test_metrics": test_metrics,
            "onnx_parity": parity_results
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Customer Feedback Analysis Pipeline')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--model-type', type=str, choices=['classical', 'transformer'],
                       default='transformer', help='Model family: classical (CPU) or transformer (GPU)')
    parser.add_argument('--model-name', type=str, default='roberta-base',
                       help='Pretrained transformer architecture (default: roberta-base)')
    parser.add_argument('--sample-size', type=int, default=120000,
                       help='Number of high-signal reviews to sample (default: 120000)')
    parser.add_argument('--full-dataset', action='store_true',
                       help='Train on full 393,000+ review dataset instead of high-signal subset')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs for transformer (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for transformer training (default: 16)')
    parser.add_argument('--use-cosine', action='store_true', default=True,
                       help='Enable Cosine Annealing learning rate schedule (default: True)')
    parser.add_argument('--no-cosine', dest='use_cosine', action='store_false',
                       help='Disable Cosine Annealing learning rate schedule')
    parser.add_argument('--use-swa', action='store_true', default=True,
                       help='Enable Stochastic Weight Averaging on final epoch (default: True)')
    parser.add_argument('--no-swa', dest='use_swa', action='store_false',
                       help='Disable Stochastic Weight Averaging')
    parser.add_argument('--export-onnx', action='store_true', default=True,
                       help='Automatically export to INT8 ONNX after training (default: True)')
    parser.add_argument('--no-export-onnx', dest='export_onnx', action='store_false',
                       help='Skip INT8 ONNX export')

    args = parser.parse_args()

    config = load_config(args.config)
    config_dict = config.to_dict()

    pipeline = FeedbackAnalysisPipeline(config_dict)

    if args.model_type == 'transformer':
        results = pipeline.run_transformer_pipeline(
            sample_size=args.sample_size,
            full_dataset=args.full_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model_name,
            use_cosine=args.use_cosine,
            use_swa=args.use_swa,
            export_onnx=args.export_onnx
        )
        print("\n=== TRANSFORMER GPU TRAINING COMPLETE ===")
        m = results['test_metrics']['calibrated']
        print(f"Model: {args.model_name}")
        print(f"Accuracy: {m['accuracy']:.4f}")
        print(f"Weighted F1: {m['weighted_f1']:.4f}")
        print(f"Macro F1: {m['macro_f1']:.4f}")
        print(f"Negative Recall: {m['negative_recall']:.4f}")
        print(f"Negative Precision: {m['negative_precision']:.4f}")
        print(f"Calibrated Thresholds: {results['calibration_results']['thresholds']}")
        if results.get('onnx_parity'):
            print(f"ONNX INT8 Parity Verified: {results['onnx_parity'].get('verified')}")
    else:
        results = pipeline.run_full_pipeline()
        print(f"\nBest Model Performance:")
        print(f"Accuracy: {results['test_results']['basic_metrics']['accuracy']:.4f}")
        print(f"F1-Score: {results['test_results']['basic_metrics']['f1']:.4f}")
        print(f"Precision: {results['test_results']['basic_metrics']['precision']:.4f}")
        print(f"Recall: {results['test_results']['basic_metrics']['recall']:.4f}")

        retrain_decision = results['retrain_decision']
        print("\n=== MONITORING SUMMARY ===")
        print(f"Should Retrain? {retrain_decision['should_retrain']}")
        if retrain_decision['reasons']:
            print(f"Reasons: {', '.join(retrain_decision['reasons'])}")


if __name__ == '__main__':
    main()
