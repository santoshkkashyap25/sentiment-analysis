# E2E Test Infrastructure Specification: Sentiment Analysis Showcase

## 1. Test Philosophy
- **Requirement-Driven & Opaque-Box**: Tests validate external contracts, public APIs, input/output behaviors, and system state transitions (filesystem artifacts, database records, HTTP response codes). There is zero dependency on private implementation variables or internals (`_`-prefixed attributes).
- **Progressive Testability & Determinism**: Verification mechanisms use the simplest reliable assert paths. Tiers 1-2 isolate individual features with mock boundaries; Tiers 3-4 exercise end-to-end multi-module pipelines.
- **Fast, Deterministic & Isolated Execution**: Heavy GPU transformer training is cleanly mocked or substituted with lightweight estimators so the entire test suite executes deterministically in < 15 seconds. SQLite operations utilize temporary, isolated databases with WAL journal mode (`PRAGMA journal_mode=WAL;`). Matplotlib operations use headless backends (`matplotlib.use('Agg')`) without GUI blocks.
- **Independence & Self-Containment**: Each test initializes its own state, utilizes temporary directories (`tmp_path`) for file and database operations, and cleans up cleanly without execution order dependencies.

---

## 2. Methodology & Tier Architecture

The test suite in `tests/test_e2e_showcase.py` is partitioned into 4 distinct verification tiers:

### Tier 1: Feature Coverage (Happy Paths)
Category-partition testing for each requirement (R1.1 through R7.3) in isolation using standard valid inputs and documented public APIs.
- Validates that TruncatedSVD is functional and imported (R1.1).
- Validates that stylistic signals (exclamations, questions, casing, sentences) extract from raw unstripped text (R1.2).
- Validates that all 8 canonical negation words are retained during text cleaning (R2.1).
- Validates that NLTK resource checks avoid redundant remote downloads (R2.2).
- Validates that API ingestion applies a 30s timeout and multi-source ingestion handles Kaggle paths and source tags (R4.1, R4.2, R4.3).
- Validates that balanced sampling preserves index tracking to prevent duplicate rows (R3.1).
- Validates that 1–5 star ratings map strictly to classes 0, 1, 2 for CrossEntropyLoss compatibility (R3.2).
- Validates that model selection ranks candidates by weighted F1 rather than raw accuracy (R5.1).
- Validates that redundant second 5-fold cross-validation after GridSearchCV is eliminated (R5.2).
- Validates that confusion matrix generation runs headlessly without `plt.show()` and saves images (R5.3).
- Validates that per-class metrics pass `labels=[0, 1, 2]` and `zero_division=0` (R5.4).
- Validates that CPU-bound API routes are synchronous `def` functions (R6.1).
- Validates that API prediction errors yield HTTP 500 without writing bogus neutral rows to SQLite (R6.2).
- Validates that SQLite connections configure `timeout=30.0` and `PRAGMA journal_mode=WAL;` (R6.3).
- Validates that environment and package configurations declare ONNX, PyTorch, Transformers, and Kaggle token templates (R7.1, R7.2, R7.3).

### Tier 2: Boundary & Corner Cases
Boundary Value Analysis (BVA), negative testing, and error handling for all requirements:
- Empty strings, `None`/`NaN`, punctuation-only, all-caps, and whitespace texts in stylistic feature extraction (R1.2).
- Unfitted SVD dimensionality reduction during inference safeguards (R1.1).
- Negation preservation across uppercase, mixed-case, and boundary-attached punctuation (R2.1).
- Filtering of rows reducing to empty strings in text normalization (R2.1).
- Graceful download fallback when an NLTK resource is missing from local cache (R2.2).
- Unsupported ingestion source formats raising `ValueError` (R4.2).
- Ingestion network timeouts raising `requests.exceptions.Timeout` (R4.1).
- Balanced sampling under extreme class imbalance and non-contiguous gapped DataFrame indices (R3.1).
- Star rating boundary formats (floats, whitespace strings, out-of-bounds ratings filtered out) (R3.2).
- Per-class metric computation when only a single class is present/predicted (R5.4).
- Model selection under tied or zero scores (R5.1).
- Headless confusion matrix saving into non-existent nested directories (R5.3).
- High-concurrency multithreaded SQLite writes verifying zero `OperationalError: database is locked` (R6.3).
- Validation and rejection of empty/whitespace review payloads at the REST API boundary (R6.2).
- PEP 508 specification compliance for dependency declarations (R7.1, R7.2).

### Tier 3: Cross-Feature Combinations
Pairwise integration testing across pipeline boundaries:
1. **Ingestion ➔ Preprocessing**: Multi-channel raw data (CSV + API) ingested and passed directly into preprocessing, ensuring HTML removal, negation preservation, and score conversion.
2. **Preprocessing ➔ Feature Engineering**: Preprocessed DataFrames simultaneously supply cleaned text to TF-IDF and raw unstripped text (`reviewText`) to stylistic feature extractors.
3. **Dataset Prep ➔ Model Training & Evaluation**: Balanced sampling and star-rating normalization feed model training and evaluation, verifying weighted F1 selection and robust per-class metric computation under absent classes.
4. **API Serving ➔ SQLite Drift Telemetry**: REST serving `/predict` endpoint logs telemetry on success and raises HTTP 500 on failure with zero bogus records under WAL mode.

### Tier 4: Real-World Sentiment Pipeline Scenarios
End-to-end multi-step operational workflows mirroring production systems:
1. **Customer Feedback Pipeline Lifecycle**: Full execution from heterogeneous raw reviews with noisy formatting to cleaning, feature extraction, estimator training, model selection via weighted F1, headless confusion matrix artifact creation, and SQLite metric logging.
2. **Production Drift Detection & Retraining Lifecycle**: Simulates continuous production monitoring: healthy baseline traffic, detection of sudden out-of-distribution drift via two-sample Kolmogorov-Smirnov tests, degradation alerting, and automated retraining plan generation.
3. **Adversarial & Edge-Case Review Serving Robustness**: Stresses the live serving API with challenging real-world user inputs (sarcasm, heavy negations, uppercase screaming, punctuation repetition, HTML/script injection, and whitespace padding) ensuring graceful handling and valid probability outputs.

---

## 3. Feature Inventory & Coverage Mapping

| # | Requirement | Feature Description | Tier 1 Tests | Tier 2 Tests | Tier 3 Integration | Tier 4 Workflow |
|---|:-----------:|:--------------------|:------------:|:------------:|:------------------:|:---------------:|
| 1 | R1.1 | TruncatedSVD Embedding Reduction | `test_tier1_r1_truncated_svd_embeddings` | `test_tier2_r1_unfitted_svd_inference_safeguard` | `test_tier3_preprocessing_to_feature_engineering_stylistic_features` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 2 | R1.2 | Stylistic Features from Raw Text | `test_tier1_r1_stylistic_features_raw_text` | `test_tier2_r1_stylistic_boundary_cases` | `test_tier3_preprocessing_to_feature_engineering_stylistic_features` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle`, `test_tier4_adversarial_and_edge_case_review_robustness` |
| 3 | R2.1 | Negation Words Preservation | `test_tier1_r2_negation_words_preserved` | `test_tier2_r2_negation_boundaries_and_casing`, `test_tier2_r2_normalize_text_empty_filtering` | `test_tier3_ingestion_to_preprocessing_negation_preservation` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle`, `test_tier4_adversarial_and_edge_case_review_robustness` |
| 4 | R2.2 | Guarded NLTK Downloads | `test_tier1_r2_guard_nltk_downloads` | `test_tier2_r2_nltk_download_fallback_when_missing` | — | — |
| 5 | R4.1 | Ingestion API Timeout (30s) | `test_tier1_r4_api_ingestion_timeout` | `test_tier2_r4_api_timeout_exception_handling` | `test_tier3_ingestion_to_preprocessing_negation_preservation` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 6 | R4.2 | Ingestion Unsupported Source Types | `test_tier1_r4_ingest_multiple_sources_and_paths` | `test_tier2_r4_ingestion_unsupported_source_raises_value_error` | — | — |
| 7 | R4.3 | Ingestion Cross-Platform Path Handling | `test_tier1_r4_ingest_multiple_sources_and_paths` | — | `test_tier3_ingestion_to_preprocessing_negation_preservation` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 8 | R3.1 | Balanced Sampling Index Preservation | `test_tier1_r3_balanced_sampling_preserves_unique_indices` | `test_tier2_r3_balanced_sampling_with_unequal_classes_and_index_gaps` | `test_tier3_dataset_prep_to_training_and_evaluation_zero_division` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 9 | R3.2 | 1-5 Star Rating Mapping to 3 Classes | `test_tier1_r3_star_rating_mapping_to_three_classes` | `test_tier2_r3_star_rating_boundary_types_and_unmapped` | `test_tier3_dataset_prep_to_training_and_evaluation_zero_division` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 10 | R5.1 | Model Selection by Weighted F1 | `test_tier1_r5_select_best_model_uses_weighted_f1` | `test_tier2_r5_select_best_model_tied_or_zero_scores` | `test_tier3_dataset_prep_to_training_and_evaluation_zero_division` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 11 | R5.2 | Removal of Redundant Secondary CV | `test_tier1_r5_no_redundant_second_cv_after_gridsearch` | — | — | — |
| 12 | R5.3 | Headless Confusion Matrix Plot | `test_tier1_r5_plot_confusion_matrix_headless_and_save` | `test_tier2_r5_plot_confusion_matrix_nested_directory_save` | `test_tier3_dataset_prep_to_training_and_evaluation_zero_division` | `test_tier4_customer_feedback_e2e_pipeline_lifecycle` |
| 13 | R5.4 | Per-Class Metrics with Zero Division | `test_tier1_r5_calculate_per_class_metrics_labels_and_zero_division` | `test_tier2_r5_per_class_metrics_single_class_only` | `test_tier3_dataset_prep_to_training_and_evaluation_zero_division` | — |
| 14 | R6.1 | Sync CPU-Bound Route Execution | `test_tier1_r6_cpu_routes_are_synchronous_def` | — | `test_tier3_api_prediction_to_sqlite_drift_logging_success_and_error` | `test_tier4_adversarial_and_edge_case_review_robustness` |
| 15 | R6.2 | Error Propagation & Clean Telemetry | `test_tier1_r6_predict_error_returns_500_and_does_not_log_sqlite` | `test_tier2_r6_predict_endpoint_empty_and_whitespace_validation` | `test_tier3_api_prediction_to_sqlite_drift_logging_success_and_error` | — |
| 16 | R6.3 | SQLite WAL Mode & Concurrency Timeout | `test_tier1_r6_sqlite_wal_mode_and_timeout` | `test_tier2_r6_sqlite_concurrent_multithreaded_writes` | `test_tier3_api_prediction_to_sqlite_drift_logging_success_and_error` | `test_tier4_production_drift_detection_and_retraining_lifecycle` |
| 17 | R7.1 | Dev Dependency `onnx>=1.14.0` | `test_tier1_r7_requirements_dev_contains_onnx` | `test_tier2_r7_packaging_specifiers_valid_pep508` | — | — |
| 18 | R7.2 | Dependencies in `pyproject.toml` | `test_tier1_r7_pyproject_toml_dependencies` | `test_tier2_r7_packaging_specifiers_valid_pep508` | — | — |
| 19 | R7.3 | Template `.env.example` with Kaggle | `test_tier1_r7_env_example_contains_kaggle_token` | — | — | — |

---

## 4. Test Execution Instructions

### Complete Showcase Suite
```powershell
pytest tests/test_e2e_showcase.py -v
```

### Execution by Tier
```powershell
# Tier 1: Feature Coverage (Happy Paths)
pytest tests/test_e2e_showcase.py -k "tier1" -v

# Tier 2: Boundary & Corner Cases
pytest tests/test_e2e_showcase.py -k "tier2" -v

# Tier 3: Cross-Feature Integration
pytest tests/test_e2e_showcase.py -k "tier3" -v

# Tier 4: Real-World Application Workflows
pytest tests/test_e2e_showcase.py -k "tier4" -v
```

### Full Repository Regression Verification
```powershell
pytest tests/ -v
```
