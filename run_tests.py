"""Direct test runner with unbuffered output and progress markers."""

import sys
import traceback

def run():
    print("=== Step 1: Importing modules ===", flush=True)
    from tests import test_preprocessing
    print("  Imported test_preprocessing", flush=True)
    from tests import test_api
    print("  Imported test_api", flush=True)

    print("=== Step 2: Instantiating fixtures ===", flush=True)
    from src.data.preprocessing import DataPreprocessor
    from src.features.feature_engineering import FeatureEngineer
    prep = DataPreprocessor({})
    feat = FeatureEngineer({'tfidf_max_features': 100})

    tests = [
        ("test_clean_text_removes_html", test_preprocessing.test_clean_text_removes_html),
        ("test_clean_text_removes_punctuation_and_lowercases", test_preprocessing.test_clean_text_removes_punctuation_and_lowercases),
        ("test_sentiment_label_mapping", test_preprocessing.test_sentiment_label_mapping),
        ("test_handle_missing_values", test_preprocessing.test_handle_missing_values),
        ("test_extract_basic_features_no_target_leakage", test_preprocessing.test_extract_basic_features_no_target_leakage),
        ("test_tfidf_feature_shape", test_preprocessing.test_tfidf_feature_shape),
        ("test_health_endpoint", test_api.test_health_endpoint),
        ("test_stats_endpoint", test_api.test_stats_endpoint),
        ("test_predict_endpoint_success", test_api.test_predict_endpoint_success),
        ("test_predict_endpoint_empty_text_400", test_api.test_predict_endpoint_empty_text_400),
        ("test_telemetry_endpoint", test_api.test_telemetry_endpoint),
        ("test_simulate_drift_endpoint", test_api.test_simulate_drift_endpoint),
    ]

    print(f"=== Step 3: Running {len(tests)} tests ===", flush=True)
    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            if "preprocessor" in fn.__code__.co_varnames:
                fn(prep)
            elif "feature_engineer" in fn.__code__.co_varnames:
                fn(feat)
            else:
                fn()
            print(f"  [PASS] {name}", flush=True)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}", flush=True)
            traceback.print_exc()
            failed += 1

    print(f"\n=== Summary: {passed} passed, {failed} failed ===", flush=True)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(run())
