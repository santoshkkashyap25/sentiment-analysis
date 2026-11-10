"""Empirical Adversarial Stress Test Suite for R3 (TransformerTrainer.prepare_dataset).

Covers:
- Non-standard index structures (sparse, negative, non-consecutive, duplicate).
- Extreme class ratios and balanced sampling row deduplication.
- Strict disjointness across train, val, and test splits.
- String rating representations, boundary values, invalid value pruning.
- Pre-encoded label preservation ({0, 1, 2}).
- Strict integer-typed {0, 1, 2} output labels.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.transformer_trainer import TransformerTrainer


@pytest.fixture
def trainer():
    return TransformerTrainer(model_name="roberta-base")


# =========================================================================
# Suite 1: Index Structure Adversarial Testing
# =========================================================================

def test_large_sparse_indices(trainer):
    """Verify prepare_dataset handles large sparse indices without corruption."""
    indices = [100_000 * i for i in range(1, 31)]  # 30 items
    df = pd.DataFrame({
        "reviewText": [f"This is review number {i} for testing." for i in range(30)],
        "sentiment": ([1, 3, 5] * 10)
    }, index=indices)

    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=30, random_state=42)

    all_indices = list(train_df.index) + list(val_df.index) + list(test_df.index)
    assert len(all_indices) == len(set(all_indices)), "Duplicate indices detected in splits!"
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))
    assert len(set(train_df.index) | set(val_df.index) | set(test_df.index)) == len(df)


def test_negative_indices(trainer):
    """Verify prepare_dataset handles negative indices gracefully."""
    indices = list(range(-30, 0))
    df = pd.DataFrame({
        "reviewText": [f"Review with negative index {idx}" for idx in indices],
        "sentiment": ([1, 3, 5] * 10)
    }, index=indices)

    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=30, random_state=42)

    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))
    assert len(set(train_df.index) | set(val_df.index) | set(test_df.index)) == 30
    assert all(idx < 0 for idx in train_df.index)


def test_non_consecutive_gapped_indices(trainer):
    """Verify prepare_dataset handles non-consecutive arbitrary gaps in index."""
    indices = [1, 5, 20, 100, 500, 1000, 2000, 5000, 10000, 50000] * 3
    # Make them strictly unique by adding offset
    indices = [indices[i] + i * 3 for i in range(30)]
    assert len(set(indices)) == 30

    df = pd.DataFrame({
        "reviewText": [f"Non-consecutive item {i} sample text" for i in range(30)],
        "sentiment": ([1, 2, 3] * 10)
    }, index=indices)

    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=30, random_state=42)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    union_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert len(union_indices) == total_rows
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))


def test_duplicate_indices_handled_safely(trainer):
    """Verify duplicate index input safely resets index and prevents KeyError/ValueError."""
    # Deliberate duplicate indices: [0, 0, 1, 1, 2, 2, ...]
    indices = [i // 2 for i in range(30)]
    df = pd.DataFrame({
        "reviewText": [f"Review with duplicate index entry {i}" for i in range(30)],
        "sentiment": ([1, 3, 5] * 10)
    }, index=indices)

    assert not df.index.is_unique

    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=30, random_state=42)

    # Output indices should be unique RangeIndex or unique indices
    total_rows = len(train_df) + len(val_df) + len(test_df)
    union_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert len(union_indices) == total_rows
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))


# =========================================================================
# Suite 2: Extreme Class Ratios & Balanced Sampling Stress
# =========================================================================

def test_extreme_class_ratio_balanced_sampling(trainer):
    """Verify balanced sampling with extreme ratios (1000 Positive, 5 Neutral, 2 Negative).
    
    Checks that clean_df.drop accurately excludes already sampled rows without KeyError
    and ensures zero row duplication across splits.
    """
    n_pos = 1000
    n_neu = 5
    n_neg = 2
    total = n_pos + n_neu + n_neg

    reviews = [f"Positive review text {i}" for i in range(n_pos)] + \
              [f"Neutral review text {i}" for i in range(n_neu)] + \
              [f"Negative review text {i}" for i in range(n_neg)]
    sentiments = [5] * n_pos + [3] * n_neu + [1] * n_neg

    df = pd.DataFrame({"reviewText": reviews, "sentiment": sentiments})

    sample_size = 300
    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=sample_size, random_state=42)

    assert len(train_df) + len(val_df) + len(test_df) == sample_size
    union_indices = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert len(union_indices) == sample_size
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))

    # Check that each split has valid labels
    for split_df in [train_df, val_df, test_df]:
        assert split_df["label"].isin([0, 1, 2]).all()
        assert np.issubdtype(split_df["label"].dtype, np.integer)


def test_extreme_single_sample_fallback(trainer):
    """Verify fallback when a class has only 1 sample (stratification impossible)."""
    reviews = [f"Positive review text {i}" for i in range(50)] + \
              ["Neutral review single sample"] + \
              ["Negative review single sample"]
    sentiments = [5] * 50 + [3] * 1 + [1] * 1
    df = pd.DataFrame({"reviewText": reviews, "sentiment": sentiments})

    # Should not raise ValueError in train_test_split
    train_df, val_df, test_df = trainer.prepare_dataset(df, sample_size=30, random_state=42)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == 30
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))


# =========================================================================
# Suite 3: Label Mapping Stress Testing
# =========================================================================

def test_string_rating_representations(trainer):
    """Verify string representations like '1', ' 2.0 ', '5' map accurately."""
    ratings = ["1", " 2.0 ", "3", "3.0", " 4 ", "5", " 5.0 ", "2", "4"]
    df = pd.DataFrame({
        "reviewText": [f"Review number {i} for string testing" for i in range(len(ratings) * 4)],
        "sentiment": ratings * 4
    })

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True)
    combined = pd.concat([train_df, val_df, test_df])

    assert len(combined) == 36
    assert combined["label"].isin([0, 1, 2]).all()
    assert np.issubdtype(combined["label"].dtype, np.integer)

    # Let's inspect specific mappings
    expected_map = {
        "1": 0, " 2.0 ": 0, "3": 1, "3.0": 1, " 4 ": 2, "5": 2, " 5.0 ": 2, "2": 0, "4": 2
    }
    for orig_sent, exp_lbl in expected_map.items():
        matched_rows = combined[combined["sentiment"] == orig_sent]
        assert len(matched_rows) == 4
        assert (matched_rows["label"] == exp_lbl).all(), f"Failed for {orig_sent}: expected {exp_lbl}, got {matched_rows['label'].iloc[0]}"


def test_boundary_and_invalid_values_dropped(trainer):
    """Verify out-of-range/invalid values (0, 0.5, 2.5, 5.5, 6, None, nan, 'unknown') are dropped."""
    invalid_sentiments = [0, 0.5, 2.5, 5.5, 6, -1, None, np.nan, "unknown", "N/A", ""]
    valid_sentiments = [1, 2, 3, 4, 5] * 4  # 20 valid rows

    reviews = [f"Valid or invalid review text {i}" for i in range(len(invalid_sentiments) + len(valid_sentiments))]
    sentiments = invalid_sentiments + valid_sentiments

    df = pd.DataFrame({"reviewText": reviews, "sentiment": sentiments})

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True)
    combined = pd.concat([train_df, val_df, test_df])

    # All invalid sentiments should have been pruned
    assert len(combined) == 20
    assert not combined["sentiment"].isin(invalid_sentiments).any()
    assert combined["label"].isin([0, 1, 2]).all()
    assert np.issubdtype(combined["label"].dtype, np.integer)


def test_pre_encoded_labels_preserved(trainer):
    """Verify pre-encoded labels {0, 1, 2} preserve 1 as Neutral and do not re-map 1 to Negative."""
    df = pd.DataFrame({
        "reviewText": [f"Pre-encoded review text {i}" for i in range(30)],
        "sentiment": ([0, 1, 2] * 10)
    })

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True)
    combined = pd.concat([train_df, val_df, test_df])

    assert len(combined) == 30
    for original_label in [0, 1, 2]:
        subset = combined[combined["sentiment"] == original_label]
        assert len(subset) == 10
        assert (subset["label"] == original_label).all(), f"Pre-encoded label {original_label} was corrupted!"


def test_pre_encoded_float_and_string_labels(trainer):
    """Verify pre-encoded float (0.0, 1.0, 2.0) and string ('0', '1', '2') labels work."""
    # Float pre-encoded
    df_float = pd.DataFrame({
        "reviewText": [f"Float review text {i}" for i in range(30)],
        "sentiment": ([0.0, 1.0, 2.0] * 10)
    })
    train_df, val_df, test_df = trainer.prepare_dataset(df_float, full_dataset=True)
    comb_float = pd.concat([train_df, val_df, test_df])
    assert len(comb_float) == 30
    assert np.issubdtype(comb_float["label"].dtype, np.integer)
    assert set(comb_float["label"].unique()) == {0, 1, 2}
    assert (comb_float[comb_float["sentiment"] == 1.0]["label"] == 1).all()

    # String pre-encoded
    df_str = pd.DataFrame({
        "reviewText": [f"String review text {i}" for i in range(30)],
        "sentiment": (["0", "1", "2"] * 10)
    })
    train_df, val_df, test_df = trainer.prepare_dataset(df_str, full_dataset=True)
    comb_str = pd.concat([train_df, val_df, test_df])
    assert len(comb_str) == 30
    assert np.issubdtype(comb_str["label"].dtype, np.integer)
    assert set(comb_str["label"].unique()) == {0, 1, 2}
    assert (comb_str[comb_str["sentiment"] == "1"]["label"] == 1).all()


# =========================================================================
# Suite 4: Text Column Fallback & Summary Fusion Edge Cases
# =========================================================================

def test_summary_fusion_edge_cases(trainer):
    """Verify headline + review body fusion with NaN, whitespace, and empty summaries."""
    df = pd.DataFrame({
        "reviewText": [
            "This is a fantastic product that exceeded all my expectations.",
            "Average quality, nothing special to write home about.",
            "Terrible experience, broke after two days of use.",
            "Another positive review that is detailed and helpful.",
            "Another neutral review with moderate satisfaction.",
            "Another terrible item, complete waste of money.",
        ] * 5,
        "summary": [
            None,
            np.nan,
            "   ",
            "Great Product!",
            "So-so",
            "Worst Ever"
        ] * 5,
        "sentiment": [5, 3, 1, 5, 3, 1] * 5
    })

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True)
    combined = pd.concat([train_df, val_df, test_df])

    assert len(combined) == 30
    assert "fused_text" in combined.columns

    # Rows with valid summary should have " — "
    fused_samples = combined[combined["summary"] == "Great Product!"]["fused_text"]
    for text in fused_samples:
        assert text.startswith("Great Product! — ")

    # Rows with None or NaN or whitespace summary should NOT have " — " prefix
    fused_none = combined[combined["summary"].isna()]["fused_text"]
    for text in fused_none:
        assert not text.startswith(" — ")
        assert "This is a fantastic product" in text or "Average quality" in text


def test_review_text_clean_fallback(trainer):
    """Verify fallback to 'reviewText_clean' when 'reviewText' is not present in columns."""
    df = pd.DataFrame({
        "reviewText_clean": [
            "excellent reliable durable device that works wonders",
            "fair average standard functionality nothing impressive",
            "poor defective broken junk returned immediately"
        ] * 10,
        "sentiment": [5, 3, 1] * 10
    })

    assert "reviewText" not in df.columns

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True)
    combined = pd.concat([train_df, val_df, test_df])

    assert len(combined) == 30
    assert "fused_text" in combined.columns
    assert (combined["fused_text"] == combined["reviewText_clean"]).all()

