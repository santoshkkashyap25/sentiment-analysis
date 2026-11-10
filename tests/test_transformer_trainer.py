import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np
import torch
from src.models.transformer_trainer import TransformerTrainer


@pytest.fixture
def trainer():
    return TransformerTrainer()


def test_prepare_dataset_balanced_sampling_no_duplicates(trainer):
    """Verify balanced sampling preserves original indices and avoids duplicates during fill."""
    # Create dataset with imbalanced classes: 20 Neg, 5 Neu, 50 Pos (total 75)
    reviews = [f"This is unique review text number {i:03d} for testing." for i in range(75)]
    sentiments = ["Negative"] * 20 + ["Neutral"] * 5 + ["Positive"] * 50

    df = pd.DataFrame({
        "reviewText": reviews,
        "sentiment": sentiments
    })

    # sample_size=30: samples_per_class=10
    # Neg has 20 -> takes 10
    # Neu has 5 -> takes 5
    # Pos has 50 -> takes 10
    # Subtotal = 25 < 30 -> triggers fill of 5 samples from remaining 50 reviews
    train_df, val_df, test_df = trainer.prepare_dataset(
        df, sample_size=30, full_dataset=False, random_state=42
    )

    combined = pd.concat([train_df, val_df, test_df])

    # Ensure total samples equal sample_size
    assert len(combined) == 30

    # Ensure zero duplicate reviews were sampled
    assert len(combined["reviewText"].unique()) == 30
    assert combined["reviewText"].duplicated().sum() == 0

    # Ensure indices are unique and clean
    assert train_df.index.is_unique
    assert val_df.index.is_unique
    assert test_df.index.is_unique
    assert set(train_df.index).isdisjoint(set(val_df.index))
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert set(val_df.index).isdisjoint(set(test_df.index))
    all_indices = list(train_df.index) + list(val_df.index) + list(test_df.index)
    assert len(all_indices) == len(set(all_indices))

    # Verify labels in {0, 1, 2}
    assert set(combined["label"].unique()).issubset({0, 1, 2})


def test_prepare_dataset_1_to_5_star_rating_mapping_int(trainer):
    """Verify integer 1-5 star ratings map to 0 (Negative), 1 (Neutral), 2 (Positive)."""
    # 10 samples for each star rating (total 50)
    data = []
    for star in [1, 2, 3, 4, 5]:
        for i in range(10):
            data.append({
                "reviewText": f"Rating {star} sample text review number {i:02d}.",
                "sentiment": star
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    assert set(all_df["label"].unique()).issubset({0, 1, 2})

    # Check mappings: 1, 2 -> 0; 3 -> 1; 4, 5 -> 2
    label_map = dict(zip(all_df["reviewText"], all_df["label"]))
    assert label_map["Rating 1 sample text review number 00."] == 0
    assert label_map["Rating 2 sample text review number 00."] == 0
    assert label_map["Rating 3 sample text review number 00."] == 1
    assert label_map["Rating 4 sample text review number 00."] == 2
    assert label_map["Rating 5 sample text review number 00."] == 2

    # Verify PyTorch CrossEntropyLoss compatibility (num_labels=3, max label < 3)
    labels_tensor = torch.tensor(all_df["label"].values, dtype=torch.long)
    logits = torch.randn(len(all_df), 3)
    loss = torch.nn.CrossEntropyLoss()(logits, labels_tensor)
    assert not torch.isnan(loss)


def test_prepare_dataset_1_to_5_star_rating_mapping_float(trainer):
    """Verify float 1.0-5.0 star ratings map to 0, 1, 2."""
    data = []
    for star in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for i in range(10):
            data.append({
                "reviewText": f"Float rating {star} review content number {i:02d}.",
                "sentiment": star
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    assert set(all_df["label"].unique()).issubset({0, 1, 2})
    label_map = dict(zip(all_df["reviewText"], all_df["label"]))
    assert label_map["Float rating 1.0 review content number 00."] == 0
    assert label_map["Float rating 2.0 review content number 00."] == 0
    assert label_map["Float rating 3.0 review content number 00."] == 1
    assert label_map["Float rating 4.0 review content number 00."] == 2
    assert label_map["Float rating 5.0 review content number 00."] == 2


def test_prepare_dataset_1_to_5_star_rating_mapping_string(trainer):
    """Verify string '1'-'5' star ratings map to 0, 1, 2."""
    data = []
    for star in ["1", "2", "3", "4", "5"]:
        for i in range(10):
            data.append({
                "reviewText": f"String rating {star} review text example {i:02d}.",
                "sentiment": star
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    assert set(all_df["label"].unique()).issubset({0, 1, 2})
    label_map = dict(zip(all_df["reviewText"], all_df["label"]))
    assert label_map["String rating 1 review text example 00."] == 0
    assert label_map["String rating 2 review text example 00."] == 0
    assert label_map["String rating 3 review text example 00."] == 1
    assert label_map["String rating 4 review text example 00."] == 2
    assert label_map["String rating 5 review text example 00."] == 2


def test_prepare_dataset_string_sentiment_mapping(trainer):
    """Verify string sentiment mapping is case-insensitive ('Negative', 'neutral', 'POSITIVE')."""
    data = []
    sentiments = [
        ("Negative", 0), ("negative", 0), ("NEGATIVE", 0),
        ("Neutral", 1), ("neutral", 1), ("NEUTRAL", 1),
        ("Positive", 2), ("positive", 2), ("POSITIVE", 2)
    ]
    for s_str, expected_lbl in sentiments:
        for i in range(5):
            data.append({
                "reviewText": f"Case test for sentiment {s_str} sample {i:02d}.",
                "sentiment": s_str
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    label_map = dict(zip(all_df["reviewText"], all_df["label"]))
    for s_str, expected_lbl in sentiments:
        assert label_map[f"Case test for sentiment {s_str} sample 00."] == expected_lbl


def test_prepare_dataset_existing_0_1_2_labels_preserved(trainer):
    """Verify existing integer labels {0, 1, 2} are preserved and 1 is not treated as 1-star."""
    data = []
    for lbl in [0, 1, 2]:
        for i in range(15):
            data.append({
                "reviewText": f"Pre-encoded label {lbl} review item {i:02d}.",
                "sentiment": lbl
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    label_map = dict(zip(all_df["reviewText"], all_df["label"]))
    assert label_map["Pre-encoded label 0 review item 00."] == 0
    assert label_map["Pre-encoded label 1 review item 00."] == 1  # must be 1, NOT 0
    assert label_map["Pre-encoded label 2 review item 00."] == 2


def test_prepare_dataset_headline_summary_fusion(trainer):
    """Verify that 'summary' column fuses with 'reviewText' in 'fused_text'."""
    data = []
    for lbl in [0, 1, 2]:
        for i in range(10):
            data.append({
                "summary": f"Headline {i}",
                "reviewText": f"Detailed review body text for review {i:02d}.",
                "sentiment": lbl
            })

    df = pd.DataFrame(data)
    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    assert "fused_text" in all_df.columns
    sample_row = all_df[all_df["reviewText"] == "Detailed review body text for review 00."].iloc[0]
    assert sample_row["fused_text"] == "Headline 0 — Detailed review body text for review 00."


def test_prepare_dataset_filters_short_text_and_nas(trainer):
    """Verify that short texts (<= 3 characters) and NA labels/texts are dropped."""
    df = pd.DataFrame({
        "reviewText": ["Valid long review text 1", "No", "   ", None, "Valid long review text 2"],
        "sentiment": ["Positive", "Positive", "Positive", "Positive", None]
    })
    # Only the first row is valid (text len > 3 and sentiment not NA)
    # Note: to split without stratify error, we add more valid rows
    valid_rows = [
        {"reviewText": f"Valid long review text sample {i}", "sentiment": ["Negative", "Neutral", "Positive"][i % 3]}
        for i in range(15)
    ]
    df = pd.concat([df, pd.DataFrame(valid_rows)], ignore_index=True)

    train_df, val_df, test_df = trainer.prepare_dataset(df, full_dataset=True, random_state=42)
    all_df = pd.concat([train_df, val_df, test_df])

    assert "No" not in all_df["reviewText"].values
    assert None not in all_df["reviewText"].values
    assert len(all_df) == 16  # 1 from first block + 15 from second block
