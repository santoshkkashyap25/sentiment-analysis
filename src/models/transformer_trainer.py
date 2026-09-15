"""PyTorch GPU Fine-Tuning Module for Pretrained Transformers (RoBERTa / DistilBERT)."""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

logger = logging.getLogger(__name__)


def f1_score_calc(y_true, y_pred, average="weighted") -> float:
    """Helper to compute F1 score safely."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


class FeedbackDataset(Dataset):
    """PyTorch Dataset for customer review texts."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class TransformerTrainer:
    """Fine-tunes a Pretrained Transformer (RoBERTa / DistilBERT) on NVIDIA GPU."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 3,
        max_length: int = 256,
        batch_size: int = 16,
        lr: float = 2e-5,
        epochs: int = 3,
        label_smoothing: float = 0.1,
        use_cosine: bool = True,
        use_swa: bool = True,
        swa_start_epoch: int = 3,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.label_smoothing = label_smoothing
        self.use_cosine = use_cosine
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Initialized TransformerTrainer: model={self.model_name}, device={self.device}, "
            f"batch_size={self.batch_size}, lr={self.lr}, max_length={self.max_length}, "
            f"cosine={self.use_cosine}, swa={self.use_swa}"
        )
        if self.device.type == "cuda":
            logger.info(f"Active GPU: {torch.cuda.get_device_name(0)}")

        self.tokenizer = None
        self.model = None

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        sample_size: int = 120000,
        full_dataset: bool = False,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract high-signal dataset and create stratified train/val/test splits."""
        text_col = "reviewText" if "reviewText" in df.columns else "reviewText_clean"
        label_col = "sentiment"

        clean_df = df.dropna(subset=[text_col, label_col]).copy()
        clean_df = clean_df[clean_df[text_col].astype(str).str.strip().str.len() > 3]

        # Lever 1: Headline + Review Body Fusion
        if "summary" in clean_df.columns:
            summaries = clean_df["summary"].fillna("").astype(str).str.strip()
            bodies = clean_df[text_col].astype(str).str.strip()
            clean_df["fused_text"] = np.where(
                summaries.str.len() > 1,
                summaries + " — " + bodies,
                bodies
            )
            logger.info("Fused review headlines ('summary') with review bodies for amplified sentiment signal.")
        else:
            clean_df["fused_text"] = clean_df[text_col]

        # Map sentiment strings to {0: Negative, 1: Neutral, 2: Positive}
        label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        first_val = str(clean_df[label_col].iloc[0]).strip()
        if first_val in label_map or not first_val.isdigit():
            clean_df["label"] = clean_df[label_col].astype(str).str.strip().map(label_map)
        else:
            clean_df["label"] = clean_df[label_col].astype(int)

        clean_df = clean_df.dropna(subset=["label"])
        clean_df["label"] = clean_df["label"].astype(int)

        if not full_dataset and len(clean_df) > sample_size:
            logger.info(f"Selecting {sample_size} high-signal balanced reviews...")
            # Balanced sampling across classes
            samples_per_class = sample_size // 3
            sampled_dfs = []
            for lbl in [0, 1, 2]:
                cls_df = clean_df[clean_df["label"] == lbl]
                n_take = min(len(cls_df), samples_per_class)
                sampled_dfs.append(cls_df.sample(n_take, random_state=random_state))

            sub_df = pd.concat(sampled_dfs, ignore_index=True)
            if len(sub_df) < sample_size:
                rem_needed = sample_size - len(sub_df)
                remaining_clean = clean_df.drop(index=sub_df.index, errors="ignore")
                if len(remaining_clean) > 0:
                    fill = remaining_clean.sample(min(rem_needed, len(remaining_clean)), random_state=random_state)
                    sub_df = pd.concat([sub_df, fill], ignore_index=True)
            clean_df = sub_df

        logger.info(f"Dataset ready: {len(clean_df)} reviews. Class distribution:\n{clean_df['label'].value_counts().to_dict()}")

        # Stratified 80 / 10 / 10 Split
        train_df, temp_df = train_test_split(
            clean_df, test_size=0.20, stratify=clean_df["label"], random_state=random_state
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df["label"], random_state=random_state
        )

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str = "data/models/transformer_sentiment",
        text_column: str = "reviewText"
    ) -> Dict[str, Any]:
        """Fine-tune Transformer on GPU with mixed precision, Cosine Annealing, and SWA."""
        col = "fused_text" if "fused_text" in train_df.columns else (text_column if text_column in train_df.columns else "reviewText_clean")

        logger.info(f"Loading pretrained tokenizer & model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
            label2id={"Negative": 0, "Neutral": 1, "Positive": 2}
        ).to(self.device)

        train_dataset = FeedbackDataset(train_df[col].values, train_df["label"].values, self.tokenizer, self.max_length)
        val_dataset = FeedbackDataset(val_df[col].values, val_df["label"].values, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Class-weighted loss with label smoothing
        class_counts = np.bincount(train_df["label"].values, minlength=self.num_labels)
        total_samples = len(train_df)
        class_weights = total_samples / (self.num_labels * np.maximum(class_counts, 1))
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=self.label_smoothing)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(0.1 * total_steps)

        if self.use_cosine:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            logger.info(f"Using Cosine Annealing learning rate schedule ({warmup_steps} warmup steps).")
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            logger.info(f"Using Linear Warmup learning rate schedule ({warmup_steps} warmup steps).")

        swa_model = None
        if self.use_swa:
            swa_model = AveragedModel(self.model)
            logger.info(f"SWA initialized; parameter averaging activates on epoch {self.swa_start_epoch}.")

        use_cuda = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_cuda else None

        logger.info(f"Starting GPU training for {self.epochs} epochs ({total_steps} total steps, batch_size={self.batch_size})...")
        best_val_f1 = 0.0
        best_val_metrics = {}
        training_history = []

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()

                if use_cuda:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()

                # Update SWA model parameters during designated epoch(s)
                if self.use_swa and swa_model is not None and epoch >= self.swa_start_epoch:
                    swa_model.update_parameters(self.model)

                total_loss += loss.item()

                if (step + 1) % 250 == 0 or (step + 1) == len(train_loader):
                    avg_step_loss = total_loss / (step + 1)
                    vram_mb = torch.cuda.memory_allocated() / (1024**2) if use_cuda else 0.0
                    logger.info(
                        f"Epoch {epoch}/{self.epochs} | Step {step+1}/{len(train_loader)} | "
                        f"Loss: {avg_step_loss:.4f} | VRAM: {vram_mb:.0f} MB"
                    )

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            val_probs, val_labels = self.predict_probabilities(val_loader)
            val_preds = np.argmax(val_probs, axis=1)
            acc = accuracy_score(val_labels, val_preds)
            _, _, f1_per_class, _ = precision_recall_fscore_support(val_labels, val_preds, average=None, zero_division=0)
            weighted_f1 = f1_score_calc(val_labels, val_preds, average="weighted")
            macro_f1 = f1_score_calc(val_labels, val_preds, average="macro")

            epoch_time = time.time() - epoch_start
            logger.info(
                f"--- Epoch {epoch} Results ({epoch_time:.1f}s) --- "
                f"Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | "
                f"Val F1 (Weighted): {weighted_f1:.4f} | Val F1 (Macro): {macro_f1:.4f}"
            )

            history_entry = {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_accuracy": float(acc),
                "val_weighted_f1": float(weighted_f1),
                "val_macro_f1": float(macro_f1),
                "epoch_seconds": round(epoch_time, 2)
            }
            training_history.append(history_entry)

            if weighted_f1 > best_val_f1:
                best_val_f1 = weighted_f1
                best_val_metrics = history_entry
                self.save(output_dir)
                logger.info(f"Saved new best model checkpoint to {output_dir} (Val F1: {best_val_f1:.4f})")

        # Evaluate SWA model if enabled
        if self.use_swa and swa_model is not None:
            logger.info("Evaluating SWA Averaged Model on Validation Set...")
            swa_probs, val_labels = self.predict_probabilities(val_loader, model=swa_model)
            swa_preds = np.argmax(swa_probs, axis=1)
            swa_acc = accuracy_score(val_labels, swa_preds)
            swa_weighted_f1 = f1_score_calc(val_labels, swa_preds, average="weighted")
            swa_macro_f1 = f1_score_calc(val_labels, swa_preds, average="macro")

            logger.info(
                f"--- SWA Model Validation --- "
                f"Acc: {swa_acc:.4f} | F1 (Weighted): {swa_weighted_f1:.4f} | F1 (Macro): {swa_macro_f1:.4f}"
            )

            if swa_weighted_f1 > best_val_f1:
                logger.info(
                    f"SWA model outperformed best checkpoint ({swa_weighted_f1:.4f} > {best_val_f1:.4f})! "
                    f"Saving SWA weights as final model."
                )
                best_val_f1 = swa_weighted_f1
                best_val_metrics = {
                    "epoch": "SWA",
                    "val_accuracy": float(swa_acc),
                    "val_weighted_f1": float(swa_weighted_f1),
                    "val_macro_f1": float(swa_macro_f1),
                    "is_swa": True
                }
                self.model = swa_model.module
                self.save(output_dir)
            else:
                logger.info(
                    f"Best checkpoint ({best_val_f1:.4f}) outperformed SWA ({swa_weighted_f1:.4f}). "
                    f"Retaining checkpoint weights."
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(self.device)

        total_duration = time.time() - start_time
        logger.info(f"GPU Training completed in {total_duration/60:.2f} minutes! Best Val F1: {best_val_f1:.4f}")

        # Save training report
        report_path = Path(output_dir) / "training_report.json"
        report = {
            "model_name": self.model_name,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if use_cuda else "CPU",
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "use_cosine": self.use_cosine,
            "use_swa": self.use_swa,
            "total_duration_seconds": round(total_duration, 2),
            "best_validation_metrics": best_val_metrics,
            "training_history": training_history
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def predict_probabilities(
        self,
        data_loader: DataLoader,
        model: Optional[torch.nn.Module] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate softmax probabilities and true labels from a DataLoader."""
        eval_model = model if model is not None else self.model
        eval_model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].cpu().numpy()

                outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(labels)

        return np.vstack(all_probs), np.concatenate(all_labels)

    def evaluate_test_set(
        self,
        test_df: pd.DataFrame,
        text_column: str = "reviewText"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute probabilities and labels for test DataFrame."""
        col = "fused_text" if "fused_text" in test_df.columns else (text_column if text_column in test_df.columns else "reviewText_clean")
        test_dataset = FeedbackDataset(test_df[col].values, test_df["label"].values, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self.predict_probabilities(test_loader)

    def save(self, output_dir: str):
        """Save model and tokenizer."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_path)
        self.tokenizer.save_pretrained(out_path)
        logger.info(f"Model and tokenizer saved to {output_dir}")

    @classmethod
    def load(cls, model_dir: str, device: Optional[str] = None):
        """Load fine-tuned model and tokenizer from directory."""
        trainer = cls(device=device)
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(trainer.device)
        trainer.model.eval()
        return trainer
