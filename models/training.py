#!/usr/bin/env python3
"""
Shared components for standardised PHI de-identification cross-validation.
Implements stratified 5-fold CV, oversampling, class weights, token-classification
training and evaluation with seqeval, and optional test-time evaluation.
"""

import os
os.environ["WANDB_DISABLED"] = "true"  # avoid accidental W&B logging

import json
import re
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from collections import Counter

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# ----------------------------- Configuration -----------------------------

SEED = 42
N_FOLDS = 5
MAX_LEN = 256

PER_DEVICE_BATCH = 16
GRAD_ACCUM = 2  # effective batch size 32
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 3

OUTPUT_ROOT = "./standardised_runs"

TARGET_LABELS = [
    "O",
    "B-NAME", "I-NAME",
    "B-DATE", "I-DATE",
    "B-ID", "I-ID",
    "B-LOCATION", "I-LOCATION",
    "B-PHONE", "I-PHONE",
    "B-HOSPITAL", "I-HOSPITAL",
]
LABEL2ID = {lab: i for i, lab in enumerate(TARGET_LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# Training-only oversampling multipliers
OVER_MULT = {"LOCATION": 6, "PHONE": 3}


# ----------------------------- Reproducibility -----------------------------

def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(SEED)


# ----------------------------- Data I/O & Cleaning -----------------------------

def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    raw = raw.lstrip("\ufeff").replace("&apos;", "'").replace("&quot;", "\"")
    raw = re.sub(r",\s*([\]}])", r"\1", raw)
    if not raw.endswith("]"):
        raw += "]"
    return json.loads(raw)


def load_bio_json(path: str, name: str) -> List[Dict[str, Any]]:
    data = _load_json_list(path)
    cleaned, issues = [], Counter()
    for ex in data:
        tokens = ex.get("tokens", [])
        labels = ex.get("labels", [])
        if len(tokens) != len(labels):
            issues["length_mismatch"] += 1
            continue

        fixed = []
        for lab in labels:
            if lab in TARGET_LABELS:
                fixed.append(lab)
            else:
                if lab in ("B-O", "I-O"):
                    fixed.append("O"); issues["invalid_O_prefix"] += 1
                elif "-" in lab:
                    _, ent = lab.split("-", 1)
                    if f"B-{ent}" in TARGET_LABELS:
                        fixed.append(f"B-{ent}"); issues[f"fixed_{lab}"] += 1
                    elif f"I-{ent}" in TARGET_LABELS:
                        fixed.append(f"B-{ent}"); issues[f"fixed_{lab}"] += 1
                    else:
                        fixed.append("O"); issues[f"unknown_{lab}"] += 1
                else:
                    fixed.append("O"); issues[f"non_bio_{lab}"] += 1

        cleaned.append({"tokens": tokens, "labels": fixed})

    if issues:
        print(f"[{name}] cleaned with notes: " + ", ".join(f"{k}={v}" for k, v in issues.most_common()))
    print(f"[{name}] {len(data)} → {len(cleaned)} samples")
    return cleaned


# ----------------------------- CV Stratification -----------------------------

def stratify_bucket(ex: Dict[str, Any]) -> int:
    labs = ex["labels"]
    has_loc = any(l != "O" and l.split("-")[-1] == "LOCATION" for l in labs)
    has_phn = any(l != "O" and l.split("-")[-1] == "PHONE" for l in labs)
    return (1 if has_loc else 0) * 2 + (1 if has_phn else 0)  # 0..3


# ----------------------------- Oversampling -----------------------------

def oversample_training(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in examples:
        ents = set(l.split("-")[-1] for l in ex["labels"] if l != "O")
        mult = 1
        for ent, m in OVER_MULT.items():
            if ent in ents:
                mult = max(mult, m)
        out.extend([ex] * mult)
    return out


# ----------------------------- Class Weights -----------------------------

def compute_class_weights(train_examples: List[Dict[str, Any]]) -> torch.Tensor:
    counts = Counter()
    for ex in train_examples:
        for lab in ex["labels"]:
            if lab in LABEL2ID:
                counts[LABEL2ID[lab]] += 1
    total = sum(counts.values())
    weights = []
    for i in range(len(TARGET_LABELS)):
        c = counts.get(i, 1)
        w = total / (len(TARGET_LABELS) * c)
        w = min(max(w, 0.3), 10.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)


# ----------------------------- Dataset -----------------------------

class NERDataset(Dataset):
    """
    For training/eval: labels are applied only to the first subword of each word.
    Subsequent subwords are set to -100 (ignored by loss and metrics).
    """
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_len: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        tokens, labels = ex["tokens"], ex["labels"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = enc.word_ids(batch_index=0)
        label_ids = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev_wid:
                # first subword of the word → keep label
                lab = labels[wid] if wid < len(labels) else "O"
                label_ids.append(LABEL2ID.get(lab, LABEL2ID["O"]))
            else:
                # subsequent subword → ignore in loss/metrics
                label_ids.append(-100)
            prev_wid = wid

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)
        return item


# ----------------------------- Prediction Alignment -----------------------------

def align_predictions(pred_logits: np.ndarray, label_ids: np.ndarray) -> Tuple[List[List[str]], List[List[str]]]:
    preds = np.argmax(pred_logits, axis=2)
    y_true, y_pred = [], []
    for p_row, l_row in zip(preds, label_ids):
        t_seq, p_seq = [], []
        for pi, li in zip(p_row, l_row):
            if li == -100:
                continue
            t_seq.append(ID2LABEL[int(li)])
            p_seq.append(ID2LABEL[int(pi)])
        y_true.append(t_seq)
        y_pred.append(p_seq)
    return y_true, y_pred


# ----------------------------- Metrics -----------------------------

def make_metrics():
    def _compute(p):
        preds, labels = p
        y_true, y_pred = align_predictions(preds, labels)
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "micro_f1": f1_score(y_true, y_pred, average="micro"),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }
    return _compute


# ----------------------------- Trainer -----------------------------

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ----------------------------- Runner -----------------------------

@dataclass
class ModelCfg:
    name: str
    model_id: str
    epochs: int
    tok_kwargs: Dict[str, Any]


def run_model(cfg: ModelCfg, train_examples: List[Dict[str, Any]], test_examples: List[Dict[str, Any]]) -> None:
    print(f"\nModel: {cfg.name} | HF id: {cfg.model_id} | epochs={cfg.epochs}")

    out_dir = os.path.join(OUTPUT_ROOT, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, **cfg.tok_kwargs)

    strata = np.array([stratify_bucket(ex) for ex in train_examples])
    indices = np.arange(len(train_examples))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics = []

    for fold_id, (tr_idx, dv_idx) in enumerate(skf.split(indices, strata), start=1):
        fold_dir = os.path.join(out_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        train_part = [train_examples[i] for i in tr_idx]
        dev_part   = [train_examples[i] for i in dv_idx]

        # training-only oversampling and class weights from original train split
        train_aug = oversample_training(train_part)
        class_weights = compute_class_weights(train_part)

        ds_train = NERDataset(train_aug, tokenizer, MAX_LEN)
        ds_dev   = NERDataset(dev_part,   tokenizer, MAX_LEN)

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_id, num_labels=len(TARGET_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
        )

        steps_per_epoch = max(1, math.ceil(len(ds_train) / max(1, PER_DEVICE_BATCH)))
        logging_steps = max(1, steps_per_epoch // 5)

        args = TrainingArguments(
            output_dir=fold_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=PER_DEVICE_BATCH,
            num_train_epochs=cfg.epochs,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_dir=os.path.join(fold_dir, "logs"),
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            save_total_limit=1,
            report_to="none",
            lr_scheduler_type="linear",
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=GRAD_ACCUM,
            dataloader_pin_memory=False
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=ds_train,
            eval_dataset=ds_dev,
            tokenizer=tokenizer,
            compute_metrics=make_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
            class_weights=class_weights
        )

        print(f"  Fold {fold_id}: training...")
        trainer.train()
        trainer.save_model(fold_dir)
        tokenizer.save_pretrained(fold_dir)

        preds = trainer.predict(ds_dev)
        y_true, y_pred = align_predictions(preds.predictions, preds.label_ids)
        fold_result = {
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": classification_report(y_true, y_pred, digits=4),
        }
        with open(os.path.join(fold_dir, "dev_eval.json"), "w") as f:
            json.dump(fold_result, f, indent=2)
        fold_metrics.append(fold_result)
        print(f"  Fold {fold_id}: macro-F1={fold_result['macro_f1']:.4f} micro-F1={fold_result['micro_f1']:.4f}")

    def mean(key: str) -> float:
        return float(np.mean([m[key] for m in fold_metrics]))

    summary = {
        "model": cfg.model_id,
        "epochs": cfg.epochs,
        "folds": len(fold_metrics),
        "avg_precision": mean("precision"),
        "avg_recall": mean("recall"),
        "avg_micro_f1": mean("micro_f1"),
        "avg_macro_f1": mean("macro_f1"),
    }
    with open(os.path.join(out_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"CV summary ({cfg.name}): macro-F1={summary['avg_macro_f1']:.4f}, micro-F1={summary['avg_micro_f1']:.4f}")

    # Optional test pass
    if test_examples:
        best_fold = 1 + int(np.argmax([m["macro_f1"] for m in fold_metrics]))
        best_dir = os.path.join(out_dir, f"fold{best_fold}")
        tokenizer = AutoTokenizer.from_pretrained(best_dir, use_fast=True, **cfg.tok_kwargs)
        model = AutoModelForTokenClassification.from_pretrained(best_dir)
        test_ds = NERDataset(test_examples, tokenizer, MAX_LEN)
        test_tr = Trainer(model=model, tokenizer=tokenizer)
        preds = test_tr.predict(test_ds)

        if "labels" in test_examples[0]:
            y_true, y_pred = align_predictions(preds.predictions, preds.label_ids)
            test_metrics = {
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "classification_report": classification_report(y_true, y_pred, digits=4),
            }
            with open(os.path.join(out_dir, "test_eval.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test macro-F1={test_metrics['macro_f1']:.4f}, micro-F1={test_metrics['micro_f1']:.4f}")
        else:
            pred_ids = np.argmax(preds.predictions, axis=-1)
            out = []
            for i, ex in enumerate(test_examples):
                out.append({
                    "tokens": ex["tokens"],
                    "pred_labels": [ID2LABEL[int(pid)] for pid, li in zip(pred_ids[i], preds.label_ids[i]) if li != -100]
                })
            with open(os.path.join(out_dir, "test_predictions.json"), "w") as f:
                json.dump(out, f, indent=2)
            print("Test predictions saved (no gold labels found).")
