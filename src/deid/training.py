"""Class weighting, weighted trainer, nested-CV fold training, and locked final eval."""
import os
import time
import random
import inspect
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import EarlyStoppingCallback

from . import labels as L
from .data import DocExample, SlidingWindowDataset
from .metrics import merge_logits_to_docs, score_docs_by_mode


def _make_training_args(**kwargs) -> TrainingArguments:
    """
    Use eval_strategy everywhere in this notebook.
    This helper remaps eval_strategy <-> evaluation_strategy for older/newer versions.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys()); valid.discard("self")

    if "evaluation_strategy" in valid and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "eval_strategy" in valid and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return TrainingArguments(**filtered)

def _trainer_tokenizer_kw(tokenizer):
    sig = inspect.signature(Trainer.__init__)
    return {"tokenizer": tokenizer} if "tokenizer" in sig.parameters else {}

def _extract_logits(predictions):
    if isinstance(predictions, (tuple, list)):
        return predictions[0]
    return predictions


def compute_class_weights_from_docs(docs: List[DocExample],
                                   o_mult=0.7, clip_min=0.2, clip_max=5.0,
                                   **type_mults) -> torch.Tensor:
    type_counts = {t:0 for t in L.ENTITY_TYPES}
    for d in docs:
        for tag in d.labels:
            if tag == "O":
                continue
            et = tag.split("-",1)[1]
            if et in type_counts:
                type_counts[et] += 1
    total = sum(type_counts.values()) + 1e-12
    base = {et: (total/(c+1e-12)) for et,c in type_counts.items()}

    w = np.ones(len(L.BIO_LABELS), dtype=np.float32)
    w[L.LABEL2ID["O"]] = float(o_mult)

    for et in L.ENTITY_TYPES:
        mult = float(type_mults.get(et, 1.0))
        v = float(np.clip(base.get(et, 1.0) * mult, clip_min, clip_max))
        for pref in ("B","I"):
            lab = f"{pref}-{et}"
            if lab in L.LABEL2ID:
                w[L.LABEL2ID[lab]] = v

    return torch.tensor(w, dtype=torch.float)


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_folds_doclevel(n_docs: int, n_folds: int, seed: int) -> List[List[int]]:
    idx = list(range(n_docs))
    random.Random(seed).shuffle(idx)
    folds = [[] for _ in range(n_folds)]
    for i, v in enumerate(idx):
        folds[i % n_folds].append(v)
    return folds

def split_inner_val_doclevel(train_fold_docs: List[DocExample], val_frac: float, seed: int) -> Tuple[List[DocExample], List[DocExample]]:
    n = len(train_fold_docs)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = set(idx[:n_val])
    tr_in = [d for i,d in enumerate(train_fold_docs) if i not in val_idx]
    va_in = [d for i,d in enumerate(train_fold_docs) if i in val_idx]
    return tr_in, va_in


@dataclass
class ModelCfg:
    name: str
    model_id: str
    max_len: int = 512
    stride: int = 256

    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    fp16: bool = True

    epochs_cv: int = 8
    epochs_final: int = 10

    n_folds: int = 5
    inner_val_frac: float = 0.10

    window_boosts: Optional[Dict[str, float]] = None

    use_class_weights: bool = False
    class_weight_multipliers: Optional[Dict[str, float]] = None

    early_stop_patience: int = 3
    early_stop_threshold: float = 0.0

    repair_train_gold: bool = True
    repair_eval_gold: bool = False


def _cfg_defaults(cfg: ModelCfg):
    if cfg.window_boosts is None:
        cfg.window_boosts = {"LOCATION": 1.3, "PHONE": 1.0}
    if cfg.class_weight_multipliers is None:
        cfg.class_weight_multipliers = {"LOCATION": 1.5, "PHONE": 1.1}
    return cfg


def train_one_fold_nested(cfg: ModelCfg,
                          tokenizer,
                          train_fold_docs: List[DocExample],
                          test_fold_docs:  List[DocExample],
                          out_dir: str,
                          seed: int,
                          mode: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = _cfg_defaults(cfg)

    tr_in, va_in = split_inner_val_doclevel(train_fold_docs, cfg.inner_val_frac, seed + 1337)

    train_ds = SlidingWindowDataset(tr_in, tokenizer, cfg.max_len, cfg.stride,
                                    compute_sample_weights=True, window_boosts=cfg.window_boosts)
    val_ds   = SlidingWindowDataset(va_in, tokenizer, cfg.max_len, cfg.stride,
                                    compute_sample_weights=False)
    test_ds  = SlidingWindowDataset(test_fold_docs, tokenizer, cfg.max_len, cfg.stride,
                                    compute_sample_weights=False)

    sampler = None
    if train_ds.sample_weights is not None:
        weights = torch.tensor(train_ds.sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    collator = DataCollatorForTokenClassification(tokenizer)
    tkw = _trainer_tokenizer_kw(tokenizer)

    config = AutoConfig.from_pretrained(cfg.model_id, num_labels=len(L.BIO_LABELS), id2label=L.ID2LABEL, label2id=L.LABEL2ID)
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_id, config=config)

    class_weights = None
    if cfg.use_class_weights:
        mults = dict(cfg.class_weight_multipliers or {})
        class_weights = compute_class_weights_from_docs(tr_in, o_mult=0.7, clip_min=0.2, clip_max=5.0, **mults)

    args = _make_training_args(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs_cv,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,      # selects using INNER-VAL only
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        save_total_limit=1,
        logging_strategy="epoch",
        fp16=(cfg.fp16 and torch.cuda.is_available()),
    )

    val_raw_map = {d.doc_id: d.labels_raw for d in va_in}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = _extract_logits(logits)
        doc_order, yt, yp = merge_logits_to_docs(val_ds, logits, labels)

        main = score_docs_by_mode(yt, yp, mode)
        yt_raw = [val_raw_map[doc_id] for doc_id in doc_order]
        raw  = score_docs_by_mode(yt_raw, yp, mode)

        return {
            "precision": main["overall"]["precision"],
            "recall": main["overall"]["recall"],
            "f1": main["overall"]["f1"],
            "macro_f1": main["overall"]["macro_f1"],
            "f1_raw_gold": raw["overall"]["f1"],
        }

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=cfg.early_stop_patience,
        early_stopping_threshold=cfg.early_stop_threshold
    )]

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        **tkw
    )

    if sampler is not None:
        def _get_train_dataloader_override():
            return torch.utils.data.DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=sampler,
                collate_fn=collator,
            )
        trainer.get_train_dataloader = _get_train_dataloader_override

    t0 = time.time()
    trainer.train()
    train_seconds = float(time.time() - t0)
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

    # One-shot fold test (NO selection)
    test_raw_map = {d.doc_id: d.labels_raw for d in test_fold_docs}
    preds = trainer.predict(test_ds)
    logits = _extract_logits(preds.predictions)
    doc_order, yt, yp = merge_logits_to_docs(test_ds, logits, preds.label_ids)

    main = score_docs_by_mode(yt, yp, mode)
    yt_raw = [test_raw_map[doc_id] for doc_id in doc_order]
    raw  = score_docs_by_mode(yt_raw, yp, mode)

    w = np.array(train_ds.sample_weights, dtype=np.float64) if train_ds.sample_weights is not None else np.array([1.0])
    w_summary = {
        "min": float(w.min()), "max": float(w.max()), "mean": float(w.mean()),
        "pct_boosted": float((w > 1.0).mean()),
        "window_boosts": cfg.window_boosts
    }

    return {
        "best_checkpoint": best_ckpt,
        "train_seconds": train_seconds,
        "mode": mode,
        "train_fold_docs": len(train_fold_docs),
        "train_inner_docs": len(tr_in),
        "val_inner_docs": len(va_in),
        "test_fold_docs": len(test_fold_docs),
        "window_sampling": w_summary,
        "test_main": main,
        "test_raw_gold": raw,
    }


def train_final_locked(cfg: ModelCfg,
                       tokenizer,
                       train_docs: List[DocExample],
                       eval_docs:  List[DocExample],
                       out_dir: str,
                       seed: int,
                       mode: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = _cfg_defaults(cfg)

    train_ds = SlidingWindowDataset(train_docs, tokenizer, cfg.max_len, cfg.stride,
                                    compute_sample_weights=True, window_boosts=cfg.window_boosts)
    eval_ds  = SlidingWindowDataset(eval_docs, tokenizer, cfg.max_len, cfg.stride,
                                    compute_sample_weights=False)

    sampler = None
    if train_ds.sample_weights is not None:
        weights = torch.tensor(train_ds.sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    collator = DataCollatorForTokenClassification(tokenizer)
    tkw = _trainer_tokenizer_kw(tokenizer)

    config = AutoConfig.from_pretrained(cfg.model_id, num_labels=len(L.BIO_LABELS), id2label=L.ID2LABEL, label2id=L.LABEL2ID)
    model = AutoModelForTokenClassification.from_pretrained(cfg.model_id, config=config)

    class_weights = None
    if cfg.use_class_weights:
        mults = dict(cfg.class_weight_multipliers or {})
        class_weights = compute_class_weights_from_docs(train_docs, o_mult=0.7, clip_min=0.2, clip_max=5.0, **mults)

    args = _make_training_args(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs_final,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,      # critical: no selection on eval
        seed=seed,
        report_to="none",
        save_total_limit=1,
        logging_strategy="epoch",
        fp16=(cfg.fp16 and torch.cuda.is_available()),
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        class_weights=class_weights,
        **tkw
    )

    if sampler is not None:
        def _get_train_dataloader_override():
            return torch.utils.data.DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                sampler=sampler,
                collate_fn=collator,
            )
        trainer.get_train_dataloader = _get_train_dataloader_override

    t0 = time.time()
    trainer.train()
    train_seconds = float(time.time() - t0)

    # ONE-SHOT eval
    preds = trainer.predict(eval_ds)
    logits = _extract_logits(preds.predictions)
    doc_order, yt, yp = merge_logits_to_docs(eval_ds, logits, preds.label_ids)

    main = score_docs_by_mode(yt, yp, mode)
    raw  = score_docs_by_mode([d.labels_raw for d in eval_docs], yp, mode)

    final_model_dir = os.path.join(out_dir, "final_model")
    trainer.save_model(final_model_dir)

    return {
        "train_seconds": train_seconds,
        "final_model_dir": final_model_dir,
        "eval_main": main,
        "eval_raw_gold": raw,
        "eval_doc_order": doc_order,
    }
