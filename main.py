import json
import random
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from seqeval.metrics import f1_score

from models.clinicalbert_finetune import (
    NERDataset,
    WeightedSmoothTrainer,
    chunk_predict,
    zip_dir
)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # ─── 0. Seed ───────────────────────────────────────────────
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ─── 1. Load Data ──────────────────────────────────────────
    print("🔄 Loading data…")
    train_data       = load_json("data/train.json")
    groundtruth_data = load_json("data/test_gt.json")
    input_test_data  = load_json("data/test.json")

    # ─── 2. Build label list ───────────────────────────────────
    print("🔖 Extracting label set from training + ground‑truth…")
    all_labels = set()
    for d in train_data + groundtruth_data:
        all_labels.update(d["labels"])
    label_list = sorted(all_labels)
    print("Labels:", label_list)

    # Map labels ↔ IDs
    label_to_id = {l:i for i,l in enumerate(label_list)}
    id_to_label = {i:l for l,i in label_to_id.items()}

    # ─── 3. Prepare tokens & tags ──────────────────────────────
    train_tokens = [d["tokens"] for d in train_data]
    train_tags   = [d["labels"] for d in train_data]
    val_tokens   = [d["tokens"] for d in groundtruth_data]
    val_tags     = [d["labels"] for d in groundtruth_data]
    test_tokens  = [d["tokens"] for d in input_test_data]

    # ─── 4. Initialize tokenizer & datasets ───────────────────
    print("\n== [CLINICALBERT Training] ==")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", use_fast=True
    )
    train_ds = NERDataset(train_tokens, train_tags, tokenizer, label_to_id)
    val_ds   = NERDataset(val_tokens,   val_tags,   tokenizer, label_to_id)

    # ─── 5. Compute class weights ──────────────────────────────
    flat = [lbl for sent in train_tags for lbl in sent]
    tot, cnt = len(flat), Counter(flat)
    class_w = torch.tensor([tot / cnt.get(l,1) for l in label_list])
    class_w /= class_w.mean()

    # ─── 6. Model & Trainer setup ─────────────────────────────
    model = AutoModelForTokenClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_list)
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        refs  = p.label_ids
        yp, yr = [], []
        for pr, rf in zip(preds, refs):
            sp, sr = [], []
            for p_i, r_i in zip(pr, rf):
                if r_i == -100: continue
                sp.append(id_to_label[p_i])
                sr.append(id_to_label[r_i])
            yp.append(sp); yr.append(sr)
        f1 = f1_score(yr, yp)
        print(f"[Eval] Macro F1: {f1:.4f}")
        return {"eval_f1": f1}

    args = TrainingArguments(
        output_dir="./outputs/clinicalbert",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none"
    )

    trainer = WeightedSmoothTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        smooth=0.1,
        class_w=class_w
    )

    # ─── 7. Train ──────────────────────────────────────────────
    trainer.train()
    trainer.save_model("./outputs/clinicalbert")
    print("✅ Training complete and model saved to ./outputs/clinicalbert")

    # ─── 8. Predict on unlabeled test ─────────────────────────
    print("\n== [CLINICALBERT Predict: Unlabeled Test] ==")
    pred_results = chunk_predict(
        trainer, tokenizer, test_tokens, id_to_label
    )
    with open("outputs/clinicalbert_test_preds.json", "w", encoding="utf-8") as f:
        json.dump(pred_results, f, indent=2, ensure_ascii=False)
    print("✅ Predictions saved to outputs/clinicalbert_test_preds.json")

    # ─── 9. Zip outputs ────────────────────────────────────────
    zip_dir("./outputs/clinicalbert", "./outputs/clinicalbert.zip")
    print("✅ Zipped model & tokenizer to ./outputs/clinicalbert.zip")

    print("\nAll done!")

if __name__ == "__main__":
    main()
