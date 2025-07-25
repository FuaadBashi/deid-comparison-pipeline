import os
import torch
import zipfile
import numpy as np
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import f1_score

# ──────── Dataset ─────────
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, tags, tokenizer, label_to_id, max_len=512):
        self.tokens, self.tags = tokens, tags
        self.tok, self.l2id, self.max_len = tokenizer, label_to_id, max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        words, labels = self.tokens[idx], self.tags[idx]
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True
        )
        wids     = enc.word_ids()
        aligned, prev = [], None

        for wid in wids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(self.l2id[labels[wid]])
            else:
                aligned.append(-100)
            prev = wid

        # keep only tensors needed for model
        out = {
            "input_ids":     torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels":        torch.tensor(aligned)
        }
        return out

# ──────── Custom Trainer ─────────
class WeightedSmoothTrainer(Trainer):
    def __init__(self, smooth: float, class_w: torch.Tensor, **kw):
        super().__init__(**kw)
        self.smooth, self.class_w = smooth, class_w.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        mask    = labels.ne(-100)

        logits, labels = logits[mask], labels[mask]
        ncls = logits.size(-1)

        true_dist = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        true_dist = true_dist * (1 - self.smooth) + self.smooth / ncls

        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(true_dist * logp * self.class_w).sum(-1).mean()

        return (loss, outputs) if return_outputs else loss

# ──────── Prediction helper ─────────
def chunk_predict(trainer, tokenizer, docs, id2label, chunk=512, stride=256):
    device = trainer.model.device
    results = []
    for words in docs:
        preds, start = [], 0
        while start < len(words):
            win = words[start : start + chunk]
            enc = tokenizer(
                win,
                is_split_into_words=True,
                truncation=True,
                max_length=chunk,
                padding="max_length",
                return_tensors="pt"
            )
            wids = enc.word_ids()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = trainer.model(**enc).logits
            top_ids = logits.argmax(-1)[0].cpu().numpy()

            prev = None
            for i, wid in enumerate(wids):
                if wid is None or wid == prev:
                    continue
                preds.append(id2label[top_ids[i]])
                prev = wid

            start += stride
            if len(preds) >= len(words):
                break

        results.append({"tokens": words, "predicted_labels": preds[: len(words)]})
    return results

# ──────── Utility: zip a folder ─────────
def zip_dir(input_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                arc = os.path.relpath(fp, input_dir)
                zf.write(fp, arcname=arc)
