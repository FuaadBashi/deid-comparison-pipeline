"""Corpus loading, document container, and the sliding-window dataset."""
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import labels as L
from .labels import (
    DOC_ID_KEY, TOK_KEY, TAG_KEY,
    infer_entity_types_from_records, set_label_schema, repair_bio_sequence,
)


@dataclass
class DocExample:
    doc_id: str
    words: List[str]
    labels: List[str]     # repaired/normalised gold used for training/scoring
    labels_raw: List[str] # raw gold from file (for reporting/ablation)


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        obj = obj["data"]
    if not isinstance(obj, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(obj)}")
    return obj

def load_raw_records(path: str) -> List[Dict[str, Any]]:
    arr = load_json_array(path)
    out = []
    for i, x in enumerate(arr):
        if DOC_ID_KEY not in x:
            raise ValueError(f"[{path}] record {i} missing DOC_ID_KEY='{DOC_ID_KEY}'. keys={list(x.keys())}")
        if TOK_KEY not in x:
            raise ValueError(f"[{path}] record {i} missing TOK_KEY='{TOK_KEY}'. keys={list(x.keys())}")
        if TAG_KEY not in x:
            raise ValueError(f"[{path}] record {i} missing TAG_KEY='{TAG_KEY}'. keys={list(x.keys())}")

        doc_id = str(x[DOC_ID_KEY])
        words  = list(x[TOK_KEY])
        tags   = x[TAG_KEY]

        if tags is None:
            raise ValueError(
                f"[{path}] record {i} has TAG_KEY='{TAG_KEY}' = None for doc_id={doc_id}. "
                f"If this is a 'test' file with missing labels, use the GOLD eval file here."
            )
        tags = list(tags)

        if len(words) != len(tags):
            raise ValueError(f"[{path}] record {i} len mismatch doc={doc_id}: words={len(words)} tags={len(tags)}")

        out.append({"id": doc_id, "words": words, "tags": tags})
    return out


def records_to_docs(records: List[Dict[str, Any]], do_repair: bool) -> Tuple[List[DocExample], Dict[str, Any]]:
    changed_total = 0
    total_tags = 0
    docs = []
    for r in records:
        doc_id = r["id"]
        words = r["words"]
        tags_raw = r["tags"]
        tags = tags_raw[:]
        if do_repair:
            tags, ch = repair_bio_sequence(tags)
            changed_total += ch
        total_tags += len(tags)
        docs.append(DocExample(doc_id=doc_id, words=words, labels=tags, labels_raw=tags_raw))
    rep = {"changed": int(changed_total), "total": int(total_tags), "changed_pct": float(changed_total/(total_tags+1e-12))}
    return docs, rep


def load_i2b2_train_and_eval(train_path: str, eval_path: str,
                            repair_train_gold: bool = True, repair_eval_gold: bool = False) -> Tuple[List[DocExample], List[DocExample], Dict[str, Any]]:
    train_rec = load_raw_records(train_path)
    eval_rec  = load_raw_records(eval_path)

    inferred = infer_entity_types_from_records(train_rec)
    if not inferred:
        raise ValueError("No entity types inferred from TRAIN. Expected BIO labels like B-XXX/I-XXX.")

    set_label_schema(inferred)

    train_docs, rep_tr = records_to_docs(train_rec, do_repair=repair_train_gold)
    eval_docs,  rep_ev = records_to_docs(eval_rec,  do_repair=repair_eval_gold)

    tr_ids = set(d.doc_id for d in train_docs)
    ev_ids = set(d.doc_id for d in eval_docs)
    overlap = len(tr_ids & ev_ids)
    if overlap != 0:
        raise ValueError(f"Doc ID overlap train∩eval={overlap} (must be 0)")

    rep = {
        "inferred_entity_types": inferred,
        "repair_train_gold": bool(repair_train_gold),
        "repair_eval_gold": bool(repair_eval_gold),
        "train_changed": rep_tr["changed"], "train_total": rep_tr["total"], "train_changed_pct": rep_tr["changed_pct"],
        "eval_changed": rep_ev["changed"],  "eval_total": rep_ev["total"],  "eval_changed_pct": rep_ev["changed_pct"],
        "train_len": len(train_docs),
        "eval_len": len(eval_docs),
        "id_overlap_train_eval": int(overlap),
    }
    return train_docs, eval_docs, rep


def build_tokenizer(model_id: str):
    kwargs = {}
    if "roberta" in model_id.lower():
        kwargs["add_prefix_space"] = True
    return AutoTokenizer.from_pretrained(model_id, use_fast=True, **kwargs)


class SlidingWindowDataset(Dataset):
    def __init__(self,
                 docs: List[DocExample],
                 tokenizer,
                 max_len: int,
                 stride: int,
                 compute_sample_weights: bool = False,
                 window_boosts: Optional[Dict[str, float]] = None):
        self.docs = docs
        self.tok = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.compute_sample_weights = compute_sample_weights
        self.window_boosts = window_boosts or {}

        self.windows = []  # {doc_i, start, end}
        for di, d in enumerate(docs):
            n = len(d.words)
            if n == 0:
                continue
            win_words = max(32, max_len // 2)
            step = max(1, win_words - max(1, stride//2))
            s = 0
            while s < n:
                e = min(n, s + win_words)
                self.windows.append({"doc_i": di, "start": s, "end": e})
                if e == n:
                    break
                s += step

        # cache word_ids for merge (do not batch)
        self._word_ids_cache: List[Optional[List[Optional[int]]]] = [None] * len(self.windows)

        self.sample_weights = None
        if self.compute_sample_weights:
            self.sample_weights = []
            for w in self.windows:
                d = self.docs[w["doc_i"]]
                seg = d.labels[w["start"]:w["end"]]
                wgt = 1.0
                for t in seg:
                    if t == "O":
                        continue
                    et = t.split("-", 1)[1]
                    wgt *= float(self.window_boosts.get(et, 1.0))
                wgt = min(10.0, max(0.1, wgt))
                self.sample_weights.append(wgt)

    def __len__(self):
        return len(self.windows)

    def _encode_window(self, idx: int):
        w = self.windows[idx]
        d = self.docs[w["doc_i"]]
        start, end = w["start"], w["end"]
        words = d.words[start:end]
        tags  = d.labels[start:end]

        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
        )
        word_ids = enc.word_ids()
        self._word_ids_cache[idx] = word_ids

        labels = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_wid:
                labels.append(L.LABEL2ID.get(tags[wid], L.LABEL2ID["O"]))
            else:
                labels.append(-100)
            prev_wid = wid

        return enc, labels

    def get_padded_word_ids(self, idx: int, target_len: int) -> List[Optional[int]]:
        if self._word_ids_cache[idx] is None:
            _enc, _labels = self._encode_window(idx)
        wid = self._word_ids_cache[idx]
        assert wid is not None
        if len(wid) < target_len:
            return wid + [None] * (target_len - len(wid))
        return wid[:target_len]

    def __getitem__(self, idx: int):
        enc, labels = self._encode_window(idx)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
