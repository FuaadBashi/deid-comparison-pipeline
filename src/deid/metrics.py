"""Scoring: token-level typed micro-F1 and strict entity exact-match micro-F1."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Set

import numpy as np

from . import labels as L
from .labels import _safe_div, _f1, repair_bio_sequence


def merge_logits_to_docs(dataset: SlidingWindowDataset,
                         logits: np.ndarray,
                         label_ids: np.ndarray) -> Tuple[List[str], List[List[str]], List[List[str]]]:
    doc_sums = []
    doc_cnts = []
    doc_gold = []
    for d in dataset.docs:
        n = len(d.words)
        doc_sums.append(np.zeros((n, len(L.BIO_LABELS)), dtype=np.float64))
        doc_cnts.append(np.zeros((n,), dtype=np.float64))
        doc_gold.append(d.labels)

    for i in range(len(dataset)):
        meta = dataset.windows[i]
        di = meta["doc_i"]
        start = meta["start"]

        li = logits[i]
        labs = label_ids[i]
        target_len = labs.shape[0]

        word_ids = dataset.get_padded_word_ids(i, target_len)

        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if labs[pos] == -100:
                continue
            abs_w = start + wid
            if 0 <= abs_w < doc_sums[di].shape[0]:
                doc_sums[di][abs_w] += li[pos]
                doc_cnts[di][abs_w] += 1.0

    yt_docs, yp_docs, doc_order = [], [], []
    for di, d in enumerate(dataset.docs):
        n = len(d.words)
        if n == 0:
            continue
        avg = doc_sums[di] / np.maximum(doc_cnts[di][:, None], 1.0)
        pred_ids = avg.argmax(axis=-1).tolist()
        pred_tags = [L.ID2LABEL[int(pid)] for pid in pred_ids]
        yt_docs.append(doc_gold[di])
        yp_docs.append(pred_tags)
        doc_order.append(d.doc_id)

    return doc_order, yt_docs, yp_docs


def _to_type(tag: str) -> str:
    if tag == "O": return "O"
    if "-" not in tag: return "O"
    return tag.split("-", 1)[1]

def token_typed_breakdown(yt_docs: List[List[str]], yp_docs: List[List[str]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    per = {e: {"tp":0,"fp":0,"fn":0,"gold":0,"pred":0} for e in L.ENTITY_TYPES}

    for yt, yp in zip(yt_docs, yp_docs):
        for g, p in zip(yt, yp):
            gt = _to_type(g)
            pt = _to_type(p)

            if gt != "O" and gt in per: per[gt]["gold"] += 1
            if pt != "O" and pt in per: per[pt]["pred"] += 1

            if gt == pt and gt != "O":
                tp += 1
                per[gt]["tp"] += 1
            else:
                if pt != "O":
                    fp += 1
                    per[pt]["fp"] += 1
                if gt != "O":
                    fn += 1
                    per[gt]["fn"] += 1

    P = _safe_div(tp, tp+fp)
    R = _safe_div(tp, tp+fn)
    F = _f1(P, R)

    per_out, macro_parts = {}, []
    for e, c in per.items():
        ep = _safe_div(c["tp"], c["tp"]+c["fp"])
        er = _safe_div(c["tp"], c["tp"]+c["fn"])
        ef = _f1(ep, er)
        macro_parts.append(ef)
        gold, pred = c["gold"], c["pred"]
        per_out[e] = {
            **{k:int(v) for k,v in c.items()},
            "bias": int(pred-gold),
            "bias_pct": float((pred-gold)/(gold+1e-12)),
            "precision": float(ep),
            "recall": float(er),
            "f1": float(ef),
        }
    macro_f1 = float(sum(macro_parts)/len(macro_parts)) if macro_parts else 0.0

    return {
        "overall": {"precision": float(P), "recall": float(R), "f1": float(F), "macro_f1": float(macro_f1),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn)},
        "per_entity": per_out
    }


def extract_entities_bio(tags: List[str]) -> List[Tuple[int, int, str]]:
    tags_rep, _ = repair_bio_sequence(tags)
    spans = []
    cur_type = None
    start = None

    def close(i):
        nonlocal cur_type, start
        if cur_type is not None and start is not None:
            spans.append((start, i, cur_type))
        cur_type, start = None, None

    for i, t in enumerate(tags_rep):
        if t == "O":
            close(i); continue
        if "-" not in t:
            close(i); continue
        pref, et = t.split("-", 1)
        if pref == "B":
            close(i)
            cur_type, start = et, i
        elif pref == "I":
            if cur_type == et and start is not None:
                continue
            close(i)
            cur_type, start = et, i
        else:
            close(i)

    close(len(tags_rep))
    return spans

def entity_exact_breakdown(yt_docs: List[List[str]], yp_docs: List[List[str]]) -> Dict[str, Any]:
    tp = fp = fn = 0
    per = {e: {"tp":0,"fp":0,"fn":0,"gold":0,"pred":0} for e in L.ENTITY_TYPES}

    for yt, yp in zip(yt_docs, yp_docs):
        gold_spans = extract_entities_bio(yt)
        pred_spans = extract_entities_bio(yp)
        gold_set: Set[Tuple[int,int,str]] = set(gold_spans)
        pred_set: Set[Tuple[int,int,str]] = set(pred_spans)

        inter = gold_set & pred_set
        tp += len(inter)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        for (s,e,t) in gold_spans:
            if t in per: per[t]["gold"] += 1
        for (s,e,t) in pred_spans:
            if t in per: per[t]["pred"] += 1
        for (s,e,t) in inter:
            if t in per: per[t]["tp"] += 1
        for (s,e,t) in (pred_set - gold_set):
            if t in per: per[t]["fp"] += 1
        for (s,e,t) in (gold_set - pred_set):
            if t in per: per[t]["fn"] += 1

    P = _safe_div(tp, tp+fp)
    R = _safe_div(tp, tp+fn)
    F = _f1(P, R)

    per_out, macro_parts = {}, []
    for e, c in per.items():
        ep = _safe_div(c["tp"], c["tp"]+c["fp"])
        er = _safe_div(c["tp"], c["tp"]+c["fn"])
        ef = _f1(ep, er)
        macro_parts.append(ef)
        gold, pred = c["gold"], c["pred"]
        per_out[e] = {
            **{k:int(v) for k,v in c.items()},
            "bias": int(pred-gold),
            "bias_pct": float((pred-gold)/(gold+1e-12)),
            "precision": float(ep),
            "recall": float(er),
            "f1": float(ef),
        }
    macro_f1 = float(sum(macro_parts)/len(macro_parts)) if macro_parts else 0.0

    return {
        "overall": {"precision": float(P), "recall": float(R), "f1": float(F), "macro_f1": float(macro_f1),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn)},
        "per_entity": per_out
    }

def score_docs_by_mode(yt_docs, yp_docs, mode: str) -> Dict[str, Any]:
    if mode == "token_typed":
        return token_typed_breakdown(yt_docs, yp_docs)
    if mode == "entity_exact":
        return entity_exact_breakdown(yt_docs, yp_docs)
    raise ValueError(f"Unknown mode: {mode}")
