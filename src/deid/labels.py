"""Label schema, BIO tag repair, and reproducibility helpers."""
import random
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

# JSON keys expected in the input corpus
DOC_ID_KEY = "record_id"
TOK_KEY = "tokens"
TAG_KEY = "labels"

# Populated by set_label_schema() after inferring types from TRAIN
ENTITY_TYPES: List[str] = []
BIO_LABELS: List[str] = []
LABEL2ID: Dict[str, int] = {}
ID2LABEL: Dict[int, str] = {}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def _f1(p, r):
    return (2*p*r/(p+r)) if (p+r) else 0.0


def infer_entity_types_from_records(records: List[Dict[str, Any]]) -> List[str]:
    types = set()
    for r in records:
        for t in r["tags"]:
            if t == "O" or t is None:
                continue
            if "-" not in t:
                continue
            pref, et = t.split("-", 1)
            if pref in ("B", "I") and et and et != "O":
                types.add(et)
    return sorted(types)

def set_label_schema(entity_types: List[str]):
    global ENTITY_TYPES, BIO_LABELS, LABEL2ID, ID2LABEL
    ENTITY_TYPES = list(entity_types)
    BIO_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ["B", "I"]]
    LABEL2ID = {l:i for i,l in enumerate(BIO_LABELS)}
    ID2LABEL = {i:l for l,i in LABEL2ID.items()}


def repair_bio_sequence(tags: List[str]) -> Tuple[List[str], int]:
    changed = 0
    out = []
    prev_type = "O"

    for t in tags:
        if t in ("B-O", "I-O"):
            out.append("O"); changed += 1
            prev_type = "O"
            continue

        if t == "O":
            out.append("O"); prev_type = "O"
            continue

        if "-" not in t:
            out.append("O"); changed += 1
            prev_type = "O"
            continue

        pref, et = t.split("-", 1)
        lab = f"{pref}-{et}"
        if lab not in LABEL2ID:
            out.append("O"); changed += 1
            prev_type = "O"
            continue

        if pref == "B":
            out.append(lab)
            prev_type = et
        elif pref == "I":
            if prev_type == et:
                out.append(lab)
            else:
                out.append(f"B-{et}")
                changed += 1
                prev_type = et
        else:
            out.append("O"); changed += 1
            prev_type = "O"

    return out, changed
