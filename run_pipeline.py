#!/usr/bin/env python3
"""
Run the full nested-CV + locked-eval pipeline for one model configuration.

Example
-------
    python run_pipeline.py --model roberta --mode entity_exact \
        --train data/train.json --eval data/eval.json --out outputs/

Scoring modes
-------------
    entity_exact  strict entity-level exact-match micro-F1 (span boundaries and
                  type must both match). This is the headline metric.
    token_typed   token-level typed micro-F1 (BIO collapsed to entity type).
                  More forgiving; reported alongside for comparability with
                  papers that use token-level scoring.
"""
import os
import sys
import json
import argparse
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from deid import labels as L
from deid.labels import set_seed
from deid.data import load_i2b2_train_and_eval, build_tokenizer
from deid.training import (
    ModelCfg, _cfg_defaults, build_folds_doclevel,
    train_one_fold_nested, train_final_locked,
)
from deid.configs import clinical_cfg, biobert_cfg, roberta_cfg

REGISTRY = {
    "clinicalbert": clinical_cfg,
    "biobert": biobert_cfg,
    "roberta": roberta_cfg,
}


def run(cfg: ModelCfg, mode: str, seed: int, train_json: str, eval_json: str,
        out_root: str) -> str:
    set_seed(seed)
    cfg = _cfg_defaults(cfg)

    train_docs, eval_docs, rep = load_i2b2_train_and_eval(
        train_json, eval_json,
        repair_train_gold=cfg.repair_train_gold,
        repair_eval_gold=cfg.repair_eval_gold,
    )
    print(f"Inferred entity types: {rep['inferred_entity_types']}")
    print(f"Train docs: {rep['train_len']}  Eval docs: {rep['eval_len']}  "
          f"ID overlap: {rep['id_overlap_train_eval']}")

    tokenizer = build_tokenizer(cfg.model_id)

    tag = "TOKEN_TYPED" if mode == "token_typed" else "ENTITY_EXACT"
    out_base = os.path.join(out_root, f"{cfg.name.lower()}_{tag}")
    os.makedirs(out_base, exist_ok=True)

    folds = build_folds_doclevel(len(train_docs), cfg.n_folds, seed)
    fold_rows = []
    for fi, test_idx in enumerate(folds, start=1):
        test_set = set(test_idx)
        train_fold = [d for i, d in enumerate(train_docs) if i not in test_set]
        test_fold = [d for i, d in enumerate(train_docs) if i in test_set]

        fold_dir = os.path.join(out_base, f"fold_{fi}")
        print(f"\n[{cfg.name}][{tag}][fold {fi}] "
              f"train={len(train_fold)} test={len(test_fold)}")
        res = train_one_fold_nested(cfg, tokenizer, train_fold, test_fold,
                                    fold_dir, seed + fi, mode=mode)
        res["fold"] = fi
        fold_rows.append(res)
        o = res["test_main"]["overall"]
        print(f"[{cfg.name}][{tag}][fold {fi}] one-shot test "
              f"F1={o['f1']:.4f} P={o['precision']:.4f} R={o['recall']:.4f}")

    final_dir = os.path.join(out_base, "final_locked_eval")
    final = train_final_locked(cfg, tokenizer, train_docs, eval_docs,
                               final_dir, seed, mode=mode)
    eo = final["eval_main"]["overall"]
    print(f"\n[{cfg.name}][{tag}][FINAL locked eval] "
          f"F1={eo['f1']:.4f} P={eo['precision']:.4f} R={eo['recall']:.4f}")

    metric_def = (
        "token-level typed micro-F1 (BIO collapsed; wrong type counts as FP+FN)"
        if mode == "token_typed" else
        "STRICT entity-level exact-match micro-F1 (typed; span boundaries must match exactly)"
    )

    payload = {
        "schema_version": "i2b2_2006_papergrade_v2",
        "metric_definition": metric_def,
        "seed": seed,
        "paths": {"train_json": train_json, "eval_json": eval_json},
        "repair_report": rep,
        "model_cfg": asdict(cfg),
        "cv_folds_nested": fold_rows,
        "final_locked_eval": final,
        "lock_eval_policy": (
            "Nested CV selection on inner-val only; eval is one-shot and never "
            "used for checkpoint selection."
        ),
    }

    out_json = os.path.join(out_base, f"{cfg.name.lower()}_{tag.lower()}_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Saved results JSON:", out_json)
    return out_json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY),
                    help="which encoder configuration to run")
    ap.add_argument("--mode", default="entity_exact",
                    choices=["entity_exact", "token_typed"],
                    help="scoring mode (default: entity_exact)")
    ap.add_argument("--train", required=True, help="path to train.json")
    ap.add_argument("--eval", dest="eval_json", required=True, help="path to eval.json")
    ap.add_argument("--out", default="outputs", help="output root directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run(REGISTRY[args.model], args.mode, args.seed,
        args.train, args.eval_json, args.out)


if __name__ == "__main__":
    main()
