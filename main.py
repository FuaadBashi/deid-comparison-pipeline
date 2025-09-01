#!/usr/bin/env python3
"""
Main runner: trains/tests one or all models with the standardised pipeline.
Usage:
  python main.py --model all
  python main.py --model roberta
  python main.py --model clinicalbert
  python main.py --model biobert
"""

import argparse
import os

from training_core import run_model, load_bio_json
from model_roberta import MODEL_CFG as ROBERTA_CFG
from model_clinicalbert import MODEL_CFG as CLINICALBERT_CFG
from model_biobert import MODEL_CFG as BIOBERT_CFG

MODEL_MAP = {
    "roberta": ROBERTA_CFG,
    "clinicalbert": CLINICALBERT_CFG,
    "biobert": BIOBERT_CFG,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["all", "roberta", "clinicalbert", "biobert"], default="all")
    ap.add_argument("--train", default="train.json", help="Path to training JSON (BIO format)")
    ap.add_argument("--test", default="test.json", help="Optional test JSON (BIO format)")
    args = ap.parse_args()

    if not os.path.exists(args.train):
        raise FileNotFoundError(f"{args.train} not found.")

    train_raw = load_bio_json(args.train, "train")
    test_raw = load_bio_json(args.test, "test") if os.path.exists(args.test) else []

    if args.model == "all":
        for key in ["roberta", "clinicalbert", "biobert"]:
            cfg = MODEL_MAP[key]
            run_model(cfg, train_raw, test_raw)
    else:
        cfg = MODEL_MAP[args.model]
        run_model(cfg, train_raw, test_raw)

if __name__ == "__main__":
    print("CUDA available:", __import__("torch").cuda.is_available())
    if __import__("torch").cuda.is_available():
        print("Device:", __import__("torch").cuda.get_device_name())
    main()
