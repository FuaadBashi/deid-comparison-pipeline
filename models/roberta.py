#!/usr/bin/env python3
from training_core import ModelCfg, run_model, load_bio_json

MODEL_CFG = ModelCfg(
    name="roberta_large",
    model_id="roberta-large",
    epochs=12,
    tok_kwargs={"add_prefix_space": True},
)

def main():
    train_raw = load_bio_json("train.json", "train")
    test_raw = load_bio_json("test.json", "test") if __import__("os").path.exists("test.json") else []
    run_model(MODEL_CFG, train_raw, test_raw)

if __name__ == "__main__":
    main()
