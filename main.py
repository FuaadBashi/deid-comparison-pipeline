import json
from models.clinicalbert_finetune import train_clinicalbert, predict_clinicalbert

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # --- 1. Load Data ---
    train_data = load_json("data/train.json")
    groundtruth_data = load_json("data/test_groundtruth.json")
    input_test_data = load_json("data/test.json")  # your "wild" test set

    # --- 2. Get all unique labels from your datasets ---
    all_labels = set()
    for d in train_data + groundtruth_data:
        all_labels.update(d["labels"])
    label_list = sorted(list(all_labels))

    print(set(label_list))  # Debug: See all labels

    # --- 3. Train ClinicalBERT ---
    print("\n== [CLINICALBERT Training] ==")
    trainer, tokenizer, label_to_id, id_to_label = train_clinicalbert(
        train_data, groundtruth_data, label_list
    )

    # --- 4. Predict on Unlabeled Test ---
    print("\n== [CLINICALBERT Predict: Unlabeled Test] ==")
    input_tokens = [d["tokens"] for d in input_test_data]
    pred_results = predict_clinicalbert(trainer, tokenizer, input_tokens, label_to_id, id_to_label)

    # --- 5. Save predictions to file ---
    with open("outputs/clinicalbert_test_preds.json", "w", encoding="utf-8") as f:
        json.dump(pred_results, f, indent=2, ensure_ascii=False)
        
    print("done!")

if __name__ == "__main__":
    main()
