import json
from seqeval.metrics import classification_report, f1_score

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def align_labels(pred, gold, max_print_tokens=200, verbose=True, max_mismatches=5):
    y_true = []
    y_pred = []
    mismatch_count = 0
    for i, (p, g) in enumerate(zip(pred, gold)):
        # Check both tokens AND label lengths
        if p['tokens'] != g['tokens'] or \
           len(p['predicted_labels']) != len(g['labels']):
            mismatch_count += 1
            if verbose and mismatch_count <= max_mismatches:
                print(f"\n[!] Misalignment at index {i}")
                print("Predicted tokens (first 200):", p['tokens'][:max_print_tokens])
                print("Gold tokens (first 200):    ", g['tokens'][:max_print_tokens])
                print("Pred labels (first 200):", p.get('predicted_labels', [])[:max_print_tokens])
                print("Gold labels (first 200): ", g.get('labels', [])[:max_print_tokens])
                print("Pred text:", " ".join(p['tokens'][:max_print_tokens]))
                print("Gold text:", " ".join(g['tokens'][:max_print_tokens]))
                print(f"Pred label len: {len(p['predicted_labels'])}, Gold label len: {len(g['labels'])}")
            continue  # Skip this record
        y_true.append(g['labels'])
        y_pred.append(p['predicted_labels'])
    print(f"\nTotal mismatches skipped: {mismatch_count}")
    print(f"Aligned records used for evaluation: {len(y_true)}")
    return y_true, y_pred


def main():
    pred = load_json("outputs/clinicalbert_test_preds.json")
    gold = load_json("data/test_groundtruth.json")  # Or your actual ground truth path

    y_true, y_pred = align_labels(pred, gold)
    print("=== EVALUATION ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("Micro F1:", f1_score(y_true, y_pred))

if __name__ == "__main__":
    main()
