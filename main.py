import json
from models.clinicalbert_finetune_copy import train_clinicalbert
# from models.biobert_finetune import train_biobert
# from models.roberta_finetune import train_roberta
from models.traditional_deid import rule_based_deid

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1. Load Data
    train_data = load_json("data/train.json")
    test_data = load_json("data/test_groundtruth.json")
    

    # 2. Train + Evaluate Models
    print("\n== [CLINICALBERT] ==")
    clinicalbert_results = train_clinicalbert(train_data, test_data)
    print("ClinicalBERT:", clinicalbert_results)

    # print("\n== [BIOBERT] ==")
    # biobert_results = train_biobert(train_data, test_data)
    # print("BioBERT:", biobert_results)

    # print("\n== [ROBERTA] ==")
    # roberta_results = train_roberta(train_data, test_data)
    # print("RoBERTa:", roberta_results)

    print("\n== [RULE-BASED] ==")
    rule_results = rule_based_deid(test_data)
    print("Rule-Based:", rule_results)

if __name__ == "__main__":
    main()


