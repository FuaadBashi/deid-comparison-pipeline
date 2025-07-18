from models.traditional_deid import rule_based_deid
from models.transformer_deid import transformer_deid
from utils.evaluation import evaluate_predictions

def main():
    text = """Patient John Smith was admitted on 12/02/2023 to Johns Hopkins Hospital.
    His contact is (123) 456-7890. Diagnosed with Type 2 Diabetes."""

    print("Original Text:")
    print(text)

    print("\n[Rule-Based De-Identification]")
    rule_output, rule_tags = rule_based_deid(text)
    print(rule_output)

    print("\n[Transformer-Based De-Identification]")
    transformer_output, transformer_tags = transformer_deid(text)
    print(transformer_output)

    gold_tags = [
        ("John", "B-PERSON"), ("Smith", "I-PERSON"),
        ("12/02/2023", "B-DATE"),
        ("Johns", "B-ORG"), ("Hopkins", "I-ORG"), ("Hospital", "I-ORG"),
        ("(123)", "B-CONTACT"), ("456-7890", "I-CONTACT")
    ]

    print("\n[Evaluation - Rule-Based vs. Gold]")
    evaluate_predictions(gold_tags, rule_tags)

    print("\n[Evaluation - Transformer vs. Gold]")
    evaluate_predictions(gold_tags, transformer_tags)

if __name__ == "__main__":
    main()
