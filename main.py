from models.traditional_deid import rule_based_deid
from models.transformer_deid import transformer_deid
from utils.evaluation import evaluate_predictions
import numpy

def main():
    sample_text = "Patient John Smith visited Johns Hopkins Hospital on April 3rd, 2023."
    redacted, entities = rule_based_deid(sample_text)

    print("Redacted Text:\n", redacted)
    print("Detected Entities:\n", entities)


if __name__ == "__main__":
    main()
