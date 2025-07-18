# De-ID Comparison Pipeline

This project compares rule-based and transformer-based methods for PHI (Protected Health Information) de-identification in clinical text.

## Structure
- `main.py`: Entry point
- `traditional_deid.py`: Regex and dictionary-based system
- `transformer_deid.py`: BERT/RoBERTa de-identification pipeline
- `utils/`: Preprocessing, evaluation scripts

## Goal
Compare performance across PHI categories (name, date, org, etc.) on annotated and unannotated clinical notes.

