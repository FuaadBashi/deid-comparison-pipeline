# PHI De-identification — Standardised Cross-Validation

This repository trains **three token-classification models** for PHI de-identification using a **fully standardised** pipeline:

- **RoBERTa-Large** (12 epochs)
- **ClinicalBERT** (8 epochs)
- **BioBERT** (8 epochs)

All other training settings are identical across models: 5-fold CV, training-only oversampling, inverse-frequency class weights, AdamW with the same hyperparameters, early stopping on macro-F1, and entity-level exact-match scoring (seqeval).

---

## Requirements

- Python 3.9+ (3.11 works)
- CUDA (optional, for GPU training)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA/CPU wheel
pip install transformers==4.41.1 seqeval==1.2.2 scikit-learn==1.4.2 numpy==1.26.4
