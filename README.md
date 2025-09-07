# PHI De-identification — Standardised Cross-Validation

This repository trains and evaluates **three token-classification models** for Protected Health Information (PHI) de-identification using a **single, standardised pipeline** that follows the thesis methodology. The **only difference** between models is the number of training epochs:

- **RoBERTa-Large** — 5 epochs  
- **ClinicalBERT** — 5 epochs  
- **BioBERT** — 5 epochs

Everything else (preprocessing, cross-validation, oversampling, class weighting, optimiser, scheduler, metrics) is identical across models to ensure a fair comparison.

---

## Contents

- [Features](#features)
- [Environment & Installation](#environment--installation)
- [Data Format](#data-format)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Methodology (Standardised)](#methodology-standardised)
- [Configuration Details](#configuration-details)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Publishing to GitHub](#publishing-to-github)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

---

## Features

- **5-fold stratified cross-validation** (stratification by rare-entity presence)
- **Training-only oversampling** (LOCATION×6, PHONE×3)
- **Inverse-frequency class weights** with clamping \[0.3, 10.0]
- **Token classification** with entity-level exact-match scoring (seqeval)
- **AdamW** optimiser, fixed LR and schedule across all models
- **Early stopping** on dev macro-F1 (patience=3)
- **FP16** training when CUDA is available
- **Single script** runs all three models end-to-end

---

## Environment & Installation

Tested with Python 3.9–3.11.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Choose the correct PyTorch wheel for your platform/CUDA:
# See https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # example for CUDA 12.1

pip install transformers==4.41.1 seqeval==1.2.2 scikit-learn==1.4.2 numpy==1.26.4
CPU-only users should install the CPU wheel from pytorch.org.

Data Format
Place train.json (and optionally test.json) in the project root. Each file must be a JSON list of examples with aligned tokens and labels.

json
Copy code
[
  {
    "tokens": ["Patient", "John", "Smith", "visited", "on", "03/21/2020", "."],
    "labels": ["O", "B-NAME", "I-NAME", "O", "O", "B-DATE", "O"]
  }
]
Supported labels (BIO):

css
Copy code
O
B-NAME   I-NAME
B-DATE   I-DATE
B-ID     I-ID
B-LOCATION   I-LOCATION
B-PHONE  I-PHONE
B-HOSPITAL   I-HOSPITAL
Requirement: len(tokens) == len(labels) for every example.

How to Run
Run the entire standardised pipeline (all three models):

bash
Copy code
python standardised_phi_cv.py
If test.json is present:

With gold labels → writes test_eval.json (metrics)

Without gold labels → writes test_predictions.json (token-level predictions)

Outputs
Default output root: standardised_runs/

python
Copy code
standardised_runs/
  roberta_large/
    fold1/
      config.json
      dev_eval.json          # per-fold validation metrics + seqeval report
      logs/
      pytorch_model.bin
      tokenizer.json
      ...
    fold2/ ... fold5/
    cv_summary.json          # mean CV metrics across folds
    test_eval.json           # (if test.json with labels)
    test_predictions.json    # (if test.json without labels)
  clinicalbert/
    ...
  biobert/
    ...
Key files:

dev_eval.json: precision, recall, micro-F1, macro-F1, and the full seqeval classification report for the dev fold.

cv_summary.json: average metrics across the 5 folds for the model.

test_eval.json / test_predictions.json: test results where applicable.

Methodology (Standardised)
The pipeline follows the thesis methodology with the following standardisations:

Cross-validation: 5-fold StratifiedKFold (strata by the presence of LOCATION and PHONE) to ensure rare entities are more evenly distributed across folds.

Oversampling: Training-only duplication of examples that contain rare entities:

LOCATION ×6

PHONE ×3

Class weighting: Inverse frequency weights computed from the training portion (before oversampling), clamped to [0.3, 10.0].

Subword labelling: Only the first subword per word is labelled; subsequent subwords are set to -100 for loss/metrics.

Optimiser & schedule: AdamW (LR=2e-5, weight_decay=0.01), linear scheduler with warmup ratio 0.1.

Batching: Per-device batch size 16 with gradient accumulation 2 → effective batch size 32.

Sequence length: Max length = 256.

Early stopping: Patience 3 on dev macro-F1.

Evaluation: seqeval entity-level exact match; we report macro-F1, micro-F1, precision, recall.

Tokeniser specifics: RoBERTa uses add_prefix_space=True. Others use defaults.

Epochs: The only model-specific variation:

RoBERTa-Large: 12

ClinicalBERT: 18

BioBERT: 18

Configuration Details
Setting	Value / Notes
Folds	5 (StratifiedKFold)
Max sequence length	256
Per-device batch size	16
Gradient accumulation	2 (effective batch 32)
Optimiser	AdamW
Learning rate	2e-5
Weight decay	0.01
Warmup	ratio = 0.1
Scheduler	Linear
Early stopping	patience = 3 on macro-F1
Oversampling	LOCATION×6, PHONE×3 (training only)
Class weights	Inverse frequency, clamped [0.3, 10.0]
Subword labels	First subword gets label, others -100
Tokeniser (RoBERTa)	add_prefix_space=True
Metrics	seqeval precision, recall, macro-F1, micro-F1
Output root	standardised_runs/

Reproducibility
Fixed seed (42) for Python, NumPy, Torch; CuDNN is set deterministic.

Identical hyperparameters across models (except epochs).

Standardised preprocessing, stratification, oversampling, class weights, and evaluation.

Subword alignment is consistent across tokenisers.

Hardware variability (e.g., different GPUs) can still introduce minor non-determinism despite deterministic settings.

Repository Structure
bash
Copy code
.
├── standardised_phi_cv.py     # main training/eval script (all three models)
├── README.md                  # this file
├── train.json                 # your training data (not committed if private)
├── test.json                  # optional test data (not committed if private)
└── standardised_runs/         # generated outputs
Suggested .gitignore:

gitignore
Copy code
__pycache__/
*.pyc
.venv/
.env
.DS_Store
Thumbs.db

standardised_runs/
logs/
*.bin
*.safetensors
Publishing to GitHub
Create an empty repository on GitHub (no README/license), e.g. phi-deid-standardised.

In your local project directory:

bash
Copy code
git init
git add standardised_phi_cv.py README.md
# If data is public (usually it is not), you may add train.json/test.json
git add .gitignore
git commit -m "Initial commit: standardised PHI CV pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/phi-deid-standardised.git
git push -u origin main
For private datasets, do not commit train.json / test.json. Distribute them separately.

Troubleshooting
FileNotFoundError: train.json not found.
Place train.json in the project root and ensure it’s valid JSON with aligned tokens and labels.

CUDA OOM (out of memory).
Lower per-device batch size (e.g., 8) or reduce MAX_LEN. You can also disable FP16 by forcing CPU or using a smaller model.

Tokeniser warnings about add_prefix_space.
RoBERTa requires add_prefix_space=True (already handled).

Mismatch or invalid labels.
The script sanitises common issues (e.g., B-O, unknown tags). Ensure you use the label set listed above.

Seqeval warnings about scheme.
Ensure your BIO tags are valid and aligned to tokens (one tag per token).

