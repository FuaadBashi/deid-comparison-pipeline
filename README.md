# Clinical PHI De-Identification — Multi-Encoder Comparison

Fine-tunes and benchmarks three transformer encoders — **RoBERTa-Large**, **ClinicalBERT**, and **BioBERT** — for token-level Protected Health Information (PHI) detection in clinical notes, under an evaluation protocol built to resist the failure modes that inflate NER numbers in published work.

The interesting result is not that a big model wins. It is **how much** it wins by, and how much of the reported gap between clinical NER systems disappears once you fix the metric definition and stop selecting checkpoints on your own test set.

---

## Why this repo exists

Most clinical NER numbers are not comparable, for three reasons:

1. **The metric is unstated.** "F1 = 0.96" means very different things under token-level scoring and strict entity-level exact-match scoring. This repo computes both, from the same predictions, and labels which is which.
2. **The test set leaks into model selection.** Early stopping on the test set is the most common silent form of this. Here, checkpoint selection happens on an inner validation split; the evaluation set is scored exactly once, with `load_best_model_at_end=False`.
3. **Gold labels are quietly repaired.** Malformed BIO sequences get fixed, which changes the target. This repo does repair them (models can't learn from `I-` tags with no antecedent) but reports scores against **both** repaired and raw gold, so the effect size is visible.

### How much the metric definition matters

From `tests/test_metrics.py` — one prediction, two metrics:

| Gold | Predicted | entity-exact F1 | token-typed F1 |
|---|---|---|---|
| `B-NAME I-NAME` + `B-DATE` | `B-NAME O` + `B-DATE` | **0.50** | **0.80** |

Predicting "John" where the gold span is "John Smith" earns zero credit under strict scoring and costs both a false positive and a false negative. Under token-level scoring it costs one false negative and keeps full precision. A 30-point swing, same model, same output. This is why every number below carries its metric definition.

---

## Evaluation protocol

```
                     ┌─ inner train ──► fine-tune
  train.json ──► 5-fold nested CV ─┤
                     └─ inner val ───► early stopping + checkpoint selection
                                              │
                     outer test fold ◄────────┘  scored once, no selection

  eval.json ───────────────────────────────► scored EXACTLY ONCE
                                              (load_best_model_at_end=False)
```

- **Document-level splits.** Folds are built over documents, never windows, so no sentence from a note appears in both train and test. Train/eval document-ID overlap is asserted to be zero at load time and the run aborts if it isn't.
- **Sliding-window inference with logit averaging.** Long notes are chunked into overlapping windows; window logits are summed back onto absolute word positions and averaged before argmax, so a token near a window edge is not decided by one truncated view.
- **Label schema inferred from train only.** Entity types are read off the training tags; unseen types in eval are never silently added to the label set.
- **Class imbalance handled explicitly**, not incidentally — `WeightedRandomSampler` over windows containing rare entities, plus optional inverse-frequency class weights in the loss, both configurable per entity type.

---

## Results

> **Provenance note — read this before quoting any number.**
> The primary metrics below are from the **camera-ready v5 replication**: seeded document-level five-fold cross-validation, inner-validation-only checkpoint selection, and one locked 220-record test evaluated once. Strict entity-exact and token-typed metrics are computed from the same predictions. The earlier v1 experiment remains under [`results/v1/`](results/v1/) for historical provenance, but should not be compared directly with v5.

### Locked final test (220 records)

| Model | Strict P | Strict R | Strict F1 | Strict macro-F1 | Token F1 |
|---|---:|---:|---:|---:|---:|
| **BioBERT** | **0.9738** | **0.9796** | **0.9767** | **0.9503** | 0.9877 |
| RoBERTa-Large | 0.9609 | 0.9754 | 0.9681 | 0.9446 | **0.9920** |
| ClinicalBERT | 0.9557 | 0.9680 | 0.9618 | 0.9012 | 0.9834 |

BioBERT has the strongest strict entity-exact result on the locked test, while RoBERTa-Large has the strongest token-typed result. This difference is exactly why the repository reports both metrics: partially correct boundaries can look strong at token level while still failing exact-span de-identification.

### Five-fold out-of-fold validation

| Model | Strict F1 | Strict macro-F1 | Token F1 | Risk-weighted F1 |
|---|---:|---:|---:|---:|
| **RoBERTa-Large** | **0.9893** | **0.9679** | **0.9935** | **0.9892** |
| BioBERT | 0.9815 | 0.9381 | 0.9885 | 0.9809 |
| ClinicalBERT | 0.9763 | 0.9155 | 0.9870 | 0.9763 |

The validation ranking does not fully carry over to the locked test. That is useful evidence against selecting a system from one headline score alone: model choice should account for metric definition, per-entity failures, and behaviour on genuinely unseen data.

**Camera-ready aggregate artefacts:** [`results/v5/`](results/v5/) contains the cross-model summary and sanitized per-model aggregates, including per-entity and per-fold metrics. Record-level predictions, document identifiers, clinical text, checkpoints, and archives are intentionally excluded.

---

## Repo layout

```
deid-comparison-pipeline/
├── run_pipeline.py           # CLI entry point
├── src/deid/
│   ├── labels.py             # label schema inference, BIO repair, seeding
│   ├── data.py               # corpus loading, leakage checks, sliding-window dataset
│   ├── metrics.py            # token-typed + entity-exact scoring, logit merging
│   ├── training.py           # weighted trainer, nested CV, locked final eval
│   └── configs.py            # the three published model configurations
├── tests/test_metrics.py     # 12 tests over BIO repair, span extraction, scoring
├── results/
│   ├── v1/                   # historical v1 evaluation JSON
│   └── v5/                   # sanitized camera-ready aggregate results
├── notebooks/
│   ├── deid_pipeline_v2.ipynb             # original modular v2 notebook
│   └── camera_ready_replication_v5.ipynb  # crash-safe camera-ready replication workflow
```

## Usage

```bash
pip install -r requirements.txt

python run_pipeline.py \
    --model roberta \
    --mode entity_exact \
    --train data/train.json \
    --eval  data/eval.json \
    --out   outputs/
```

`--model` is one of `roberta`, `clinicalbert`, `biobert`. `--mode` is `entity_exact` (strict, default) or `token_typed`.

Run the tests:

```bash
PYTHONPATH=src pytest tests/ -v
```

### Camera-ready Colab workflow

[`notebooks/camera_ready_replication_v5.ipynb`](notebooks/camera_ready_replication_v5.ipynb)
contains the complete ICACIn camera-ready replication workflow. It keeps the
same frozen six-category protocol across RoBERTa-Large, ClinicalBERT, and
BioBERT, while adding:

- a pre-training audit of record counts, document identifiers, labels, and
  train/test separation;
- five-fold document-level cross-validation with inner validation used only
  for checkpoint selection;
- label-free blind inference before the gold test file is opened;
- token-typed and strict entity-exact metrics from the same out-of-fold
  predictions;
- atomic result writes, per-fold checkpoint bundles, Google Drive backup, and
  downloadable per-model and all-model result archives; and
- automatic generation of the camera-ready comparison tables.

The notebook expects the authorised XML corpora at the Colab paths declared in
its first code cell. No clinical data, model checkpoints, credentials, or
executed notebook outputs are stored in this repository.

### Input format

A JSON array of pre-tokenised documents with BIO tags:

```json
[
  {
    "record_id": "doc_001",
    "tokens":  ["Patient", "John", "Smith", "seen", "on", "3/14"],
    "labels":  ["O", "B-NAME", "I-NAME", "O", "O", "B-DATE"]
  }
]
```

Entity types are inferred from the training file, so the schema is not hardcoded — any BIO-tagged corpus works.

## Data

Developed against i2b2/n2c2-style de-identification corpora. **No clinical data is included in this repository.** i2b2 data requires a signed data use agreement and cannot be redistributed; obtain it via the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/).

## Related publication

Findings from this work informed a co-authored conference paper on medical NER.

> **[PLACEHOLDER — replace before making the repo public]**
> Authors, "Title", *Venue*, Year. DOI / arXiv link.
> If not yet published, state exactly one of: `Under Review at <venue>` or `Preprint`, and link the manuscript.

## Limitations

- Research pipeline, not a validated clinical de-identification tool. PHI recall below 100% means residual re-identification risk; do not deploy against real patient data without independent validation and a human review layer.
- v1 results are single-run on one evaluation set. No confidence intervals, no seed variance — the v2 nested-CV run exists to supply both.
- `LOCATION` and `PHONE` results rest on 20–30 gold spans each. Treat as directional.

## Licence

MIT — see [LICENSE](LICENSE).
