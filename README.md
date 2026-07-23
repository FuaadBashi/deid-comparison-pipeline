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
> The metrics below come from the **v1 pipeline** (fixed 256-token windows, single held-out evaluation, no nested CV). The code in `src/` is the **v2 pipeline** (512-token sliding windows, 5-fold nested CV, locked one-shot eval, dual metrics). **They are different experiments.** The v2 re-run is pending and its results will be committed to `results/v2/` with the full per-fold JSON. Until then, `results/v1/` is what the numbers are, and it is labelled as such rather than presented as output of the published code.

**v1 results** — 220 evaluation documents, micro-averaged across the six PHI entity types, token-level typed scoring with evaluation preprocessing matched to training:

| Model | F1 | Precision | Recall |
|---|---|---|---|
| **RoBERTa-Large** + oversampling | **0.9624** | 0.9499 | 0.9752 |
| ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) | 0.8970 | 0.8559 | 0.9422 |
| BioBERT (`dmis-lab/biobert-base-cased-v1.1`) | 0.8386 | 0.7980 | 0.8836 |

The headline finding: **general-domain RoBERTa-Large beat both biomedically pretrained encoders by 6.5 and 12.4 F1 points.** In-domain pretraining did not compensate for capacity on this task. That is the opposite of the usual assumption in clinical NLP tooling, and it is the result worth defending in an interview.

### Where the models actually differ (per-entity F1, v1)

| Entity | RoBERTa-L | ClinicalBERT | BioBERT | Gold spans |
|---|---|---|---|---|
| ID | **0.996** | 0.989 | 0.933 | ~900 |
| HOSPITAL | **0.952** | 0.822 | 0.909 | ~340 |
| NAME | **0.940** | 0.833 | 0.518 | ~250 |
| DATE | **0.808** | 0.774 | 0.613 | ~90 |
| LOCATION | **0.780** | 0.537 | 0.458 | ~28 |
| PHONE | **1.000** | 0.449 | 0.455 | ~20 |

Two things this table shows that the headline F1 hides:

- **The aggregate is dominated by `ID`.** Roughly 55% of gold spans are identifiers, which every model handles well. A single micro-F1 flatters all three systems. Macro-F1 across entity types is the more honest summary for a de-identification system, where a missed `NAME` and a missed `ID` are not equivalent failures.
- **The rare classes are where the gap lives.** `LOCATION` and `PHONE` carry the fewest gold spans and show the widest spread — RoBERTa-Large reaches 1.000 on `PHONE` where both BERT variants sit near 0.45. On ~20 gold spans, that is a handful of decisions and the confidence interval is wide. Not a result to over-claim from.

Per-entity gold counts differ slightly across models because v1 scored at the token level after each model's own subword alignment, and the three tokenizers segment differently. This is a further argument for entity-level scoring, which is tokenizer-independent — and is why v2 reports it.

**Raw evaluation artefacts:** [`results/v1/`](results/v1/) — unedited JSON per model, including per-class precision/recall/support and predicted-vs-true entity counts.

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
├── results/v1/               # evaluation JSON from the v1 pipeline
├── notebooks/                # original Colab notebook (v2 pipeline, as run)
└── docs/
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
