# Camera-ready v5 aggregate results

These files contain sanitized aggregate outputs from the camera-ready replication workflow in [`notebooks/camera_ready_replication_v5.ipynb`](../../notebooks/camera_ready_replication_v5.ipynb).

## Files

- [`comparison_summary.json`](comparison_summary.json): compact cross-model comparison for five-fold out-of-fold validation and the locked final test.
- [`biobert/aggregate_results.json`](biobert/aggregate_results.json): BioBERT configuration, audit counts, fold-level aggregates, and per-entity metrics.
- [`clinicalbert/aggregate_results.json`](clinicalbert/aggregate_results.json): ClinicalBERT configuration, audit counts, fold-level aggregates, and per-entity metrics.
- [`roberta_large/aggregate_results.json`](roberta_large/aggregate_results.json): RoBERTa-Large configuration, audit counts, fold-level aggregates, and per-entity metrics.

## Evaluation boundaries

- Cross-validation used five document-level folds. Checkpoint selection used only an inner validation split drawn from the outer-training documents.
- The final 220-record test was locked and evaluated once after blind predictions had been generated.
- Strict entity-exact and token-typed metrics were calculated from the same predictions.
- The seed was 42 and train/test record-ID overlap was zero.

## Privacy and reproducibility

Only aggregate metrics and non-identifying audit counts are published. The source result bundles also contain record-level predictions, fold assignments, document identifiers, local paths, archives, and trained model artefacts; those files are deliberately excluded because the underlying clinical corpus is restricted by its data-use agreement.

The JSON files retain enough information to audit the reported headline, macro, per-entity, and fold-level values without redistributing clinical content.
