"""Model configurations used in the published comparison."""
from .training import ModelCfg

SEED = 42

clinical_cfg = ModelCfg(
    name="ClinicalBERT",
    model_id="emilyalsentzer/Bio_ClinicalBERT",
    max_len=512, stride=256,
    batch_size=4, grad_accum=2, lr=3e-5,
    epochs_cv=8, epochs_final=10,
    window_boosts={"LOCATION": 1.3, "PHONE": 1.0},
    use_class_weights=True,
    class_weight_multipliers={"LOCATION": 1.5, "PHONE": 1.1},
)

biobert_cfg = ModelCfg(
    name="BioBERT",
    model_id="dmis-lab/biobert-base-cased-v1.1",
    max_len=512, stride=256,
    batch_size=4, grad_accum=2, lr=3e-5,
    epochs_cv=8, epochs_final=10,
    window_boosts={"LOCATION": 1.3, "PHONE": 1.0},
    use_class_weights=True,
    class_weight_multipliers={"LOCATION": 1.5, "PHONE": 1.1},
)

roberta_cfg = ModelCfg(
    name="RoBERTaLarge",
    model_id="roberta-large",
    max_len=512, stride=256,
    batch_size=2, grad_accum=4, lr=2e-5,
    epochs_cv=8, epochs_final=10,
    window_boosts={"LOCATION": 1.3, "PHONE": 1.0},
    use_class_weights=True,
)
