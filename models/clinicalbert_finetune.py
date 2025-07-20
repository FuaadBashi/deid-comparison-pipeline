from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

def ensure_labels_are_ids(data, label_to_id):
    """Convert string labels to integer ids (in-place)."""
    for rec in data:
        rec["labels"] = [label_to_id[lbl] if isinstance(lbl, str) else lbl for lbl in rec["labels"]]
    return data

def train_clinicalbert(train_data, eval_data, label_list):
    # 1. Build label dicts
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    # 2. Convert labels to ids
    train_data = ensure_labels_are_ids(train_data, label_to_id)
    eval_data = ensure_labels_are_ids(eval_data, label_to_id)

    # 3. Build HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        "tokens": [d["tokens"] for d in train_data],
        "labels": [d["labels"] for d in train_data]
    })
    eval_dataset = Dataset.from_dict({
        "tokens": [d["tokens"] for d in eval_data],
        "labels": [d["labels"] for d in eval_data]
    })

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512
        )
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx])
                prev_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_and_align_labels, batched=True)

    # 5. Model
    model = AutoModelForTokenClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=len(label_list)
    )

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir="./outputs/clinicalbert",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./outputs/logs",
        report_to="none"
    )

    # 7. Metrics
    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=2)
        true_labels = p.label_ids
        pred_tags = [
            [id_to_label[p_] for (p_, l_) in zip(prediction, label) if l_ != -100]
            for prediction, label in zip(predictions, true_labels)
        ]
        true_tags = [
            [id_to_label[l_] for (p_, l_) in zip(prediction, label) if l_ != -100]
            for prediction, label in zip(predictions, true_labels)
        ]
        return {
            "f1": f1_score(true_tags, pred_tags),
            "report": classification_report(true_tags, pred_tags)
        }

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 9. Train!
    trainer.train()
    return trainer, tokenizer, label_to_id, id_to_label

def predict_clinicalbert(trainer, tokenizer, input_tokens, label_to_id, id_to_label, max_length=128):
    from datasets import Dataset

    # Fake labels needed for HuggingFace Dataset compatibility (not used)
    fake_labels = [[0] * len(seq) for seq in input_tokens]

    def tokenize_pred(examples):
        return tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    pred_dataset = Dataset.from_dict({"tokens": input_tokens, "labels": fake_labels})
    pred_dataset = pred_dataset.map(tokenize_pred, batched=True)
    pred_dataset = pred_dataset.remove_columns("labels")  # not needed

    preds = trainer.predict(pred_dataset)
    pred_ids = np.argmax(preds.predictions, axis=2)
    results = []
    for i, record in enumerate(input_tokens):
        word_ids = pred_dataset[i]["input_ids"]  # For reference (not used)
        tokens = record
        pred_labels = []
        pred_for_record = pred_ids[i]
        # Only keep predictions for original tokens (skip padding)
        non_pad_count = len(tokens)
        pred_labels = [id_to_label[pred] for pred in pred_for_record[:non_pad_count]]
        results.append({
            "tokens": tokens,
            "predicted_labels": pred_labels
        })
    return results
