
import json
import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# === Helper: Read your json files ===
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === 1. Load Data ===
train_data = read_json("../data/train.json")
test_groudtruth_data = read_json("../data/test_groudtruth.json")

# === 2. Extract unique labels and map to IDs ===
labels = sorted({lab.replace('B-', '').replace('I-', '') for doc in train_data + test_groudtruth_data for lab in doc['labels'] if lab != 'O'})
# For IOB2 labels
unique_labels = ['O'] + [prefix + label for label in labels for prefix in ['B-', 'I-']]
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

# === 3. Tokenizer and Align Labels ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding='max_length', max_length=256)
    labels = []
    word_ids = tokenized.word_ids()
    prev_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != prev_word_idx:
            label_ids.append(label2id[example["labels"][word_idx]])
        else:
            # For subwords inside a token, use I- prefix (or -100 if you want to ignore)
            label = example["labels"][word_idx]
            if label.startswith("B-"):
                label = "I-" + label[2:]
            label_ids.append(label2id[label])
        prev_word_idx = word_idx
    tokenized["labels"] = label_ids
    return tokenized

# Convert data to HuggingFace Dataset
ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test": Dataset.from_list(test_groudtruth_data),
})
ds = ds.map(tokenize_and_align, batched=False)

# === 4. Model ===
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# === 5. TrainingArguments and Trainer ===
training_args = TrainingArguments(
    output_dir="./clinicalbert_ner",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(-1)
    # Remove ignored index (special tokens)
    true_labels = [
        [id2label[l] for l, m in zip(label, mask) if m != -100]
        for label, mask in zip(labels, labels)
    ]
    true_preds = [
        [id2label[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    return {"seqeval": classification_report(true_labels, true_preds, output_dict=True)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# === 6. Train and Evaluate ===
if __name__ == "__main__":
    trainer.train()
    results = trainer.evaluate()
    print(results)
    # Classification report
    predictions, labels, _ = trainer.predict(ds["test"])
    preds = predictions.argmax(-1)
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    print(classification_report(true_labels, true_preds, digits=4))
