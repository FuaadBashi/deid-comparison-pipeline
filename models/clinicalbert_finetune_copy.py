from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification
from datasets import Dataset
import numpy as np

# --- EXAMPLE: Replace with your real data loading ---
# Each "tokens" is a list of word strings
# Each "labels" is a list of integer IDs (0, 1, 2, ...) matching the IOB2 tagset for each token
train_tokens = [["Mr.", "John", "Smith", "was", "admitted", "to", "St.", "Luke", "Medical", "Center", "on", "03/15/2005", "."]]
train_labels = [[1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0]]  # e.g., 1="B-NAME", 2="I-NAME", 3="B-DATE", 0="O"
test_tokens = [["Patient", "ID", "12345", "visited", "Dr.", "Grey", "on", "07/12/2020", "."]]
test_labels = [[0, 0, 4, 0, 0, 1, 0, 3, 0]]  # 4="B-ID", etc.

# Build the Dataset
train_dataset = Dataset.from_dict({"tokens": train_tokens, "labels": train_labels})
test_dataset = Dataset.from_dict({"tokens": test_tokens, "labels": test_labels})

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

label_list = ["O", "B-NAME", "I-NAME", "B-DATE", "B-ID"]  # add all label types you use!
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Mask for ignored tokens
            elif word_idx != prev_word_idx:
                label_ids.append(label[word_idx])
            else:
                # For subwords, we mark them as I-label if needed (depends on your scheme)
                label_ids.append(label[word_idx])
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization to dataset
train_tokenized = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized = test_dataset.map(tokenize_and_align_labels, batched=True)

# Print a sample to verify (important!):
print(train_tokenized[0])

# --- Model ---
model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(label_list))

# --- TrainingArguments (old HF, use eval_strategy) ---
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


# --- Trainer ---
from seqeval.metrics import classification_report, f1_score

def compute_metrics(p):
    # Get predictions and gold labels, ignoring -100
    predictions = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids
    pred_tags = [[id_to_label[p_] for (p_, l_) in zip(prediction, label) if l_ != -100]
                 for prediction, label in zip(predictions, true_labels)]
    true_tags = [[id_to_label[l_] for (p_, l_) in zip(prediction, label) if l_ != -100]
                 for prediction, label in zip(predictions, true_labels)]
    return {"f1": f1_score(true_tags, pred_tags), "report": classification_report(true_tags, pred_tags)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Number of train examples:", len(train_tokenized))
print("First train example:", train_tokenized[0])
print("Number of test examples:", len(test_tokenized))
print("First test example:", test_tokenized[0])
# --- Train the model ---
trainer.train()
