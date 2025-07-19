import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score

import transformers
print("Transformers version:", transformers.__version__)
print("Transformers module file:", transformers.__file__)
from transformers import TrainingArguments
print("TrainingArguments:", TrainingArguments)
print("TrainingArguments doc:", TrainingArguments.__init__.__doc__)

import transformers
print("Transformers version:", transformers.__version__)
print("Transformers module file:", transformers.__file__)
from transformers import TrainingArguments
print("TrainingArguments:", TrainingArguments)


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

def preprocess_data(data, tokenizer, label2id, max_length=256):
    encodings = tokenizer(
        [x['tokens'] for x in data],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

    labels = []
    for i, x in enumerate(data):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[x['labels'][word_idx]])
            else:
                label_ids.append(label2id[x['labels'][word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    encodings['labels'] = torch.tensor(labels)
    return encodings

def train_clinicalbert(train_data, test_data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Collect all unique labels
    unique_labels = sorted(list(set(lab for x in train_data+test_data for lab in x['labels'])))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    def tokenize_fn(examples):
        return preprocess_data(examples, tokenizer, label2id)

    # Data collator will pad the data
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

    # Training arguments (tweak as needed)
    training_args = TrainingArguments(
    output_dir="./outputs/clinicalbert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    eval_strategy="epoch",  # <-- change here!
    save_strategy="epoch",
    logging_dir="./outputs/logs",
    report_to="none"
    )       

    # Trainer expects HuggingFace datasets with batchable tokenize fn
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    preds = trainer.predict(test_dataset)
    pred_labels = [[id2label[i] for i in pred if i != -100] for pred in preds.predictions.argmax(-1)]
    true_labels = [[id2label[i] for i in label if i != -100] for label in preds.label_ids]

    f1 = f1_score(true_labels, pred_labels)
    return {
        "f1": f1,
        "classification_report": classification_report(true_labels, pred_labels, digits=4)
    }
