import csv
import json
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

input_file = "deid_test_groudtruth_iob2.csv"
output_file = "deid_test_groudtruth_hf.json"

all_docs = []
tokens, labels = [], []

with open(input_file, encoding="utf-8") as f:
    reader = csv.reader(f, quotechar='"')
    for row in reader:
        if not row or all(r.strip() == '' for r in row):
            if tokens:
                all_docs.append({"tokens": tokens, "labels": labels})
                tokens, labels = [], []
        else:
            if len(row) > 2:
                tok = ",".join(row[:-1]).strip()
                lab = row[-1].strip()
            elif len(row) == 2:
                tok, lab = row
                tok = tok.strip()
                lab = lab.strip()
            else:
                continue  # skip malformed lines
            tokens.append(tok)
            labels.append(lab)
    if tokens:
        all_docs.append({"tokens": tokens, "labels": labels})

print(f"Processed {len(all_docs)} documents.")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print(f"Wrote Hugging Face JSON to: {output_file}")
