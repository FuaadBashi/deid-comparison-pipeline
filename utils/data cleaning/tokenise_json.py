#!/usr/bin/env python3
import json, sys

def convert(input_json, output_json):
    reports = json.load(open(input_json, 'r', encoding='utf-8'))
    out = []
    for rec in reports:
        text = rec.get('phi_text', '')
        # simple whitespace tokenizer:
        toks = text.split()
        labs = ['O'] * len(toks)
        out.append({
            "tokens": toks,
            "labels": labs
        })
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("Usage: make_tok_json.py input.json output.json")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
