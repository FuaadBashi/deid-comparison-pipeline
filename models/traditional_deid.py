import spacy

nlp = spacy.load("en_core_web_sm")

# Mapping entity labels to PHI token replacements
ENTITY_LABELS = {
    "PERSON": "[NAME]",
    "ORG": "[ORG]",
    "GPE": "[LOCATION]",
    "DATE": "[DATE]",
    "EMAIL": "[EMAIL]",
}

def rule_based_deid(text):
    doc = nlp(text)
    redacted_text = text
    offset = 0
    annotations = []

    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            replacement = ENTITY_LABELS[ent.label_]
            start = ent.start_char + offset
            end = ent.end_char + offset
            annotations.append((ent.text, ent.label_))
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            offset += len(replacement) - (end - start)

    return redacted_text, annotations
