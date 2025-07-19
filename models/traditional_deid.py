import re

# Define regex patterns for each PHI category
ENTITY_PATTERNS = {
    "PHONE": r"\b(?:\+?\d{1,4}[-.\s]+)?(?:\(?\d{2,4}\)?[-.\s]+)?\d{2,4}[-.\s]+\d{2,4}(?:[-.\s]+\d{2,4})?\b",
    "ID": r"\b\d{6,}\b",  # Match IDs that are 6+ digits, like MRNs
    "DATE": r"\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b",
    "EMAIL": r"\b[\w.-]+@[\w.-]+\.\w+\b",
    "NAME": r"\b(Mr\.|Ms\.|Mrs\.|Dr\.) [A-Z][a-z]+(?: [A-Z][a-z]+)?\b",
    "HOSPITAL": r"\b(?:St\.?\s)?[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Medical Center|Hospital|Clinic)\b"
}


def rule_based_deid(text):
    annotations = []
    redacted_text = text

    for label, pattern in ENTITY_PATTERNS.items():
        matches = list(re.finditer(pattern, text))
        for match in matches:
            entity_text = match.group()
            start = match.start()
            end = match.end()
            replacement = f"[{label}]"

            annotations.append((entity_text, label))
            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]
            text = redacted_text  # update for next match to not shift indices

    return redacted_text, annotations
