from lxml import etree
import re

# FILES
input_file = "deid_surrogate_train_all_version2.xml"
output_file = "deid_train_iob2.csv"

NAME_TYPES = {"PATIENT", "DOCTOR"}
ALL_LABELS = ["PATIENT", "DOCTOR", "HOSPITAL", "ID", "DATE", "PHONE", "LOCATION", "AGE"]

def group_label(label):
    if label in NAME_TYPES:
        return "NAME"
    return label

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def process_text_element(elem):
    # Walks through text + PHI children in order, using lxml
    for chunk in elem.iter():
        if chunk is elem:
            if chunk.text:
                for token in tokenize(chunk.text):
                    yield token, "O"
        elif chunk.tag == "PHI":
            label = group_label(chunk.attrib.get("TYPE", "O"))
            phi_tokens = tokenize(chunk.text or "")
            for i, token in enumerate(phi_tokens):
                prefix = "B-" if i == 0 else "I-"
                yield token, f"{prefix}{label}"
            if chunk.tail:
                for token in tokenize(chunk.tail):
                    yield token, "O"

def xml_to_iob2(input_file, output_file):
    print(f"Parsing {input_file} with error recovery ...")
    root = etree.parse(input_file, etree.XMLParser(recover=True)).getroot()
    records = root.findall(".//RECORD")
    all_rows = []
    for rec in records:
        text_elems = rec.findall(".//TEXT")
        for t in text_elems:
            toks_labels = list(process_text_element(t))
            if toks_labels:
                all_rows.append(('', ''))  # blank line to separate documents
                all_rows.extend(toks_labels)
    print(f"Writing to {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        for tok, lab in all_rows:
            if tok == '' and lab == '':
                f.write("\n")
            else:
                f.write(f"{tok},{lab}\n")

if __name__ == "__main__":
    xml_to_iob2(input_file, output_file)
    print("Done! Output:", output_file)
