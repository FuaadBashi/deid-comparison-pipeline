#!/usr/bin/env python3
import os
import json
import pandas as pd
from lxml import etree

# -- CONFIGURE THESE --
XML_FILES = {
    'test_groundtruth': 'deid_surrogate_test_all_groundtruth_version2.xml',
    'test'             : 'deid_surrogate_test_all_version2.xml',
    'train'            : 'deid_surrogate_train_all_version2.xml'
}

OUTPUT_DIR = '.'        # where to write the CSV/JSON
# ----------------------

def parse_records(xml_path):
    """
    Parse a de‑identified XML report file into a flat list of dicts:
      { 'record_id': ..., 'phi_type': ..., 'phi_text': ... }
    """
    parser = etree.XMLParser(recover=True)  # recover malformed bits
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    rows = []
    for rec in root.findall('.//RECORD'):
        rec_id = rec.get('ID')
        text_el = rec.find('TEXT')
        if text_el is None:
            continue
        # collect every PHI tag
        phis = text_el.findall('.//PHI')
        if phis:
            for phi in phis:
                rows.append({
                    'record_id': rec_id,
                    'phi_type' : phi.get('TYPE'),
                    'phi_text' : (phi.text or '').strip()
                })
        else:
            # if no <PHI>, capture all text
            all_text = ''.join(text_el.itertext()).strip().replace('\n', ' ')
            rows.append({
                'record_id': rec_id,
                'phi_type' : None,
                'phi_text' : all_text
            })
    return pd.DataFrame(rows)

def main():
    for tag, fname in XML_FILES.items():
        in_path  = os.path.abspath(fname)
        csv_out  = os.path.join(OUTPUT_DIR, f"{tag}.csv")
        json_out = os.path.join(OUTPUT_DIR, f"{tag}.json")

        print(f"Parsing {in_path} → {csv_out}, {json_out}")
        df = parse_records(in_path)
        df.to_csv(csv_out, index=False)
        df.to_json(json_out, orient='records', lines=False, force_ascii=False)

    print("Done! Your CSVs/JSONs are in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
