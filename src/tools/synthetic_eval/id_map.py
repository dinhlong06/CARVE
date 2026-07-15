import json

def load_anns(ann_paths):
    rows = []
    for p in ann_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line: rows.append(json.loads(line))
    return rows

def build_id_source_map(ann_paths):
    m = {}
    for r in load_anns(ann_paths):
        if "image_id" in r and "source_id" in r:
            m[r["image_id"]] = r["source_id"]
        if r.get("hard_i_id") and r.get("hard_source_id"):
            m[r["hard_i_id"]] = r["hard_source_id"]
    return m
