import json, os, glob, random, argparse
from tools.synthetic_eval.id_map import load_anns

def select_val_records(anns, val_source_ids):
    vs = set(val_source_ids)
    return [r for r in anns if r["source_id"] in vs]

def select_bucket(anns, val_source_ids, bucket):
    """bucket='val' -> records whose source IS held out; 'train' -> the complement."""
    vs = set(val_source_ids)
    if bucket == "val":
        return [r for r in anns if r["source_id"] in vs]
    return [r for r in anns if r["source_id"] not in vs]

def subsample_per_source(records, cap, seed):
    from collections import defaultdict
    import random
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source_id"]].append(r)
    rng = random.Random(seed)
    out = []
    for s in sorted(by_src):
        recs = by_src[s]
        out.extend(rng.sample(recs, cap) if len(recs) > cap else recs)
    return out

def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    # Builds a bucket's IMAGES (gallery) + CLEAN captions. Works for both val (held-out sources)
    # and the train finetune subset (the complement). Noisy captions are produced separately by
    # tools/synthetic_eval/gen_noisy_val.py (Qwen LLM style-rewrites) on the *_queries_clean.jsonl.
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="data/PAB/annotation/train")
    ap.add_argument("--split", default="data/synthetic_eval/split_ids.json")
    ap.add_argument("--out-dir", default="data/synthetic_eval")
    ap.add_argument("--bucket", choices=["train", "val"], default="val")
    ap.add_argument("--out-prefix", default="", help="output filename prefix (default = bucket name)")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--per-source-cap", type=int, default=15,
                    help="max images kept per source (diversity / cost control)")
    ap.add_argument("--max-records", type=int, default=0,
                    help="0 = all; else a global random cap (use to size the train finetune subset)")
    a = ap.parse_args()
    prefix = a.out_prefix or a.bucket
    split = json.load(open(a.split))
    anns = load_anns(sorted(glob.glob(f"{a.ann_dir}/attr_*.json")))
    recs = select_bucket(anns, split["val_source_ids"], a.bucket)
    recs = subsample_per_source(recs, a.per_source_cap, a.seed)
    if a.max_records and len(recs) > a.max_records:
        recs = random.Random(a.seed).sample(recs, a.max_records)
    gallery = [{"image": r["image"], "image_id": r["image_id"]} for r in recs]
    clean   = [{"caption": r["caption"], "image_id": r["image_id"]} for r in recs]
    _write(f"{a.out_dir}/{prefix}_gallery.jsonl", gallery)
    _write(f"{a.out_dir}/{prefix}_queries_clean.jsonl", clean)
    print(f"[{a.bucket}] gallery={len(gallery)} clean={len(clean)} -> {a.out_dir}/{prefix}_*.jsonl  "
          f"(noisy: gen_noisy_val.py --clean {a.out_dir}/{prefix}_queries_clean.jsonl "
          f"--out {a.out_dir}/{prefix}_queries_noisy.jsonl)")
