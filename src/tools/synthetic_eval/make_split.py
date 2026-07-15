import json, random, argparse, glob
from collections import defaultdict
from tools.synthetic_eval.id_map import load_anns

def assign_split(records, val_frac=None, seed=20260622, n_val_sources=None):
    if val_frac is None and n_val_sources is None:
        raise ValueError("assign_split: provide either val_frac or n_val_sources")
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source_id"]].append(r["image_id"])
    total = sum(len(v) for v in by_src.values())
    srcs = sorted(by_src.keys())
    random.Random(seed).shuffle(srcs)
    val_srcs, n_val = [], 0
    if n_val_sources is not None:
        val_srcs = srcs[:n_val_sources]
        n_val = sum(len(by_src[s]) for s in val_srcs)
    else:
        target_val = int(round(total * val_frac))
        for s in srcs:
            if n_val >= target_val: break
            val_srcs.append(s); n_val += len(by_src[s])
    val_set = set(val_srcs)
    train_srcs = [s for s in srcs if s not in val_set]
    return {
        "seed": seed, "val_frac": val_frac if n_val_sources is None else None,
        "val_source_ids": sorted(val_srcs),
        "train_source_ids": sorted(train_srcs),
        "n_val_images": n_val,
        "n_train_images": total - n_val,
        "n_val_sources": len(val_srcs),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="data/PAB/annotation/train")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--n-val-sources", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="data/synthetic_eval/split_ids.json")
    a = ap.parse_args()
    recs = load_anns(sorted(glob.glob(f"{a.ann_dir}/attr_*.json")))
    split = assign_split(recs, val_frac=a.val_frac, seed=a.seed, n_val_sources=a.n_val_sources)
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: json.dump(split, f, indent=2)
    print(f"val sources={len(split['val_source_ids'])} val imgs={split['n_val_images']} "
          f"train sources={len(split['train_source_ids'])} train imgs={split['n_train_images']} -> {a.out}")
