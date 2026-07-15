"""Pre-write a frozen, degraded copy of the val gallery to disk (one-time).

Why: degrade-axis eval (--degrade-gallery in run_eval) re-degrades every image on the fly in a
SINGLE process (nw=0, for determinism), paying ~8 min of WebP decode each run and blocking the
GPU. Freezing the degraded gallery once lets eval read ready-made low-res/JPEG images with
workers>0 (fast) while staying deterministic (frozen on disk + per-image seed).

Runs on the HOST (pure PIL, no torch): reads source train images, writes the degraded JPEGs +
a manifest. POSE is NOT degraded (it is a rendered skeleton) -> the manifest keeps the original
relative path so run_eval still resolves pose from the untouched pose tree.

Example (host, from repo root .../vannk):
  python SSDC-lo/tools/synthetic_eval/dump_degraded_gallery.py \
    --gallery  SSDC-luan/data/synthetic_eval/val_gallery.jsonl \
    --src-root dataset \
    --out-dir  dataset/val_gallery \
    --manifest SSDC-luan/data/synthetic_eval/val_gallery_degraded.jsonl \
    --workers 32

Consume in run_eval (after the degraded dir is reachable by the container):
  python tools/synthetic_eval/run_eval.py --split noisy \
    --gallery-jsonl data/synthetic_eval/val_gallery_degraded.jsonl \
    --rgb-root data/PAB --config-tag cmp_degrade   # do NOT pass --degrade-gallery
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degrade import degrade


def _one(args):
    rec, src_root, out_dir, rgb_prefix, seed, smin, smax, max_aspect, p_landscape = args
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    iid = rec["image_id"]
    src = os.path.join(src_root, rec["image"])
    try:
        img = Image.open(src).convert("RGB")
        # landscape-biased center crop (non-square, true proportions) + low-res + JPEG;
        # CMP's square Resize(224^2) then warps it once, like a real test image
        img, q = degrade(img, iid, seed=seed, smin=smin, smax=smax,
                         max_aspect=max_aspect, p_landscape=p_landscape)
        img.save(os.path.join(out_dir, f"{iid}.jpg"), "JPEG", quality=q)
    except Exception as e:
        return {"image_id": iid, "_err": f"{src}: {e}"}
    # keep original 'image' (pose lookup) + degraded 'rgb' (relative to run_eval --rgb-root)
    return {"image": rec["image"], "image_id": iid, "rgb": f"{rgb_prefix}/{iid}.jpg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="SSDC-luan/data/synthetic_eval/val_gallery.jsonl")
    ap.add_argument("--src-root", default="dataset",
                    help="prefix joined to each manifest 'image' to find the source")
    ap.add_argument("--out-dir", default="dataset/val_gallery",
                    help="where degraded JPEGs are written (<image_id>.jpg)")
    ap.add_argument("--manifest", default="SSDC-luan/data/synthetic_eval/val_gallery_degraded.jsonl")
    ap.add_argument("--rgb-prefix", default="val_gallery",
                    help="path stored in manifest 'rgb', relative to run_eval --rgb-root")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--smin", type=int, default=128,
                    help="downscale short side min (test gallery short-side median ~180)")
    ap.add_argument("--smax", type=int, default=256, help="downscale short side max")
    ap.add_argument("--max-aspect", type=float, default=1.6,
                    help="max crop aspect (test median 1.50, p90 1.78); landscape crop removes "
                         "top/bottom background so detail survives. Raise to 1.78 for full test "
                         "fidelity, lower for safety.")
    ap.add_argument("--p-landscape", type=float, default=0.7,
                    help="fraction landscape crops (test gallery is ~70% landscape / 16:9)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (for a quick timing probe)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    recs = [json.loads(l) for l in open(a.gallery) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    tasks = [(r, a.src_root, a.out_dir, a.rgb_prefix, a.seed, a.smin, a.smax,
              a.max_aspect, a.p_landscape) for r in recs]
    print(f"### degrading {len(tasks)} imgs  src={a.src_root}  out={a.out_dir}  workers={a.workers}", flush=True)

    t0 = time.time()
    rows, errs, done = [], [], 0
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(_one, tasks, chunksize=16):
            done += 1
            (errs if "_err" in res else rows).append(res.get("_err") or res)
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)

    md = os.path.dirname(a.manifest)
    if md:
        os.makedirs(md, exist_ok=True)
    with open(a.manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    print(f"### wrote {len(rows)} degraded imgs -> {a.out_dir}  "
          f"({dt:.0f}s, {len(rows)/max(dt,1e-9):.0f} img/s)", flush=True)
    print(f"### manifest -> {a.manifest}  (rgb -> {a.rgb_prefix}/<id>.jpg, pose unchanged)", flush=True)
    if errs:
        print(f"### {len(errs)} errors. First: {errs[:5]}", flush=True)


if __name__ == "__main__":
    main()
