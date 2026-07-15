"""Meaning-gate (stage B): keep a noisy variant only if its CMP-text cosine to its own clean
caption is >= gate (default 0.85). A rewrite that drifted (changed/dropped/added content) would
make the query no longer describe its gt image -> corrupt the eval. Runs in ssdc-eval (CMP needs
transformers 4.x).

  python tools/synthetic_eval/filter_noisy.py \
    --clean      data/synthetic_eval/val_queries_clean.jsonl \
    --candidates data/synthetic_eval/val_queries_noisy_candidates.jsonl \
    --out        data/synthetic_eval/val_queries_noisy.jsonl --gate 0.85

Candidates may carry K variants per image_id (gen_noisy_val --variants K), joined to their clean
caption by image_id. Per query: keep the passing variants; drop a query's failing variants if it
still has a passing one. If ALL of a query's variants drift, --on-drift decides (default 'clean':
replace with the clean caption -> coverage kept, gt NOT corrupted; the row is relabelled
level='clean' so the difficulty bins stay pure).
"""
import os
import sys
import json
import argparse
import collections
import numpy as np
import torch

# repo root on sys.path so `tools.synthetic_eval.*` imports work from any cwd (like run_eval)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tools.synthetic_eval.run_eval import encode_text_feats, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/ssdc_openpose_ver2.yaml')
    ap.add_argument('--checkpoint', default='checkpoint/cmp.pth')
    ap.add_argument('--clean', default='data/synthetic_eval/val_queries_clean.jsonl')
    ap.add_argument('--candidates', default='data/synthetic_eval/val_queries_noisy_candidates.jsonl')
    ap.add_argument('--out', default='data/synthetic_eval/val_queries_noisy.jsonl')
    ap.add_argument('--gate', type=float, default=0.85, help="min CMP-text cosine(clean, noisy) to keep")
    ap.add_argument('--on-drift', choices=['clean', 'drop', 'keep'], default='clean',
                    help="when ALL of a query's variants drift below gate: 'clean'=replace with the "
                         "clean caption (keep coverage, no gt corruption, relabel level='clean'); "
                         "'drop'=remove the query; 'keep'=use the drifted rewrite anyway (corrupts gt)")
    a = ap.parse_args()

    from ruamel.yaml import YAML
    from transformers import BertTokenizer
    from models.model_search import Search

    config = YAML(typ='safe').load(open(a.config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Search(config=config)
    model.load_pretrained(a.checkpoint)
    model = model.to(device).eval()
    tok = BertTokenizer.from_pretrained(config['text_encoder'])
    mw = config.get('max_words', 56)

    clean_by_id = {r['image_id']: r['caption'] for r in load_jsonl(a.clean)}
    cand = load_jsonl(a.candidates)
    miss = [c['image_id'] for c in cand if c['image_id'] not in clean_by_id]
    if miss:
        print(f"  !! {len(miss)} candidates have no clean match (first {miss[:3]})", flush=True)
        cand = [c for c in cand if c['image_id'] in clean_by_id]

    print(f"### encoding {len(cand)} (clean,noisy) pairs in CMP text space...", flush=True)
    cf = encode_text_feats(model, tok, [clean_by_id[c['image_id']] for c in cand], device, mw)
    nf = encode_text_feats(model, tok, [c['caption'] for c in cand], device, mw)
    cos = (cf * nf).sum(1).numpy()
    keep = cos >= a.gate

    by_id = collections.defaultdict(list)
    for i, c in enumerate(cand):
        by_id[c['image_id']].append(i)

    out_rows, n_pass, n_fb, n_drop = [], 0, 0, 0
    for iid, idxs in by_id.items():
        passing = [i for i in idxs if keep[i]]
        if passing:                                   # keep passing variants; drop this query's failing ones
            for i in passing:
                r = dict(cand[i]); r['cos'] = round(float(cos[i]), 4)
                out_rows.append(r); n_pass += 1
        else:                                         # all variants drifted -> on-drift policy
            best = max(idxs, key=lambda i: cos[i])
            if a.on_drift == 'drop':
                n_drop += 1
            elif a.on_drift == 'keep':
                r = dict(cand[best]); r['cos'] = round(float(cos[best]), 4)
                out_rows.append(r); n_pass += 1
            else:                                     # 'clean': use the clean caption, relabel as a clean bin
                out_rows.append({'caption': clean_by_id[iid], 'image_id': iid,
                                 'style': 'clean-fallback', 'level': 'clean',
                                 'cos': round(float(cos[best]), 4)})
                n_fb += 1

    with open(a.out, 'w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n### gate={a.gate}  on-drift={a.on_drift}", flush=True)
    print(f"### kept noisy: {n_pass}  clean-fallback: {n_fb}  dropped: {n_drop}  "
          f"-> {len(out_rows)} rows, {len({r['image_id'] for r in out_rows})}/{len(clean_by_id)} queries covered", flush=True)
    # per ORIGINAL level: pass rate (how much each difficulty drifts) -- the real diagnostic
    lv = collections.defaultdict(lambda: [0, 0, []])
    for i, c in enumerate(cand):
        l = c.get('level', '?')
        lv[l][0] += int(keep[i]); lv[l][1] += 1; lv[l][2].append(cos[i])
    print("### per-level  pass%(cos>=gate)  mean-cos")
    for l in sorted(lv):
        k, n, cs = lv[l]
        print(f"  [{l:7s}] {100*k/n:5.1f}%  cos={np.mean(cs):.3f}  (n={n})", flush=True)


if __name__ == '__main__':
    main()
