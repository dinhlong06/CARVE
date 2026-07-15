"""Measure semantic drift between clean and noisy val queries, in the CMP text-encoder space.

For each (clean, noisy) pair (aligned by line order; image_id verified), compute the cosine of
their L2-normalised CMP text features. Reports the distribution, per-style means, and the worst
(lowest-cosine) pairs to eyeball whether the LLM rewrite drifted (changed/dropped/added content)
or is a faithful restyle. Run inside ssdc-eval, cwd /workspace/SSDC.
"""
import json
import argparse
import collections
import numpy as np
import torch
from tools.synthetic_eval.run_eval import encode_text_feats, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/ssdc_openpose_ver2.yaml')
    ap.add_argument('--checkpoint', default='checkpoint/cmp.pth')
    ap.add_argument('--clean', default='data/synthetic_eval/val_queries_clean.jsonl')
    ap.add_argument('--noisy', default='data/synthetic_eval/val_queries_noisy.jsonl')
    ap.add_argument('--worst', type=int, default=18)
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

    clean = load_jsonl(a.clean)
    noisy = load_jsonl(a.noisy)
    assert len(clean) == len(noisy), f"len mismatch {len(clean)} vs {len(noisy)}"
    mism = sum(1 for c, n in zip(clean, noisy) if c['image_id'] != n['image_id'])
    print(f"### pairs={len(clean)}  image_id mismatches={mism}", flush=True)

    cf = encode_text_feats(model, tok, [c['caption'] for c in clean], device, mw)  # [N,D] L2-normed
    nf = encode_text_feats(model, tok, [n['caption'] for n in noisy], device, mw)
    cos = (cf * nf).sum(1).numpy()  # per-pair cosine

    print(f"\n### cosine clean<->noisy (CMP text space)")
    print(f"  mean={cos.mean():.3f}  p50={np.percentile(cos,50):.3f}  "
          f"p10={np.percentile(cos,10):.3f}  p1={np.percentile(cos,1):.3f}  min={cos.min():.3f}")
    for thr in (0.95, 0.9, 0.85, 0.8, 0.7):
        print(f"  fraction < {thr}: {100*(cos < thr).mean():.1f}%")

    by = collections.defaultdict(list)
    for n, c in zip(noisy, cos):
        by[n.get('style', '?')].append(c)
    print("\n### per-style mean cosine")
    for s in sorted(by):
        v = np.array(by[s])
        print(f"  [{s:11s}] mean={v.mean():.3f}  %<0.8={100*(v<0.8).mean():.1f}  (n={len(v)})")

    print(f"\n### {a.worst} WORST (lowest-cosine) pairs")
    for i in np.argsort(cos)[:a.worst]:
        print(f"cos={cos[i]:.3f} [{noisy[i].get('style')}]")
        print(f"  CLEAN: {clean[i]['caption'][:170]}")
        print(f"  NOISY: {noisy[i]['caption'][:170]}")


if __name__ == '__main__':
    main()
