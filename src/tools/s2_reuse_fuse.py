"""Stage-2 on a NEW stage-1 pool by REUSING existing verify caches, image-keyed.

LLM verify(query, image) is position-independent, so a verify score cached for one stage-1
run is valid for the same (query, image) pair under a different pool (e.g. fuse_gated_best).
This re-keys old caches by image name and reranks the new pool's top-`round` exactly like
sweep_stage2.rank_for (gate: minmax top-1 struct <= xi; reorder by lam*minmax(struct)+(1-lam)*sem).

Missing (query,image) pairs (candidates never verified by any cache) are handled per --missing:
  no       -> sem=0.0  (as if LLM said No)  -> pessimistic LOWER bound
  neutral  -> sem=minmax(struct)_i          -> LLM leaves it at struct rank -> optimistic upper-ish
It also dumps the missing top-`round` pairs to <out>/missing_pairs.json for a targeted Qwen fill.

  python3 tools/s2_reuse_fuse.py --run fuse_gated_best \
     --caches s2cache_w03_r10_fp16 s2cache_w03_r20 --missing neutral --dump_missing 10
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from eval_submission import score  # noqa

stem = lambda x: os.path.splitext(x)[0]


def build_img_sem(cache_runs, out_root):
    """merged {query: {image_stem: sem}} from each cache, aligning its sem list to its own run's pool."""
    merged = {}
    for cr in cache_runs:
        cache = json.load(open(os.path.join(out_root, cr, 'verify_cache.json')))
        run = cache['run']
        sem = cache['sem']
        rs = json.load(open(os.path.join(out_root, run, 'scores.json')))
        for q, lst in sem.items():
            if q not in rs:
                continue
            cand = [stem(n) for n, _ in rs[q][:len(lst)]]
            d = merged.setdefault(q, {})
            for img, sc in zip(cand, lst):
                d[img] = sc          # last cache wins on overlap (same LLM, identical)
    return merged


def rank_for(scores, img_sem, norm1, qids, xi, lam, rnd, missing, top_n=10):
    submission = {}
    for q in qids:
        pool = [n for n, _ in scores[q]]
        if norm1[q] <= xi and q in img_sem:
            k = min(rnd, len(pool))
            seg = np.array([s for _, s in scores[q][:k]], dtype=np.float64)
            sn = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
            sem = np.empty(k)
            for i in range(k):
                v = img_sem[q].get(stem(pool[i]))
                if v is None:
                    sem[i] = 0.0 if missing == 'no' else sn[i]   # bound handling
                else:
                    sem[i] = v
            s_final = lam * sn + (1 - lam) * sem
            new_idx = list(np.argsort(-s_final))
            pool = [pool[i] for i in new_idx] + pool[k:]
        submission[q] = pool[:top_n]
    return submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='fuse_gated_best')
    ap.add_argument('--caches', nargs='+', required=True)
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--xis', default='0.3,0.4,0.5,0.6,0.674,0.75,0.85,1.0')
    ap.add_argument('--lams', default='0.2,0.3,0.4,0.5,0.6')
    ap.add_argument('--rounds', default='5,10')
    ap.add_argument('--missing', choices=['no', 'neutral'], default='neutral')
    ap.add_argument('--dump_missing', type=int, default=0,
                    help='if >0, write missing top-R (query,image) pairs to <run>/missing_pairs.json')
    a = ap.parse_args()

    scores = json.load(open(os.path.join(a.out_root, a.run, 'scores.json')))
    gt = json.load(open(a.gt))
    qids = list(scores.keys())
    img_sem = build_img_sem(a.caches, a.out_root)

    top1 = np.array([scores[q][0][1] for q in qids], dtype=np.float64)
    nrm = (top1 - top1.min()) / (top1.max() - top1.min() + 1e-9)
    norm1 = {q: float(n) for q, n in zip(qids, nrm)}

    base = score({q: [n for n, _ in scores[q]][:10] for q in qids}, gt)
    print(f"stage-1 baseline ({a.run}): R@1={base['R@1']} R@5={base['R@5']} R@10={base['R@10']} mAP={base['mAP']}  (n={len(qids)})")
    print(f"missing-mode={a.missing}\n")

    xis = [float(x) for x in a.xis.split(',')]
    lams = [float(x) for x in a.lams.split(',')]
    rounds = [int(x) for x in a.rounds.split(',')]

    rows = []
    for xi in xis:
        for rnd in rounds:
            for lam in lams:
                m = score(rank_for(scores, img_sem, norm1, qids, xi, lam, rnd, a.missing), gt)
                rows.append((m['R@1'], m['mAP'], m['R@5'], m['R@10'], xi, lam, rnd))
    rows.sort(reverse=True)
    print(f"{'R@1':>6} {'mAP':>6} {'R@5':>6} {'R@10':>6} | {'xi':>5} {'lam':>4} {'rnd':>3}")
    for r in rows[:12]:
        print(f"{r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f} | {r[4]:5.3f} {r[5]:4.1f} {r[6]:3d}")
    b = rows[0]
    print(f"\nBEST R@1={b[0]} mAP={b[1]} R@10={b[3]}  @ xi={b[4]} lam={b[5]} round={b[6]}  (missing={a.missing})")

    if a.dump_missing:
        R = a.dump_missing
        miss = {}
        for q in qids:
            need = []
            for n, _ in scores[q][:R]:
                if stem(n) not in img_sem.get(q, {}):
                    need.append(n)
            if need:
                miss[q] = need
        outp = os.path.join(a.out_root, a.run, 'missing_pairs.json')
        json.dump(miss, open(outp, 'w'))
        print(f"# missing top-{R}: {sum(len(v) for v in miss.values())} pairs over {len(miss)} queries -> {outp}")


if __name__ == '__main__':
    main()
