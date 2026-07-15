"""GATED RRF fusion (no training): scale the ext (ViT-H/SigLIP) RRF weight PER QUERY by CMP
confidence. CMP top1-top2 margin is a strong confidence signal (low-margin queries: CMP top-1
acc 41.6%; high-margin: 94.7%). So lean on ext where CMP is unsure, protect CMP where it is
confident -> aim to capture high-w recall WITHOUT the R@1 cost of uniform high w.

Stage-1 ONLY (no LLM). Evals top-10 vs gt.json. Sweeps (w_lo, w_hi, gate shape) cheaply over
precomputed ranks. Uniform RRF = w_lo==w_hi (reproduces fuse_cmp512_vith_siglip_w03 at 0.3).

  python3 tools/fuse_gated.py --cmp submissions/stage1_deepitm/scores.json \
     --ext submissions/ext_openclip_vith/features --ext submissions/ext_siglip/features
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from eval_submission import score  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmp', default='submissions/stage1_deepitm/scores.json')
    ap.add_argument('--ext', action='append', required=True)
    ap.add_argument('--gt', default='submissions/gt.json')
    ap.add_argument('--krrf', type=int, default=60)
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--write_best', default=None, help='run name to materialize the mAP-best config')
    ap.add_argument('--write_best_r1', default=None, help='run name to materialize the R@1-best config')
    ap.add_argument('--w_lo_grid', type=float, nargs='+', default=[0.0, 0.05, 0.1])
    ap.add_argument('--w_hi_grid', type=float, nargs='+', default=[0.3, 0.35, 0.4, 0.45, 0.5])
    ap.add_argument('--gamma_grid', type=float, nargs='+', default=[1.5, 2.0, 2.5, 3.0])
    a = ap.parse_args()

    cmp_scores = json.load(open(os.path.join(ROOT, a.cmp)))
    gt = json.load(open(os.path.join(ROOT, a.gt)))
    stem = lambda x: os.path.splitext(x)[0]

    ext_data = []
    for ed in a.ext:
        idx = json.load(open(os.path.join(ROOT, ed, 'index.json')))
        names = [stem(x) for x in idx['names']]
        sims = np.load(os.path.join(ROOT, ed, 'sims_ext.npy'))
        ext_data.append({'names': names, 'sims': sims, 'qidx': idx['qidx'],
                         'col': {n: i for i, n in enumerate(names)}})
    names = ext_data[0]['names']; col = ext_data[0]['col']; ng = len(names)
    k = a.krrf
    qids = [q for q in cmp_scores if q in set(ext_data[0]['qidx'])]

    # CMP sparse RRF term + confidence (top1-top2 margin -> percentile)
    cmp_rrf = {}
    margin = np.zeros(len(qids))
    for i, qi in enumerate(qids):
        lst = cmp_scores[qi]
        d = {}
        for r, (nm, _) in enumerate(lst):
            c = col.get(stem(nm))
            if c is not None:
                d[c] = 1.0 / (k + r + 1)
        cmp_rrf[qi] = d
        margin[i] = lst[0][1] - lst[1][1]
    pct = margin.argsort().argsort() / (len(qids) - 1)        # 0=lowest margin .. 1=highest
    p = {qi: float(pct[i]) for i, qi in enumerate(qids)}

    # ext dense RRF term (summed over members), per query
    ext_rrf = {}
    for ed in ext_data:
        order = np.argsort(-ed['sims'], axis=1)
        rankmat = np.empty(order.shape, dtype=np.int32)
        rows = np.arange(order.shape[0])[:, None]
        rankmat[rows, order] = np.arange(ng, dtype=np.int32)[None, :] + 1
        qr = {ed['qidx'][i]: rankmat[i] for i in range(len(ed['qidx']))}
        for qi in qids:
            term = 1.0 / (k + qr[qi].astype(np.float64))
            ext_rrf[qi] = term if qi not in ext_rrf else ext_rrf[qi] + term

    def rank_cfg(w_lo, w_hi, gamma):
        sub = {}
        for qi in qids:
            wq = w_lo + (w_hi - w_lo) * (1.0 - p[qi]) ** gamma  # high w at low margin (p small)
            s = wq * ext_rrf[qi]
            for c, v in cmp_rrf[qi].items():
                s[c] += v
            top = np.argpartition(-s, 10)[:10]
            top = top[np.argsort(-s[top])]
            sub[qi] = [names[c] + '.jpg' for c in top]
        return sub

    print(f"n={len(qids)}  (gate: wq = w_lo + (w_hi-w_lo)*(1-marginPctile)^gamma)")
    print(f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'mAP':>6} | {'w_lo':>5} {'w_hi':>5} {'gam':>4}  note")
    rows = []
    # uniform baselines first
    for w in [0.0, 0.3, 0.5, 0.7]:
        m = score(rank_cfg(w, w, 1.0), gt)
        tag = 'uniform' + (' (=base)' if w == 0.3 else (' CMP-only' if w == 0 else ''))
        print(f"{m['R@1']:6.2f} {m['R@5']:6.2f} {m['R@10']:6.2f} {m['mAP']:6.2f} | {w:5.2f} {w:5.2f} {1.0:4.1f}  {tag}")
    # gated grid
    for w_lo in a.w_lo_grid:
        for w_hi in a.w_hi_grid:
            for gamma in a.gamma_grid:
                m = score(rank_cfg(w_lo, w_hi, gamma), gt)
                rows.append((m['R@1'], m['mAP'], m['R@5'], m['R@10'], w_lo, w_hi, gamma))
                print(f"{m['R@1']:6.2f} {m['R@5']:6.2f} {m['R@10']:6.2f} {m['mAP']:6.2f} | {w_lo:5.2f} {w_hi:5.2f} {gamma:4.1f}  gated")
    rows.sort(reverse=True)
    b = rows[0]
    print(f"\nBEST(by R@1): R@1={b[0]} mAP={b[1]} R@5={b[2]} R@10={b[3]}  @ w_lo={b[4]} w_hi={b[5]} gamma={b[6]}")
    bm = max(rows, key=lambda r: r[1])
    print(f"BEST(by mAP): R@1={bm[0]} mAP={bm[1]} R@10={bm[3]}  @ w_lo={bm[4]} w_hi={bm[5]} gamma={bm[6]}")
    br = max(rows, key=lambda r: r[3])
    print(f"BEST(by R@10): R@1={br[0]} R@10={br[3]} mAP={br[1]}  @ w_lo={br[4]} w_hi={br[5]} gamma={br[6]}")

    def materialize(name, w_lo, w_hi, gamma, crit):
        sub = rank_cfg(w_lo, w_hi, gamma)
        # full pool_k scores for downstream
        full = {}
        for qi in qids:
            wq = w_lo + (w_hi - w_lo) * (1.0 - p[qi]) ** gamma
            s = wq * ext_rrf[qi]
            for c, v in cmp_rrf[qi].items():
                s[c] += v
            order = np.argsort(-s)[:128]
            full[qi] = [[names[c] + '.jpg', float(s[c])] for c in order]
        d = os.path.join(ROOT, a.out_root, name); os.makedirs(d, exist_ok=True)
        json.dump({qi: [n for n, _ in full[qi]][:10] for qi in qids}, open(d + '/submission.json', 'w'))
        json.dump(full, open(d + '/scores.json', 'w'))
        open(d + '/answer.txt', 'w').write('\n'.join(' '.join(stem(x) for x in sub[qi]) for qi in qids) + '\n')
        m = score(sub, gt)
        json.dump({'name': name, 'method': 'gated RRF (CMP-margin-gated ext weight)',
                   'cmp': a.cmp, 'selected_by': crit,
                   'w_lo': w_lo, 'w_hi': w_hi, 'gamma': gamma, 'krrf': k, 'stage': 1, 'metrics': m},
                  open(d + '/meta.json', 'w'), indent=2)
        print(f"wrote {d}/  ({crit}-best)  metrics={m}")

    if a.write_best:
        materialize(a.write_best, bm[4], bm[5], bm[6], 'mAP')
    if a.write_best_r1:
        materialize(a.write_best_r1, b[4], b[5], b[6], 'R@1')


if __name__ == '__main__':
    main()
